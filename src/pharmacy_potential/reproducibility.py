"""Can a published run be rebuilt from its own evidence?

Two independent verifications, never mixed:
  (a) re-score with the models SAVED by the run (artifacts/gbm_q*.txt) -> checksum must equal the stored one;
  (b) re-train from the same inputs and re-score -> checksum must equal the stored one.
Both are preceded by drift checks. If the data, the code, the config files or the saved models differ from what
the run recorded, nothing is re-run: the answer is 'fail' (with the re-runs reported as skipped = incomplete),
because a comparison against changed ingredients would prove nothing.
"""
from __future__ import annotations

from typing import Any

from .config import Settings
from .db import connect, input_hash
from .pipeline import run_scoring
from .provenance import file_hashes, source_tree_hash
from .utils import read_json, sha256_file


def _check(name: str, ok: bool | None, detail: Any = "") -> dict[str, Any]:
    return {"check": name, "status": "skipped" if ok is None else ("pass" if ok else "fail"), "detail": detail}


def verify_run(settings: Settings, run_id: str, repeat: int = 2) -> list[dict[str, Any]]:
    run_dir = settings.reports_root / run_id
    art = settings.artifacts_root / run_id
    man = read_json(run_dir / "manifest.json")
    prov = man.get("provenance")
    if not prov:
        return [_check("provenance_recorded", False, "run predates (no provenance): a new run is required")]

    con = connect(settings.warehouse_path)
    try:
        now_input = input_hash(con)
    finally:
        con.close()
    current_files = file_hashes(settings.config_files(), settings.root)
    drift = [
        _check("input_hash_unchanged", now_input == man["input_hash"],
               "" if now_input == man["input_hash"] else "inputs changed since the run: a new run is required"),
        _check("source_unchanged", source_tree_hash(settings.root) == prov["source_tree_sha256"],
               "code differs from the run's code: check out the recorded commit"),
        _check("config_files_unchanged", current_files == prov["config_file_sha256"],
               sorted(k for k in set(current_files) | set(prov["config_file_sha256"])
                      if current_files.get(k) != prov["config_file_sha256"].get(k))),
    ]
    saved = man.get("model_files_sha256", {})
    on_disk = {n: sha256_file(art / n) for n in saved if (art / n).exists()}
    drift.append(_check("model_files_unchanged", on_disk == saved,
                        sorted(n for n in saved if on_disk.get(n) != saved[n])))
    if any(c["status"] == "fail" for c in drift):
        return drift + [_check(n, None, "not run: ingredients changed") for n in
                        ("rescore_with_saved_models_repeatable", "rescore_with_saved_models_matches_stored",
                         "retrain_matches_stored")]

    settings.overrides["score_origin_month"] = man["origin_month"]
    settings.overrides["score_cutoff_utc"] = man["cutoff_utc"]
    common = dict(exploratory=man["exploratory"], skip_backtest=True, champion=man["champion"], write_outputs=False)
    rescored = [run_scoring(settings, run_label=f"repro_saved_{i}", model_dir=art, **common)["score_checksum"]
                for i in range(repeat)]
    retrained = run_scoring(settings, run_label="repro_retrain", **common)["score_checksum"]
    stored = man["score_checksum"]
    return drift + [
        _check("rescore_with_saved_models_repeatable", len(set(rescored)) == 1, rescored),
        _check("rescore_with_saved_models_matches_stored", rescored[0] == stored,
               {"stored": stored, "recomputed": rescored[0]}),
        _check("retrain_matches_stored", retrained == stored, {"stored": stored, "recomputed": retrained}),
    ]
