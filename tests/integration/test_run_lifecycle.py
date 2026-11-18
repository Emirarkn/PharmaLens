""": run ids are reserved up front, outputs are published atomically,
a failed or repeated call never touches an earlier run, and the manifest binds the run to its code/config/models."""
from __future__ import annotations

import pytest

from pharmacy_potential import pipeline
from pharmacy_potential.db import connect
from pharmacy_potential.utils import read_json, sha256_file

pytestmark = pytest.mark.integration


def tree_hashes(settings, run_id: str) -> dict[str, str]:
    roots = [settings.reports_root, settings.artifacts_root, settings.outputs_root, settings.data_root / "canonical"]
    return {f"{r.name}/{p.relative_to(r).as_posix()}": sha256_file(p)
            for r in roots for p in sorted((r / run_id).rglob("*")) if p.is_file()}


def run_status(settings, run_id: str):
    con = connect(settings.warehouse_path)
    try:
        row = con.execute("select status from score_run where run_id = ?", [run_id]).fetchone()
        n = con.execute("select count(*) from score_result where run_id = ?", [run_id]).fetchone()[0] \
            if con.execute("select count(*) from information_schema.tables where table_name = 'score_result'"
                           ).fetchone()[0] else 0
        return (row[0] if row else None), n
    finally:
        con.close()


def no_staging(settings, run_id: str) -> bool:
    roots = [settings.reports_root, settings.artifacts_root, settings.outputs_root, settings.data_root / "canonical"]
    return not any((r / "_staging" / run_id).exists() for r in roots)


@pytest.fixture(scope="module")
def first_run(small_universe):
    res = pipeline.run_scoring(small_universe, skip_backtest=True, run_label="lc_ok")
    return small_universe, res


def test_published_run_is_registered_and_manifest_binds_code_config_and_models(first_run):
    s, res = first_run
    assert run_status(s, "lc_ok")[0] == "succeeded" and run_status(s, "lc_ok")[1] > 0
    assert no_staging(s, "lc_ok")
    assert res["excel"].exists() and res["excel"].parent == s.outputs_root / "lc_ok"
    man = read_json(res["reports"] / "manifest.json")
    prov = man["provenance"]
    assert set(prov) >= {"git", "source_tree_sha256", "environment", "config_file_sha256", "effective_config",
                         "effective_config_sha256"}
    assert prov["environment"]["requirements_lock_sha256"] and prov["environment"]["installed_packages_sha256"]
    assert "configs/pilot.yaml" in prov["effective_config"]["files"]
    assert prov["effective_config"]["files"]["configs/pilot.yaml"]["brand"] == "DemoBrand"
    assert set(man["model_files_sha256"]) == {"gbm_q10.txt", "gbm_q50.txt", "gbm_q90.txt", "model_card.json"}
    for name, digest in man["model_files_sha256"].items():
        assert sha256_file(res["artifacts"] / name) == digest


def test_repeating_a_run_id_is_refused_and_published_files_stay_byte_identical(first_run):
    s, _ = first_run
    before = tree_hashes(s, "lc_ok")
    assert before
    with pytest.raises(pipeline.RunExists):
        pipeline.run_scoring(s, skip_backtest=True, run_label="lc_ok")
    assert tree_hashes(s, "lc_ok") == before
    assert run_status(s, "lc_ok")[0] == "succeeded"


def test_failed_run_publishes_nothing_and_leaves_earlier_runs_untouched(first_run, monkeypatch):
    s, _ = first_run
    before = tree_hashes(s, "lc_ok")

    def boom(*a, **k):
        raise RuntimeError("export failed")

    monkeypatch.setattr(pipeline, "export_workbook", boom)
    with pytest.raises(RuntimeError, match="export failed"):
        pipeline.run_scoring(s, skip_backtest=True, run_label="lc_fail")
    assert tree_hashes(s, "lc_fail") == {}
    assert no_staging(s, "lc_fail")
    assert run_status(s, "lc_fail") == ("failed", 0)
    assert tree_hashes(s, "lc_ok") == before
    # A failed id stays reserved: it cannot silently be reused over an unknown partial state.
    monkeypatch.undo()
    with pytest.raises(pipeline.RunExists):
        pipeline.run_scoring(s, skip_backtest=True, run_label="lc_fail")


@pytest.mark.parametrize("label", ["../evil", "a/b", "", "_staging x", "x" * 200])
def test_run_label_cannot_escape_the_output_folders(small_universe, label):
    if label == "":
        pytest.skip("empty label means 'generate an id'")
    with pytest.raises(ValueError, match="invalid run id"):
        pipeline.run_scoring(small_universe, skip_backtest=True, run_label=label)


# ----------------------------------------------------------------------------- reproducibility (K13d, B18)

def test_untouched_run_rebuilds_from_saved_models_and_from_scratch(first_run):
    from pharmacy_potential.reproducibility import verify_run
    s, _ = first_run
    checks = {c["check"]: c for c in verify_run(s, "lc_ok", repeat=2)}
    assert {c["status"] for c in checks.values()} == {"pass"}, checks
    assert {"rescore_with_saved_models_matches_stored", "retrain_matches_stored", "model_files_unchanged",
            "source_unchanged", "config_files_unchanged", "input_hash_unchanged"} <= set(checks)


def test_drift_stops_the_comparison_before_any_rerun(first_run, monkeypatch):
    from pharmacy_potential import reproducibility
    s, _ = first_run
    calls = []
    monkeypatch.setattr(reproducibility, "source_tree_hash", lambda root: "0" * 64)
    monkeypatch.setattr(reproducibility, "run_scoring", lambda *a, **k: calls.append(1))
    checks = {c["check"]: c["status"] for c in reproducibility.verify_run(s, "lc_ok")}
    assert calls == [] and checks["source_unchanged"] == "fail"
    assert checks["retrain_matches_stored"] == "skipped" and checks["rescore_with_saved_models_matches_stored"] == "skipped"


def test_tampered_saved_model_is_detected(first_run):
    from pharmacy_potential.reproducibility import verify_run
    s, _ = first_run
    path = s.artifacts_root / "lc_ok" / "gbm_q50.txt"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        checks = {c["check"]: c["status"] for c in verify_run(s, "lc_ok")}
    finally:
        path.write_bytes(original)
    assert checks["model_files_unchanged"] == "fail" and checks["retrain_matches_stored"] == "skipped"


def test_run_without_provenance_cannot_be_verified(small_universe, tmp_path):
    from pharmacy_potential.reproducibility import verify_run
    from pharmacy_potential.utils import write_json
    s = small_universe
    legacy = s.reports_root / "legacy_run"
    write_json(legacy / "manifest.json", {"run_id": "legacy_run", "input_hash": "x", "score_checksum": "y"})
    try:
        checks = verify_run(s, "legacy_run")
        assert [c["status"] for c in checks] == ["fail"] and checks[0]["check"] == "provenance_recorded"
    finally:
        import shutil
        shutil.rmtree(legacy)
