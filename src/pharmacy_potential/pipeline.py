"""End-to-end orchestration: ingest -> reconcile -> readiness -> demand -> features -> models ->
backtest -> score -> reports -> Excel. Same path for synthetic and (gated) real data."""
from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from . import __version__
from .config import Settings
from .db import connect, input_hash
from .demand.external import build_external_opportunity
from .evaluation.backtest import build_context, choose_champion, run_backtest
from .export.excel import export_workbook
from .features.asof import AsOfPanels
from .features.build import FEATURE_DICTIONARY, asof_audit
from .gates import GateBlocked, check_gates, data_is_synthetic, gate_status  # noqa: F401  (re-exported)
from .ingest.loader import batch_summary, ingest_path
from .models.explain import global_importance, save_summary_plot
from .models.gbm import load_quantile_models
from .provenance import run_provenance
from .readiness.readiness import stock_reconciliation
from .reconcile.canonical import build_canonical, coverage_report, export_canonical, reconciliation_report
from .scoring.score import assemble
from .utils import (frame_checksum, get_logger, hash_files, month_start, parse_utc, read_json, sha256_file,
                    sha256_text, to_naive_utc, utc_now, utc_stamp, write_json)

log = get_logger()


class RunExists(Exception):
    """The run id (or one of its output folders) is already taken; published runs are never overwritten."""


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
OUTPUT_FAMILIES = ("reports", "artifacts", "outputs", "canonical")


# ----------------------------------------------------------------------------- steps

def ingest_all(settings: Settings, con: duckdb.DuckDBPyConnection, root: Path) -> list[dict[str, Any]]:
    results = []
    for ds in settings.get("ingest_order"):
        folder = root / ds
        if not folder.exists():
            log.warning("no files for dataset %s under %s", ds, folder)
            continue
        results += [r.as_dict() for r in ingest_path(settings, con, ds, folder)]
    return results


_input_hash = input_hash   # kept for scripts/verify_reproducibility.py


def _quality_table(con, canon_report: dict, ext_checks: dict, stock: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in batch_summary(con).iterrows():
        rows.append({"section": "ingest", "metric": f"{r['dataset']}: batches/rows read/accepted/quarantined",
                     "value": f"{int(r['batches'])} / {int(r['rows_read'] or 0)} / {int(r['rows_accepted'] or 0)} / "
                              f"{int(r['rows_quarantined'] or 0)}"})
    for code, n in con.execute("select error_code, count(*) from quarantine group by 1 order by 2 desc").fetchall():
        rows.append({"section": "quarantine", "metric": code, "value": str(n)})
    for k in ("rows", "pk_unique", "fk_orphans_gln", "fk_orphans_barcode", "sources_not_summed", "conflict_flag_cells",
              "observed_without_lineage", "canonical_total_net_units", "mutabakat_total_net_units", "vendor_total_net_units"):
        rows.append({"section": "canonical", "metric": k, "value": str(canon_report.get(k))})
    for k, v in canon_report.get("status_counts", {}).items():
        rows.append({"section": "observation_status", "metric": k, "value": str(v)})
    for k, v in canon_report.get("chosen_source_counts", {}).items():
        rows.append({"section": "chosen_source", "metric": k, "value": str(v)})
    for k in ("mutabakat_dedup", "vendor_dedup"):
        for kk, vv in canon_report.get(k, {}).items():
            rows.append({"section": k, "metric": kk, "value": str(vv)})
    for k, v in ext_checks.items():
        rows.append({"section": "external_demand", "metric": k, "value": str(v)})
    if len(stock):
        rows.append({"section": "stock_balance", "metric": "comparable parent-months",
                     "value": str(int((stock["status"] == "comparable").sum()))})
        rows.append({"section": "stock_balance", "metric": "skipped_with_reason",
                     "value": str(int((stock["status"] != "comparable").sum()))})
    return pd.DataFrame(rows)


def _evaluation_table(ev: dict) -> pd.DataFrame:
    rows = []
    for o in ev.get("origins", []):
        for subset in ("overall_excl_holdout", "low_sales_excl_holdout", "holdout_city"):
            sub = o["subsets"].get(subset)
            if not sub:
                continue
            for cand, m in sub["candidates"].items():
                for k in ("P@50", "P@100", "P@250"):
                    rows.append({"origin": o["origin"], "role": o["role"], "subset": subset, "candidate": cand,
                                 "metric": k, "value": m[k]["precision"], "lift": m[k]["lift"], "k_eff": m[k]["k_eff"],
                                 "n_eligible": sub["n"], "positives": sub["positives"]})
        for meth, fm in o["forecast"].items():
            rows.append({"origin": o["origin"], "role": o["role"], "subset": "forecast_excl_holdout",
                         "candidate": meth, "metric": "WAPE", "value": fm["wape"], "lift": None, "k_eff": None,
                         "n_eligible": fm["n"], "positives": None})
    for o in ev.get("origins", []):
        alt = o.get("alt_label_v002")
        for cand, m in ((alt or {}).get("candidates") or {}).items():
            for k in ("P@50", "P@100", "P@250"):
                rows.append({"origin": o["origin"], "role": "validation_alt_label_DRAFT", "subset": "overall_excl_holdout",
                             "candidate": cand, "metric": k, "value": m[k]["precision"], "lift": m[k]["lift"],
                             "k_eff": m[k]["k_eff"], "n_eligible": alt["n"], "positives": alt["positives"]})
    acc = ev.get("acceptance", {})
    for name, c in acc.get("checks", {}).items():
        rows.append({"origin": "test", "role": "G3_check", "subset": "", "candidate": acc.get("champion"),
                     "metric": name, "value": c.get("value"), "lift": None, "k_eff": None,
                     "n_eligible": c.get("n"), "positives": "PASS" if c.get("pass") else "FAIL"})
    rows.append({"origin": "test", "role": "G3_decision_proposal", "subset": "", "candidate": acc.get("champion"),
                 "metric": acc.get("decision_proposal"), "value": None, "lift": None, "k_eff": None, "n_eligible": None,
                 "positives": None})
    return pd.DataFrame(rows)


def evaluation_markdown(ev: dict, run_id: str) -> str:
    acc, ch = ev["acceptance"], ev["champion"]
    lines = [f"# Evaluation report — {run_id}", "",
             "> Labels are **proxies** (sustained excess growth versus the district median). They are not evidence of a causal effect or ROI. "
             "Thresholds are frozen for the demo; G2 owner approval is pending.", "",
             f"- Protokol: `{ev['protocol']['protocol_version']}` hash `{ev['protocol']['protocol_hash'][:12]}`",
             f"- Ranking champion (validation): **{ch['ranking_champion']}**; forecast champion: **{ch['forecast_champion']}**",
             f"- G3 decision proposal (test {ev['protocol']['test_origin']}): **{acc['decision_proposal']}** — gate acceptance belongs to the owner.",
             "", "## P@100 (excluding holdout)", "", "| origin | role | n | positives | " +
             " | ".join(ev["origins"][0]["subsets"]["overall_excl_holdout"]["candidates"].keys()) + " |",
             "|---|---|---|---|" + "---|" * len(ev["origins"][0]["subsets"]["overall_excl_holdout"]["candidates"])]
    for o in ev["origins"]:
        s = o["subsets"]["overall_excl_holdout"]
        vals = []
        for m in s["candidates"].values():
            p, l = m["P@100"]["precision"], m["P@100"]["lift"]
            vals.append("—" if p is None else f"{p:.3f} (lift {l:.2f})" if l else f"{p:.3f}")
        lines.append(f"| {o['origin']} | {o['role']} | {s['n']} | {s['positives']} | " + " | ".join(vals) + " |")
    alts = [o for o in ev["origins"] if o.get("alt_label_v002")]
    if alts:
        cand_names = list(alts[0]["alt_label_v002"]["candidates"])
        lines += ["", "## Alternative label DRAFT (approaches the peer gap; validation only, not computed at the test origin)", "",
                  f"> `{alts[0]['alt_label_v002']['label_version']}` — {alts[0]['alt_label_v002']['approval_status']}. "
                  "Label choice is an owner decision; this table does not feed champion selection or any gate.", "",
                  "| origin | n | positives | " + " | ".join(cand_names) + " |", "|---|---|---|" + "---|" * len(cand_names)]
        for o in alts:
            a = o["alt_label_v002"]
            vals = []
            for c in cand_names:
                p = a["candidates"][c]["P@100"]
                vals.append("—" if p["precision"] is None else f"{p['precision']:.3f}")
            lines.append(f"| {o['origin']} | {a['n']} | {a['positives']} | " + " | ".join(vals) + " |")
    lines += ["", "## Forecast WAPE (next 3 months total, excluding holdout; common universe where every candidate produced a forecast)", "", "| origin | " +
              " | ".join(ev["origins"][0]["forecast"].keys()) + " |", "|---|" + "---|" * len(ev["origins"][0]["forecast"])]
    for o in ev["origins"]:
        lines.append(f"| {o['origin']} | " + " | ".join("—" if v["wape"] is None else f"{v['wape']:.3f}"
                                                       for v in o["forecast"].values()) + " |")
    lines += ["", "## G3 checks", ""]
    for k, c in acc["checks"].items():
        lines.append(f"- {'✅' if c['pass'] else '❌'} `{k}`: " + ", ".join(f"{kk}={vv}" for kk, vv in c.items() if kk != "pass"))
    b = ev["bootstrap_test_P@100_vs_best_simple"]
    lines += ["", f"Bootstrap (district-clustered) champion − {b.get('best_simple')} P@100 difference: mean {b.get('diff_mean')}, "
                  f"95% CI {b.get('diff_ci95')}. {b.get('note', '')}", "",
              "## Source sensitivity (test origin, peer-gap top-100 overlap)", ""]
    for mode, s in ev["source_sensitivity"].items():
        lines.append(f"- {mode}: overlap {s['top100_overlap_with_default']:.2f}, eligible GLNs {s['eligible_glns']} — {s['note']}")
    return "\n".join(lines) + "\n"


TRUTH_CANDIDATES = ("peer_gap", "b_composite", "b_hybrid", "ml_growth", "current_sales", "external_demand",
                    "random_stratified")


def truth_topk(df: pd.DataFrame, score: str, prevalence: float | None, ks=(50, 100, 250)) -> dict[str, Any]:
    """Hidden-truth hit rate of the top-k of one ranking (rows without a score are never ranked)."""
    ranked = df.dropna(subset=[score]).sort_values([score, "_tiebreak", "gln"], ascending=[False, False, True],
                                                   kind="mergesort")
    out: dict[str, Any] = {"n_ranked": int(len(ranked))}
    for k in ks:
        top = ranked.head(k)
        p = float(top["underserved"].mean()) if len(top) else None
        out[f"top{k}"] = {"underserved_precision": p,
                          "lift": (p / prevalence) if (p is not None and prevalence) else None,
                          "mean_realizable_gap": float(top["realizable_gap"].mean()) if len(top) else None}
    return out


def synthetic_truth_check(settings: Settings, result: pd.DataFrame, origin: pd.Timestamp,
                          candidate_scores: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """Demo-only: compare rankings with the hidden truth (never produced for real data).

    Reports the champion track B and external track A as before, and, when ``candidate_scores`` (one row per GLN with
    every candidate's score) is given, EACH candidate separately - so a champion that misses the real target is visible
    next to the simple peer gap (the proxy label rewards growth that happens anyway)."""
    tdir = settings.data_root / "synthetic_truth"
    if not (tdir / "truth_pharmacy_month.parquet").exists():
        return None
    t = pd.read_parquet(tdir / "truth_pharmacy_month.parquet")
    t = t[t["month"] == origin].copy()
    t["underserved"] = t["realization"] < 0.5
    t["realizable_gap"] = t["potential"] * (1 - t["realization"])
    out: dict[str, Any] = {"definition": "underserved = hidden realization rate < 0.5 at the score origin",
                           "prevalence": float(t.loc[t["active"], "underserved"].mean())}
    for track in ("A_external_only", "B_sales_aware"):
        r = result[(result["track"] == track) & (result["form_or_total"] == "TOTAL")].merge(t, on="gln", how="left")
        r = r.sort_values(["rank", "gln"], na_position="last", kind="mergesort")
        out[track] = {}
        for k in (50, 100, 250):
            top = r.head(k)
            p = float(top["underserved"].mean())
            out[track][f"top{k}"] = {"underserved_precision": p, "lift": p / out["prevalence"] if out["prevalence"] else None,
                                     "mean_realizable_gap": float(top["realizable_gap"].mean())}
        out[track]["population_mean_realizable_gap"] = float(t["realizable_gap"].mean())
    cur = result[(result["track"] == "B_sales_aware") & (result["form_or_total"] == "TOTAL")].merge(t, on="gln")
    top_sales = cur.sort_values("baseline_units_per_month", ascending=False, na_position="last").head(100)
    out["current_sales_top100_underserved_precision"] = float(top_sales["underserved"].mean())
    if candidate_scores is not None:
        cs = candidate_scores.merge(t[["gln", "underserved", "realizable_gap"]], on="gln", how="inner")
        out["candidates"] = {c: truth_topk(cs, c, out["prevalence"]) for c in TRUTH_CANDIDATES if c in cs.columns}
        out["candidates_note"] = ("each candidate ranks only GLNs for which it has a score; sales-aware candidates need "
                                  "an observed baseline. Synthetic ground truth - not evidence about real data.")
    return out


# ----------------------------------------------------------------------------- main run

def _run_dirs(settings: Settings, run_id: str) -> dict[str, tuple[Path, Path]]:
    """(final, staging) folder per output family. Staging sits beside the final folder so the publish is a rename."""
    roots = {"reports": settings.reports_root, "artifacts": settings.artifacts_root,
             "outputs": settings.outputs_root, "canonical": settings.data_root / "canonical"}
    return {k: (r / run_id, r / "_staging" / run_id) for k, r in roots.items()}


def _reserve_run(con: duckdb.DuckDBPyConnection, dirs: dict[str, tuple[Path, Path]], run_id: str, origin: pd.Timestamp,
                 cutoff, input_hash: str, config_hash: str, settings: Settings, gates: dict[str, Any],
                 kind: str = "score") -> None:
    """Claim the run id before any work: a taken id or output folder is an error, never an overwrite."""
    taken = [final.as_posix() for final, _ in dirs.values() if final.exists()]
    if taken:
        raise RunExists(f"run {run_id}: output folder(s) already exist: {taken}")
    try:
        con.execute("insert into score_run (run_id, run_kind, origin_month, cutoff_utc, input_hash, config_hash, "
                    "code_version, selection_policy, seed, exploratory, synthetic, status, created_at) "
                    "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)",
                    [run_id, kind, origin.date(), cutoff, input_hash, config_hash, __version__,
                     settings.policy["policy_version"], int(settings.get("seed")), gates["exploratory"],
                     gates["synthetic"], to_naive_utc(utc_now())])
    except duckdb.ConstraintException as exc:
        raise RunExists(f"run id {run_id} is already registered") from exc


def _abandon_run(con: duckdb.DuckDBPyConnection, run_id: str, dirs: dict[str, tuple[Path, Path]],
                 promoted: list[tuple[Path, Path]]) -> None:
    """A failed run leaves no served score and no half-published folder; earlier runs are never touched."""
    for staging, final in reversed(promoted):          # undo a partial publish
        try:
            os.replace(final, staging)
        except OSError:
            log.error("could not move %s back to staging", final)
    for _, staging in dirs.values():
        shutil.rmtree(staging, ignore_errors=True)
    try:
        con.execute("rollback")
    except duckdb.Error:
        pass
    con.execute("update score_run set status = 'failed' where run_id = ? and status = 'running'", [run_id])


def _publish(dirs: dict[str, tuple[Path, Path]], promoted: list[tuple[Path, Path]]) -> None:
    for final, staging in dirs.values():
        if final.exists():
            raise RunExists(f"output folder appeared while the run was in progress: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        promoted.append((staging, final))


def run_scoring(settings: Settings, exploratory: bool = False, skip_backtest: bool = False,
                champion: dict | None = None, include_names: bool = False, run_label: str | None = None,
                write_outputs: bool = True, model_dir: Path | None = None) -> dict[str, Any]:
    """``model_dir``: score with the models saved by an earlier run instead of training (verification only;
    needs ``write_outputs=False``)."""
    if model_dir is not None and write_outputs:
        raise ValueError("model_dir is for verification re-scoring and never publishes a run")
    t0 = time.time()
    con = connect(settings.warehouse_path)
    try:
        gates = check_gates(settings, con, exploratory, skip_backtest)
        origin = month_start(settings.get("score_origin_month"))
        cutoff = to_naive_utc(settings.get("score_cutoff_utc"))
        input_hash = _input_hash(con)
        config_hash = hash_files(settings.config_files())
        run_id = run_label or (f"{origin:%Y%m}_{input_hash[:6]}{config_hash[:4]}_{utc_stamp()}_{secrets.token_hex(2)}")
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}: letters, digits, '_', '-', '.' only")
        log.info("run %s: origin %s cutoff %s (synthetic=%s, exploratory=%s)", run_id, origin.date(), cutoff,
                 gates["synthetic"], gates["exploratory"])
        exploring = gates["mode"] == "exploration_only"
        dirs = _run_dirs(settings, run_id) if write_outputs else None
        if exploring and not dirs:
            raise GateBlocked("an exploration-only run always writes its outputs")
        if dirs:
            _reserve_run(con, dirs, run_id, origin, cutoff, input_hash, config_hash, settings, gates,
                         "exploration" if exploring else "score")
        promoted: list[tuple[Path, Path]] = []
        try:
            if exploring:
                return _explore_and_publish(con, settings, gates, origin, cutoff, input_hash, config_hash, run_id, t0,
                                            dirs, promoted)
            return _score_and_publish(con, settings, gates, origin, cutoff, input_hash, config_hash, run_id, t0,
                                      skip_backtest or model_dir is not None, champion, include_names, dirs, promoted,
                                      model_dir)
        except BaseException:
            if dirs:
                _abandon_run(con, run_id, dirs, promoted)
            raise
    finally:
        con.close()


def _explore_and_publish(con: duckdb.DuckDBPyConnection, settings: Settings, gates: dict[str, Any], origin: pd.Timestamp,
                         cutoff, input_hash: str, config_hash: str, run_id: str, t0: float,
                         dirs: dict[str, tuple[Path, Path]], promoted: list[tuple[Path, Path]]) -> dict[str, Any]:
    """Exploration only (real data, G1 not accepted): data quality, reconciliation and the external track A.
    No feature panel is built for a model, no model is trained and no sales-aware score exists."""
    for _, staging in dirs.values():
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
    rep, art = dirs["reports"][1], dirs["artifacts"][1]
    canon = build_canonical(con, settings, cutoff, origin, "explore")
    reconciliation_report(con, canon.table).to_csv(rep / "reconciliation.csv", index=False)
    coverage_report(con).to_csv(rep / "coverage.csv", index=False)
    ext, ext_checks = build_external_opportunity(con, cutoff, origin, settings.scenario)
    ext.to_parquet(art / "external_track_a.parquet", index=False)
    write_json(rep / "quality.json", {
        "label": "EXPLORATORY — G1 not accepted; no model trained, no track B score",
        "ingest": batch_summary(con).to_dict("records"),
        "quarantine": dict(con.execute("select error_code, count(*) from quarantine group by 1").fetchall()),
        "canonical": canon.report, "external_demand": ext_checks})
    duration = round(time.time() - t0, 1)
    checksum = frame_checksum(ext, ["gln"])
    write_json(rep / "manifest.json", {
        "run_id": run_id, "run_kind": "exploration", "code_version": __version__,
        "origin_month": origin.strftime("%Y-%m"), "cutoff_utc": settings.get("score_cutoff_utc"),
        "input_hash": input_hash, "config_hash": config_hash, "gates": gates["gates"], "exploratory": True,
        "synthetic": False, "models_trained": False, "external_checksum": checksum, "duration_s": duration,
        "provenance": run_provenance(settings),
        "outputs": {"reports": dirs["reports"][0], "artifacts": dirs["artifacts"][0]}})
    _publish(dirs, promoted)
    con.execute("update score_run set status = 'succeeded', score_checksum = ?, duration_s = ? "
                "where run_id = ? and status = 'running'", [checksum, duration, run_id])
    con.execute("drop table if exists canon_explore")
    log.info("exploration run %s finished in %.0fs (no model trained)", run_id, duration)
    return {"run_id": run_id, "score_checksum": checksum, "reports": dirs["reports"][0],
            "artifacts": dirs["artifacts"][0], "excel": None, "evaluation": None, "truth": None,
            "mode": "exploration_only", "duration_s": duration}


def _score_and_publish(con: duckdb.DuckDBPyConnection, settings: Settings, gates: dict[str, Any], origin: pd.Timestamp,
                       cutoff, input_hash: str, config_hash: str, run_id: str, t0: float, skip_backtest: bool,
                       champion: dict | None, include_names: bool, dirs: dict[str, tuple[Path, Path]] | None,
                       promoted: list[tuple[Path, Path]], model_dir: Path | None = None) -> dict[str, Any]:
    for _, staging in (dirs or {}).values():
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
    asof = AsOfPanels(con, settings)
    ctx = build_context(con, settings, origin, cutoff, "score", exclude_city=None, tune=gates["tuning_allowed"],
                        asof=asof, preloaded=load_quantile_models(model_dir) if model_dir else None)
    # Reports that read the score-cutoff temp tables must run before the backtest rebuilds them.
    recon_df = reconciliation_report(con, ctx.canonical.table)
    cov_df = coverage_report(con)
    stock = stock_reconciliation(con, ctx.canonical.table, cutoff, settings.get("brand"))
    ev = None
    if not skip_backtest:
        ev = run_backtest(con, settings, ctx.panel, asof=asof)
        champion = ev["champion"]
    champion = champion or {"ranking_champion": "b_composite", "forecast_champion": "gbm_p50",
                            "rule": "default (no backtest run)"}
    ext, ext_checks = build_external_opportunity(con, cutoff, origin, settings.scenario)
    model_tuning = "tuned" if gates["tuning_allowed"] else "untuned_engineering_default"
    out = assemble(ctx, ext, champion, settings, run_id, model_tuning)
    result = out["score_result"]
    checksum = frame_checksum(result.drop(columns=["run_id"]), ["track", "form_or_total", "gln"])
    paths: dict[str, Any] = {"run_id": run_id, "score_checksum": checksum}
    if dirs is None:
        con.execute("drop table if exists canon_score")
        paths["score_result"] = result
        return paths

    # Everything is written to the staging folders and renamed into place only after every output succeeded.
    rep, art, outd = dirs["reports"][1], dirs["artifacts"][1], dirs["outputs"][1]
    rep_final, art_final, outd_final = dirs["reports"][0], dirs["artifacts"][0], dirs["outputs"][0]

    result.to_parquet(art / "score_result.parquet", index=False)
    export_canonical(con, ctx.canonical.table, dirs["canonical"][1] / "fact_canonical_movement.parquet")

    recon_df.to_csv(rep / "reconciliation.csv", index=False)
    cov_df.to_csv(rep / "coverage.csv", index=False)
    stock.to_csv(rep / "stock_reconciliation.csv", index=False)
    write_json(rep / "asof_audit.json", {**asof_audit(ctx.panel, origin, cutoff),
                                         "training_panels": ctx.train_audit})
    quality = {"ingest": batch_summary(con).to_dict("records"),
               "quarantine": dict(con.execute("select error_code, count(*) from quarantine group by 1").fetchall()),
               "canonical": ctx.canonical.report, "external_demand": ext_checks,
               "stock_balance": stock["status"].value_counts().to_dict() if len(stock) else {}}
    write_json(rep / "quality.json", quality)

    imp = global_importance(out["shap_values"], ctx.gbm.features)
    imp.to_csv(rep / "shap_global_importance.csv", index=False)
    pd.DataFrame(out["shap_values"], columns=ctx.gbm.features).assign(gln=out["features"]["gln"].to_numpy()) \
        .to_parquet(art / "shap_values.parquet", index=False)
    png = save_summary_plot(out["shap_values"], out["shap_X"], rep / "shap_global.png")
    for q, m in ctx.gbm.models.items():  # LightGBM's C writer cannot open non-ASCII paths on Windows
        (art / f"gbm_q{int(q * 100)}.txt").write_text(m.model_to_string(), encoding="utf-8")
    write_json(art / "model_card.json", {
        "model": "LightGBM quantile (p10/p50/p90) on log1p(next-3-month gross outflow: paid+free, returns excluded)",
        "features": ctx.gbm.features, "tuned": ctx.gbm.tuned, "model_tuning": model_tuning,
        "quantiles": list(ctx.gbm.models), "max_log_target": ctx.gbm.max_log_target,
        "train_domain": ctx.gbm.train_domain,      # frozen 1st/99th percentiles used for the confidence domain
        "params": ctx.gbm.params, "rounds": ctx.gbm.best_rounds, "selection": ctx.gbm.selection,
        "train_rows": ctx.gbm.train_rows, "train_origins": ctx.gbm.train_origins,
        "holt_winters": ctx.hw_params, "champion": champion,
        "intended_use": "Ranking support for channel managers; proxy-validated only; not causal; no automatic retrain.",
        "excluded_inputs": ["GLN", "pharmacy name", "coordinates (none exist)"]})

    if ev:
        write_json(rep / "evaluation.json", ev)
        (rep / "evaluation.md").write_text(evaluation_markdown(ev, run_id), encoding="utf-8")
        write_json(rep / "acceptance.json", ev["acceptance"])
        write_json(art / "evaluation_protocol_v001.json", ev["protocol"])
    truth = None
    if gates["synthetic"]:
        cand = ctx.scores.merge(ctx.features[["gln", "baseline_units"]], on="gln", how="left")
        truth = synthetic_truth_check(settings, result, origin, cand)
    if truth:
        write_json(rep / "synthetic_truth_check.json", truth)
    write_json(rep / "cost.json", {"estimate_usd_month": 0.0, "actual_usd_month": 0.0,
                                   "basis": "local compute only; no paid service, hosting or external API used",
                                   "budget_warn_usd": 40, "budget_stop_usd": 50})

    master = con.execute("select gln, brick_code, pharmacy_name from dim_pharmacy").df()
    if not include_names:
        master["pharmacy_name"] = np.nan
    gates_text = ", ".join(f"{g}={s}" for g, s in gates["gates"].items())
    g3 = gates["gates"].get("G3")
    xlsx = outd / f"pharmacy_opportunity_{origin:%Y-%m}_{run_id}.xlsx"
    export_workbook(xlsx, result, master, {
        "run_id": run_id, "score_month": origin.strftime("%Y-%m"), "cutoff": settings.get("score_cutoff_utc"),
        "model_tuning": model_tuning,
        "gates_text": gates_text + (" | G3 proposal: " + ev["acceptance"]["decision_proposal"] if ev else ""),
        "g3_decision": g3, "champion": champion["ranking_champion"],
        "quality_table": _quality_table(con, ctx.canonical.report, ext_checks, stock),
        "evaluation_table": _evaluation_table(ev) if ev else pd.DataFrame({"note": ["backtest skipped"]}),
        "shap_importance": imp, "shap_png": png, "feature_dictionary": FEATURE_DICTIONARY,
    }, synthetic=gates["synthetic"], exploratory=gates["exploratory"])

    duration = round(time.time() - t0, 1)
    model_files = {f.name: sha256_file(f) for f in sorted(art.glob("gbm_q*.txt"))}
    model_files["model_card.json"] = sha256_file(art / "model_card.json")
    xlsx_final = outd_final / xlsx.name
    manifest = {
        "run_id": run_id, "code_version": __version__, "origin_month": origin.strftime("%Y-%m"),
        "cutoff_utc": settings.get("score_cutoff_utc"), "input_hash": input_hash, "config_hash": config_hash,
        "config_files": [p.relative_to(settings.root).as_posix() for p in settings.config_files()],
        "selection_policy": settings.policy["policy_version"], "seed": settings.get("seed"),
        "champion": champion, "score_checksum": checksum, "synthetic": gates["synthetic"],
        "exploratory": gates["exploratory"], "gates": gates["gates"], "duration_s": duration,
        "model_tuning": model_tuning, "usable_as_g3_evidence": model_tuning == "tuned" and ev is not None,
        "provenance": run_provenance(settings), "model_files_sha256": model_files,
        "batches": con.execute("select batch_id, sha256, state from ingestion_batch order by batch_id").df().to_dict("records"),
        "outputs": {"excel": xlsx_final, "reports": rep_final, "artifacts": art_final},
    }
    write_json(rep / "manifest.json", manifest)
    # Publish (rename staging -> final) and register the run only after every output succeeded:
    # a failed run never becomes a served score and never touches an earlier run.
    _publish(dirs, promoted)
    con.execute("begin transaction")
    con.register("_sr", result)
    con.execute("create table if not exists score_result as select * from _sr where false")
    con.execute("insert into score_result select * from _sr")
    con.unregister("_sr")
    con.execute("update score_run set status = 'succeeded', model_version = ?, training_end = ?, score_checksum = ?, "
                "duration_s = ? where run_id = ? and status = 'running'",
                [manifest["champion"]["ranking_champion"], (origin - pd.DateOffset(months=3)).date(), checksum,
                 duration, run_id])
    con.execute("commit")
    con.execute("drop table if exists canon_score")
    log.info("run %s finished in %.0fs -> %s", run_id, duration, xlsx_final)
    paths.update({"excel": xlsx_final, "reports": rep_final, "artifacts": art_final,
                  "evaluation": ev and ev["acceptance"], "truth": truth, "duration_s": duration})
    return paths
