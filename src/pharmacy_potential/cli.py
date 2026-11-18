"""Command line interface: `ppi <command>`. Exit codes: 0 ok, 1 validation failure, 2 config/input unavailable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_settings
from .utils import get_logger, to_naive_utc, write_json

app = typer.Typer(add_completion=False, help="Pharmacy Potential Intelligence (synthetic demo first, gated real data).")
log = get_logger()
CONFIG = typer.Option("configs/pilot.yaml", "--config", help="Pilot config file.")
DATA_ROOT = typer.Option(None, "--data-root", help="Override data root (approved secure location).")


def _settings(config: str, data_root: Optional[str]):
    try:
        return load_settings(config, data_root=data_root)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)


@app.command()
def synth(config: str = CONFIG, data_root: Optional[str] = DATA_ROOT,
          scale: float = typer.Option(1.0, help="Fraction of the pilot universe size (tests use ~0.05)."),
          synthetic_config: str = typer.Option("configs/synthetic_v001.yaml", help="Synthetic universe config.")):
    """Generate the synthetic universe as raw source files + manifests."""
    from .synth.generator import generate
    s = _settings(config, data_root)
    summary = generate(s, Path(synthetic_config), scale=scale)
    typer.echo(f"{summary['pharmacies']} GLN, {summary['districts']} districts, {len(summary['files'])} files -> {summary['out_root']}")


@app.command()
def manifest(dataset: str = typer.Option(...), file: Path = typer.Option(..., exists=True),
             period_start: str = typer.Option(..., help="YYYY-MM-DD"), period_end: str = typer.Option(..., help="YYYY-MM-DD"),
             owner: str = typer.Option("unknown"), license_id: str = typer.Option("unknown"),
             confidentiality: str = typer.Option("restricted"), revision: int = 1,
             supersedes: Optional[str] = None, available_at: Optional[str] = typer.Option(None, help="ISO-8601 UTC"),
             availability_mode: str = typer.Option("batch", help="batch | retrospective_lag"),
             retrospective_lag_days: Optional[int] = None, config: str = CONFIG):
    """Write a sidecar manifest (<file>.manifest.json) for a delivered file."""
    from .ingest.manifest import build_manifest, validate_manifest, write_manifest
    s = _settings(config, None)
    src = s.source_config(dataset)
    m = build_manifest(dataset, src["source_name"], file, period_start=period_start, period_end=period_end,
                       revision=revision, supersedes_batch_id=supersedes, confidentiality=confidentiality,
                       license_id=license_id, owner=owner, available_at=available_at, availability_mode=availability_mode,
                       retrospective_lag_days=retrospective_lag_days)
    errs = validate_manifest(m)
    if errs:
        typer.echo("Manifest invalid: " + "; ".join(errs), err=True)
        raise typer.Exit(1)
    typer.echo(f"manifest written: {write_manifest(m, file)} (batch_id {m['batch_id']})")
    if owner == "unknown" or license_id == "unknown":
        typer.echo("WARNING: owner/license unknown -> G0 open; scoring will require --exploratory.", err=True)


@app.command()
def ingest(dataset: str = typer.Option(...), path: Path = typer.Option(..., exists=True),
           manifest_path: Optional[Path] = typer.Option(None, "--manifest"), config: str = CONFIG,
           data_root: Optional[str] = DATA_ROOT):
    """Ingest one file (or a folder of files with sidecar manifests) for a dataset."""
    from .db import connect
    from .gates import GateBlocked
    from .ingest.loader import IngestError, ingest_file, ingest_path
    s = _settings(config, data_root)
    con = connect(s.warehouse_path)
    try:
        results = [ingest_file(s, con, dataset, path, manifest_path)] if manifest_path else ingest_path(s, con, dataset, path)
    except IngestError as e:
        typer.echo(f"ingest failed: {e}", err=True)
        raise typer.Exit(e.exit_code)
    except GateBlocked as e:
        typer.echo(f"BLOCKED: {e}", err=True)
        raise typer.Exit(2)
    for r in results:
        typer.echo(json.dumps(r.as_dict(), ensure_ascii=False, default=str))
    if any(r.state == "rejected" for r in results):
        raise typer.Exit(1)


@app.command("ingest-all")
def ingest_all_cmd(root: Path = typer.Option(..., exists=True, help="Folder with one sub-folder per dataset."),
                   config: str = CONFIG, data_root: Optional[str] = DATA_ROOT):
    """Ingest every dataset folder under ROOT in the configured dependency order."""
    from .db import connect
    from .gates import GateBlocked
    from .ingest.loader import batch_summary
    from .pipeline import ingest_all
    s = _settings(config, data_root)
    con = connect(s.warehouse_path)
    try:
        res = ingest_all(s, con, root)
    except GateBlocked as e:
        typer.echo(f"BLOCKED: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(batch_summary(con).to_string(index=False))
    if any(r["state"] == "rejected" for r in res):
        raise typer.Exit(1)


@app.command()
def template(dataset: Optional[str] = typer.Option(None, help="Dataset (see configs/sources); omit with --all"),
             all_datasets: bool = typer.Option(False, "--all", help="One template per mapped dataset"),
             out: Optional[Path] = typer.Option(None, help="Output folder (default: outputs/templates)"),
             config: str = CONFIG):
    """Write versioned Excel upload template(s) generated from the source mappings (labelled synthetic example sheet)."""
    from .ingest.templates import available_templates, build_template
    s = _settings(config, None)
    names = available_templates(s) if all_datasets else [dataset] if dataset else []
    if not names:
        typer.echo("give --dataset D or --all", err=True)
        raise typer.Exit(2)
    for name in names:
        try:
            typer.echo(str(build_template(s, name, out or s.outputs_root / "templates")))
        except FileNotFoundError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(2)


@app.command()
def preview(dataset: str = typer.Option(...), path: Path = typer.Option(..., exists=True),
            error_report: Optional[Path] = typer.Option(None, help="Write rejected rows + reasons here (.xlsx)"),
            config: str = CONFIG, data_root: Optional[str] = DATA_ROOT):
    """Show what loading FILE would do (rows, period, errors, duplicates) without loading anything."""
    from .db import connect
    from .gates import GateBlocked, require_gates
    from .ingest.loader import IngestError
    from .ingest.preview import preview_file, write_error_report
    s = _settings(config, data_root)
    if (s.data_root / "raw" / "synthetic").resolve() not in path.resolve().parents:
        try:                      # a file outside the synthetic demo folder is treated as real data
            require_gates(s, "data", "upload preview")
        except GateBlocked as e:
            typer.echo(f"BLOCKED: {e}", err=True)
            raise typer.Exit(2)
    con = connect(s.warehouse_path)
    try:
        res = preview_file(s, con, dataset, path)
    except IngestError as e:
        typer.echo(f"preview failed: {e}", err=True)
        raise typer.Exit(e.exit_code)
    finally:
        con.close()
    errors = res.pop("error_rows")
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    if error_report and len(errors):
        res["error_rows"] = errors
        typer.echo(f"error report -> {write_error_report(res, error_report)}")
    if not res["ok"]:
        raise typer.Exit(1)


@app.command()
def profile(path: Path = typer.Option(..., exists=True), dataset: Optional[str] = typer.Option(None),
            out: Optional[Path] = typer.Option(None, help="Where to write the JSON profile."),
            config: str = CONFIG):
    """Profile a new file (no raw values printed) and optionally draft a mapping YAML for DATASET."""
    from .gates import GateBlocked, require_gates
    from .ingest.profile import draft_mapping, profile_file, write_draft
    s = _settings(config, None)
    if (s.data_root / "raw" / "synthetic").resolve() not in path.resolve().parents:
        try:                      # a file outside the synthetic demo folder is treated as real data
            require_gates(s, "data", "profiling")
        except GateBlocked as e:
            typer.echo(f"BLOCKED: {e}", err=True)
            raise typer.Exit(2)
    prof = profile_file(path)
    out = out or s.reports_root / "profiling" / f"{path.stem}.profile.json"
    write_json(out, prof)
    typer.echo(f"profile -> {out}")
    for w in prof["warnings"]:
        typer.echo(f"WARNING: {w}")
    if dataset:
        draft = write_draft(draft_mapping(s, dataset, prof), s.reports_root / "profiling" / f"{dataset}.draft.yaml")
        typer.echo(f"draft mapping -> {draft} (review, then copy to configs/sources/{dataset}.yaml)")


@app.command()
def reconcile(origin: str = typer.Option(..., help="YYYY-MM"), cutoff: Optional[str] = typer.Option(None, help="ISO UTC"),
              mode: str = typer.Option("default"), config: str = CONFIG, data_root: Optional[str] = DATA_ROOT):
    """Build canonical movement as of a cutoff and write quality/reconciliation/coverage reports."""
    from .db import connect
    from .gates import GateBlocked, has_real_batches, require_gates
    from .reconcile.canonical import build_canonical, coverage_report, reconciliation_report
    s = _settings(config, data_root)
    con = connect(s.warehouse_path)
    if has_real_batches(con):
        try:
            require_gates(s, "data", "reconciliation and quality reports")
        except GateBlocked as e:
            typer.echo(f"BLOCKED: {e}", err=True)
            raise typer.Exit(2)
    cut = to_naive_utc(cutoff) if cutoff else None
    res = build_canonical(con, s, cut, origin, tag="adhoc", mode=mode)
    rep = s.reports_root / f"reconcile_{origin.replace('-', '')}_{mode}"
    rep.mkdir(parents=True, exist_ok=True)
    write_json(rep / "quality.json", res.report)
    reconciliation_report(con, res.table).to_csv(rep / "reconciliation.csv", index=False)
    coverage_report(con).to_csv(rep / "coverage.csv", index=False)
    typer.echo(f"canonical {res.table}: {res.report['status_counts']} -> {rep}")
    if not (res.report["pk_unique"] and res.report["fk_orphans_gln"] == 0 and res.report["fk_orphans_barcode"] == 0):
        raise typer.Exit(1)


@app.command()
def run(config: str = CONFIG, data_root: Optional[str] = DATA_ROOT,
        exploratory: bool = typer.Option(False, help="Real data without accepted G1: exploration only (data quality, "
                                                     "reconciliation, external track A). No model is trained. G0 is still required."),
        skip_backtest: bool = typer.Option(False, help="Score only, using the default champion. Required on real data "
                                                       "until G2 is accepted (then fixed, untuned model settings)."),
        with_names: bool = typer.Option(False, help="Include restricted pharmacy names in the Excel export."),
        score_origin: Optional[str] = typer.Option(None, help="Override score origin YYYY-MM."),
        score_cutoff: Optional[str] = typer.Option(None, help="Override score cutoff ISO UTC.")):
    """Full monthly run on ingested data: reconcile -> features -> models -> backtest -> score -> reports -> Excel."""
    from .pipeline import GateBlocked, run_scoring
    s = _settings(config, data_root)
    if score_origin:
        s.overrides["score_origin_month"] = score_origin
    if score_cutoff:
        s.overrides["score_cutoff_utc"] = score_cutoff
    try:
        res = run_scoring(s, exploratory=exploratory, skip_backtest=skip_backtest, include_names=with_names)
    except GateBlocked as e:
        typer.echo(f"BLOCKED: {e}", err=True)
        raise typer.Exit(2)
    _print_result(res)


@app.command()
def demo(config: str = CONFIG, data_root: Optional[str] = DATA_ROOT,
         scale: float = typer.Option(1.0, help="Universe scale (1.0 = ~3.7k GLN)."),
         fresh: bool = typer.Option(False, help="Delete the existing synthetic data root first."),
         skip_backtest: bool = typer.Option(False)):
    """One command: synth -> ingest all -> full run on synthetic data (names included: they are synthetic)."""
    from .db import connect
    from .ingest.loader import batch_summary
    from .pipeline import ingest_all, run_scoring
    from .synth.cleanup import UnsafeCleanup, clean_synthetic_root
    from .synth.generator import generate
    s = _settings(config, data_root)
    if fresh:
        try:
            clean_synthetic_root(s.data_root, s.raw["paths"]["warehouse_file"])
        except UnsafeCleanup as e:
            typer.echo(f"Refusing --fresh (nothing deleted): {e}", err=True)
            raise typer.Exit(2)
    generate(s, scale=scale)
    con = connect(s.warehouse_path)
    ingest_all(s, con, s.data_root / "raw" / "synthetic")
    typer.echo(batch_summary(con).to_string(index=False))
    con.close()
    res = run_scoring(s, skip_backtest=skip_backtest, include_names=True)
    _print_result(res)


@app.command()
def status(config: str = CONFIG, data_root: Optional[str] = DATA_ROOT):
    """Show batches, gate statuses and recent runs."""
    from .db import connect
    from .ingest.loader import batch_summary
    from .pipeline import gate_status
    s = _settings(config, data_root)
    con = connect(s.warehouse_path)
    typer.echo(batch_summary(con).to_string(index=False))
    typer.echo("Gates: " + ", ".join(f"{g}={gate_status(s, g)}" for g in ("G0", "G1", "G2", "G3", "G4", "G5")))
    typer.echo(con.execute("select run_id, status, synthetic, exploratory, score_checksum from score_run "
                           "order by created_at desc limit 5").df().to_string(index=False))


@app.command()
def ui(config: str = CONFIG, data_root: Optional[str] = DATA_ROOT,
       port: int = typer.Option(8501, help="Local port. The viewer binds to 127.0.0.1 only.")):
    """Open PharmaLens, the local read-only viewer of finished runs (needs `pip install -e .[ui]`)."""
    import importlib.util
    import os
    import subprocess
    import sys
    if importlib.util.find_spec("streamlit") is None:
        typer.echo("The viewer needs the optional UI dependency: pip install -e \".[ui]\"", err=True)
        raise typer.Exit(2)
    s = _settings(config, data_root)
    env = {**os.environ, "PHARMALENS_REPORTS": str(s.reports_root), "PHARMALENS_ARTIFACTS": str(s.artifacts_root),
           "PHARMALENS_OUTPUTS": str(s.outputs_root)}
    script = Path(__file__).parent / "ui" / "app.py"
    typer.echo(f"PharmaLens on http://127.0.0.1:{port} (Ctrl+C to stop)")
    raise typer.Exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(script), "--server.address", "127.0.0.1", "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"], env=env))


def _print_result(res: dict) -> None:
    typer.echo("")
    typer.echo(f"run_id      : {res['run_id']}" + ("  [exploration only, no model trained]" if res.get("mode") == "exploration_only" else ""))
    typer.echo(f"excel       : {res.get('excel')}")
    typer.echo(f"reports     : {res.get('reports')}")
    typer.echo(f"checksum    : {res['score_checksum']}")
    if res.get("evaluation"):
        typer.echo(f"G3 proposal : {res['evaluation']['decision_proposal']} (champion {res['evaluation']['champion']})")
    if res.get("truth"):
        t = res["truth"]
        typer.echo(f"truth check : prevalence {t['prevalence']:.3f} | B top100 {t['B_sales_aware']['top100']['underserved_precision']:.3f}"
                   f" | A top100 {t['A_external_only']['top100']['underserved_precision']:.3f}"
                   f" | current-sales top100 {t['current_sales_top100_underserved_precision']:.3f}")


if __name__ == "__main__":
    app()
