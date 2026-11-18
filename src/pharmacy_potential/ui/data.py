"""Read-only access to the artifacts of finished runs, for the local PharmaLens viewer.

No Streamlit import here: everything is plain pandas/json so it can be tested without a browser. A run id
comes from the URL/session, so it is validated before it is joined to a path (no path traversal).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,120}$")

TRACKS = {
    "B_total": "Track B - sales-aware, total",
    "B_form": "Track B - sales-aware, by form",
    "A_external": "Track A - external demand only",
}


@dataclass(frozen=True)
class Roots:
    reports: Path
    artifacts: Path
    outputs: Path


def _run_dir(base: Path, run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    p = (base / run_id).resolve()
    if base.resolve() not in p.parents:
        raise ValueError(f"run id escapes the run folder: {run_id!r}")
    return p


def list_runs(roots: Roots) -> list[str]:
    """Run ids that have both a manifest and a score table, newest first."""
    found: list[tuple[float, str]] = []
    if not roots.reports.exists():
        return []
    for d in roots.reports.iterdir():
        manifest = d / "manifest.json"
        if d.is_dir() and RUN_ID_RE.match(d.name) and manifest.exists() \
                and (roots.artifacts / d.name / "score_result.parquet").exists():
            found.append((manifest.stat().st_mtime, d.name))
    return [name for _, name in sorted(found, reverse=True)]


def read_json(roots: Roots, run_id: str, name: str) -> dict[str, Any] | None:
    p = _run_dir(roots.reports, run_id) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_text(roots: Roots, run_id: str, name: str) -> str | None:
    p = _run_dir(roots.reports, run_id) / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def read_csv(roots: Roots, run_id: str, name: str) -> pd.DataFrame | None:
    p = _run_dir(roots.reports, run_id) / name
    return pd.read_csv(p, dtype={"gln": str}) if p.exists() else None


def image_path(roots: Roots, run_id: str, name: str) -> Path | None:
    p = _run_dir(roots.reports, run_id) / name
    return p if p.exists() else None


def read_scores(roots: Roots, run_id: str) -> pd.DataFrame:
    p = _run_dir(roots.artifacts, run_id) / "score_result.parquet"
    df = pd.read_parquet(p)
    df["gln"] = df["gln"].astype(str)
    return df


def workbook_path(roots: Roots, run_id: str) -> Path | None:
    d = _run_dir(roots.outputs, run_id)
    files = sorted(d.glob("*.xlsx")) if d.exists() else []
    return files[0] if files else None


def select_track(scores: pd.DataFrame, track: str) -> pd.DataFrame:
    if track == "B_total":
        return scores[(scores["track"] == "B_sales_aware") & (scores["form_or_total"] == "TOTAL")]
    if track == "B_form":
        return scores[(scores["track"] == "B_sales_aware") & (scores["form_or_total"] != "TOTAL")]
    if track == "A_external":
        return scores[scores["track"] == "A_external_only"]
    raise ValueError(f"unknown track: {track}")


def filter_scores(df: pd.DataFrame, *, cities: list[str] | None = None, districts: list[str] | None = None,
                  eligibility: list[str] | None = None, readiness: list[str] | None = None,
                  forms: list[str] | None = None, gln_contains: str = "") -> pd.DataFrame:
    """Row filters of the ranking page. Empty selection means 'no filter', never 'nothing'."""
    out = df
    for col, vals in (("city_code", cities), ("district_code", districts), ("eligibility", eligibility),
                      ("readiness", readiness), ("form_or_total", forms)):
        if vals:
            out = out[out[col].astype(str).isin([str(v) for v in vals])]
    if gln_contains.strip():
        out = out[out["gln"].str.contains(gln_contains.strip(), regex=False)]
    return out


def ranking_view(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Best rank first. Unranked rows (no observed baseline etc.) sort last and keep an empty rank."""
    return df.sort_values(["rank", "gln"], na_position="last", kind="mergesort").head(top_n)


def pharmacy_rows(scores: pd.DataFrame, gln: str) -> pd.DataFrame:
    return scores[scores["gln"] == gln.strip()].sort_values(["track", "form_or_total"])


def shap_for(roots: Roots, run_id: str, gln: str, top: int = 10) -> pd.DataFrame | None:
    """Largest per-pharmacy SHAP contributions (log scale) if the run stored them, else None."""
    p = _run_dir(roots.artifacts, run_id) / "shap_values.parquet"
    if not p.exists():
        return None
    shap = pd.read_parquet(p)
    shap["gln"] = shap["gln"].astype(str)
    hit = shap[shap["gln"] == gln.strip()]
    if hit.empty:
        return None
    row = hit.drop(columns=["gln"]).iloc[0]
    out = pd.DataFrame({"feature": row.index, "shap": row.to_numpy(dtype=float)}).dropna()
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
