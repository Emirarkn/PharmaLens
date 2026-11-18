"""Ranking candidates evaluated on the same eligible universe.

Higher score = higher priority. Ties are broken by a fixed deterministic hash of the GLN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import stable_unit_hash

SIMPLE_BASELINES = ["random_stratified", "current_sales", "external_demand", "peer_gap"]
SALES_AWARE = ["peer_gap", "b_composite", "b_hybrid", "ml_growth"]
COMPLEXITY = {"peer_gap": 0, "b_composite": 1, "b_hybrid": 2, "ml_growth": 3}


def pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def ml_growth_signal(f: pd.DataFrame, p50_next3: pd.Series) -> pd.Series:
    """Predicted excess growth vs the district median prediction (not the size of the forecast)."""
    # The forecast is gross outflow, so the baseline is gross too where known; net-negative baselines (returns
    # only) are floored at 0: returns are not negative demand.
    base = f["baseline_gross"].fillna(f["baseline_units"]) if "baseline_gross" in f else f["baseline_units"]
    g = (p50_next3 / 3 + 1) / (base.clip(lower=0) + 1)
    return g / g.groupby(f["district_code"]).transform("median")


def candidate_scores(f: pd.DataFrame, seed: int, weights: dict, p50_next3: pd.Series | None = None) -> pd.DataFrame:
    """All candidate scores for one origin. ``f`` = features (+ optional ML forecast). Readiness is not an input."""
    s = pd.DataFrame({"gln": f["gln"]}, index=f.index)
    rnd = pd.Series(stable_unit_hash(f["gln"], seed), index=f.index)
    stratum = f["city_code"].astype(str) + "|" + (f["n_obs12"] >= 9).map({True: "hi_cov", False: "lo_cov"})
    s["random_stratified"] = rnd.groupby(stratum).rank(pct=True)
    s["current_sales"] = f["baseline_units"]
    s["external_demand"] = f["ext_base_units"]
    s["peer_gap"] = f["peer_gap_rel"] + 1e-6 * pct(f["peer_gap_units"].fillna(0))
    comp = weights["b_composite"]
    if {"continuity", "readiness"} & set(comp):
        raise ValueError("readiness must not be a component of the opportunity score: it is a separate column "
                         "")
    s["b_composite"] = comp["peer_gap"] * pct(f["peer_gap_rel"]) + comp["external"] * pct(f["ext_base_units"])
    if p50_next3 is not None:
        growth = ml_growth_signal(f, p50_next3)
        s["ml_growth"] = growth
        hyb = weights["b_hybrid"]
        s["b_hybrid"] = (hyb["peer_gap"] * pct(f["peer_gap_rel"]) + hyb["ml_growth"] * pct(growth)
                         + hyb["external"] * pct(f["ext_base_units"]))
    # Sales-aware scores need an observed baseline; no fake sales-gap for unobserved GLNs.
    no_base = f["baseline_units"].isna()
    for c in s.columns:
        if c in ("gln", "random_stratified", "external_demand"):
            continue
        s.loc[no_base, c] = np.nan
    s["_tiebreak"] = rnd
    return s


def rank_order(scores: pd.Series, tiebreak: pd.Series) -> pd.Series:
    """1 = best. NaN scores go last. Deterministic."""
    key = pd.DataFrame({"s": scores.fillna(-np.inf), "t": tiebreak})
    order = key.sort_values(["s", "t"], ascending=[False, False], kind="mergesort").index
    r = pd.Series(np.arange(1, len(order) + 1), index=order)
    return r.reindex(scores.index)
