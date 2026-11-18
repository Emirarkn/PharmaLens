"""Ranking and forecast metrics. Undefined values are None, never 0."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..models.baselines import rank_order


def precision_at_k(score: pd.Series, label: pd.Series, tiebreak: pd.Series, k: int) -> dict[str, Any]:
    n = int(label.notna().sum())
    k_eff = min(k, n)
    if k_eff == 0:
        return {"k": k, "k_eff": 0, "precision": None, "lift": None, "positives_in_top": 0}
    r = rank_order(score, tiebreak)
    top = r <= k_eff
    prec = float(label[top].mean())
    prev = float(label.mean())
    tied_at_boundary = int((score == score[r == k_eff].iloc[0]).sum()) if score.notna().any() else 0
    return {"k": k, "k_eff": k_eff, "precision": prec, "lift": (prec / prev) if prev > 0 else None,
            "positives_in_top": int(label[top].sum()), "ties_at_boundary": tied_at_boundary}


def ndcg_at_k(score: pd.Series, gain: pd.Series, tiebreak: pd.Series, k: int) -> float | None:
    k_eff = min(k, len(gain))
    if k_eff == 0 or gain.sum() <= 0:
        return None
    r = rank_order(score, tiebreak)
    disc = 1 / np.log2(np.arange(2, k_eff + 2))
    got = gain[r.sort_values().index[:k_eff]].to_numpy()
    ideal = np.sort(gain.to_numpy())[::-1][:k_eff]
    return float((got * disc).sum() / (ideal * disc).sum())


def evaluate_ranking(score: pd.Series, label: pd.Series, gain: pd.Series, tiebreak: pd.Series,
                     k_values: list[int]) -> dict[str, Any]:
    out = {f"P@{k}": precision_at_k(score, label, tiebreak, k) for k in k_values}
    out["nDCG@100"] = ndcg_at_k(score, gain, tiebreak, 100)
    return out


def cluster_bootstrap_diff(df: pd.DataFrame, a: str, b: str, k: int, reps: int, seed: int,
                           cluster: str = "district_code") -> dict[str, Any]:
    """Block bootstrap over districts of P@k(a) - P@k(b) and P@k(a) lift (GLN/district correlation kept)."""
    empty = {"reps": 0, "diff_mean": None, "diff_ci95": None, "lift_ci95": None}
    if df is None or len(df) == 0 or df[cluster].nunique() == 0:
        return {**empty, "note": "insufficient sample: no eligible GLN in the test universe (bootstrap not run)"}
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = list(groups)
    diffs, lifts = [], []
    for _ in range(reps):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        s = df.iloc[idx].reset_index(drop=True)
        pa = precision_at_k(s[a], s["label"], s["_tiebreak"], k)
        pb = precision_at_k(s[b], s["label"], s["_tiebreak"], k)
        if pa["precision"] is None or pb["precision"] is None:
            continue
        diffs.append(pa["precision"] - pb["precision"])
        if pa["lift"] is not None:
            lifts.append(pa["lift"])
    if not diffs:
        return {**empty, "note": "no bootstrap replicate had a defined precision for both candidates"}
    return {"reps": len(diffs), "diff_mean": float(np.mean(diffs)),
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "lift_ci95": [float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))] if lifts else None,
            "note": f"cluster bootstrap over {len(keys)} districts; few cities limit generalisation"}


def forecast_metrics(actual: pd.Series, pred: pd.Series, lo: pd.Series | None = None,
                     hi: pd.Series | None = None) -> dict[str, Any]:
    ok = actual.notna() & pred.notna()
    a, p = actual[ok].to_numpy(float), pred[ok].to_numpy(float)
    den = np.abs(a).sum()
    out = {"n": int(ok.sum()), "wape": None if den == 0 else float(np.abs(a - p).sum() / den),
           "mae": float(np.abs(a - p).mean()) if len(a) else None, "bias": float((p - a).sum() / den) if den else None}
    if lo is not None and hi is not None:
        out["interval_coverage_p10_p90"] = float(((lo[ok] <= actual[ok]) & (actual[ok] <= hi[ok])).mean()) if len(a) else None
    return out
