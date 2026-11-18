"""SHAP explanations for the LightGBM median model (model contributions, not causes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import FEATURE_DICTIONARY

# Human-readable phrasing per feature and sign of its contribution to the forecast.
PHRASES = {
    "mean3": ("recent 3-month sales level raises the forecast", "recent 3-month sales level is low"),
    "mean6": ("recent 6-month sales level is high", "recent 6-month sales level is low"),
    "mean12": ("strong 12-month sales history", "weak 12-month sales history"),
    "lag0": ("last-month sales are high", "last-month sales are low"),
    "peer_benchmark": ("district peers sell at a high level", "district peers sell at a low level"),
    "ratio_to_peer": ("sales ratio to similar pharmacies is high", "sales are low relative to similar pharmacies"),
    "peer_gap_rel": ("large gap to the peer group", "small gap to the peer group"),
    "ext_base_units": ("high adult population per pharmacy in the district (scenario demand)", "low district scenario demand"),
    "adults_per_pharmacy": ("high adult population per pharmacy", "low adult population per pharmacy"),
    "pharmacies_per_10k_adults": ("pharmacy density raises the forecast", "high pharmacy density (competition)"),
    "visits_3m": ("frequent visits in the last 3 months", "few visits in the last 3 months"),
    "visits_12m": ("frequent visits over 12 months", "few visits over 12 months"),
    "order_units_3m": ("rep orders in the last 3 months", "no or few rep orders in the last 3 months"),
    "trend12": ("12-month trend is up", "12-month trend is down"),
    "yoy_ratio": ("up versus last year", "down versus last year"),
    "n_obs12": ("sufficient observation history", "few observed months"),
    "n_obs3": ("recent months observed", "observations missing in recent months"),
    "months_since_obs": ("recent observation available", "no observation for a long time"),
    "zero_share12": ("share of zero-sales months raises the forecast", "high share of zero-sales months"),
    "known_campaign_next3": ("announced campaign period", "not a campaign period"),
    "crm_coverage_flag": ("CRM (visit/order) coverage available", "no CRM coverage: visits/orders unknown"),
    "target_month": ("seasonally strong period", "seasonally weak period"),
}


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    import shap
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X.astype(float))
    return np.asarray(vals)


def top_reasons(values: np.ndarray, features: list[str], k: int = 3) -> list[list[str]]:
    out = []
    order = np.argsort(-np.abs(values), axis=1)[:, :k]
    for i, idx in enumerate(order):
        reasons = []
        for j in idx:
            name = features[j]
            up, down = PHRASES.get(name, (f"{name} raises the forecast", f"{name} lowers the forecast"))
            reasons.append(f"{up if values[i, j] > 0 else down} (SHAP {values[i, j]:+.2f})")
        out.append(reasons)
    return out


def global_importance(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp["description"] = imp["feature"].map(lambda f: FEATURE_DICTIONARY.get(f, ("", ""))[1])
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_summary_plot(values: np.ndarray, X: pd.DataFrame, path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(values, X.astype(float), show=False, max_display=15)
        plt.title("SHAP - p50 next-3-month model (model contributions, not causal)", fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close("all")
        return path
    except Exception:  # plotting is a convenience; never fail the run on it
        return None
