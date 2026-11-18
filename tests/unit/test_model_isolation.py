"""holdout isolation of tuning, K11 frozen confidence domain, K8 untuned fixed settings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pharmacy_potential.config import load_settings
from pharmacy_potential.evaluation.backtest import select_hw_params
from pharmacy_potential.features.build import MODEL_FEATURES, Panel
from pharmacy_potential.models.gbm import OOD_COLUMNS, fit_quantile_models
from pharmacy_potential.scoring.score import confidence

ROOT = Path(__file__).resolve().parents[2]
S = load_settings(ROOT / "configs" / "pilot.yaml")
MC = S.get("model")


def seasonal_panel(n=60, months=40, seed=3) -> Panel:
    rng = np.random.default_rng(seed)
    city = np.array(["34"] * (n // 2) + ["21"] * (n // 2))                 # "21" is the holdout city
    t = np.arange(months)
    base = rng.uniform(5, 30, n)[:, None] * (1 + 0.3 * np.sin(2 * np.pi * t / 12))[None, :]
    Y = np.clip(base + rng.normal(0, 2, (n, months)), 0, None)
    return Panel(glns=np.array([f"g{i}" for i in range(n)]), city=city, district=np.array(["d"] * n),
                 months=pd.date_range("2022-01-01", periods=months, freq="MS"), Y=Y.copy(),
                 active=np.ones((n, months), bool), fallback=np.zeros((n, months)), Yg=Y.copy())


def hw(panel, **kw):
    return select_hw_params(panel, "2024-12", 3, MC["holt_winters_grid"], "2022-01", **kw)


def test_k9_holt_winters_selection_never_sees_the_holdout_city():
    panel = seasonal_panel()
    chosen = hw(panel, exclude_city=MC["holdout_city"])
    hold = panel.city == MC["holdout_city"]
    rng = np.random.default_rng(99)
    panel.Yg[hold] = rng.uniform(0, 900, panel.Yg[hold].shape)             # wreck the holdout rows
    panel.Y[hold] = panel.Yg[hold]
    assert hw(panel, exclude_city=MC["holdout_city"]) == chosen
    # Sanity: the selection does react to data it is allowed to see, and would react to the holdout if it saw it.
    assert hw(panel, exclude_city=None) != chosen
    panel.Yg[~hold] = rng.uniform(0, 900, panel.Yg[~hold].shape)
    assert hw(panel, exclude_city=MC["holdout_city"])["wape"] != chosen["wape"]


def test_k8_untuned_mode_uses_fixed_settings_and_never_selects():
    fixed = select_hw_params(None, "2024-12", 3, MC["holt_winters_grid"], "2022-01", None, tune=False)
    g = MC["holt_winters_grid"]
    assert (fixed["alpha"], fixed["beta"], fixed["gamma"]) == (g["alpha"][0], g["beta"][0], g["gamma"][0])
    assert "not allowed" in fixed["selection"]


def training_frame(n_origins=8, per=300, seed=1, shift=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_origins):
        o = pd.Timestamp("2024-01-01") + pd.DateOffset(months=3 * k)
        d = pd.DataFrame(rng.uniform(0, 10, (per, len(MODEL_FEATURES))) + shift, columns=MODEL_FEATURES)
        d["gln"] = [f"g{i}" for i in range(per)]
        d["origin_month"] = o
        d["target_units_next3"] = np.clip(d["mean3"] * 3 + rng.normal(0, 1, per), 0, None)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_k8_fit_without_tuning_has_no_grid_selection():
    cfg = {**MC["gbm"], "max_rounds": 30, "num_leaves_grid": [4, 8], "untuned": {"num_leaves": 4, "rounds": 25}}
    m = fit_quantile_models(training_frame(), cfg, 1, 1, 3, tune=False)
    assert m.tuned is False and len(m.selection) == 1 and "not selected" in m.selection[0]["note"]
    assert set(m.best_rounds.values()) == {25}
    t = fit_quantile_models(training_frame(), cfg, 1, 1, 3, tune=True)
    assert t.tuned is True and len(t.selection) == 2


def test_k11_training_domain_is_frozen_from_the_training_frame():
    cfg = {**MC["gbm"], "max_rounds": 20, "num_leaves_grid": [4], "untuned": {"num_leaves": 4, "rounds": 20}}
    train = training_frame()
    m = fit_quantile_models(train, cfg, 1, 1, 3)
    assert set(m.train_domain) == set(OOD_COLUMNS)
    for c, (lo, hi) in m.train_domain.items():
        assert lo == float(np.nanpercentile(train[c], 1)) and hi == float(np.nanpercentile(train[c], 99))

    # A universe that moved far away from the training distribution.
    n = 200
    f = pd.DataFrame({"mean3": np.linspace(500, 900, n), "ext_base_units": np.linspace(300, 800, n),
                      "n_obs12": 12, "months_since_obs": 0, "fallback_share12": 0.0, "n_peers": 30})
    fc = pd.DataFrame({"fc_gbm_p10": 1.0, "fc_gbm_p50": 2.0, "fc_gbm_p90": 3.0}, index=f.index)
    th = S.thresholds
    frozen = confidence(f, fc, m, th, m.train_domain)
    assert (frozen["domain"] == 0.4).all()                                  # everything is outside the training range
    # The pre-K11 behaviour (limits from the scored frame itself) would have called the same rows in-domain.
    own = {c: (np.nanpercentile(f[c], 1), np.nanpercentile(f[c], 99)) for c in OOD_COLUMNS}
    assert (confidence(f, fc, m, th, own)["domain"] == 1.0).mean() > 0.9
    legacy = confidence(f, fc, m, th, own)["confidence_score"]
    assert (frozen["confidence_score"] <= legacy).all() and (frozen["confidence_score"] < legacy).mean() > 0.9
