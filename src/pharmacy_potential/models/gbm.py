"""Global LightGBM quantile models for next-3-month DemoBrand units.

Target is log1p(gross outflow over the next 3 months: paid + free, returns excluded - ). It can
never be negative; a non-finite log target is a bug and stops the fit. Quantiles are invariant under the
monotone transform, so p10/p50/p90 are back-transformed directly. Hyper-parameters are chosen on the
last train origins (time-ordered, purged) only - never on the locked test or holdout city - and only when the
protocol gate allows tuning; otherwise the fixed ``untuned`` settings of the config are used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import MODEL_FEATURES

# Features whose training range defines the confidence "domain" component (frozen at training, ).
OOD_COLUMNS = ("mean3", "ext_base_units")


@dataclass
class QuantileModels:
    models: dict[float, lgb.Booster]
    params: dict[str, Any]
    features: list[str]
    best_rounds: dict[float, int]
    selection: list[dict] = field(default_factory=list)
    train_rows: int = 0
    train_origins: list[str] = field(default_factory=list)
    max_log_target: float = 20.0
    train_domain: dict[str, tuple[float, float]] = field(default_factory=dict)   # 1st/99th percentile of the training frame
    tuned: bool = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=df.index)
        cap = self.max_log_target * 1.25   # quantiles never extrapolate far beyond the observed target range
        for q, m in self.models.items():
            raw = np.clip(m.predict(X, num_iteration=self.best_rounds[q]), 0, cap)
            out[f"p{int(q * 100)}"] = np.expm1(raw)
        cols = sorted(out.columns, key=lambda c: int(c[1:]))
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)   # no quantile crossing
        return out


def _params(cfg: dict, num_leaves: int, alpha: float, seed: int, threads: int) -> dict:
    return {
        "objective": "quantile", "alpha": alpha, "learning_rate": cfg["learning_rate"], "num_leaves": num_leaves,
        "min_data_in_leaf": cfg["min_data_in_leaf"], "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"], "bagging_freq": cfg["bagging_freq"], "lambda_l2": cfg["lambda_l2"],
        "seed": seed, "deterministic": True, "force_row_wise": True, "num_threads": threads, "verbose": -1,
    }


def _pinball(y: np.ndarray, p: np.ndarray, q: float) -> float:
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def load_quantile_models(art_dir: Path) -> tuple[QuantileModels, dict]:
    """Rebuild the exact model set of a published run from its artifacts (model_card.json + gbm_q*.txt).
    Returns the models and the Holt-Winters parameters recorded with them."""
    card = json.loads((Path(art_dir) / "model_card.json").read_text(encoding="utf-8"))
    if card["features"] != MODEL_FEATURES:
        raise ValueError("saved model was trained on a different feature list than the current code")
    models = {}
    for q in (float(x) for x in card["quantiles"]):
        text = (Path(art_dir) / f"gbm_q{int(round(q * 100))}.txt").read_text(encoding="utf-8")
        models[q] = lgb.Booster(model_str=text)
    qm = QuantileModels(models, card["params"], list(card["features"]),
                        {float(k): int(v) for k, v in card["rounds"].items()}, card.get("selection", []),
                        int(card["train_rows"]), list(card["train_origins"]), float(card["max_log_target"]),
                        {c: (float(lo), float(hi)) for c, (lo, hi) in card["train_domain"].items()},
                        bool(card.get("tuned", True)))
    return qm, card["holt_winters"]


def check_finite_log_target(y: pd.Series) -> np.ndarray:
    """log1p of the target, refusing anything that would give -inf/NaN (a negative or non-numeric target)."""
    arr = np.log1p(np.asarray(y, dtype=float))
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{bad} target value(s) are < -1 or non-finite after log1p: the model target must be a "
                         "non-negative gross outflow (returns are kept out of the target)")
    return arr


def fit_quantile_models(train: pd.DataFrame, cfg: dict, seed: int, threads: int, horizon: int = 3,
                        tune: bool = True) -> QuantileModels:
    data = train[train["target_units_next3"].notna()].copy()
    if (data["target_units_next3"] < 0).any():
        raise ValueError("negative model target: use gross outflow (paid + free), not net units")
    data = data.sort_values(["origin_month", "gln"], kind="mergesort").reset_index(drop=True)
    domain = {c: (float(np.nanpercentile(data[c], 1)), float(np.nanpercentile(data[c], 99))) for c in OOD_COLUMNS}
    origins = sorted(data["origin_month"].unique())
    # Inner validation = last 3 origins; purge inner-train origins whose label window reaches them.
    val_origins = origins[-3:]
    purge_before = pd.Timestamp(val_origins[0]) - pd.DateOffset(months=horizon)
    inner_train = data[data["origin_month"] <= purge_before]
    inner_val = data[data["origin_month"].isin(val_origins)]
    y_tr, y_va = (pd.Series(check_finite_log_target(inner_train["target_units_next3"])),
                  pd.Series(check_finite_log_target(inner_val["target_units_next3"])))
    X_tr, X_va = inner_train[MODEL_FEATURES].astype(float), inner_val[MODEL_FEATURES].astype(float)

    selection, best = [], None
    if not tune:
        fixed = cfg["untuned"]      # G2 not accepted: no validation-based selection of any hyper-parameter
        best = {"num_leaves": int(fixed["num_leaves"]), "best_iteration": int(fixed["rounds"] / 1.1),
                "val_pinball_q50": None}
        selection.append({"num_leaves": best["num_leaves"], "best_iteration": None, "val_pinball_q50": None,
                          "note": "fixed settings; hyper-parameters not selected (G2 not accepted)"})
    for nl in (cfg["num_leaves_grid"] if tune else []):
        p = _params(cfg, nl, 0.5, seed, threads)
        booster = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=cfg["max_rounds"],
                            valid_sets=[lgb.Dataset(X_va, y_va)],
                            callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
        loss = _pinball(y_va.to_numpy(), booster.predict(X_va, num_iteration=booster.best_iteration), 0.5)
        selection.append({"num_leaves": nl, "best_iteration": int(booster.best_iteration), "val_pinball_q50": loss})
        if best is None or loss < best["val_pinball_q50"]:
            best = selection[-1]

    models, rounds = {}, {}
    X_all, y_all = data[MODEL_FEATURES].astype(float), pd.Series(check_finite_log_target(data["target_units_next3"]))
    for q in cfg["quantiles"]:
        p = _params(cfg, best["num_leaves"], q, seed, threads)
        n_rounds = int(cfg["untuned"]["rounds"]) if not tune else max(50, int(best["best_iteration"] * 1.1))
        models[q] = lgb.train(p, lgb.Dataset(X_all, y_all), num_boost_round=n_rounds)
        rounds[q] = n_rounds
    return QuantileModels(models, {"num_leaves": best["num_leaves"], **{k: v for k, v in cfg.items()
                                                                         if k != "num_leaves_grid"}},
                          list(MODEL_FEATURES), rounds, selection, len(data),
                          [pd.Timestamp(o).strftime("%Y-%m") for o in origins], float(y_all.max()), domain, tune)
