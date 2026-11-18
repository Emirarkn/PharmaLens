"""Rolling-origin backtest with a locked final test and a holdout city.

Protocol: for every evaluation origin E the evaluation features use only what was known at cutoff(E), and
every training origin o uses a panel built as of cutoff(o). Labels and the model target come
from the matured panel of E (gross outflow, K3). Validation origins choose the champion; the locked test
origin and the holdout city are reported, never used for tuning or selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..features.asof import AsOfPanels
from ..features.build import (Panel, build_panel, features_for_origin, labels_for_origin, labels_peer_closure,
                              training_frame, training_origins)
from ..models.baselines import COMPLEXITY, SALES_AWARE, SIMPLE_BASELINES, candidate_scores
from ..models.gbm import QuantileModels, fit_quantile_models
from ..models.ts import holt_winters, moving_average, seasonal_naive, tune_holt_winters
from ..readiness.readiness import build_readiness
from ..reconcile.canonical import build_canonical
from ..utils import add_months, backtest_cutoff, get_logger, month_range, month_start, sha256_text
from .metrics import cluster_bootstrap_diff, evaluate_ranking, forecast_metrics

log = get_logger()


@dataclass
class Context:
    origin: pd.Timestamp
    cutoff: datetime
    canonical: Any
    panel: Panel
    features: pd.DataFrame
    gbm: QuantileModels
    hw_params: dict
    forecasts: pd.DataFrame
    scores: pd.DataFrame
    train_audit: list[dict] = field(default_factory=list)   # (origin, cutoff, last month) of each training panel


def protocol(settings) -> dict[str, Any]:
    th = settings.thresholds
    mc = settings.get("model")
    p = {
        "protocol_version": "evaluation_protocol_v001",
        "status": "frozen_for_demo_by_agent__G2_owner_approval_pending",
        "horizon_months": settings.get("horizon_months"),
        "validation_origins": mc["validation_origins"], "test_origin": mc["test_origin"],
        "holdout_city": mc["holdout_city"], "min_train_origin": mc["min_train_origin"],
        "purge": "train label windows end <= evaluation origin; inner validation purged by horizon",
        "train_origin_step_months": mc.get("train_origin_step_months", 3),
        "train_features": "each training origin uses a panel built as of cutoff(origin)",
        "model_target": "next-3-month gross outflow: paid+free (mutabakat) / non-negative VENDOR units; returns excluded",
        "cutoff_rule": f"month_end(origin) + {settings.get('backtest_cutoff_lag_days')} days",
        "proxy_label": th["proxy_label"], "low_sales_cohort": th["low_sales_cohort"], "peer": th["peer"],
        "k_values": th["k_values"], "g3_targets": th["g3_targets"], "bootstrap": th["bootstrap"],
        "tie_break": "fixed blake2b hash of (seed, GLN)", "seed": settings.get("seed"),
        "candidates": {"simple_baselines": SIMPLE_BASELINES, "sales_aware": SALES_AWARE},
        "forecast_candidates": ["seasonal_naive", "moving_average_3", "holt_winters", "gbm_p50"],
        "eligible_universe": "active at origin, baseline observed (>=2 of last 3 months), label not censored",
    }
    p["protocol_hash"] = sha256_text(repr(sorted((k, repr(v)) for k, v in p.items())))
    return p


def select_hw_params(panel: Panel, origin, horizon: int, grid: dict, min_train_origin, exclude_city: str | None = None,
                     tune: bool = True) -> dict:
    """Holt-Winters smoothing parameters. The pooled-WAPE selection never sees the holdout city and, while
    tuning is not allowed (G2 pending on real data), is replaced by the first grid value of each parameter."""
    if not tune:
        return {"alpha": grid["alpha"][0], "beta": grid["beta"][0], "gamma": grid["gamma"][0], "wape": None,
                "selection": "fixed (hyper-parameter selection not allowed: G2 not accepted)"}
    keep = np.ones(len(panel.glns), dtype=bool) if not exclude_city else (panel.city != exclude_city)
    first = max(month_start(min_train_origin), add_months(origin, -horizon - 5))
    tune_idx = [panel.month_idx(o) for o in month_range(first, add_months(origin, -horizon))]
    o_idx = panel.month_idx(origin)
    return tune_holt_winters(panel.Yg[keep, :o_idx + 1], tune_idx, grid, horizon)["best"]


def build_context(con: duckdb.DuckDBPyConnection, settings, origin, cutoff: datetime, tag: str,
                  exclude_city: str | None, final_panel: Panel | None = None, tune: bool = True,
                  asof: AsOfPanels | None = None, preloaded: tuple[QuantileModels, dict] | None = None) -> Context:
    """``preloaded`` = (models, holt-winters params) of a published run: scoring then uses them as they are -
    no training frame, no training, no tuning (reproducibility check (a), )."""
    th, scen, mc = settings.thresholds, settings.scenario, settings.get("model")
    h = int(settings.get("horizon_months"))
    seed = int(settings.get("seed"))
    origin = month_start(origin)
    canon = build_canonical(con, settings, cutoff, origin, tag)
    panel = build_panel(con, canon.table, cutoff)        # evaluation panel = matured label panel for training rows

    if preloaded is None:
        asof = asof or AsOfPanels(con, settings)
        n_audit = len(asof.log)
        train_origins = training_origins(mc, origin, h)
        train = training_frame(asof.panel, panel, train_origins, th, scen, int(settings.get("backtest_cutoff_lag_days")))
        if exclude_city:
            train = train[train["city_code"] != exclude_city]
        gbm = fit_quantile_models(train, mc["gbm"], seed, int(settings.get("num_threads")), h, tune=tune)
    else:
        gbm = preloaded[0]
        train_origins, n_audit = [], 0
        asof = asof or SimpleNamespace(log=[], hits=0, builds=0)

    feats = features_for_origin(panel, origin, cutoff, th, scen)
    ready = build_readiness(con, cutoff, origin, th, settings.get("brand"))
    feats = feats.merge(ready, on="gln", how="left")
    preds = gbm.predict(feats)

    o_idx = panel.month_idx(origin)
    gi = pd.Index(panel.glns).get_indexer(feats["gln"])
    hw = preloaded[1] if preloaded else select_hw_params(panel, origin, h, mc["holt_winters_grid"],
                                                         mc["min_train_origin"], exclude_city, tune)
    Yh = panel.Yg[gi, :o_idx + 1]                          # forecasts are compared with the gross target
    fc = pd.DataFrame({
        "gln": feats["gln"],
        "fc_seasonal_naive": seasonal_naive(Yh, h).sum(axis=1),
        "fc_moving_average_3": moving_average(Yh, h).sum(axis=1),
        "fc_holt_winters": holt_winters(Yh, hw["alpha"], hw["beta"], hw["gamma"], h).sum(axis=1),
        "fc_gbm_p10": preds["p10"].to_numpy(), "fc_gbm_p50": preds["p50"].to_numpy(),
        "fc_gbm_p90": preds["p90"].to_numpy(),
    })
    scores = candidate_scores(feats, seed, settings.get("score_weights"), preds["p50"])
    log.info("context %s: %d GLN, train rows %d (%d quarterly origins; canonical cache %d hit / %d built), "
             "gbm leaves %s, HW %s", tag, len(feats), gbm.train_rows, len(train_origins), asof.hits, asof.builds,
             gbm.params["num_leaves"], {k: hw[k] for k in ("alpha", "beta", "gamma")})
    return Context(origin, cutoff, canon, panel, feats, gbm, hw, fc, scores, asof.log[n_audit:])


FORECAST_METHODS = ("seasonal_naive", "moving_average_3", "holt_winters", "gbm_p50")


def forecast_block(df: pd.DataFrame, holdout: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Forecast metrics on ONE universe: GLNs with an observed target (holdout city excluded) for which EVERY
    candidate produced a forecast. A candidate that cannot forecast the hard rows no longer looks better by being
    scored on fewer rows; each candidate's own coverage of the observed-target universe is reported."""
    observed = df["target_units_next3"].notna() & (df["city_code"] != holdout)
    common = observed.copy()
    for m in FORECAST_METHODS:
        common &= df[f"fc_{m}"].notna()
    out = {}
    for m in FORECAST_METHODS:
        out[m] = forecast_metrics(df.loc[common, "target_units_next3"], df.loc[common, f"fc_{m}"],
                                  df.loc[common, "fc_gbm_p10"] if m == "gbm_p50" else None,
                                  df.loc[common, "fc_gbm_p90"] if m == "gbm_p50" else None)
        out[m]["coverage"] = float((observed & df[f"fc_{m}"].notna()).sum() / observed.sum()) if observed.any() else None
    return out, {"n_common": int(common.sum()), "n_actual_observed": int(observed.sum())}


def _evaluate_origin(ctx: Context, final_panel: Panel, settings, role: str) -> dict[str, Any]:
    th = settings.thresholds
    k_values = th["k_values"]
    holdout = settings.get("model")["holdout_city"]
    lab = labels_for_origin(final_panel, ctx.origin, th)
    df = ctx.features[["gln", "city_code", "district_code", "baseline_units", "low_sales_cohort"]].merge(
        lab[["gln", "proxy_positive", "graded_label", "target_units_next3", "label_status"]], on="gln", how="left")
    df = df.merge(ctx.scores, on="gln", how="left")
    df = df.merge(ctx.forecasts, on="gln", how="left")
    df["label"] = df["proxy_positive"]
    eligible = df["label"].notna() & df["baseline_units"].notna()
    res: dict[str, Any] = {
        "role": role, "origin": ctx.origin.strftime("%Y-%m"), "cutoff_utc": ctx.cutoff,
        "active_glns": int(len(df)), "eligible": int(eligible.sum()),
        "censored_or_ineligible": int((~eligible).sum()),
        "label_status_counts": df["label_status"].value_counts().to_dict(),
        "canonical_checks": {k: ctx.canonical.report[k] for k in ("pk_unique", "fk_orphans_gln", "status_counts",
                                                                   "sources_not_summed", "conflict_flag_cells")},
    }
    cands = [c for c in SIMPLE_BASELINES + SALES_AWARE if c in df.columns]
    cands = list(dict.fromkeys(cands))
    subsets = {"overall_excl_holdout": eligible & (df["city_code"] != holdout),
               "holdout_city": eligible & (df["city_code"] == holdout),
               "low_sales_excl_holdout": eligible & (df["city_code"] != holdout) & df["low_sales_cohort"].astype(bool)}
    for city in sorted(df["city_code"].unique()):
        subsets[f"city_{city}"] = eligible & (df["city_code"] == city)
    res["subsets"] = {}
    for name, mask in subsets.items():
        sub = df[mask]
        entry = {"n": int(len(sub)), "positives": int(sub["label"].sum()),
                 "prevalence": float(sub["label"].mean()) if len(sub) else None, "candidates": {}}
        for c in cands:
            entry["candidates"][c] = evaluate_ranking(sub[c], sub["label"], sub["graded_label"], sub["_tiebreak"], k_values)
        res["subsets"][name] = entry

    res["forecast"], res["forecast_universe"] = forecast_block(df, holdout)
    res["gbm"] = {"num_leaves": ctx.gbm.params["num_leaves"], "rounds": {str(k): v for k, v in ctx.gbm.best_rounds.items()},
                  "train_rows": ctx.gbm.train_rows, "train_origins": [ctx.gbm.train_origins[0], ctx.gbm.train_origins[-1]],
                  "selection": ctx.gbm.selection}
    res["holt_winters"] = ctx.hw_params
    res["_frame"] = df[eligible]
    res["_all"] = df
    return res


def evaluate_alt_label(ctx: Context, final_panel: Panel, settings, df_scores: pd.DataFrame) -> dict[str, Any] | None:
    """Draft alternative label (thresholds_v002), VALIDATION origins only, same eligible-universe logic as the
    primary label. Reported side by side; it never selects a champion and is never computed for the locked test."""
    alt = settings.alt_thresholds
    if not alt:
        return None
    holdout = settings.get("model")["holdout_city"]
    lab = labels_peer_closure(final_panel, ctx.origin, ctx.features, alt)
    df = df_scores.merge(lab, on="gln", how="left")
    df["label"] = df["alt_positive"]
    sub = df[df["label"].notna() & df["baseline_units"].notna() & (df["city_code"] != holdout)]
    k_values = settings.thresholds["k_values"]
    cands = list(dict.fromkeys(c for c in SIMPLE_BASELINES + SALES_AWARE if c in df.columns))
    return {"label_version": alt["thresholds_version"], "approval_status": alt["approval_status"],
            "n": int(len(sub)), "positives": int(sub["label"].sum()),
            "prevalence": float(sub["label"].mean()) if len(sub) else None,
            "candidates": {c: evaluate_ranking(sub[c], sub["label"], sub["label"], sub["_tiebreak"], k_values)
                           for c in cands}}


def _p(res: dict, subset: str, cand: str, k: int = 100) -> float | None:
    try:
        return res["subsets"][subset]["candidates"][cand][f"P@{k}"]["precision"]
    except KeyError:
        return None


def choose_champion(val_results: list[dict]) -> dict[str, Any]:
    """Highest mean validation P@100 among sales-aware candidates; a more complex candidate must
    beat the best simpler one by >= 2% relative, else the simpler wins. Without any validation
    evidence the simplest candidate is returned."""
    means = {}
    for c in SALES_AWARE:
        vals = [_p(r, "overall_excl_holdout", c) for r in val_results]
        vals = [v for v in vals if v is not None]
        if vals:
            means[c] = float(np.mean(vals))
    ordered = sorted(means, key=lambda c: COMPLEXITY[c])
    champ = ordered[0] if ordered else min(SALES_AWARE, key=lambda c: COMPLEXITY[c])
    # A more complex candidate must beat the BEST simpler candidate (not merely the one chosen so far) by >= 2%.
    for i, c in enumerate(ordered[1:], start=1):
        best_simpler = max(means[k] for k in ordered[:i])
        if means[c] >= best_simpler * 1.02 and means[c] > best_simpler:
            champ = c
    fc_means = {}
    for m in ("seasonal_naive", "moving_average_3", "holt_winters", "gbm_p50"):
        w = [r["forecast"][m]["wape"] for r in val_results if r["forecast"][m]["wape"] is not None]
        if w:
            fc_means[m] = float(np.mean(w))
    simple_fc = min((fc_means[m] for m in ("seasonal_naive", "moving_average_3") if m in fc_means), default=None)
    fc_champ = min(fc_means, key=fc_means.get) if fc_means else "moving_average_3"
    if fc_champ in ("holt_winters", "gbm_p50") and simple_fc is not None and fc_means[fc_champ] > simple_fc * 0.98:
        fc_champ = min(("seasonal_naive", "moving_average_3"), key=lambda m: fc_means.get(m, np.inf))
    return {"ranking_champion": champ, "validation_mean_P@100": means, "forecast_champion": fc_champ,
            "validation_mean_wape": fc_means,
            "rule": "simplest sales-aware candidate unless a more complex one is >=2% better on mean validation P@100; "
                    "complex forecasters must beat the best simple forecaster WAPE by >=2%"}


def acceptance(test: dict, val_results: list[dict], champion: str, settings) -> dict[str, Any]:
    g = settings.thresholds["g3_targets"]
    k = 100
    overall = test["subsets"]["overall_excl_holdout"]
    hold = test["subsets"]["holdout_city"]
    low = test["subsets"]["low_sales_excl_holdout"]
    simple = [c for c in SIMPLE_BASELINES if c != champion]
    best_simple = max(simple, key=lambda c: _p(test, "overall_excl_holdout", c) or -1)
    p_ch = _p(test, "overall_excl_holdout", champion)
    p_bs = _p(test, "overall_excl_holdout", best_simple)
    lift = overall["candidates"][champion][f"P@{k}"]["lift"]
    rel = (p_ch / p_bs - 1) if (p_ch is not None and p_bs) else None
    low_best = max(simple, key=lambda c: _p(test, "low_sales_excl_holdout", c) or -1)
    low_ok = (_p(test, "low_sales_excl_holdout", champion) or 0) > (_p(test, "low_sales_excl_holdout", low_best) or 0)
    consistent = sum(1 for r in val_results if (_p(r, "overall_excl_holdout", champion) or 0) >
                     max(_p(r, "overall_excl_holdout", c) or 0 for c in simple))
    hold_best = max(simple, key=lambda c: _p(test, "holdout_city", c) or -1)
    hold_ok = (_p(test, "holdout_city", champion) or 0) > (_p(test, "holdout_city", hold_best) or 0)
    sample_ok = (overall["n"] >= g["min_eligible_overall"] and overall["positives"] >= g["min_positive_overall"])
    hold_sample_ok = (hold["n"] >= g["min_eligible_holdout"] and hold["positives"] >= g["min_positive_holdout"])
    checks = {
        "lift_vs_random_at_100": {"value": lift, "target": g["min_lift_vs_random_at_100"],
                                  "pass": lift is not None and lift >= g["min_lift_vs_random_at_100"]},
        "rel_improvement_vs_best_simple_at_100": {"value": rel, "best_simple": best_simple,
                                                  "target": g["min_rel_improvement_vs_best_simple_at_100"],
                                                  "pass": rel is not None and rel >= g["min_rel_improvement_vs_best_simple_at_100"]},
        "low_sales_cohort_improves": {"best_simple_in_cohort": low_best, "pass": bool(low_ok)},
        "validation_origins_consistent": {"value": consistent, "target": g["min_origins_consistent"],
                                          "pass": consistent >= g["min_origins_consistent"]},
        "holdout_direction_consistent": {"best_simple_in_holdout": hold_best, "pass": bool(hold_ok)},
        "min_sample_overall": {"n": overall["n"], "positives": overall["positives"], "pass": bool(sample_ok)},
        "min_sample_holdout": {"n": hold["n"], "positives": hold["positives"], "pass": bool(hold_sample_ok)},
    }
    if not (sample_ok and hold_sample_ok):
        decision = "inconclusive"
    else:
        decision = "pass" if all(c["pass"] for c in checks.values()) else "fail"
    return {"gate": "G3", "decision_proposal": decision, "champion": champion, "checks": checks,
            "evidence": "proxy label only; not causal; G2 owner approval pending (thresholds frozen for demo)",
            "note": "The coding agent never marks a gate accepted; this is machine-readable evidence for the owner."}


def run_backtest(con: duckdb.DuckDBPyConnection, settings, final_panel: Panel,
                 asof: AsOfPanels | None = None) -> dict[str, Any]:
    mc = settings.get("model")
    asof = asof or AsOfPanels(con, settings)
    lag = int(settings.get("backtest_cutoff_lag_days"))
    holdout = mc["holdout_city"]
    results, frames = [], {}
    for role, origin in [("validation", o) for o in mc["validation_origins"]] + [("test", mc["test_origin"])]:
        cutoff = backtest_cutoff(origin, lag)
        ctx = build_context(con, settings, origin, cutoff, f"bt_{origin.replace('-', '')}", holdout, asof=asof)
        res = _evaluate_origin(ctx, final_panel, settings, role)
        frame, everyone = res.pop("_frame"), res.pop("_all")
        if role == "validation":      # the locked test origin is never evaluated with the draft label
            res["alt_label_v002"] = evaluate_alt_label(ctx, final_panel, settings, everyone)
        frames[origin] = frame
        results.append(res)
        if role == "test":
            sens = source_sensitivity(con, settings, ctx, final_panel)
        con.execute(f"drop table if exists {ctx.canonical.table}")
    val = [r for r in results if r["role"] == "validation"]
    test = [r for r in results if r["role"] == "test"][0]
    champ = choose_champion(val)
    best_simple = max([c for c in SIMPLE_BASELINES if c != champ["ranking_champion"]],
                      key=lambda c: _p(test, "overall_excl_holdout", c) or -1)
    tf = frames[mc["test_origin"]]
    tf = tf[tf["city_code"] != holdout]
    boot = cluster_bootstrap_diff(tf, champ["ranking_champion"], best_simple, 100,
                                  settings.thresholds["bootstrap"]["reps"], int(settings.get("seed")))
    return {"protocol": protocol(settings), "origins": results, "champion": champ,
            "bootstrap_test_P@100_vs_best_simple": {"best_simple": best_simple, **boot},
            "acceptance": acceptance(test, val, champ["ranking_champion"], settings),
            "source_sensitivity": sens}


def source_sensitivity(con, settings, ctx: Context, final_panel: Panel) -> dict[str, Any]:
    """Source-fixed sensitivity: how the simple peer-gap ranking moves under primary-only / fallback-only."""
    th, scen = settings.thresholds, settings.scenario
    base_rank = ctx.scores.set_index("gln")["peer_gap"].rank(ascending=False, method="first")
    top_base = set(base_rank[base_rank <= 100].index)
    out = {}
    for mode in ("primary_only", "fallback_only"):
        canon = build_canonical(con, settings, ctx.cutoff, ctx.origin, f"sens_{mode}", mode=mode)
        panel = build_panel(con, canon.table, ctx.cutoff)
        if mode == "fallback_only":
            from ..reconcile.canonical import gln_month_totals
            tot = gln_month_totals(con, canon.table, partial_as_observed=True)
            from ..features.build import _pivot
            panel.Y = _pivot(tot, panel.glns, panel.months, "units")
        f = features_for_origin(panel, ctx.origin, ctx.cutoff, th, scen)
        r = f.set_index("gln")["peer_gap_rel"].rank(ascending=False, method="first")
        top = set(r[r <= 100].index)
        out[mode] = {"eligible_glns": int(f["baseline_units"].notna().sum()),
                     "top100_overlap_with_default": len(top & top_base) / 100.0,
                     "canonical_status_counts": canon.report["status_counts"],
                     "note": "fallback_only treats VENDOR-partial months as lower bounds (sensitivity only)"
                     if mode == "fallback_only" else "mutabakat cells only; no VENDOR fallback"}
        con.execute(f"drop table if exists {canon.table}")
    return out
