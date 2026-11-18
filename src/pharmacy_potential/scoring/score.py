"""Assemble versioned score results for tracks A and B, total and form level.

- Track A (external-only) never changes with sales; sales/readiness are attached only as
  explanatory columns.
- Track B uses the champion chosen on validation origins. Unobserved baselines get no fake gap.
- ``incremental_gap = max(0, benchmark - baseline)`` per month; a comparative gap, not a
  promised return. Scenario range, forecast interval and gap are separate columns.
- Score 0-100 is a relative percentile index, not a probability.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.backtest import Context
from ..features.build import _window, loo_median
from ..models.baselines import pct, rank_order
from ..models.explain import shap_values, top_reasons

FORMS = ["GUM", "PATCH", "LOZENGE", "SPRAY", "UNKNOWN"]


def confidence(f: pd.DataFrame, fc: pd.DataFrame, gbm, thresholds: dict, train_ranges: dict) -> pd.DataFrame:
    """Coverage, freshness, source mixing, peer size, interval width and training-domain similarity."""
    min_peers = thresholds["peer"]["min_peers"]
    comp = pd.DataFrame(index=f.index)
    comp["coverage"] = f["n_obs12"] / 12.0
    comp["freshness"] = np.select([f["months_since_obs"] == 0, f["months_since_obs"] <= 2], [1.0, 0.5], 0.0)
    comp["source_consistency"] = 1.0 - f["fallback_share12"].clip(0, 1)
    comp["peer_size"] = (f["n_peers"] / min_peers).clip(upper=1.0)
    width = (fc["fc_gbm_p90"] - fc["fc_gbm_p10"]) / (fc["fc_gbm_p50"] + 3.0)
    comp["interval"] = (1 - (width / 3).clip(0, 1)).to_numpy()
    in_dom = np.ones(len(f), dtype=bool)
    for col, (lo, hi) in train_ranges.items():
        v = f[col]
        in_dom &= (v.isna() | ((v >= lo) & (v <= hi))).to_numpy()
    comp["domain"] = np.where(in_dom, 1.0, 0.4)
    score = comp.mean(axis=1)
    label = np.select([score >= thresholds["confidence"]["high"], score >= thresholds["confidence"]["medium"]],
                      ["high", "medium"], "low")
    comp["confidence_score"] = score
    comp["confidence"] = label
    return comp


def _reasons_b(row: pd.Series) -> list[str]:
    r = []
    if pd.notna(row.peer_gap_rel) and row.peer_gap_rel >= 0.25:
        r.append(f"boxes below the peer group ({row.baseline_units:.1f} vs peer {row.peer_benchmark:.1f} boxes/month)")
    if row.ext_pct >= 0.7:
        r.append("high adult population per pharmacy in the district (scenario demand)")
    if pd.notna(row.get("ml_growth_pct")) and row.ml_growth_pct >= 0.8:
        r.append("model forecasts relative growth over the next 3 months versus the district")
    if row.low_sales_cohort:
        r.append("in the low current-sales cohort")
    if row.n_obs12 < 9:
        r.append(f"insufficient observations ({int(row.n_obs12)}/12 months observed)")
    if row.readiness != "high":
        r.append(f"supply evidence: {row.readiness}")
    if pd.notna(row.peer_gap_rel) and row.peer_gap_rel < 0.25:
        r.append("sales near or above the peer group")
    return (r + ["", "", ""])[:3]


def _reasons_a(row: pd.Series) -> list[str]:
    r = [f"district adults per pharmacy: {row.adults_per_pharmacy:,.0f} (external, vintage {int(row.external_vintage)})"
         if pd.notna(row.adults_per_pharmacy) else "no external data",
         f"scenario demand {row.scenario_units_per_month_base:.1f} boxes/month (not a measurement)"
         if pd.notna(row.scenario_units_per_month_base) else "",
         f"equal share within the district: {int(row.n_pharmacies_master)} pharmacies share the same score (tie)"]
    return r


def form_eligibility(forms: pd.Series, gap: pd.Series) -> pd.Series:
    """Eligibility of a form row: a gap is the only thing that makes it eligible; UNKNOWN is 'form_unmapped'."""
    return pd.Series(np.select([forms.eq("UNKNOWN"), gap.notna()], ["form_unmapped", "eligible"], "no_observed_baseline"),
                     index=forms.index)


def allocate_total_gap(total_gap: pd.Series, raw_gaps: pd.DataFrame) -> pd.DataFrame:
    """Distribute the TOTAL gap over forms in proportion to their positive raw gaps.

    The total definition is primary: raw form gaps (which may be negative) are never summed to
    replace it. If no form has a positive raw gap, the total stays in an UNALLOCATED residual.
    Returns one column per form plus ``UNALLOCATED``; NaN where the total gap is unknown.
    """
    pos = raw_gaps.clip(lower=0)
    pos_sum = pos.sum(axis=1, min_count=1)
    out = pd.DataFrame(index=raw_gaps.index)
    for form in pos.columns:
        share = pos[form] / pos_sum.where(pos_sum > 0)
        out[form] = np.where(total_gap.isna(), np.nan, np.where(pos_sum > 0, total_gap * share.fillna(0), 0.0))
    out["UNALLOCATED"] = np.where(total_gap.isna(), np.nan, np.where(pos_sum > 0, 0.0, total_gap))
    return out


def form_allocation(ctx: Context, thresholds: dict, total_gap: pd.Series) -> pd.DataFrame:
    """Form baselines/benchmarks, raw form gaps and allocation of the TOTAL gap."""
    panel, f = ctx.panel, ctx.features
    o = panel.month_idx(ctx.origin)
    gi = pd.Index(panel.glns).get_indexer(f["gln"])
    observed_total = ~np.isnan(panel.Y[:, :o + 1])
    rows = []
    raw = {}
    for form in FORMS:
        fy = panel.form_Y.get(form)
        if fy is None:
            continue
        fyo = np.where(observed_total, fy[:, :o + 1], np.nan)
        w3 = _window(fyo, o, 3)
        n_obs = np.sum(~np.isnan(w3), axis=1)
        with np.errstate(all="ignore"):
            base = np.where(n_obs >= 2, np.nanmean(w3, axis=1), np.nan)[gi]
        if form == "UNKNOWN":
            raw[form] = pd.Series(np.nan, index=f.index)
            rows.append(pd.DataFrame({"gln": f["gln"], "form": form, "baseline_units": base, "benchmark_units": np.nan,
                                      "raw_form_gap": np.nan}))
            continue
        norm = np.where(f["ext_base_units"] > 0, base / f["ext_base_units"], np.nan)
        med_d, n_d = loo_median(norm, f["district_code"].to_numpy())
        med_c, _ = loo_median(norm, f["city_code"].to_numpy())
        k = float(thresholds["peer"]["shrink_k"])
        wgt = n_d / (n_d + k)
        bench = np.where(np.isnan(med_d), med_c, wgt * np.nan_to_num(med_d) + (1 - wgt) * med_c) * f["ext_base_units"]
        gap = np.where(np.isnan(base), np.nan, bench - base)
        raw[form] = pd.Series(gap, index=f.index)
        rows.append(pd.DataFrame({"gln": f["gln"], "form": form, "baseline_units": base, "benchmark_units": bench,
                                  "raw_form_gap": gap}))
    out = pd.concat(rows, ignore_index=True)
    alloc = allocate_total_gap(total_gap, pd.DataFrame({k: v for k, v in raw.items() if k != "UNKNOWN"}))
    gl = f["gln"]
    out["allocated_gap"] = np.nan
    for form in alloc.columns.drop("UNALLOCATED"):
        m = out["form"] == form
        out.loc[m, "allocated_gap"] = pd.Series(alloc[form].to_numpy(), index=gl.to_numpy()).reindex(
            out.loc[m, "gln"]).to_numpy()
    # UNKNOWN form: a gap cannot be attributed to an unmapped form - null, never 0.
    out.loc[out["form"] == "UNKNOWN", "allocated_gap"] = np.nan
    resid = pd.DataFrame({"gln": gl, "form": "UNALLOCATED", "baseline_units": np.nan, "benchmark_units": np.nan,
                          "raw_form_gap": np.nan, "allocated_gap": alloc["UNALLOCATED"].to_numpy()})
    return pd.concat([out, resid], ignore_index=True)


def assemble(ctx: Context, ext: pd.DataFrame, champion: dict, settings, run_id: str,
             model_tuning: str = "tuned") -> dict[str, Any]:
    th = settings.thresholds
    seed = int(settings.get("seed"))
    f = ctx.features.copy()
    fc = ctx.forecasts.set_index("gln").reindex(f["gln"]).reset_index()
    s = ctx.scores.set_index("gln").reindex(f["gln"]).reset_index()
    ext_cols = ["gln", "adults_per_pharmacy", "n_pharmacies_master", "external_vintage", "scenario_units_per_month_low",
                "scenario_units_per_month_base", "scenario_units_per_month_high", "a_score_0_100", "a_score_raw",
                "tie_group", "a_eligibility"]
    f = f.merge(ext[ext_cols].drop(columns=["adults_per_pharmacy"]), on="gln", how="left")
    f["ext_pct"] = pct(f["ext_base_units"]).fillna(0)
    champ = champion["ranking_champion"]
    if "ml_growth" in s:
        f["ml_growth_pct"] = pct(s["ml_growth"]).to_numpy()

    # Training-domain ranges for the OOD component of confidence: frozen at training, never re-derived from the
    # population being scored (a fully shifted universe would otherwise redefine its own limits; ).
    conf = confidence(f, fc, ctx.gbm, th, ctx.gbm.train_domain)

    # SHAP on the p50 model (log1p space) -> top-3 model reasons.
    X = f[ctx.gbm.features].astype(float)
    sv = shap_values(ctx.gbm.models[0.5], X)
    ml_reasons = top_reasons(sv, ctx.gbm.features)

    fc_method = champion["forecast_champion"]
    point = fc[f"fc_{fc_method}"] if f"fc_{fc_method}" in fc else fc["fc_gbm_p50"]
    common = pd.DataFrame({
        "run_id": run_id, "score_month": ctx.origin.strftime("%Y-%m"), "gln": f["gln"], "city_code": f["city_code"],
        "district_code": f["district_code"],
        "cohort": np.select([f["baseline_units"].isna(), f["low_sales_cohort"].astype(bool)],
                            ["no_observed_baseline", "low_sales"], "regular"),
        "observed_months_12": f["n_obs12"].astype(int),
        "observation_status_last": np.where(f["months_since_obs"] == 0, "observed", "unobserved"),
        "baseline_units_per_month": f["baseline_units"],
        "benchmark_units_per_month": np.where(f["baseline_units"].notna(), f["peer_benchmark"], np.nan),
        "incremental_gap_units_per_month": f["peer_gap_units"],
        "scenario_units_per_month_low": f["scenario_units_per_month_low"],
        "scenario_units_per_month_base": f["scenario_units_per_month_base"],
        "scenario_units_per_month_high": f["scenario_units_per_month_high"],
        "forecast_next3m_point": point.to_numpy(), "forecast_method": fc_method,
        "forecast_next3m_p10": fc["fc_gbm_p10"].to_numpy(), "forecast_next3m_p50": fc["fc_gbm_p50"].to_numpy(),
        "forecast_next3m_p90": fc["fc_gbm_p90"].to_numpy(),
        "readiness": f["readiness"], "readiness_reason": f["readiness_reason"],
        "confidence": conf["confidence"], "confidence_score": conf["confidence_score"],
        "source_policy": settings.policy["policy_version"],
        "data_cutoff_utc": pd.Timestamp(ctx.cutoff).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": f"gbm_q_leaves{ctx.gbm.params['num_leaves']}|hw_{ctx.hw_params['alpha']}_{ctx.hw_params['beta']}_"
                         f"{ctx.hw_params['gamma']}|champ_{champ}",
        # : a run that used fixed engineering defaults says so on every row.
        "model_tuning": model_tuning,
    })

    # --- track A -------------------------------------------------------------------------
    a = common.copy()
    a["track"] = "A_external_only"
    a["form_or_total"] = "TOTAL"
    a["score_0_100"] = f["a_score_0_100"]
    a["rank"] = f["ext_base_units"].rank(ascending=False, method="min").astype("Int64")
    a["tie_group"] = f["tie_group"]
    a["eligibility"] = f["a_eligibility"]
    ra = f.apply(_reasons_a, axis=1, result_type="expand")
    for i in range(3):
        a[f"reason_{i + 1}"] = ra[i]
        a[f"ml_reason_{i + 1}"] = ""

    # --- track B -------------------------------------------------------------------------
    b = common.copy()
    b["track"] = "B_sales_aware"
    b["form_or_total"] = "TOTAL"
    sc = s[champ]
    elig = f["baseline_units"].notna() & sc.notna()
    b["score_0_100"] = np.where(elig, 100 * sc.rank(pct=True, method="average"), np.nan)
    b["rank"] = rank_order(sc, s["_tiebreak"]).where(elig).astype("Int64")
    b["tie_group"] = ""
    b["eligibility"] = np.where(elig, "eligible", "no_observed_baseline")
    rb = f.assign(readiness=f["readiness"]).apply(_reasons_b, axis=1, result_type="expand")
    for i in range(3):
        b[f"reason_{i + 1}"] = rb[i]
        b[f"ml_reason_{i + 1}"] = [m[i] for m in ml_reasons]

    # --- form rows (track B) ---------------------------------------------------------------
    fa = form_allocation(ctx, th, f["peer_gap_units"])
    fa = fa.merge(common.drop(columns=["baseline_units_per_month", "benchmark_units_per_month",
                                       "incremental_gap_units_per_month"]), on="gln", how="left")
    fa["track"] = "B_sales_aware"
    fa["form_or_total"] = fa["form"]
    fa = fa.rename(columns={"baseline_units": "baseline_units_per_month", "benchmark_units": "benchmark_units_per_month",
                            "allocated_gap": "incremental_gap_units_per_month"})
    fa["score_0_100"] = fa.groupby("form")["incremental_gap_units_per_month"].rank(pct=True) * 100
    fa["rank"] = fa.groupby("form")["incremental_gap_units_per_month"].rank(ascending=False, method="min").astype("Int64")
    fa["tie_group"] = ""
    fa["eligibility"] = form_eligibility(fa["form"], fa["incremental_gap_units_per_month"])
    for i in range(3):
        fa[f"reason_{i + 1}"] = ""
        fa[f"ml_reason_{i + 1}"] = ""
    fa["reason_1"] = np.where(fa["form"] == "UNALLOCATED", "total gap could not be allocated to a positive form gap (residual)",
                     np.where(fa["form"] == "UNKNOWN", "no form mapping: kept separate, gap not allocated",
                              "total gap allocated proportionally to the positive raw form gaps"))
    fa = fa.drop(columns=["form"])

    cols = ["run_id", "score_month", "gln", "city_code", "district_code", "track", "form_or_total", "score_0_100", "rank",
            "tie_group", "eligibility", "cohort", "observed_months_12", "observation_status_last",
            "baseline_units_per_month", "benchmark_units_per_month", "incremental_gap_units_per_month", "raw_form_gap",
            "scenario_units_per_month_low", "scenario_units_per_month_base", "scenario_units_per_month_high",
            "forecast_next3m_point", "forecast_method", "forecast_next3m_p10", "forecast_next3m_p50", "forecast_next3m_p90",
            "readiness", "readiness_reason", "confidence", "confidence_score",
            "reason_1", "reason_2", "reason_3", "ml_reason_1", "ml_reason_2", "ml_reason_3",
            "source_policy", "data_cutoff_utc", "model_version", "model_tuning"]
    for d in (a, b):
        d["raw_form_gap"] = np.nan
    result = pd.concat([a[cols], b[cols], fa[cols]], ignore_index=True)
    result = result.sort_values(["track", "form_or_total", "rank", "gln"], kind="mergesort",
                                na_position="last").reset_index(drop=True)
    return {"score_result": result, "shap_values": sv, "shap_X": X, "confidence_components": conf,
            "features": f}
