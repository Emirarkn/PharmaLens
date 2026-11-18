"""Leakage-safe features, peers and proxy labels.

A ``Panel`` holds the GLN x month DemoBrand totals of ONE canonical build (as of one cutoff).
Features for origin ``o`` are read from a panel built AS OF ``cutoff(o)`` (: training origins never
see a later delivery or revision of an earlier month), and side data (visits, orders, campaigns, external
statistics) only where ``available_at <= cutoff(o)``. CRM and campaign counts are numbers only where a batch
known at the cutoff covers the window; otherwise they are missing, not zero.
Labels are built by a separate function from a (possibly later, matured) panel; features never see them.
GLN and pharmacy name are identifiers only - never model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd

from ..db import active_batches_sql, table_exists
from ..demand.external import scenario_rate
from ..reconcile.canonical import form_month_totals, gln_month_totals
from ..utils import backtest_cutoff, month_start

MODEL_FEATURES = [
    "lag0", "lag1", "lag2", "lag3", "mean3", "mean6", "mean12", "n_obs3", "n_obs12", "cv12", "trend12",
    "yoy_ratio", "months_since_obs", "zero_share12", "max12", "target_month", "peer_benchmark",
    "ratio_to_peer", "peer_gap_rel", "n_peers", "adults_per_pharmacy", "pharmacies_per_10k_adults",
    "ext_base_units", "visits_3m", "visits_12m", "order_units_3m", "known_campaign_next3", "crm_coverage_flag",
    "gum_share12", "patch_share12", "fallback_share12",
]

FEATURE_DICTIONARY = {
    "lag0": ("boxes/month", "observed total DemoBrand net boxes in the origin month (empty if not observed)"),
    "lag1": ("boxes/month", "origin-1 month"), "lag2": ("boxes/month", "origin-2 month"), "lag3": ("boxes/month", "origin-3 month"),
    "mean3": ("boxes/month", "observed mean of the last 3 months (baseline)"),
    "mean6": ("boxes/month", "observed mean of the last 6 months"), "mean12": ("boxes/month", "observed mean of the last 12 months"),
    "n_obs3": ("months", "number of observed months in the last 3"), "n_obs12": ("months", "number of observed months in the last 12"),
    "cv12": ("ratio", "12-month coefficient of variation"), "trend12": ("ratio/month", "12-month linear slope / mean"),
    "yoy_ratio": ("ratio", "last 3 months / same 3 months last year"), "months_since_obs": ("months", "months since the last observation"),
    "zero_share12": ("ratio", "share of observed zeros in 12 months"), "max12": ("boxes/month", "12-month maximum"),
    "target_month": ("months", "first month of the forecast window (1-12)"),
    "peer_benchmark": ("boxes/month", "district peer median (excluding itself), shrunk to province level, scaled by external demand"),
    "ratio_to_peer": ("ratio", "mean3 / peer_benchmark"), "peer_gap_rel": ("ratio", "max(0, benchmark-mean3)/benchmark"),
    "n_peers": ("pharmacies", "number of observed peers (excluding itself)"),
    "adults_per_pharmacy": ("people", "district adult population / active pharmacies (vintage published at the cutoff)"),
    "pharmacies_per_10k_adults": ("pharmacies", "pharmacies per 10,000 adults"),
    "ext_base_units": ("boxes/month", "scenario (base) external demand / pharmacy - not a measurement"),
    "visits_3m": ("visits", "visits in the last 3 months (record date <= cutoff); empty (not 0) if no batch covers the period"),
    "visits_12m": ("visits", "visits in the last 12 months; empty unless all 12 months are covered"),
    "order_units_3m": ("boxes", "rep orders in the last 3 months (excluding cancellations, record <= cutoff); empty if not covered"),
    "known_campaign_next3": ("count", "campaign-barcode-months announced at the cutoff for the next 3 months; empty if there is no announcement file"),
    "crm_coverage_flag": ("0/1", "1 = a batch known at the cutoff covers both visits and orders for the last 3 months"),
    "gum_share12": ("ratio", "12-month gum share"), "patch_share12": ("ratio", "12-month patch share"),
    "fallback_share12": ("ratio", "share of cells observed through VENDOR fallback in 12 months"),
}


@dataclass
class Panel:
    glns: np.ndarray
    city: np.ndarray
    district: np.ndarray
    months: pd.DatetimeIndex
    Y: np.ndarray                 # [G, M] observed total net units, NaN = not observed
    active: np.ndarray            # [G, M] pharmacy in scope
    fallback: np.ndarray          # [G, M] share of cells from fallback source
    form_Y: dict[str, np.ndarray] = field(default_factory=dict)
    side: dict[str, pd.DataFrame] = field(default_factory=dict)
    Yg: np.ndarray | None = None  # [G, M] gross outflow (paid+free; returns excluded) = model target basis

    def month_idx(self, month) -> int:
        return int(self.months.get_loc(month_start(month)))


def _pivot(df: pd.DataFrame, glns: np.ndarray, months: pd.DatetimeIndex, col: str) -> np.ndarray:
    gi = pd.Index(glns).get_indexer(df["gln"])
    mi = months.get_indexer(pd.to_datetime(df["month"]))
    out = np.full((len(glns), len(months)), np.nan)
    ok = (gi >= 0) & (mi >= 0)
    out[gi[ok], mi[ok]] = df[col].to_numpy(dtype=float)[ok]
    return out


def load_side_data(con: duckdb.DuckDBPyConnection, cutoff: datetime | None) -> dict[str, pd.DataFrame]:
    """Visits, orders, campaigns and external stats with availability timestamps (filtered later per origin)."""
    side: dict[str, pd.DataFrame] = {}

    def rows(dataset, table, cols):
        if not table_exists(con, table):
            return pd.DataFrame()
        sub, params = active_batches_sql(dataset, cutoff)
        sql = f"select {cols} from {table} where _batch_id in ({sub})"
        if cutoff is not None:
            sql += " and _available_at <= ?"
            params = params + [cutoff]
        return con.execute(sql, params).df()

    side["visits"] = rows("visits", "stg_visits", "gln, date_trunc('month', occurred_at) as month, _available_at as avail")
    side["orders"] = rows("orders", "stg_orders", "gln, date_trunc('month', occurred_at) as month, "
                          "ordered_units - coalesce(cancelled_units, 0) as units, _available_at as avail")
    side["campaigns"] = rows("commercial_terms", "stg_commercial_terms", "month, barcode, _available_at as avail")
    side["external"] = rows("external_geo", "stg_external_geo",
                            "district_code, metric_id, cast(value as double) as value, reference_period, "
                            "_available_at as avail")
    # Which periods were covered by a batch known at the cutoff. One row per active batch.
    def periods(dataset: str) -> pd.DataFrame:
        sub, params = active_batches_sql(dataset, cutoff)
        df = con.execute(f"""select period_start, period_end, effective_available_from as eff_avail,
                                    available_at as batch_avail, availability_mode as mode,
                                    coalesce(retrospective_lag_days, 0) as lag_days
                             from ingestion_batch where batch_id in ({sub})""", params).df()
        for c in ("period_start", "period_end", "eff_avail", "batch_avail"):
            df[c] = pd.to_datetime(df[c])
        return df

    side["visits_periods"] = periods("visits")
    side["orders_periods"] = periods("orders")
    side["campaigns_batches"] = periods("commercial_terms")
    for k, df in side.items():
        for c in ("month", "avail"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
    return side


def window_covered(periods: pd.DataFrame | None, cut: pd.Timestamp, months: list[pd.Timestamp]) -> bool:
    """True only if, at the cutoff, every month of the window lies inside the period of an accepted batch that was
    already known (and, for retrospective backfills, old enough to have been fully recorded). No batch = not covered."""
    if periods is None or periods.empty:
        return False
    known = periods[periods["eff_avail"] <= cut]
    for m in months:
        end = m + pd.offsets.MonthEnd(0)
        spans = (known["period_start"] <= m) & (known["period_end"] >= end)
        delivered = np.where(known["mode"] == "retrospective_lag",
                             end + pd.to_timedelta(known["lag_days"], unit="D") <= cut, known["batch_avail"] <= cut)
        if not (spans.to_numpy() & delivered).any():
            return False
    return True


def panel_frames(con: duckdb.DuckDBPyConnection, canonical_table: str) -> dict[str, pd.DataFrame]:
    """The only canonical-derived inputs of a Panel (small, cacheable per cutoff - see features/asof.py)."""
    return {"totals": gln_month_totals(con, canonical_table), "forms": form_month_totals(con, canonical_table)}


def build_panel(con: duckdb.DuckDBPyConnection, canonical_table: str, side_cutoff: datetime | None) -> Panel:
    return panel_from_frames(con, panel_frames(con, canonical_table), side_cutoff)


def panel_from_frames(con: duckdb.DuckDBPyConnection, frames: dict[str, pd.DataFrame],
                      side_cutoff: datetime | None) -> Panel:
    ph = con.execute("select gln, city_code, district_code, valid_from, valid_to from dim_pharmacy order by gln").df()
    tot = frames["totals"]
    months = pd.DatetimeIndex(sorted(tot["month"].unique()))
    glns = ph["gln"].to_numpy()
    Y = _pivot(tot, glns, months, "units")
    Yg = _pivot(tot, glns, months, "units_gross")
    active = ~np.isnan(_pivot(tot.assign(a=(tot["total_status"] != "out_of_scope").astype(float))
                              .query("a > 0"), glns, months, "a"))
    fb = _pivot(tot.assign(f=tot["n_fallback"] / tot["n_in_scope"].clip(lower=1)), glns, months, "f")
    forms = frames["forms"]
    form_Y = {f: _pivot(forms[forms["form"] == f], glns, months, "units") for f in forms["form"].unique()}
    return Panel(glns, ph["city_code"].to_numpy(), ph["district_code"].to_numpy(), months, Y, active,
                 np.nan_to_num(fb), form_Y, load_side_data(con, side_cutoff), Yg)


# ----------------------------------------------------------------------------- helpers

def _nanmean(a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        n = np.sum(~np.isnan(a), axis=1)
        s = np.nansum(a, axis=1)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _window(Y: np.ndarray, o: int, n: int) -> np.ndarray:
    lo = max(0, o - n + 1)
    w = Y[:, lo:o + 1]
    if w.shape[1] < n:
        w = np.hstack([np.full((Y.shape[0], n - w.shape[1]), np.nan), w])
    return w


def loo_median(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out median of observed peers in the same group, and the peer count."""
    out = np.full(len(values), np.nan)
    cnt = np.zeros(len(values), dtype=int)

    def pick(s: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        # j-th element of s after removing sorted position k (k = -1 removes nothing)
        return s[np.where((k >= 0) & (j >= k), j + 1, j)]

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = values[idx]
        obs = ~np.isnan(v)
        s = np.sort(v[obs])
        n = len(s)
        k = np.where(obs, np.searchsorted(s, np.nan_to_num(v)), -1)
        m = np.where(obs, n - 1, n)              # size of the peer set
        cnt[idx] = m
        valid = m > 0
        if not valid.any():
            continue
        lo = np.where(m % 2 == 1, (m - 1) // 2, m // 2 - 1)
        hi = np.where(m % 2 == 1, (m - 1) // 2, m // 2)
        lo, hi, kk = lo[valid], hi[valid], k[valid]
        out[idx[valid]] = 0.5 * (pick(s, lo, kk) + pick(s, hi, kk))
    return out, cnt


def external_asof(panel: Panel, cutoff: datetime, origin_active: np.ndarray, scenario: dict) -> pd.DataFrame:
    ext = panel.side.get("external", pd.DataFrame())
    df = pd.DataFrame({"district": panel.district})
    n_active = pd.Series(origin_active.astype(int)).groupby(panel.district).transform("sum").to_numpy()
    if ext.empty:
        df["adults"] = np.nan
    else:
        e = ext[(ext["avail"] <= pd.Timestamp(cutoff)) & (ext["metric_id"] == "ADULT_POPULATION")]
        e = e.sort_values(["reference_period", "avail"]).groupby("district_code").last()
        df["adults"] = df["district"].map(e["value"])
    df["adults_per_pharmacy"] = df["adults"] / np.maximum(n_active, 1)
    df["pharmacies_per_10k_adults"] = 1e4 * n_active / df["adults"]
    df["ext_base_units"] = df["adults_per_pharmacy"] * scenario_rate(scenario, "base")
    return df


def features_for_origin(panel: Panel, origin, cutoff: datetime, thresholds: dict, scenario: dict) -> pd.DataFrame:
    """One row per GLN active at the origin. Only months <= origin and side data known at cutoff."""
    o = panel.month_idx(origin)
    Y = panel.Y[:, :o + 1]
    w3, w6, w12 = _window(Y, o, 3), _window(Y, o, 6), _window(Y, o, 12)
    f = pd.DataFrame({"gln": panel.glns, "city_code": panel.city, "district_code": panel.district})
    for k in range(4):
        f[f"lag{k}"] = Y[:, o - k] if o - k >= 0 else np.nan
    f["mean3"], f["mean6"], f["mean12"] = _nanmean(w3), _nanmean(w6), _nanmean(w12)
    f["n_obs3"] = np.sum(~np.isnan(w3), axis=1)
    f["n_obs12"] = np.sum(~np.isnan(w12), axis=1)
    with np.errstate(all="ignore"):
        f["cv12"] = np.nanstd(w12, axis=1) / np.where(f["mean12"] > 0, f["mean12"], np.nan)
        x = np.arange(12) - 5.5
        obs = ~np.isnan(w12)
        xm = np.where(obs, x[None, :], np.nan)
        xc = xm - np.nanmean(xm, axis=1, keepdims=True)
        yc = w12 - np.nanmean(w12, axis=1, keepdims=True)
        slope = np.nansum(xc * yc, axis=1) / np.nansum(xc * xc, axis=1)
        f["trend12"] = np.where(f["n_obs12"] >= 6, slope / np.where(f["mean12"] > 0, f["mean12"], np.nan), np.nan)
        ly = _nanmean(_window(Y, o - 12, 3)) if o >= 14 else np.full(len(f), np.nan)
        f["yoy_ratio"] = (f["mean3"].clip(lower=0) + 1) / (np.clip(ly, 0, None) + 1)   # net-negative months safe
        f["zero_share12"] = np.nansum(w12 == 0, axis=1) / np.maximum(f["n_obs12"], 1)
        f["max12"] = np.where(f["n_obs12"] > 0, np.nanmax(np.where(obs, w12, -np.inf), axis=1), np.nan)
    last_obs = np.where(~np.isnan(Y), np.arange(Y.shape[1])[None, :], -1).max(axis=1)
    f["months_since_obs"] = np.where(last_obs >= 0, o - last_obs, 99)
    f["target_month"] = (panel.months[o] + pd.DateOffset(months=1)).month
    f["fallback_share12"] = _window(panel.fallback, o, 12).mean(axis=1)

    for form, name in (("GUM", "gum_share12"), ("PATCH", "patch_share12")):
        fy = panel.form_Y.get(form)
        if fy is None:
            f[name] = np.nan
            continue
        num = np.nansum(_window(fy[:, :o + 1], o, 12), axis=1)
        den = np.nansum(w12, axis=1)
        f[name] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    # External (as published by the cutoff) and peer benchmark scaled by external demand.
    active_o = panel.active[:, o]
    ext = external_asof(panel, cutoff, active_o, scenario)
    for c in ("adults_per_pharmacy", "pharmacies_per_10k_adults", "ext_base_units"):
        f[c] = ext[c].to_numpy()
    base_ok = f["n_obs3"] >= 2
    norm = np.where(base_ok & (f["ext_base_units"] > 0), f["mean3"] / f["ext_base_units"], np.nan)
    norm = np.where(active_o, norm, np.nan)
    peer = thresholds["peer"]
    med_d, n_d = loo_median(norm, panel.district)
    med_c, _ = loo_median(norm, panel.city)
    k = float(peer["shrink_k"])
    w = n_d / (n_d + k)
    shrunk = np.where(np.isnan(med_d), med_c, w * np.nan_to_num(med_d) + (1 - w) * med_c)
    f["n_peers"] = n_d
    f["peer_benchmark"] = shrunk * f["ext_base_units"]
    f["peer_source"] = np.where(n_d >= peer["min_peers"], "district", "district_shrunk_to_city")
    f["ratio_to_peer"] = np.where(base_ok, (f["mean3"] + 0.1) / (f["peer_benchmark"] + 0.1), np.nan)
    gap = np.clip(f["peer_benchmark"] - f["mean3"].clip(lower=0), 0, None)   # returns-only months floor at 0
    f["peer_gap_units"] = np.where(base_ok, gap, np.nan)
    f["peer_gap_rel"] = np.where(base_ok & (f["peer_benchmark"] > 0), gap / f["peer_benchmark"], np.nan)
    f["baseline_units"] = np.where(base_ok, f["mean3"], np.nan)
    if panel.Yg is not None:   # gross basis of the model target; used where a forecast is compared with a baseline
        wg = _window(panel.Yg[:, :o + 1], o, 3)
        f["baseline_gross"] = np.where(np.sum(~np.isnan(wg), axis=1) >= 2, _nanmean(wg), np.nan)
    else:
        f["baseline_gross"] = np.nan

    # CRM side data: only events that happened by the origin AND were recorded by the cutoff. A count is a number
    # only where a batch known at the cutoff covers the window (then "no record" is a certified zero);
    # otherwise the feature is missing and crm_coverage_flag is 0.
    m_o = panel.months[o]
    cut = pd.Timestamp(cutoff)
    gi = pd.Index(panel.glns)
    win3 = [m_o - pd.DateOffset(months=k) for k in range(3)]
    win12 = [m_o - pd.DateOffset(months=k) for k in range(12)]
    vis3 = window_covered(panel.side.get("visits_periods"), cut, win3)
    vis12 = window_covered(panel.side.get("visits_periods"), cut, win12)
    ord3 = window_covered(panel.side.get("orders_periods"), cut, win3)
    for name, src, months_back, col, covered in (("visits_3m", "visits", 3, None, vis3),
                                                 ("visits_12m", "visits", 12, None, vis12),
                                                 ("order_units_3m", "orders", 3, "units", ord3)):
        if not covered:
            f[name] = np.nan
            continue
        df = panel.side.get(src, pd.DataFrame())
        if df.empty:
            f[name] = 0.0
            continue
        sel = df[(df["avail"] <= cut) & (df["month"] <= m_o) & (df["month"] > m_o - pd.DateOffset(months=months_back))]
        agg = sel.groupby("gln")[col].sum() if col else sel.groupby("gln").size()
        f[name] = gi.map(agg).to_numpy(dtype=float)
        f[name] = f[name].fillna(0.0)
    f["crm_coverage_flag"] = float(vis3 and ord3)
    camp = panel.side.get("campaigns", pd.DataFrame())
    batches = panel.side.get("campaigns_batches")
    if batches is None or batches.empty or not (batches["eff_avail"] <= cut).any():
        f["known_campaign_next3"] = np.nan          # no announcement file known at the cutoff
    elif camp.empty:
        f["known_campaign_next3"] = 0
    else:
        nxt = camp[(camp["avail"] <= cut) & (camp["month"] > m_o) & (camp["month"] <= m_o + pd.DateOffset(months=3))]
        f["known_campaign_next3"] = len(nxt)

    f["origin_month"] = m_o
    f["cutoff_utc"] = cut
    f["active"] = active_o
    low_q = thresholds["low_sales_cohort"]["city_quantile"]
    city_q = f[base_ok & active_o].groupby("city_code")["mean3"].quantile(low_q)
    f["low_sales_cohort"] = base_ok & (f["mean3"] <= f["city_code"].map(city_q))
    f[MODEL_FEATURES] = f[MODEL_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)   # never feed +-inf
    return f[active_o].reset_index(drop=True)


def labels_for_origin(panel: Panel, origin, thresholds: dict) -> pd.DataFrame:
    """Proxy label and 3-month target from a (later) panel, both on gross outflow (paid+free, returns excluded).
    Unobserved future = censored, never negative; the target and the growth ratio can never be below zero."""
    lab = thresholds["proxy_label"]
    if panel.Yg is None:
        raise ValueError("labels need the gross outflow series (Panel.Yg): returns never enter the target")
    o = panel.month_idx(origin)
    nb, nf = lab["base_window_months"], lab["future_window_months"]
    out = pd.DataFrame({"gln": panel.glns, "city_code": panel.city, "district_code": panel.district})
    if o + nf >= len(panel.months):
        out["label_status"] = "immature"
        out["target_units_next3"] = np.nan
        out["proxy_positive"] = np.nan
        out["graded_label"] = np.nan
        return out
    base = panel.Yg[:, o - nb + 1:o + 1]
    fut = panel.Yg[:, o + 1:o + 1 + nf]
    base_ok = ~np.isnan(base).any(axis=1)
    fut_ok = ~np.isnan(fut).any(axis=1)
    in_scope = panel.active[:, o] & panel.active[:, o + nf]
    out["target_units_next3"] = np.where(fut_ok & panel.active[:, o], fut.sum(axis=1), np.nan)
    b = base.mean(axis=1)
    fm = fut.mean(axis=1)
    g = (fm + 1) / (b + 1)
    elig = base_ok & fut_ok & in_scope
    gs = pd.Series(np.where(elig, g, np.nan))
    med_d = gs.groupby(panel.district).transform("median")
    n_d = gs.groupby(panel.district).transform("count")
    med_c = gs.groupby(panel.city).transform("median")
    ref = np.where(n_d >= 5, med_d, med_c)
    excess = g / ref
    months_above = (fut > b[:, None]).sum(axis=1)
    pos = (excess >= lab["min_excess_ratio"]) & (fm - b >= lab["min_abs_increase_per_month"]) & \
          (months_above >= lab["min_future_months_above_base"])
    out["label_base_units"] = np.where(elig, b, np.nan)
    out["label_future_units"] = np.where(elig, fm, np.nan)
    out["excess_ratio"] = np.where(elig, excess, np.nan)
    out["proxy_positive"] = np.where(elig, pos.astype(float), np.nan)   # 1 / 0 / NaN (censored)
    out["graded_label"] = np.where(elig & pos, np.clip(excess - 1, 0, 3), 0.0)
    out["label_status"] = np.select([~in_scope, ~base_ok, ~fut_ok], ["out_of_scope", "censored_base", "censored_future"],
                                    "eligible")
    return out


def labels_peer_closure(panel: Panel, origin, features: pd.DataFrame, alt: dict) -> pd.DataFrame:
    """Draft alternative label (thresholds_v002): the pharmacy closes >= X of its gap to the peer benchmark.

    Gap and closure are on the gross-outflow series: gap0 = peer_benchmark - base mean; closure =
    (future mean - base mean) / gap0. Eligible only with an observed base and future window, a peer benchmark and
    a real gap; everything else is censored/ineligible (NaN), never a negative."""
    lab = alt["alt_label"]
    if panel.Yg is None:
        raise ValueError("alternative label needs the gross outflow series (Panel.Yg)")
    o = panel.month_idx(origin)
    nb, nf = lab["base_window_months"], lab["future_window_months"]
    out = pd.DataFrame({"gln": panel.glns})
    if o + nf >= len(panel.months):
        return out.assign(alt_status="immature", alt_positive=np.nan, alt_closure=np.nan)
    base = panel.Yg[:, o - nb + 1:o + 1]
    fut = panel.Yg[:, o + 1:o + 1 + nf]
    b, fm = base.mean(axis=1), fut.mean(axis=1)
    bench = pd.Series(features["peer_benchmark"].to_numpy(), index=features["gln"]).reindex(panel.glns).to_numpy()
    gap0 = bench - b
    elig = (~np.isnan(base).any(axis=1) & ~np.isnan(fut).any(axis=1) & panel.active[:, o] & panel.active[:, o + nf]
            & ~np.isnan(gap0) & (gap0 >= lab["min_peer_gap_units_per_month"]))
    closure = np.where(elig, (fm - b) / np.where(gap0 > 0, gap0, np.nan), np.nan)
    months_above = (fut > b[:, None]).sum(axis=1)
    pos = (closure >= lab["min_closure_share"]) & (months_above >= lab["min_future_months_above_base"])
    out["alt_closure"] = closure
    out["alt_positive"] = np.where(elig, pos.astype(float), np.nan)
    out["alt_status"] = np.where(elig, "eligible", "censored_or_no_gap")
    return out


def training_origins(model_cfg: dict, origin, horizon: int) -> list[pd.Timestamp]:
    """Quarterly training origins ending at ``origin - horizon`` (``train_origin_step_months``; )."""
    from ..utils import add_months, month_range
    step = int(model_cfg.get("train_origin_step_months", 3))
    last = add_months(origin, -horizon)
    return [o for o in month_range(model_cfg["min_train_origin"], last)
            if ((last.year - o.year) * 12 + last.month - o.month) % step == 0]


def training_frame(panel_for: Callable[[pd.Timestamp, datetime], Panel], label_panel: Panel, origins: list,
                   thresholds: dict, scenario: dict, cutoff_lag_days: int) -> pd.DataFrame:
    """Stacked GLN x origin rows. Features of origin ``o`` come from ``panel_for(o, cutoff(o))`` - a panel built
    with only what was known at cutoff(o) - while labels come from the matured ``label_panel``."""
    frames = []
    for o in origins:
        cut = backtest_cutoff(o, cutoff_lag_days)
        feats = features_for_origin(panel_for(month_start(o), cut), o, cut, thresholds, scenario)
        labs = labels_for_origin(label_panel, o, thresholds)[["gln", "target_units_next3", "proxy_positive", "label_status"]]
        frames.append(feats.merge(labs, on="gln", how="left"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def asof_audit(panel: Panel, origin, cutoff: datetime) -> dict[str, Any]:
    """Evidence for validate_asof: nothing used for the origin was known after the cutoff."""
    o = panel.month_idx(origin)
    cut = pd.Timestamp(cutoff)
    out: dict[str, Any] = {"origin_month": panel.months[o], "cutoff_utc": cut, "panel_last_month": panel.months[-1],
                           "panel_months_after_origin_used": 0}
    for name, df in panel.side.items():
        if "avail" not in df:        # batch-period frames (coverage evidence), not event rows
            out[f"{name}_batches"] = int(len(df))
            continue
        if df.empty:
            out[f"{name}_rows_after_cutoff_in_window"] = 0
            continue
        late = df[(df["avail"] > cut)]
        out[f"{name}_rows_loaded"] = int(len(df))
        out[f"{name}_rows_after_cutoff_excluded"] = int(len(late))
    out["master_history"] = "retrospective snapshot (no as-of master history; S16 flagged)"
    return out
