"""External-only opportunity, track A.

Uses only external district statistics, the pharmacy master (district membership) and the
scenario parameters. It never reads sales, orders, visits or supply data, so perturbing sales
cannot change track A. Pharmacies in the same district are tied.
"""
from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from ..db import active_batches_sql, table_exists
from ..utils import month_start


def latest_external(con: duckdb.DuckDBPyConnection, cutoff: datetime | None) -> pd.DataFrame:
    """Latest vintage of each district metric published by the cutoff."""
    if not table_exists(con, "stg_external_geo"):
        return pd.DataFrame(columns=["district_code", "metric_id", "value", "reference_period", "available_at"])
    sub, params = active_batches_sql("external_geo", cutoff)
    sql = f"select * from stg_external_geo where _batch_id in ({sub}) and geo_level = 'DISTRICT'"
    if cutoff is not None:
        sql += " and _available_at <= ?"
        params = params + [cutoff]
    return con.execute(f"""
        select district_code, metric_id, cast(value as double) as value, reference_period,
               _available_at as available_at, license_id
        from (select *, row_number() over (partition by district_code, metric_id
                                           order by reference_period desc, _available_at desc) rn
              from ({sql})) where rn = 1
    """, params).df()


def scenario_rate(scenario: dict, level: str) -> float:
    """Monthly DemoBrand boxes per adult under one scenario level (product of multipliers)."""
    p = scenario["parameters"]
    rate = (p["smoking_prevalence"][level] * p["quit_attempt_rate"][level] / scenario["annual_to_monthly_divisor"]
            * p["nrt_use_share"][level] * p["boxes_per_attempt"][level] * p["demo_brand_share"][level])
    return float(rate)


def build_external_opportunity(con: duckdb.DuckDBPyConnection, cutoff: datetime | None, origin_month,
                               scenario: dict) -> tuple[pd.DataFrame, dict]:
    origin = month_start(origin_month)
    ext = latest_external(con, cutoff)
    wide = ext.pivot_table(index="district_code", columns="metric_id", values="value", aggfunc="first")
    vintage = ext.groupby("district_code")["reference_period"].max()
    ph = con.execute("""
        select gln, city_code, district_code from dim_pharmacy
        where coalesce(valid_from, date '1900-01-01') <= last_day(?::date)
          and coalesce(valid_to, date '9999-12-31') >= ?::date
        order by gln
    """, [origin.date(), origin.date()]).df()
    n_master = ph.groupby("district_code")["gln"].transform("count")
    ph["n_pharmacies_master"] = n_master
    ph["adult_population"] = ph["district_code"].map(wide.get("ADULT_POPULATION", pd.Series(dtype=float)))
    ph["pharmacy_count_official"] = ph["district_code"].map(wide.get("PHARMACY_COUNT_OFFICIAL", pd.Series(dtype=float)))
    ph["external_vintage"] = ph["district_code"].map(vintage)
    ph["adults_per_pharmacy"] = ph["adult_population"] / ph["n_pharmacies_master"]
    ph["pharmacies_per_10k_adults"] = 1e4 * ph["n_pharmacies_master"] / ph["adult_population"]
    for level in ("low", "base", "high"):
        # Equal-share allocation inside the district: shares sum to 1 per district.
        ph[f"scenario_units_per_month_{level}"] = ph["adults_per_pharmacy"] * scenario_rate(scenario, level)
    ph["demand_share"] = 1.0 / ph["n_pharmacies_master"]
    has = ph["adults_per_pharmacy"].notna()
    ph["a_score_raw"] = ph["scenario_units_per_month_base"]
    ph["a_score_0_100"] = np.where(has, 100 * ph["a_score_raw"].rank(pct=True, method="average"), np.nan)
    ph["a_rank"] = ph["a_score_raw"].rank(ascending=False, method="min").astype("Int64")
    ph["tie_group"] = "district:" + ph["district_code"].astype(str)
    ph["a_eligibility"] = np.where(has, "eligible", "no_external_data")
    checks = {
        "share_sum_per_district_max_abs_error": float((ph.groupby("district_code")["demand_share"].sum() - 1).abs().max()),
        "scenario_ordering_ok": bool(((ph["scenario_units_per_month_low"] <= ph["scenario_units_per_month_base"])
                                      & (ph["scenario_units_per_month_base"] <= ph["scenario_units_per_month_high"]))[has].all()),
        "districts_without_external_data": int(ph.loc[~has, "district_code"].nunique()),
        "scenario_version": scenario["scenario_version"], "scenario_status": scenario["approval_status"],
        "district_box_total_matches_scenario": bool(np.allclose(
            ph[has].groupby("district_code")["scenario_units_per_month_base"].sum(),
            (ph[has].groupby("district_code")["adult_population"].first() * scenario_rate(scenario, "base")))),
    }
    return ph, checks
