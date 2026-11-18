"""Supply readiness signal from sell-in and depot stock.

Readiness is categorical evidence that the *supplier network* of a pharmacy holds the product.
It is never a pharmacy stock estimate: depot stock is not divided among pharmacies, parent
sell-in is never allocated to sub depots, and missing evidence yields ``unknown`` (it does not
suppress the opportunity score).
"""
from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from ..db import active_batches_sql, table_exists
from ..utils import month_start

LEVEL_SCORE = {"high": 1.0, "limited": 0.25, "unknown": 0.5}


def _asof(dataset: str, table: str, cutoff: datetime | None) -> tuple[str, list]:
    sub, params = active_batches_sql(dataset, cutoff)
    sql = f"select * from {table} where _batch_id in ({sub})"
    if cutoff is not None:
        sql += " and _available_at <= ?"
        params = params + [cutoff]
    return sql, params


SCOPE_SQL = "barcode in (select barcode from dim_product where upper(trim(brand)) = upper(trim(?)))"
READINESS_COLUMNS = ["gln", "readiness", "readiness_reason", "readiness_score", "age_days"]


def _unknown_readiness(con: duckdb.DuckDBPyConnection, reason: str) -> pd.DataFrame:
    """Same schema on every path: missing evidence is 'unknown' with the neutral score 0.5."""
    out = con.execute("select gln from dim_pharmacy order by gln").df()
    out["readiness"] = "unknown"
    out["readiness_reason"] = reason
    out["readiness_score"] = LEVEL_SCORE["unknown"]
    out["age_days"] = float("nan")
    return out[READINESS_COLUMNS]


def build_readiness(con: duckdb.DuckDBPyConnection, cutoff: datetime | None, origin_month, thresholds: dict,
                    brand: str) -> pd.DataFrame:
    """Supply evidence for the configured brand only: stock and sell-in of another product never make the target
    product 'ready'."""
    origin = month_start(origin_month)
    asof_date = (origin + pd.offsets.MonthEnd(0)).date()
    max_age = int(thresholds["readiness"]["max_stock_age_days"])
    lookback = int(thresholds["readiness"]["sellin_lookback_months"])
    if not (table_exists(con, "bridge_pharmacy_depot") and table_exists(con, "stg_depot_stock")):
        return _unknown_readiness(con, "no depot link or stock data")

    stock_sql, sp = _asof("depot_stock", "stg_depot_stock", cutoff)
    con.execute(f"""
        create or replace temp table _stock as
        select d.parent_depot_code as parent, s.sub_depot_code, s.stock_as_of, sum(s.units_on_hand) as units
        from ({stock_sql}) s join dim_depot d on d.depot_code = s.sub_depot_code
        where s.stock_as_of <= ? and s.barcode in (select barcode from dim_product where upper(trim(brand)) = upper(trim(?)))
        group by all
    """, sp + [asof_date, brand])
    # Latest snapshot per sub depot; parent evidence is partial if a sub depot has no snapshot at the latest date.
    con.execute("""
        create or replace temp table _parent_stock as
        with latest as (
            select parent, sub_depot_code, max(stock_as_of) as last_date from _stock group by all
        ), lv as (
            select l.parent, l.sub_depot_code, l.last_date, s.units
            from latest l join _stock s on s.parent = l.parent and s.sub_depot_code = l.sub_depot_code
                 and s.stock_as_of = l.last_date
        ), subs as (
            select parent_depot_code as parent, count(*) as n_subs from dim_depot where depot_level = 'SUB' group by 1
        )
        select lv.parent, max(lv.last_date) as last_date, sum(lv.units) as units_any_sub,
               count(*) as subs_reporting, any_value(subs.n_subs) as n_subs,
               count(case when lv.last_date = (select max(last_date) from lv x where x.parent = lv.parent) then 1 end)
                   as subs_at_latest
        from lv join subs using (parent) group by lv.parent
    """)
    # Sell-in is optional: absent, or not yet delivered at the cutoff, means "no sell-in evidence" -> unknown,
    # never "no sell-in happened".
    first = (origin - pd.DateOffset(months=lookback - 1)).date()
    sellin_known = False
    if table_exists(con, "stg_sellin"):
        sub, _ = active_batches_sql("sellin", cutoff)
        sellin_known = bool(con.execute(f"select count(*) from ({sub})", active_batches_sql("sellin", cutoff)[1]).fetchone()[0])
    if sellin_known:
        sellin_sql, sip = _asof("sellin", "stg_sellin", cutoff)
        con.execute(f"""
            create or replace temp table _sellin as
            select depot_code as parent, sum(received_units) as received
            from ({sellin_sql}) where month between ? and ? and barcode in (select barcode from dim_product where upper(trim(brand)) = upper(trim(?))) group by 1
        """, sip + [first, origin.date(), brand])
    else:
        con.execute("create or replace temp table _sellin as select cast(null as varchar) as parent, "
                    "cast(null as double) as received where false")
    df = con.execute("""
        select b.gln, b.depot_code as parent, ps.last_date, ps.units_any_sub, ps.subs_at_latest, ps.n_subs,
               si.received
        from bridge_pharmacy_depot b
        left join _parent_stock ps on ps.parent = b.depot_code
        left join _sellin si on si.parent = b.depot_code
        order by b.gln, b.depot_code
    """).df()
    df["age_days"] = (pd.Timestamp(asof_date) - pd.to_datetime(df["last_date"])).dt.days

    def level(r) -> tuple[str, str]:
        if pd.isna(r.last_date):
            return "unknown", "no stock snapshot for linked depot"
        if not sellin_known:
            return "unknown", "no sell-in data known at the cutoff"
        fresh = r.age_days <= max_age
        if fresh and r.units_any_sub > 0 and (r.received or 0) > 0:
            partial = " (partial sub-depot list)" if r.subs_at_latest < r.n_subs else ""
            return "high", f"linked depot holds product, stock age {int(r.age_days)}d, sell-in observed{partial}"
        if not fresh:
            return "limited", f"stock snapshot is {int(r.age_days)} days old"
        if (r.received or 0) <= 0:
            return "limited", f"no sell-in to linked depot in last {lookback} months"
        return "limited", "linked depot shows zero stock"

    if df.empty:
        return _unknown_readiness(con, "no depot link")
    lv = df.apply(level, axis=1, result_type="expand")
    df["level"], df["reason"] = lv[0], lv[1]
    order = {"high": 2, "limited": 1, "unknown": 0}
    df["rank"] = df["level"].map(order)
    best = df.sort_values(["gln", "rank", "parent"], ascending=[True, False, True], kind="mergesort").drop_duplicates("gln")
    out = con.execute("select gln from dim_pharmacy order by gln").df().merge(
        best[["gln", "level", "reason", "age_days"]], on="gln", how="left")
    out["readiness"] = out["level"].fillna("unknown")
    out["readiness_reason"] = out["reason"].fillna("no depot link")
    out["readiness_score"] = out["readiness"].map(LEVEL_SCORE)
    return out[READINESS_COLUMNS]


def stock_reconciliation(con: duckdb.DuckDBPyConnection, canonical_table: str, cutoff: datetime | None,
                         brand: str) -> pd.DataFrame:
    """Parent-depot flow balance: opening + sell-in - outflow - closing, for the configured brand only.

    / : the outflow of a parent is the sum of the movement rows whose DEPOT CODE is the parent or
    one of its sub depots - read at the source grain, so a GLN total is never copied to depots and a two-supplier
    pharmacy lands on the right depot. The v1 comparability conditions are kept: complete sub-depot snapshot list, the
    same month-end/units, sell-in known as of the cutoff, mutabakat coverage COMPLETE for the depot-month, no conflicting
    duplicate rows summed, and no rejected row that could belong to the depot-month. Anything else is
    ``skipped_with_reason``. Reads the temp tables of the canonical build named by ``canonical_table`` and of the last
    ``build_readiness`` call, so call it right after both."""
    if not table_exists(con, "_parent_stock") or not table_exists(con, "_mut_key"):
        return pd.DataFrame()
    tag = canonical_table.removeprefix("canon_")
    if table_exists(con, "stg_sellin"):
        sellin_sql, sip = _asof("sellin", "stg_sellin", cutoff)
        sellin_cte = (f"select depot_code as parent, month, sum(received_units) as received from ({sellin_sql}) "
                      f"where {SCOPE_SQL} group by all")
        sip = sip + [brand]
    else:
        sellin_cte, sip = ("select cast(null as varchar) as parent, cast(null as date) as month, "
                           "cast(null as double) as received where false"), []
    rejected = (f"exists (select 1 from _rr_{tag} r where (r.depot_parent is null or r.depot_parent = s.parent) "
                "and s.month between r.m_from and r.m_to)") if table_exists(con, f"_rr_{tag}") else "false"
    df = con.execute(f"""
        with stock as (
            select parent, date_trunc('month', stock_as_of) as month, sum(units) as closing,
                   count(distinct sub_depot_code) as subs
            from _stock group by all
        ), subs as (select parent_depot_code as parent, count(*) n_subs from dim_depot where depot_level='SUB' group by 1),
        sellin as ({sellin_cte}),
        depot_parent as (select depot_code, coalesce(nullif(parent_depot_code, ''), depot_code) as parent from dim_depot),
        flow as (
            select dp.parent, k.month,
                   sum(case when k.n_distinct = 1 then k.units_paid + coalesce(k.units_free, 0) - coalesce(k.returns_units, 0) end)
                       as out_units,
                   count(case when k.n_distinct > 1 then 1 end) as n_conflict_rows
            from _mut_key k join depot_parent dp on dp.depot_code = k.depot_code group by 1, 2
        ), cov as (
            select scope_id as parent, month, coverage_status = 'COMPLETE' as complete from _cov
            where covered_source = 'MUTABAKAT' and scope_type = 'DEPOT'
        )
        select s.parent, strftime(s.month, '%Y-%m') as month, s.closing,
               lag(s.closing) over (partition by s.parent order by s.month) as opening,
               si.received, f.out_units, s.subs, subs.n_subs, coalesce(c.complete, false) as complete,
               coalesce(f.n_conflict_rows, 0) as n_conflict_rows, {rejected} as rejected_rows_possible
        from stock s join subs using (parent) left join sellin si using (parent, month)
        left join flow f on f.parent = s.parent and f.month = s.month
        left join cov c on c.parent = s.parent and c.month = s.month
        order by 1, 2""", sip).df()
    comparable = ((df["subs"] == df["n_subs"]) & df["opening"].notna() & df["received"].notna() & df["complete"]
                  & (df["n_conflict_rows"] == 0) & ~df["rejected_rows_possible"].astype(bool))
    df["status"] = np.where(comparable, "comparable", "skipped_with_reason")
    df["reason"] = np.select(
        [df["n_conflict_rows"] > 0, df["rejected_rows_possible"].astype(bool), ~df["complete"],
         df["subs"] < df["n_subs"], df["received"].isna(), df["opening"].isna()],
        ["conflicting duplicate outflow rows are not summed",
         "a rejected movement row could belong to this depot-month",
         "outflow coverage not complete", "incomplete sub-depot snapshot list", "no sell-in known for the month",
         "no opening stock snapshot"], "")
    # Complete coverage and no rejected row that could belong here: no movement row IS the evidence of zero outflow.
    df["out_units"] = df["out_units"].fillna(0).where(df["complete"] & ~df["rejected_rows_possible"].astype(bool),
                                                      df["out_units"])
    df["residual"] = (df["opening"] + df["received"] - df["out_units"] - df["closing"]).where(comparable)
    return df
