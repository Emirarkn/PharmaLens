"""Within-source dedup, VENDOR-mutabakat reconciliation and canonical selection.

Hard rule: the two sources are alternative measurements of the same flow and are never added.
Absence of a record becomes ``observed_zero`` only where coverage proves complete reporting.
Everything is computed *as of* a cutoff: only batches/rows known at that time are used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..db import active_batches_sql, ensure_dimension_views, table_exists
from ..utils import get_logger, month_start

log = get_logger()

STATUSES = ["observed_positive", "observed_zero", "observed_net_negative", "unobserved", "out_of_scope", "quarantined"]
OBSERVED = ("observed_positive", "observed_zero", "observed_net_negative")


@dataclass
class CanonicalResult:
    table: str
    tag: str
    cutoff: datetime | None
    origin_month: pd.Timestamp
    mode: str
    report: dict[str, Any]


def _rows_sql(dataset: str, table: str, cutoff: datetime | None, origin: pd.Timestamp) -> tuple[str, list]:
    sub, params = active_batches_sql(dataset, cutoff)
    sql = f"select * from {table} where _batch_id in ({sub}) and month <= ?"
    params = params + [origin.date()]
    if cutoff is not None:
        sql += " and _available_at <= ?"
        params.append(cutoff)
    return sql, params


def _scope_products(con: duckdb.DuckDBPyConnection, settings) -> int:
    """Temp table of the approved modelling scope: products of the configured brand."""
    brand = settings.get("brand")
    if not brand or not str(brand).strip():
        raise RuntimeError("config 'brand' is required: the canonical universe is limited to the approved brand")
    con.execute("create or replace temp table _scope_products as select * from dim_product "
                "where upper(trim(brand)) = upper(trim(?))", [str(brand)])
    n = con.execute("select count(*) from _scope_products").fetchone()[0]
    if not n:
        raise RuntimeError(f"no product of brand '{brand}' in dim_product - ingest the product master first")
    return int(n)


def _out_of_brand_rows(con: duckdb.DuckDBPyConnection, raw: str) -> int:
    return int(con.execute(f"select count(*) from {raw} r where r.barcode not in (select barcode from _scope_products)"
                           ).fetchone()[0])


def _rejected_rows(con: duckdb.DuckDBPyConnection, dataset: str, cutoff: datetime | None) -> pd.DataFrame:
    """Rows rejected at ingest by batches known at the cutoff, with the typed fields that survived."""
    sub, params = active_batches_sql(dataset, cutoff)
    sql = f"""select q.batch_id, q.source_row_id, q.error_code, q.raw_record, q.key_gln, q.key_month, q.key_barcode,
                     q.key_depot_code, b.period_start, b.period_end
              from quarantine q join ingestion_batch b on b.batch_id = q.batch_id
              where q.dataset = ? and q.batch_id in ({sub})"""
    params = [dataset] + params
    if cutoff is not None:
        sql += " and (q.available_at is null or q.available_at <= ?)"
        params.append(cutoff)
    return con.execute(sql + " order by q.batch_id, q.source_row_id", params).df()


def _register(con: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame, types: str) -> None:
    con.execute(f"create or replace temp table {name} ({types})")
    if len(frame):
        con.register("_qframe", frame)
        con.execute(f"insert into {name} select * from _qframe")
        con.unregister("_qframe")


def _readable_units(raw_record: str, src: dict[str, Any]) -> float | None:
    """Boxes (paid + free) a rejected mutabakat row carried, or None when the number cannot be read or trusted."""
    try:
        rec = json.loads(raw_record)
    except ValueError:
        return None
    fmt = src.get("file", {})

    def num(field: str, required: bool) -> float | None:
        txt = str(rec.get(src["columns"][field]["source"], "")).strip()
        if txt == "":
            return None if required else 0.0
        if fmt.get("thousands"):
            txt = txt.replace(fmt["thousands"], "")
        if fmt.get("decimal", ".") != ".":
            txt = txt.replace(fmt["decimal"], ".")
        try:
            v = float(txt)
        except ValueError:
            return None
        return v if np.isfinite(v) and v >= 0 else None      # a negative or odd value is not a usable bound

    paid, free = num("units_paid", True), num("units_free", False)
    return None if paid is None or free is None else paid + free


def rule_type(g: bool, m: bool, b: bool, d: bool) -> str:
    """Which row of the table a rejected row falls under (by the fields it still shows)."""
    if g and m and b:
        return "cell"
    if g and m:
        return "gln_month_barcode_unknown"
    if g and b:
        return "gln_barcode_month_unknown"
    if g:
        return "gln_only"
    if m and b:
        return "gln_unknown_depot_month_barcode" if d else "gln_unknown_month_barcode"
    return "depot_only" if d else "batch_only"


def _reject_tables(con: duckdb.DuckDBPyConnection, settings, tag: str, cutoff: datetime | None, origin: pd.Timestamp,
                   start: pd.Timestamp) -> dict[str, dict[str, int]]:
    """Turn rejected rows into (a) fully keyed cells `_qc_<ds>` - the cell can never be proven zero nor trusted - and
    (b) mutabakat rules `_rej_<tag>`: the set of cells whose "no record -> 0" inference a row blocks."""
    stats: dict[str, dict[str, int]] = {}
    known_barcodes = {r[0] for r in con.execute("select barcode from dim_product").fetchall()}
    for dataset, key in (("movement_mutabakat", "mutabakat"), ("movement_vendor", "vendor")):
        q = _rejected_rows(con, dataset, cutoff)
        # A barcode is "resolved" only if the product master knows it; a well-formed but unknown barcode may still be
        # a target product that is missing from the master.
        q["key_barcode"] = q["key_barcode"].where(q["key_barcode"].isin(known_barcodes))
        pinned = q["key_gln"].notna() & q["key_month"].notna() & q["key_barcode"].notna()
        cells = q.loc[pinned, ["key_gln", "key_month", "key_barcode"]].drop_duplicates()
        cells.columns = ["gln", "month", "barcode"]
        cells["flag"] = True
        _register(con, f"_qc_{key}", cells, "gln varchar, month date, barcode varchar, flag boolean")
        stats[dataset] = {"rejected_rows": int(len(q)), "rejected_rows_pinned_to_cell": int(pinned.sum()),
                          "rejected_rows_unpinnable": int((~pinned).sum()), "depot_months_downgraded": 0}
        if dataset != "movement_mutabakat":
            continue
        src = settings.source_config(dataset)
        rules = []
        for r in q.itertuples():
            if pd.notna(r.key_month):
                m_from = m_to = month_start(r.key_month)
            else:
                m_from, m_to = month_start(r.period_start), month_start(r.period_end)
            m_from, m_to = max(m_from, start), min(m_to, origin)
            if m_from > m_to:
                continue
            g, m, b, d = (pd.notna(r.key_gln), pd.notna(r.key_month), pd.notna(r.key_barcode),
                          pd.notna(r.key_depot_code))
            rules.append((len(rules) + 1, dataset, r.batch_id, int(r.source_row_id), r.error_code,
                          rule_type(g, m, b, d), r.key_gln if g else None, m_from.date(), m_to.date(),
                          r.key_barcode if b else None, r.key_depot_code if d else None,
                          _readable_units(r.raw_record, src)))
        frame = pd.DataFrame(rules, columns=["rule_id", "dataset", "batch_id", "source_row_id", "error_code",
                                             "rule_type", "gln", "m_from", "m_to", "barcode", "depot", "units"])
        _register(con, f"_rej_{tag}", frame, "rule_id integer, dataset varchar, batch_id varchar, source_row_id bigint, "
                  "error_code varchar, rule_type varchar, gln varchar, m_from date, m_to date, barcode varchar, "
                  "depot varchar, units double")
    return stats


def _blocking_sets(con: duckdb.DuckDBPyConnection, tag: str) -> None:
    """Expand the rules into `_blk_<tag>(rule_id, gln, month, barcode)`; barcode NULL = every in-scope barcode.

    Product scope comes first: a row whose barcode is known and belongs to another brand blocks nothing; a barcode
    that is unknown (it might be a target product missing from the master) is not a proof of another brand."""
    depot_parent = ("coalesce(nullif(d.parent_depot_code, ''), d.depot_code)" if table_exists(con, "dim_depot")
                    else "cast(null as varchar)")
    depot_join = "left join dim_depot d on d.depot_code = r.depot" if table_exists(con, "dim_depot") else ""
    con.execute(f"""
        create or replace temp table _rr_{tag} as
        select r.*, case when sp.barcode is not null then r.barcode end as eff_barcode, {depot_parent} as depot_parent
        from _rej_{tag} r
        left join dim_product p on p.barcode = r.barcode
        left join _scope_products sp on sp.barcode = r.barcode
        {depot_join}
        where not (p.barcode is not null and sp.barcode is null)
    """)
    links = ("""
        select r.rule_id, g.gln, g.month, r.eff_barcode as barcode
        from _rr_{t} r join bridge_pharmacy_depot b on r.gln is null and r.depot_parent is not null
             and b.depot_code = r.depot_parent
        join _gm g on g.gln = b.gln and g.month between r.m_from and r.m_to
             and coalesce(b.valid_from, date '1900-01-01') <= last_day(g.month)
             and coalesce(b.valid_to, date '9999-12-31') >= g.month
        union all""".format(t=tag) if table_exists(con, "bridge_pharmacy_depot") else "")
    con.execute(f"""
        create or replace temp table _blk_{tag} as
        select r.rule_id, g.gln, g.month, r.eff_barcode as barcode
        from _rr_{tag} r join _gm g on r.gln is not null and g.gln = r.gln and g.month between r.m_from and r.m_to
        union all
        {links}
        select r.rule_id, g.gln, g.month, r.eff_barcode as barcode
        from _rr_{tag} r join _gm g on r.gln is null and r.depot_parent is null and g.month between r.m_from and r.m_to
    """)
    con.execute(f"create or replace temp table _blk_all_{tag} as select distinct gln, month from _blk_{tag} "
                "where barcode is null")
    con.execute(f"create or replace temp table _blk_b_{tag} as select distinct gln, month, barcode from _blk_{tag} "
                "where barcode is not null")


def _dedup(con: duckdb.DuckDBPyConnection, raw: str, out: str, key: list[str], measures: list[str]) -> dict[str, int]:
    """Collapse identical repeats; mark keys with conflicting repeats (they never reach canonical)."""
    k = ", ".join(key)
    sig = "concat_ws('|', " + ", ".join(f"coalesce(cast({m} as varchar), '<null>')" for m in measures) + ")"
    aggs = ", ".join(f"any_value({m}) as {m}" for m in measures)
    con.execute(f"""
        create or replace temp table {out} as
        select {k}, count(*) as n_rows, count(distinct {sig}) as n_distinct, {aggs},
               string_agg(_batch_id || '#' || cast(_source_row_id as varchar), ';' order by _batch_id, _source_row_id) as row_ids
        from {raw} group by {k}
    """)
    stats = con.execute(f"""
        select coalesce(sum(n_rows), 0), coalesce(sum(case when n_distinct = 1 then n_rows - 1 else 0 end), 0),
               coalesce(sum(case when n_distinct > 1 then n_rows else 0 end), 0),
               coalesce(sum(case when n_distinct > 1 then 1 else 0 end), 0)
        from {out}
    """).fetchone()
    return {"rows": int(stats[0]), "identical_repeats_removed": int(stats[1]),
            "conflicting_rows_quarantined": int(stats[2]), "conflicting_keys": int(stats[3])}


def build_canonical(con: duckdb.DuckDBPyConnection, settings, cutoff: datetime | None, origin_month: Any,
                    tag: str, mode: str = "default") -> CanonicalResult:
    policy = settings.policy
    origin = month_start(origin_month)
    start = month_start(settings.get("history_start_month"))
    ensure_dimension_views(con)
    for need in ("dim_pharmacy", "dim_product"):
        if not table_exists(con, need):
            raise RuntimeError(f"required table/view '{need}' missing - ingest the corresponding master first")
    # A source that has not been delivered is simply absent (single-source scope), never invented.
    from ..ingest.loader import _ensure_stage_table
    for ds in ("movement_mutabakat", "movement_vendor", "coverage_register"):
        src = settings.source_config(ds)
        _ensure_stage_table(con, src["stage_table"], src)

    report: dict[str, Any] = {"tag": tag, "mode": mode, "cutoff": cutoff, "origin_month": origin,
                              "policy_version": policy["policy_version"], "policy_status": policy["approval_status"]}

    # --- approved brand scope and rejected-row keys --------------------------------
    report["scope_brand"] = str(settings.get("brand"))
    report["scope_products"] = _scope_products(con, settings)
    evidence = policy.get("absence_zero_evidence", "z2_narrowed")
    if evidence not in ("z2_narrowed", "strict_k7_v1"):
        raise ValueError(f"unknown policy absence_zero_evidence: {evidence} (z2_narrowed | strict_k7_v1)")
    strict = evidence == "strict_k7_v1"
    report["zero_evidence_mode"] = evidence
    quarantine_impact = _reject_tables(con, settings, tag, cutoff, origin, start)
    report["quarantine_impact"] = quarantine_impact
    in_scope = "barcode in (select barcode from _scope_products)"

    # --- within-source dedup ------------------------------------------------------------
    sql, params = _rows_sql("movement_mutabakat", "stg_movement_mutabakat", cutoff, origin)
    con.execute(f"create or replace temp table _mut_all as {sql}", params)
    con.execute(f"create or replace temp table _mut_raw as select * from _mut_all where {in_scope}")
    report["mutabakat_dedup"] = _dedup(con, "_mut_raw", "_mut_key", ["gln", "month", "barcode", "depot_code"],
                                       ["units_paid", "units_free", "returns_units", "net_value_vat_inc_try"])
    # Proven depot sub-grain: aggregate inside the source only.
    con.execute("""
        create or replace temp table _mut as
        select gln, month, barcode,
               bool_or(n_distinct > 1) as conflict,
               sum(units_paid) as units_paid, sum(coalesce(units_free, 0)) as units_free,
               sum(coalesce(returns_units, 0)) as returns_units, sum(net_value_vat_inc_try) as net_value_vat_inc_try,
               string_agg(row_ids, ';') as row_ids
        from _mut_key group by gln, month, barcode
    """)
    sql, params = _rows_sql("movement_vendor", "stg_movement_vendor", cutoff, origin)
    con.execute(f"create or replace temp table _vendor_all as {sql}", params)
    con.execute(f"create or replace temp table _vendor_raw as select * from _vendor_all where {in_scope}")
    report["out_of_brand_scope_rows"] = {"movement_mutabakat": _out_of_brand_rows(con, "_mut_all"),
                                         "movement_vendor": _out_of_brand_rows(con, "_vendor_all")}
    report["vendor_dedup"] = _dedup(con, "_vendor_raw", "_vendor_key", ["gln", "month", "barcode"], ["units_total"])
    con.execute("""create or replace temp table _vendor as
                   select gln, month, barcode, n_distinct > 1 as conflict, units_total, row_ids from _vendor_key""")

    # --- coverage as known at cutoff ----------------------------------------------------
    sub, params = active_batches_sql("coverage_register", cutoff)
    cov_sql = f"select * from stg_coverage_register where _batch_id in ({sub})"
    if cutoff is not None:
        cov_sql += " and _available_at <= ?"
        params = params + [cutoff]
    con.execute(f"""
        create or replace temp table _cov as
        select covered_source, month, scope_type, scope_id, coverage_status from (
            select *, row_number() over (partition by covered_source, month, scope_type, scope_id
                                         order by _available_at desc, _batch_id desc) rn
            from ({cov_sql})) where rn = 1
    """, params)

    # --- universe: active pharmacy x month x active product ----------------------------
    con.execute("create or replace temp table _months as select cast(range as date) as month "
                "from range(?::date, ?::date + interval 1 day, interval 1 month)", [start.date(), origin.date()])
    con.execute("""
        create or replace temp table _gm as
        select p.gln, m.month,
               (coalesce(p.valid_from, date '1900-01-01') <= last_day(m.month)
                and coalesce(p.valid_to, date '9999-12-31') >= m.month) as pharmacy_active
        from dim_pharmacy p cross join _months m
    """)
    _blocking_sets(con, tag)
    if strict:
        # Sensitivity only (superseded ): an unpinned rejected row withdraws the declared COMPLETE of the
        # depot-months it names, so recorded cells of those depot-months leave the primary source too.
        con.execute(f"""
            create or replace temp table _cov_eff as
            select c.covered_source, c.month, c.scope_type, c.scope_id,
                   case when c.covered_source = 'MUTABAKAT' and c.scope_type = 'DEPOT' and c.coverage_status = 'COMPLETE'
                             and exists (select 1 from _rr_{tag} r where r.rule_type <> 'cell'
                                         and c.month between r.m_from and r.m_to
                                         and (r.depot_parent is null or r.depot_parent = c.scope_id))
                        then 'PARTIAL' else c.coverage_status end as coverage_status
            from _cov c
        """)
        quarantine_impact["movement_mutabakat"]["depot_months_downgraded"] = int(con.execute("""
            select count(*) from _cov c join _cov_eff e using (covered_source, month, scope_type, scope_id)
            where c.coverage_status = 'COMPLETE' and e.coverage_status <> 'COMPLETE'
              and c.month in (select month from _months)""").fetchone()[0])
    else:
        con.execute("create or replace temp table _cov_eff as select * from _cov")
    # Primary eligibility: every linked depot declared COMPLETE for that month.
    if table_exists(con, "bridge_pharmacy_depot"):
        con.execute("""
            create or replace temp table _elig as
            select g.gln, g.month, count(b.depot_code) as n_links,
                   count(case when c.coverage_status = 'COMPLETE' then 1 end) as n_complete,
                   count(case when c.coverage_status = 'PARTIAL' then 1 end) as n_partial
            from _gm g
            left join bridge_pharmacy_depot b on b.gln = g.gln
                 and coalesce(b.valid_from, date '1900-01-01') <= last_day(g.month)
                 and coalesce(b.valid_to, date '9999-12-31') >= g.month
            left join _cov_eff c on c.covered_source = 'MUTABAKAT' and c.scope_type = 'DEPOT'
                 and c.scope_id = b.depot_code and c.month = g.month
            group by g.gln, g.month
        """)
    else:
        con.execute("create or replace temp table _elig as select gln, month, 0 as n_links, 0 as n_complete, "
                    "0 as n_partial from _gm")

    use_primary = mode in ("default", "primary_only")
    use_fallback = mode in ("default", "fallback_only")
    primary_zero = bool(policy.get("primary_absence_is_zero_when_complete", True))
    abs_thr = float(policy["conflict_threshold"]["abs_units"])
    rel_thr = float(policy["conflict_threshold"]["rel"])
    out = f"canon_{tag}"
    con.execute(f"""
        create or replace table {out} as
        with u as (
            select g.gln, g.month, pr.barcode, pr.form, g.pharmacy_active,
                   (coalesce(pr.valid_from, date '1900-01-01') <= last_day(g.month)
                    and coalesce(pr.valid_to, date '9999-12-31') >= g.month) as product_active,
                   e.n_links > 0 and e.n_complete = e.n_links as primary_eligible,
                   case when e.n_links = 0 then 'NO_DEPOT_LINK'
                        when e.n_complete = e.n_links then 'COMPLETE'
                        when e.n_partial > 0 then 'PARTIAL' else 'UNKNOWN' end as primary_coverage
            from _gm g join _elig e using (gln, month) cross join _scope_products pr
        ), j as (
            select u.*, m.units_paid as m_paid, m.units_free as m_free, m.returns_units as m_ret,
                   m.net_value_vat_inc_try as m_value, m.conflict as m_conflict, m.row_ids as m_rows,
                   m.units_paid + m.units_free - m.returns_units as m_net,
                   i.units_total as i_net, i.conflict as i_conflict, i.row_ids as i_rows,
                   mq.flag is not null as m_quar, iq.flag is not null as i_quar,
                   -- "no record -> 0" is withdrawn for cells a rejected row could belong to
                   {"false" if strict else "(ba.gln is not null or bb.gln is not null)"} as absence_blocked
            from u left join _mut m using (gln, month, barcode) left join _vendor i using (gln, month, barcode)
                   left join _qc_mutabakat mq on mq.gln = u.gln and mq.month = u.month and mq.barcode = u.barcode
                   left join _qc_vendor iq on iq.gln = u.gln and iq.month = u.month and iq.barcode = u.barcode
                   left join _blk_all_{tag} ba on ba.gln = u.gln and ba.month = u.month
                   left join _blk_b_{tag} bb on bb.gln = u.gln and bb.month = u.month and bb.barcode = u.barcode
        ), d as (
            select *,
              case
                when not (pharmacy_active and product_active) then 'out_of_scope'
                when {str(use_primary).lower()} and primary_eligible and coalesce(m_conflict, false) = false
                     and not m_quar and (m_net is not null or ({str(primary_zero).lower()} and not absence_blocked)) then 'mutabakat'
                when {str(use_fallback).lower()} and i_net is not null and coalesce(i_conflict, false) = false
                     and not i_quar then 'vendor'
                else null end as chosen_source
            from j
        )
        select gln, month, barcode, form,
               case when chosen_source = 'out_of_scope' then null else chosen_source end as chosen_source,
               case when chosen_source = 'mutabakat' then coalesce(m_paid, 0) end as units_paid,
               case when chosen_source = 'mutabakat' then coalesce(m_free, 0) end as units_free,
               case when chosen_source = 'mutabakat' then coalesce(m_ret, 0) end as returns_units,
               case when chosen_source = 'mutabakat' then coalesce(m_net, 0)
                    when chosen_source = 'vendor' then i_net end as net_units,
               -- Model target basis: gross outflow. Returns are never mixed in; an VENDOR cell that is
               -- net-negative cannot be split into gross and returns, so it is unobserved for the target.
               case when chosen_source = 'mutabakat' then coalesce(m_paid, 0) + coalesce(m_free, 0)
                    when chosen_source = 'vendor' and i_net >= 0 then i_net end as gross_units,
               case when chosen_source = 'mutabakat' then coalesce(m_value, 0) end as net_value_vat_inc_try,
               case
                 when chosen_source = 'out_of_scope' then 'out_of_scope'
                 when chosen_source = 'mutabakat' and coalesce(m_net, 0) > 0 then 'observed_positive'
                 when chosen_source = 'mutabakat' and coalesce(m_net, 0) = 0 then 'observed_zero'
                 when chosen_source = 'mutabakat' then 'observed_net_negative'
                 when chosen_source = 'vendor' and i_net > 0 then 'observed_positive'
                 when chosen_source = 'vendor' and i_net = 0 then 'observed_zero'
                 when chosen_source = 'vendor' then 'observed_net_negative'
                 when coalesce(m_conflict, false) or coalesce(i_conflict, false) or m_quar or i_quar then 'quarantined'
                 else 'unobserved' end as observation_status,
               case
                 when chosen_source = 'out_of_scope' then 'pharmacy_or_product_inactive'
                 when chosen_source = 'mutabakat' and m_net is null then 'primary_complete_coverage_no_record_zero'
                 when chosen_source = 'mutabakat' then 'primary_coverage_complete'
                 when chosen_source = 'vendor' and coalesce(m_conflict, false) then 'fallback_primary_conflict'
                 when chosen_source = 'vendor' and m_quar then 'fallback_primary_quarantined_rows'
                 when chosen_source = 'vendor' and primary_eligible and m_net is null and absence_blocked
                      then 'fallback_absence_zero_blocked_by_rejection'
                 when chosen_source = 'vendor' then 'fallback_primary_ineligible_' || lower(primary_coverage)
                 when coalesce(m_conflict, false) or coalesce(i_conflict, false) then 'conflicting_duplicates_quarantined'
                 when m_quar or i_quar then 'source_rows_quarantined'
                 when primary_eligible and m_net is null and absence_blocked then 'absence_zero_blocked_by_rejection'
                 else 'no_eligible_source_' || lower(primary_coverage) end as selection_reason,
               '{settings.policy["policy_version"]}' as selection_policy_version,
               case when chosen_source = 'mutabakat' then m_rows when chosen_source = 'vendor' then i_rows end
                   as selected_source_row_ids,
               m_net as alt_mutabakat_net_units, i_net as alt_vendor_net_units,
               (m_net is not null and i_net is not null
                and abs(m_net - i_net) > {abs_thr} and abs(m_net - i_net) > {rel_thr} * greatest(abs(m_net), abs(i_net)))
                   as conflict_flag,
               primary_coverage
        from d
    """)
    report.update(canonical_checks(con, out))
    log.info("canonical %s (%s): %s", tag, mode, report["status_counts"])
    return CanonicalResult(out, tag, cutoff, origin, mode, report)


def canonical_checks(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    """G1-style invariants."""
    n, n_keys = con.execute(f"select count(*), count(distinct (gln, month, barcode)) from {table}").fetchone()
    orphans_gln = con.execute(f"select count(*) from {table} c anti join dim_pharmacy p using (gln)").fetchone()[0]
    orphans_bc = con.execute(f"select count(*) from {table} c anti join dim_product p using (barcode)").fetchone()[0]
    status = dict(con.execute(f"select observation_status, count(*) from {table} group by 1 order by 1").fetchall())
    sources = dict(con.execute(f"select coalesce(chosen_source, '<none>'), count(*) from {table} group by 1").fetchall())
    totals = con.execute(f"""
        select coalesce(sum(net_units), 0), coalesce(sum(alt_mutabakat_net_units), 0),
               coalesce(sum(alt_vendor_net_units), 0), coalesce(sum(case when conflict_flag then 1 else 0 end), 0),
               coalesce(sum(case when observation_status in {OBSERVED} and selected_source_row_ids is null
                                  and selection_reason <> 'primary_complete_coverage_no_record_zero' then 1 else 0 end), 0)
        from {table}
    """).fetchone()
    canonical_total, mut_total, vendor_total, n_conflict, untraceable = (float(totals[0]), float(totals[1]),
                                                                       float(totals[2]), int(totals[3]), int(totals[4]))
    return {
        "rows": int(n), "pk_unique": bool(n == n_keys), "fk_orphans_gln": int(orphans_gln),
        "fk_orphans_barcode": int(orphans_bc), "status_counts": status, "chosen_source_counts": sources,
        "canonical_total_net_units": canonical_total, "mutabakat_total_net_units": mut_total,
        "vendor_total_net_units": vendor_total, "sources_not_summed": bool(abs(canonical_total - (mut_total + vendor_total)) > 0.5
                                                                      or (mut_total == 0 or vendor_total == 0)),
        "conflict_flag_cells": n_conflict, "observed_without_lineage": untraceable,
        "returns_units_total": float(con.execute(f"select coalesce(sum(returns_units), 0) from {table}").fetchone()[0]),
        "net_negative_cells": int(con.execute(f"select count(*) from {table} where observation_status = 'observed_net_negative'"
                                              ).fetchone()[0]),
        "vendor_cells_unobserved_for_target": int(con.execute(
            f"select count(*) from {table} where chosen_source = 'vendor' and gross_units is null").fetchone()[0]),
        "invalid_month_rows": int(con.execute(f"select count(*) from {table} where month is null").fetchone()[0]),
        "absence_zero_blocked_cells": int(con.execute(
            f"select count(*) from {table} where selection_reason like '%absence_zero_blocked_by_rejection'").fetchone()[0]),
    }


def reconciliation_report(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    """Per-month comparison of the two sources on comparable cells (never summed)."""
    return con.execute(f"""
        select strftime(month, '%Y-%m') as month,
               count(case when alt_mutabakat_net_units is not null and alt_vendor_net_units is not null then 1 end) as matched_keys,
               count(case when alt_mutabakat_net_units is not null and alt_vendor_net_units is null then 1 end) as mutabakat_only,
               count(case when alt_mutabakat_net_units is null and alt_vendor_net_units is not null then 1 end) as vendor_only,
               count(case when (alt_mutabakat_net_units = 0) = (alt_vendor_net_units = 0) then 1 end) as zero_nonzero_agree,
               sum(alt_mutabakat_net_units) as mutabakat_units, sum(alt_vendor_net_units) as vendor_units,
               sum(case when alt_mutabakat_net_units is not null and alt_vendor_net_units is not null
                        then abs(alt_mutabakat_net_units - alt_vendor_net_units) end) as abs_diff_matched,
               median(case when alt_mutabakat_net_units is not null and alt_vendor_net_units is not null
                                and alt_mutabakat_net_units <> 0
                           then (alt_vendor_net_units - alt_mutabakat_net_units) / abs(alt_mutabakat_net_units) end) as median_rel_diff,
               count(case when alt_mutabakat_net_units = 0 and alt_vendor_net_units is not null then 1 end) as zero_denominator_cells,
               sum(case when conflict_flag then 1 else 0 end) as conflict_cells,
               sum(net_units) as canonical_units,
               count(case when chosen_source = 'vendor' then 1 end) as fallback_cells
        from {table} group by 1 order by 1
    """).df()


def coverage_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        select covered_source as source, strftime(month, '%Y-%m') as month, scope_type, coverage_status, count(*) as scopes
        from _cov group by all order by 2, 1, 3, 4
    """).df()


def gln_month_totals(con: duckdb.DuckDBPyConnection, table: str, partial_as_observed: bool = False) -> pd.DataFrame:
    """DemoBrand total per GLN x month with a total-level status.

    A month is 'observed' only if every in-scope barcode cell is observed; mixed months are
    'partial' (lower bound only) and are treated as missing by features unless explicitly allowed.
    """
    df = con.execute(f"""
        select gln, month,
               sum(case when observation_status in {OBSERVED} then net_units end) as units,
               sum(case when observation_status in {OBSERVED} then gross_units end) as units_gross,
               count(case when observation_status in {OBSERVED} and gross_units is not null then 1 end) as n_observed_gross,
               count(case when observation_status <> 'out_of_scope' then 1 end) as n_in_scope,
               count(case when observation_status in {OBSERVED} then 1 end) as n_observed,
               count(case when chosen_source = 'vendor' then 1 end) as n_fallback,
               count(case when conflict_flag then 1 end) as n_conflict
        from {table} group by gln, month
    """).df()
    df["total_status"] = "unobserved"
    df.loc[df["n_in_scope"] == 0, "total_status"] = "out_of_scope"
    df.loc[(df["n_in_scope"] > 0) & (df["n_observed"] == df["n_in_scope"]), "total_status"] = "observed"
    df.loc[(df["n_observed"] > 0) & (df["n_observed"] < df["n_in_scope"]), "total_status"] = "partial"
    keep = ["observed", "partial"] if partial_as_observed else ["observed"]
    df.loc[~df["total_status"].isin(keep), "units"] = float("nan")
    # Gross total: every in-scope cell must be gross-observed (a net-negative VENDOR cell makes the month unobserved).
    gross_ok = (df["n_in_scope"] > 0) & (df["n_observed_gross"] == df["n_in_scope"])
    if partial_as_observed:
        gross_ok = (df["n_observed_gross"] > 0)
    df.loc[~gross_ok, "units_gross"] = float("nan")
    df["month"] = pd.to_datetime(df["month"])
    return df


def form_month_totals(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return con.execute(f"""
        select gln, month, form, sum(net_units) as units,
               bool_and(observation_status in {OBSERVED} or observation_status = 'out_of_scope') as complete
        from {table} group by gln, month, form
    """).df()


def export_canonical(con: duckdb.DuckDBPyConnection, table: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy {table} to '{path.as_posix()}' (format parquet)")
    return path


def rejection_reports(con: duckdb.DuckDBPyConnection, result: CanonicalResult, settings) -> dict[str, pd.DataFrame]:
    """Reports of what rejected rows did, read from the tables of this canonical build.

    - ``quarantine_quality``: per batch, the quality label from the draft thresholds (never changes a cell's status);
    - ``zero_blocking``: per rejected mutabakat row, the narrowed set of cells whose "no record -> 0" it blocks;
    - ``coverage_bounds``: per GLN-month, recorded total and an upper bound that adds every readable rejected row that
      could belong to it (report only - not a model input, Z5).
    """
    tag, table = result.tag, result.table
    pol = settings.policy["batch_quality"]
    tau_rel, tau_abs = float(pol["tau_rel"]), float(pol["tau_abs"])
    parts = []
    for dataset in ("movement_mutabakat", "movement_vendor"):
        sub, params = active_batches_sql(dataset, result.cutoff)
        df = con.execute(f"""
            select b.batch_id, b.dataset, b.period_start, b.period_end, b.rows_read, b.rows_accepted,
                   count(q.batch_id) as rows_rejected,
                   count(case when q.key_gln is not null and q.key_month is not null
                                   and q.key_barcode in (select barcode from dim_product) then 1 end) as pinned_rows
            from ingestion_batch b left join quarantine q on q.batch_id = b.batch_id
            where b.batch_id in ({sub}) group by all order by b.batch_id""", params).df()
        parts.append(df)
    quality = pd.concat(parts, ignore_index=True)
    quality["unresolved_rows"] = quality["rows_rejected"] - quality["pinned_rows"]
    quality["unresolved_rate"] = quality["unresolved_rows"] / quality["rows_accepted"].clip(lower=1)
    quality["tau_rel"], quality["tau_abs"] = tau_rel, tau_abs
    quality["threshold"] = np.maximum(tau_rel * quality["rows_accepted"], tau_abs)
    quality["quality"] = np.where(quality["unresolved_rows"] <= quality["threshold"], "acceptable_with_rejections",
                                  "review_required")
    quality.loc[quality["rows_rejected"] == 0, "quality"] = "clean"

    n_products = int(con.execute("select count(*) from _scope_products").fetchone()[0])
    blocking = con.execute(f"""
        with pm as (select rule_id, count(distinct (gln, month)) as pharmacy_months from _blk_{tag} group by 1),
        cb as (select b.rule_id, count(distinct (c.gln, c.month, c.barcode)) as cells_blocked
               from _blk_{tag} b join {table} c on c.gln = b.gln and c.month = b.month
                    and (b.barcode is null or b.barcode = c.barcode)
               where c.selection_reason like '%absence_zero_blocked_by_rejection' group by 1)
        select r.rule_id, r.batch_id, r.source_row_id, r.error_code, r.rule_type, r.gln is not null as gln_known,
               r.m_from as month_from, r.m_to as month_to, r.barcode, r.depot, r.units as units_readable,
               x.rule_id is null as out_of_brand_no_effect,
               coalesce(pm.pharmacy_months, 0) as pharmacy_months,
               coalesce(pm.pharmacy_months, 0) * case when x.eff_barcode is null then {n_products} else 1 end
                   as cells_potentially_affected,
               coalesce(cb.cells_blocked, 0) as cells_blocked
        from _rej_{tag} r left join _rr_{tag} x using (rule_id) left join pm on pm.rule_id = r.rule_id
             left join cb on cb.rule_id = r.rule_id order by r.rule_id""").df()
    bounds = con.execute(f"""
        with e as (select distinct b.rule_id, b.gln, b.month, r.units
                   from _blk_{tag} b join _rej_{tag} r using (rule_id)),
        lo as (select gln, month, coalesce(sum(case when observation_status in {OBSERVED} then net_units end), 0) as lower_units
               from {table} group by 1, 2)
        select e.gln, e.month, lo.lower_units, count(*) as n_possible_rows,
               coalesce(sum(e.units), 0) as readable_extra_units,
               count(case when e.units is null then 1 end) as unreadable_rows,
               case when count(case when e.units is null then 1 end) = 0 then lo.lower_units + coalesce(sum(e.units), 0)
                    end as upper_units
        from e join lo using (gln, month) group by e.gln, e.month, lo.lower_units order by e.gln, e.month""").df()
    return {"quarantine_quality": quality, "zero_blocking": blocking, "coverage_bounds": bounds}
