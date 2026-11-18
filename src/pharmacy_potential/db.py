"""DuckDB warehouse connection, core DDL and as-of helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

ACCEPTED_STATES = ("validated", "validated_with_quarantine")

# Typed key of a quarantined row (null where the raw value itself was invalid) and the time it became known.
QUARANTINE_KEY_COLUMNS = {"key_gln": "varchar", "key_month": "date", "key_barcode": "varchar",
                          "key_depot_code": "varchar", "available_at": "timestamp"}

CORE_DDL = [
    """
    create table if not exists ingestion_batch (
        batch_id varchar primary key,
        dataset varchar not null,
        source varchar not null,
        file_name varchar,
        stored_path varchar,
        sha256 varchar not null,
        received_at_utc timestamp,
        period_start date,
        period_end date,
        schema_version varchar,
        revision integer,
        supersedes_batch_id varchar,
        confidentiality varchar,
        license_id varchar,
        owner varchar,
        row_count_declared bigint,
        rows_read bigint,
        rows_accepted bigint,
        rows_quarantined bigint,
        available_at timestamp,
        availability_mode varchar,
        retrospective_lag_days integer,
        effective_available_from timestamp,
        synthetic boolean,
        state varchar not null,
        state_reason varchar,
        ingested_at timestamp
    )
    """,
    """
    create table if not exists quarantine (
        batch_id varchar,
        dataset varchar,
        source_row_id bigint,
        error_code varchar,
        error_detail varchar,
        raw_record varchar,
        key_gln varchar,
        key_month date,
        key_barcode varchar,
        key_depot_code varchar,
        available_at timestamp
    )
    """,
    """
    create table if not exists audit_event (
        event_id varchar primary key,
        actor_role varchar,
        actor_id varchar,
        action varchar,
        resource varchar,
        occurred_at timestamp,
        outcome varchar
    )
    """,
    """
    create table if not exists score_run (
        run_id varchar primary key,
        run_kind varchar,
        origin_month date,
        cutoff_utc timestamp,
        input_hash varchar,
        config_hash varchar,
        code_version varchar,
        model_version varchar,
        selection_policy varchar,
        seed bigint,
        training_end date,
        status varchar,
        exploratory boolean,
        synthetic boolean,
        score_checksum varchar,
        created_at timestamp,
        duration_s double
    )
    """,
]


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    for ddl in CORE_DDL:
        con.execute(ddl)
    # Warehouses created before lack the typed quarantine key columns.
    for col, ctype in QUARANTINE_KEY_COLUMNS.items():
        con.execute(f"alter table quarantine add column if not exists {col} {ctype}")
    return con


def input_hash(con: duckdb.DuckDBPyConnection) -> str:
    """Hash of the accepted batches (id + content hash): the identity of the data a run was built on."""
    import hashlib
    rows = con.execute("select batch_id, sha256 from ingestion_batch where state in "
                       "('validated', 'validated_with_quarantine') order by batch_id").fetchall()
    return hashlib.sha256("|".join(f"{b}:{h}" for b, h in rows).encode("utf-8")).hexdigest()


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(con.execute(
        "select count(*) from information_schema.tables where table_name = ?", [name]).fetchone()[0])


def active_batches_sql(dataset: str, cutoff: datetime | None) -> tuple[str, list]:
    """Batches of a dataset that are accepted, known at the cutoff and not superseded by then."""
    if cutoff is None:
        sql = f"""
            select b.batch_id from ingestion_batch b
            where b.dataset = ? and b.state in {ACCEPTED_STATES}
              and not exists (select 1 from ingestion_batch s
                              where s.supersedes_batch_id = b.batch_id and s.state in {ACCEPTED_STATES})
        """
        return sql, [dataset]
    sql = f"""
        select b.batch_id from ingestion_batch b
        where b.dataset = ? and b.state in {ACCEPTED_STATES}
          and b.effective_available_from <= ?
          and not exists (select 1 from ingestion_batch s
                          where s.supersedes_batch_id = b.batch_id and s.state in {ACCEPTED_STATES}
                            and s.effective_available_from <= ?)
    """
    return sql, [dataset, cutoff, cutoff]


def stage_rows(con: duckdb.DuckDBPyConnection, dataset: str, stage_table: str,
               cutoff: datetime | None, columns: str = "*"):
    """Relation of accepted staging rows as known at the cutoff (row availability enforced)."""
    sub, params = active_batches_sql(dataset, cutoff)
    sql = f"select {columns} from {stage_table} where _batch_id in ({sub})"
    if cutoff is not None:
        sql += " and _available_at <= ?"
        params = params + [cutoff]
    return con.execute(sql, params)


def ensure_dimension_views(con: duckdb.DuckDBPyConnection) -> list[str]:
    """(Re)create dimension views from the latest accepted master batches.

    Masters in the synthetic demo are retrospective snapshots; as-of master history is
    not available so these views carry no cutoff and are flagged in audits.
    """
    created = []
    specs = {
        "dim_pharmacy": ("pharmacy_master", "stg_pharmacy_master", "gln",
                         "gln, pharmacy_name, city_code, district_code, brick_code, valid_from, valid_to"),
        "dim_product": ("product_master", "stg_product_master", "barcode",
                        "barcode, brand, upper(coalesce(form, 'UNKNOWN')) as form, strength, pack_size, "
                        "unit_definition, valid_from, valid_to, list_price_vat_inc_try"),
        "dim_depot": ("depot_master", "stg_depot_master", "depot_code",
                      "depot_code, parent_depot_code, depot_level, depot_name, city_code, valid_from, valid_to"),
        "bridge_pharmacy_depot": ("pharmacy_depot", "stg_pharmacy_depot", "gln, depot_code",
                                  "gln, depot_code, valid_from, valid_to"),
    }
    for view, (dataset, table, key, cols) in specs.items():
        if not table_exists(con, table):
            continue
        sub, params = active_batches_sql(dataset, None)
        # DuckDB views cannot hold parameters; inline the dataset literal safely (fixed identifiers).
        sub = sub.replace("?", f"'{dataset}'")
        con.execute(f"""
            create or replace view {view} as
            select {cols}, _batch_id, _available_at from (
                select *, row_number() over (partition by {key} order by _available_at desc, _batch_id desc) as _rn
                from {table} where _batch_id in ({sub})
            ) where _rn = 1
        """)
        created.append(view)
    return created
