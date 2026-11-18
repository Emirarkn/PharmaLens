"""Raw ingestion: immutable raw copy, manifest, typed staging, quarantine, batch state.

Row accounting invariant per batch: rows_read = rows_accepted + rows_quarantined.
Within-source duplicates are kept in staging and resolved in reconcile (explicitly counted there).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..config import Settings
from ..db import ACCEPTED_STATES, ensure_dimension_views, stage_rows, table_exists
from ..gates import require_gates
from ..utils import get_logger, parse_utc, to_naive_utc, utc_now
from .manifest import effective_available_from, sidecar_path, validate_manifest
from .parse import DUCK_TYPES, parse_frame

log = get_logger()

FK_REFERENCES = {
    "pharmacy": ("pharmacy_master", "stg_pharmacy_master", "gln"),
    "product": ("product_master", "stg_product_master", "barcode"),
    "depot": ("depot_master", "stg_depot_master", "depot_code"),
}


class IngestError(Exception):
    """File-level failure. exit_code 1 = validation failure, 2 = config/input unavailable."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class IngestResult:
    batch_id: str
    dataset: str
    state: str
    rows_read: int = 0
    rows_accepted: int = 0
    rows_quarantined: int = 0
    error_counts: dict[str, int] | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_MIDNIGHT = r"^(\d{4}-\d{2}-\d{2}) 00:00:00$"


def read_raw(path: Path, src: dict[str, Any]) -> pd.DataFrame:
    """Raw text frame. An Excel upload (.xlsx) goes through the same mapping as the CSV delivery: the sheet is
    ``file.sheet`` of the mapping, else the template's "Data" sheet, else the first sheet."""
    fmt = src.get("file", {})
    kind = "xlsx" if Path(path).suffix.lower() in (".xlsx", ".xlsm") else fmt.get("format", "csv")
    if kind == "csv":
        return pd.read_csv(path, sep=fmt.get("delimiter", ","), dtype=str, keep_default_na=False,
                           encoding=fmt.get("encoding", "utf-8"))
    if kind == "xlsx":
        sheet = fmt.get("sheet")
        if sheet is None:
            sheet = "Data" if "Data" in pd.ExcelFile(path).sheet_names else 0
        df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
        # Excel stores a typed date as a timestamp; a date without a time part is the same date, not a new value.
        return df.apply(lambda c: c.str.replace(_MIDNIGHT, r"\1", regex=True))
    if kind == "parquet":
        return pd.read_parquet(path).astype("string").fillna("")
    raise IngestError(f"Unsupported file format '{kind}'", exit_code=2)


def _reference_set(con: duckdb.DuckDBPyConnection, ref: str) -> set[str]:
    dataset, table, col = FK_REFERENCES[ref]
    if not table_exists(con, table):
        raise IngestError(f"Reference '{ref}' not loaded yet (ingest {dataset} first)", exit_code=2)
    rows = stage_rows(con, dataset, table, None, f"distinct {col}").fetchall()
    return {r[0] for r in rows}


def _row_available_at(typed: pd.DataFrame, src: dict[str, Any], manifest: dict[str, Any]) -> pd.Series:
    batch_avail = pd.Timestamp(to_naive_utc(manifest["available_at"]))
    n = len(typed)
    explicit_col = src.get("available_at_column")
    time_col = src.get("time_column")
    if manifest["availability_mode"] == "retrospective_lag":
        lag = timedelta(days=manifest.get("retrospective_lag_days") or 0)
        if explicit_col and explicit_col in typed:
            base = pd.to_datetime(typed[explicit_col])
        elif time_col and time_col in typed:
            t = pd.to_datetime(typed[time_col])
            base = (t + pd.offsets.MonthEnd(0)) if src["columns"][time_col]["type"] == "month" else t
            base = base + lag
        else:
            base = pd.Series(pd.Timestamp(manifest["period_start"]), index=typed.index)
        return base.fillna(batch_avail)
    # Batch mode: nothing is known before the batch arrived.
    if explicit_col and explicit_col in typed:
        explicit = pd.to_datetime(typed[explicit_col]).fillna(batch_avail)
        return explicit.where(explicit > batch_avail, batch_avail)
    return pd.Series([batch_avail] * n, index=typed.index)


def validate_rows(con: duckdb.DuckDBPyConnection, src: dict[str, Any], raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Typed frame and per-row error string (empty = accepted). The single validation path: the real load and the
    upload preview both call it, so a preview can never disagree with the load that follows."""
    typed, errs = parse_frame(raw, src)
    for field, ref in (src.get("fk") or {}).items():
        refs = _reference_set(con, ref)
        unknown = typed[field].notna() & ~typed[field].isin(refs)
        add = pd.Series(np.where(unknown, f"FK_NOT_FOUND:{field}", ""), index=errs.index, dtype="string")
        errs = errs + pd.Series(np.where((errs != "") & (add != ""), ";", ""), index=errs.index) + add
    return typed, errs


def _ensure_stage_table(con: duckdb.DuckDBPyConnection, table: str, src: dict[str, Any]) -> None:
    cols = ", ".join(f"{f} {DUCK_TYPES[s['type']]}" for f, s in src["columns"].items())
    con.execute(f"create table if not exists {table} ({cols}, _batch_id varchar, "
                f"_source_row_id bigint, _available_at timestamp)")


def load_manifest_for(path: Path, manifest: dict[str, Any] | Path | None) -> dict[str, Any]:
    if isinstance(manifest, dict):
        return manifest
    mpath = Path(manifest) if manifest else sidecar_path(path)
    if not mpath.exists():
        raise IngestError(f"No manifest for {path.name} (expected {mpath.name}); build one with "
                          "`ppi manifest` or pass --manifest", exit_code=2)
    with open(mpath, encoding="utf-8") as fh:
        return json.load(fh)


def ingest_file(settings: Settings, con: duckdb.DuckDBPyConnection, dataset: str, path: Path,
                manifest: dict[str, Any] | Path | None = None) -> IngestResult:
    path = Path(path)
    src = settings.source_config(dataset)
    m = load_manifest_for(path, manifest)
    if m.get("dataset") != dataset:
        raise IngestError(f"Manifest dataset '{m.get('dataset')}' != '{dataset}'", exit_code=2)
    errors = validate_manifest(m)
    if errors:
        raise IngestError("Invalid manifest: " + "; ".join(errors), exit_code=1)
    if not m.get("synthetic"):      # real data needs G0 (access and rights) accepted before it enters the warehouse
        require_gates(settings, "data", f"ingest of {m.get('batch_id', path.name)}")

    from ..utils import sha256_file
    sha = sha256_file(path)
    if sha != m["sha256"]:
        raise IngestError(f"SHA-256 mismatch for {path.name}: file changed after manifest was written")

    # Idempotency: same content for the same dataset is a no-op.
    dup = con.execute("select batch_id, state from ingestion_batch where dataset = ? and sha256 = ? "
                      "and state <> 'rejected'", [dataset, sha]).fetchone()
    if dup:
        return IngestResult(dup[0], dataset, "duplicate", message=f"identical file already ingested as {dup[0]}")
    if con.execute("select count(*) from ingestion_batch where batch_id = ?", [m["batch_id"]]).fetchone()[0]:
        raise IngestError(f"batch_id {m['batch_id']} already exists with different content")
    if m.get("supersedes_batch_id"):
        prev = con.execute("select state from ingestion_batch where batch_id = ?", [m["supersedes_batch_id"]]).fetchone()
        if not prev:
            raise IngestError(f"supersedes_batch_id {m['supersedes_batch_id']} not found", exit_code=2)

    # Immutable raw copy (kept even when the file later fails validation).
    raw_dir = settings.data_root / "raw" / m["source"] / m["batch_id"]
    stored = raw_dir / path.name
    if not stored.exists() and settings.data_root / "raw" not in path.resolve().parents:
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, stored)
        shutil.copy2(sidecar_path(path), sidecar_path(stored)) if sidecar_path(path).exists() else None
    stored_path = stored if stored.exists() else path.resolve()

    eff = effective_available_from(m)
    con.execute(
        """insert into ingestion_batch (batch_id, dataset, source, file_name, stored_path, sha256, received_at_utc,
               period_start, period_end, schema_version, revision, supersedes_batch_id, confidentiality, license_id,
               owner, row_count_declared, available_at, availability_mode, retrospective_lag_days,
               effective_available_from, synthetic, state, ingested_at)
           values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'received', ?)""",
        [m["batch_id"], dataset, m["source"], path.name, str(stored_path), sha, to_naive_utc(m["received_at_utc"]),
         m["period_start"], m["period_end"], m["schema_version"], m["revision"], m.get("supersedes_batch_id"),
         m["confidentiality"], m["license_id"], m["owner"], m.get("row_count"), to_naive_utc(m["available_at"]),
         m["availability_mode"], m.get("retrospective_lag_days"), to_naive_utc(eff), bool(m.get("synthetic", False)),
         to_naive_utc(utc_now())])

    try:
        raw = read_raw(path, src)
        missing_cols = [s["source"] for s in src["columns"].values() if s.get("required") and s["source"] not in raw.columns]
        if missing_cols:
            raise IngestError(f"Missing required columns: {missing_cols}")
        if m.get("row_count") is not None and int(m["row_count"]) != len(raw):
            raise IngestError(f"Row count {len(raw)} != manifest row_count {m['row_count']}")

        typed, errs = validate_rows(con, src, raw)

        bad = errs != ""
        typed["_batch_id"] = m["batch_id"]
        typed["_source_row_id"] = np.arange(1, len(typed) + 1, dtype="int64")
        typed["_available_at"] = _row_available_at(typed, src, m)

        table = src["stage_table"]
        accepted = typed[~bad]
        quarantined = raw[bad.to_numpy()]
        q = pd.DataFrame({
            "batch_id": m["batch_id"], "dataset": dataset,
            "source_row_id": typed.loc[bad, "_source_row_id"].to_numpy(),
            "error_code": errs[bad].str.split(";").str[0].str.split(":").str[0].to_numpy(),
            "error_detail": errs[bad].to_numpy(),
            "raw_record": [json.dumps(r, ensure_ascii=False) for r in quarantined.to_dict("records")],
            # Typed key of the rejected row (null where the raw value was itself invalid): reconcile uses it
            # so that a rejected row can never be read as "no record -> observed zero".
            **{f"key_{f}": (typed.loc[bad, f].to_numpy() if f in typed else None)
               for f in ("gln", "month", "barcode", "depot_code")},
            "available_at": typed.loc[bad, "_available_at"].to_numpy(),
        })

        con.execute("begin transaction")
        _ensure_stage_table(con, table, src)
        cols = list(src["columns"].keys())
        select = ", ".join(f"cast({c} as {DUCK_TYPES[src['columns'][c]['type']]}) as {c}" for c in cols)
        con.register("_accepted", accepted)
        con.execute(f"insert into {table} select {select}, _batch_id, _source_row_id, _available_at from _accepted")
        con.unregister("_accepted")
        if len(q):
            con.register("_q", q)
            qcols = ("batch_id, dataset, source_row_id, error_code, error_detail, raw_record, "
                     "key_gln, key_month, key_barcode, key_depot_code, available_at")
            con.execute(f"insert into quarantine ({qcols}) select {qcols} from _q")
            con.unregister("_q")
        state = "validated_with_quarantine" if len(q) else "validated"
        con.execute("update ingestion_batch set rows_read = ?, rows_accepted = ?, rows_quarantined = ?, state = ? "
                    "where batch_id = ?", [len(raw), len(accepted), len(q), state, m["batch_id"]])
        con.execute("commit")
    except Exception as exc:  # atomic failure: nothing staged, batch kept as rejected with reason
        try:
            con.execute("rollback")
        except duckdb.Error:
            pass
        reason = str(exc)[:500]
        con.execute("update ingestion_batch set state = 'rejected', state_reason = ? where batch_id = ?",
                    [reason, m["batch_id"]])
        if isinstance(exc, IngestError):
            return IngestResult(m["batch_id"], dataset, "rejected", message=reason)
        raise

    if dataset in {"pharmacy_master", "product_master", "depot_master", "pharmacy_depot"}:
        ensure_dimension_views(con)
    counts = q["error_code"].value_counts().to_dict() if len(q) else {}
    log.info("ingested %s: read=%d accepted=%d quarantined=%d", m["batch_id"], len(raw), len(accepted), len(q))
    return IngestResult(m["batch_id"], dataset, state, len(raw), len(accepted), len(q), counts)


def ingest_path(settings: Settings, con: duckdb.DuckDBPyConnection, dataset: str, path: Path) -> list[IngestResult]:
    """Ingest one file or every data file (with sidecar manifest) in a directory, oldest revision first."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.endswith(".manifest.json"))
    else:
        files = [path]

    def order(p: Path) -> tuple:
        mp = sidecar_path(p)
        if mp.exists():
            with open(mp, encoding="utf-8") as fh:
                mm = json.load(fh)
            return (parse_utc(mm["available_at"]), mm["revision"], p.name)
        return (parse_utc("1970-01-01T00:00:00Z"), 0, p.name)

    return [ingest_file(settings, con, dataset, f) for f in sorted(files, key=order)]


def batch_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        select dataset, count(*) batches,
               sum(case when state in {ACCEPTED_STATES} then 1 else 0 end) accepted_batches,
               sum(case when state = 'rejected' then 1 else 0 end) rejected_batches,
               sum(rows_read) rows_read, sum(rows_accepted) rows_accepted, sum(rows_quarantined) rows_quarantined
        from ingestion_batch group by dataset order by dataset
    """).df()
