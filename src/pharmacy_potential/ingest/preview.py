"""Upload preview: what a file would do if it were loaded, without loading it.

It runs the same validation as the real load (``validate_rows``) and writes nothing to the warehouse, so a preview
can never disagree with the load that follows. The error report lists rejected rows with the reason, ready to be
corrected and re-uploaded; it holds raw values and is therefore restricted data like the quarantine table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..config import Settings
from ..utils import sha256_file
from .loader import read_raw, validate_rows
from .templates import template_version


def preview_file(settings: Settings, con: duckdb.DuckDBPyConnection, dataset: str, path: Path) -> dict[str, Any]:
    path = Path(path)
    src = settings.source_config(dataset)
    raw = read_raw(path, src)
    out: dict[str, Any] = {"dataset": dataset, "file": path.name, "template_version": template_version(settings, dataset),
                           "rows_read": int(len(raw)), "ok": False}
    out["unknown_columns"] = sorted(set(raw.columns) - {s["source"] for s in src["columns"].values()})
    missing = [s["source"] for s in src["columns"].values() if s.get("required") and s["source"] not in raw.columns]
    if missing:
        out["fatal"] = f"Missing required columns: {missing}"
        return out
    typed, errs = validate_rows(con, src, raw)
    bad = errs != ""
    out["rows_accepted"], out["rows_rejected"] = int((~bad).sum()), int(bad.sum())
    codes = errs[bad].str.split(";").explode().str.split(":").str[0]
    out["error_counts"] = {k: int(v) for k, v in codes.value_counts().items()}
    tc = src.get("time_column")
    if tc and tc in typed and typed[tc].notna().any():
        out["period_min"], out["period_max"] = str(typed[tc].min())[:10], str(typed[tc].max())[:10]
    key = [k for k in src.get("key", []) if k in typed]
    ok_rows = typed[~bad]
    out["duplicate_key_rows"] = int(ok_rows.duplicated(key, keep=False).sum()) if key and len(ok_rows) else 0
    sha = sha256_file(path)
    prev = con.execute("select batch_id from ingestion_batch where dataset = ? and sha256 = ? and state <> 'rejected'",
                       [dataset, sha]).fetchone()
    out["already_ingested_as"] = prev[0] if prev else None       # loading it again would add nothing
    out["ok"] = bool(len(raw)) and not bad.all()
    out["error_rows"] = pd.concat([pd.DataFrame({"source_row": (raw.index[bad.to_numpy()] + 1).astype(int),
                                                 "error": errs[bad].to_numpy()}),
                                   raw[bad.to_numpy()].reset_index(drop=True)], axis=1)
    return out


def write_error_report(preview: dict[str, Any], path: Path) -> Path:
    """Rejected rows with reasons as an Excel file; every cell is written as plain text (no formula/number/URL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = preview["error_rows"].astype(str)
    with pd.ExcelWriter(path, engine="xlsxwriter", engine_kwargs={"options": {
            "strings_to_numbers": False, "strings_to_formulas": False, "strings_to_urls": False}}) as xw:
        df.to_excel(xw, sheet_name="Hatali_satirlar", index=False)
        xw.sheets["Hatali_satirlar"].set_column(0, len(df.columns), 18, xw.book.add_format({"num_format": "@"}))
    return path
