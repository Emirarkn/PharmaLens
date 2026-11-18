"""Typed, vectorised parsing of raw string columns with per-row error codes.

Identifiers stay strings (leading zeros preserved). Nothing is coerced silently:
unparseable values produce an error code and the row goes to quarantine.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DUCK_TYPES = {
    "gln": "varchar", "barcode": "varchar", "code": "varchar", "string": "varchar",
    "month": "date", "date": "date", "int": "bigint", "decimal": "decimal(18,4)",
}

_GLN = r"^\d{13}$"
_BARCODE = r"^\d{8,14}$"


def _blank_to_na(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s == "", pd.NA)


def _num_text(s: pd.Series, decimal: str, thousands: str | None) -> pd.Series:
    if thousands:
        s = s.str.replace(thousands, "", regex=False)
    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)
    return s


def parse_column(raw: pd.Series, ctype: str, fmt: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Return (typed values, error code or NA) for one column."""
    s = _blank_to_na(raw)
    present = s.notna()
    err = pd.Series(pd.NA, index=s.index, dtype="string")

    if ctype == "gln":
        ok = s.str.fullmatch(_GLN).fillna(False).astype(bool)
        sci = s.str.contains(r"[eE]\+", regex=True).fillna(False).astype(bool)
        err = err.mask(present & ~ok, "INVALID_GLN").mask(present & sci, "INVALID_GLN_NUMERIC_CONVERSION")
        return s.where(ok), err
    if ctype == "barcode":
        ok = s.str.fullmatch(_BARCODE).fillna(False).astype(bool)
        sci = s.str.contains(r"[eE]\+", regex=True).fillna(False).astype(bool)
        err = err.mask(present & ~ok, "INVALID_BARCODE").mask(present & sci, "INVALID_BARCODE_NUMERIC_CONVERSION")
        return s.where(ok), err
    if ctype == "month":
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
        for f in fmt.get("month_formats", ["%Y-%m"]):
            parsed = pd.to_datetime(s.where(out.isna()), format=f, errors="coerce")
            out = out.fillna(parsed)
        out = out.dt.to_period("M").dt.to_timestamp()
        err = err.mask(present & out.isna(), "INVALID_MONTH")
        return out, err
    if ctype == "date":
        out = pd.to_datetime(s, format=fmt.get("date_format", "%Y-%m-%d"), errors="coerce")
        iso = pd.to_datetime(s.where(out.isna()), format="%Y-%m-%d", errors="coerce")
        out = out.fillna(iso)
        err = err.mask(present & out.isna(), "INVALID_DATE")
        return out, err
    if ctype in ("int", "decimal"):
        txt = _num_text(s, fmt.get("decimal", "."), fmt.get("thousands"))
        num = pd.to_numeric(txt, errors="coerce")
        bad = present & num.isna()
        if ctype == "int":
            frac = num.notna() & (np.floor(num.fillna(0)) != num.fillna(0))
            bad = bad | frac
            num = num.where(~frac).round().astype("Int64")
        else:
            num = num.astype("Float64")
        err = err.mask(bad, "INVALID_NUMBER")
        return num, err
    if ctype in ("code", "string"):
        return s, err
    raise ValueError(f"Unknown column type '{ctype}'")


def parse_frame(df_raw: pd.DataFrame, src: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Parse all mapped columns. Returns typed frame and a per-row ';'-joined error string."""
    fmt = src.get("file", {})
    typed = {}
    errors = pd.Series("", index=df_raw.index, dtype="string")

    def add(mask: pd.Series, code_series: pd.Series | str) -> None:
        nonlocal errors
        codes = code_series if isinstance(code_series, pd.Series) else pd.Series(code_series, index=errors.index)
        add_txt = codes.where(mask, "").fillna("")
        sep = np.where((errors != "") & (add_txt != ""), ";", "")
        errors = errors + sep + add_txt

    for field, spec in src["columns"].items():
        col = spec["source"]
        raw = df_raw[col] if col in df_raw.columns else pd.Series(pd.NA, index=df_raw.index, dtype="string")
        values, err = parse_column(raw, spec["type"], fmt)
        typed[field] = values
        add(err.notna(), err.fillna("") + ":" + field)
        if spec.get("required"):
            missing = _blank_to_na(raw).isna()
            add(missing, f"MISSING_REQUIRED:{field}")
    for field in src.get("nonnegative", []):
        v = typed.get(field)
        if v is not None:
            neg = pd.Series(v).fillna(0) < 0
            add(neg.astype(bool), f"NEGATIVE_VALUE:{field}")
    return pd.DataFrame(typed, index=df_raw.index), errors
