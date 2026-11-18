"""Profile a newly delivered file and draft a source-mapping YAML.

Only structural statistics are reported (null rate, distinct count, length range, value
pattern classes) - never raw values - so the profile can be shared with the data owner.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import Settings

PATTERNS = [
    ("gln13", r"^\d{13}$"), ("barcode", r"^\d{8,14}$"), ("yyyy-mm", r"^\d{4}-\d{2}$"), ("yyyymm", r"^\d{6}$"),
    ("date_iso", r"^\d{4}-\d{2}-\d{2}$"), ("date_dmy_dot", r"^\d{2}\.\d{2}\.\d{4}$"), ("int", r"^-?\d+$"),
    ("decimal_comma", r"^-?\d+,\d+$"), ("decimal_dot", r"^-?\d+\.\d+$"), ("scientific", r"^-?\d+(\.\d+)?[eE][+-]?\d+$"),
]

SYNONYMS = {
    "gln": ["gln", "eczanegln", "pharmacygln", "eczane_gln", "glnno"],
    "month": ["donem", "ay", "month", "period", "yilay"],
    "barcode": ["barkod", "barcode", "ean", "urunbarkod"],
    "depot_code": ["depokodu", "depo", "depot", "depotcode", "depo_kodu"],
    "units_paid": ["satisadet", "adet", "units", "kutu", "miktar"],
    "units_free": ["mfadet", "malfazlasi", "mf", "freeunits"],
    "returns_units": ["iadeadet", "iade", "returns"],
    "net_value_vat_inc_try": ["nettutarkdvdahil", "tutar", "net_tutar", "value", "ciro"],
    "units_total": ["units", "adet", "kutu"],
    "pharmacy_name": ["eczaneadi", "eczane", "name", "unvan"],
    "city_code": ["ilkodu", "il", "city", "citycode"],
    "district_code": ["ilcekodu", "ilce", "district"],
    "brick_code": ["brickkodu", "brick"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().translate(str.maketrans("çğıöşüİ", "cgiosui")))


def sniff_delimiter(path: Path, encoding: str = "utf-8") -> str:
    with open(path, encoding=encoding, errors="replace") as fh:
        sample = fh.read(20000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        return ","


def profile_file(path: Path, encoding: str = "utf-8", sheet: str | int = 0, max_rows: int | None = 200_000) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet, dtype=str, nrows=max_rows).fillna("")
        fmt = {"format": "xlsx", "sheet": sheet}
    elif suffix == ".parquet":
        df = pd.read_parquet(path).astype("string").fillna("")
        fmt = {"format": "parquet"}
    else:
        delim = sniff_delimiter(path, encoding)
        df = pd.read_csv(path, sep=delim, dtype=str, keep_default_na=False, encoding=encoding, nrows=max_rows)
        fmt = {"format": "csv", "delimiter": delim, "encoding": encoding}
    cols = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        nonblank = s[s != ""]
        pats = {}
        for name, rx in PATTERNS:
            share = float(nonblank.str.fullmatch(rx).mean()) if len(nonblank) else 0.0
            if share > 0:
                pats[name] = round(share, 4)
        cols.append({"column": c, "null_or_blank_share": round(float((s == "").mean()), 4),
                     "distinct": int(nonblank.nunique()),
                     "length_min": int(nonblank.str.len().min()) if len(nonblank) else None,
                     "length_max": int(nonblank.str.len().max()) if len(nonblank) else None,
                     "patterns": pats})
    if fmt.get("format") == "csv":
        dec = sum(cp["patterns"].get("decimal_comma", 0) for cp in cols)
        fmt["decimal"] = "," if dec > 0 else "."
    return {"file": path.name, "rows_profiled": int(len(df)), "file_format": fmt, "columns": cols,
            "warnings": [f"{cp['column']}: scientific notation found (Excel numeric conversion of identifiers?)"
                         for cp in cols if cp["patterns"].get("scientific")] +
                        [f"{cp['column']}: 13-digit identifiers with length < 13 present (lost leading zeros?)"
                         for cp in cols if cp["patterns"].get("gln13", 0) > 0.5 and (cp["length_min"] or 13) < 13]}


def draft_mapping(settings: Settings, dataset: str, prof: dict[str, Any]) -> dict[str, Any]:
    """Start from the dataset's existing contract and guess `source` columns by name similarity."""
    base = settings.source_config(dataset)
    cols = {_norm(c["column"]): c["column"] for c in prof["columns"]}
    mapped = {}
    for field, spec in base["columns"].items():
        guess = None
        for cand in [field] + SYNONYMS.get(field, []):
            if _norm(cand) in cols:
                guess = cols[_norm(cand)]
                break
        mapped[field] = {**spec, "source": guess or f"<<MAP ME: was {spec['source']}>>"}
    out = dict(base)
    out["file"] = {**base.get("file", {}), **prof["file_format"]}
    out["columns"] = mapped
    return out


def write_draft(mapping: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# DRAFT mapping generated by `ppi profile` - review every `source` and type, then copy to configs/sources/\n"
    path.write_text(header + yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
