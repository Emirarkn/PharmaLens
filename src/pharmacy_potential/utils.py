"""Small shared helpers: hashing, months, JSON reports, redacted logging."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_GLN_RE = re.compile(r"\b\d{12,14}\b")


class RedactingFilter(logging.Filter):
    """Masks GLN/barcode-like identifiers in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        masked = _GLN_RE.sub(lambda m: m.group(0)[:3] + "*" * (len(m.group(0)) - 3), msg)
        if masked != msg:
            record.msg, record.args = masked, ()
        return True


def get_logger(name: str = "ppi") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_stamp(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_naive_utc(value: str | datetime) -> datetime:
    """DuckDB TIMESTAMP columns hold naive UTC."""
    return parse_utc(value).astimezone(timezone.utc).replace(tzinfo=None)


# --- months -----------------------------------------------------------------

def month_start(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(str(value)[:7] + "-01")


def month_str(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def month_end(value: Any) -> pd.Timestamp:
    return month_start(value) + pd.offsets.MonthEnd(0)


def add_months(value: Any, n: int) -> pd.Timestamp:
    return month_start(value) + pd.DateOffset(months=n)


def month_range(start: Any, end: Any) -> list[pd.Timestamp]:
    return list(pd.date_range(month_start(start), month_start(end), freq="MS"))


def month_index(value: Any) -> int:
    ts = pd.Timestamp(value)
    return ts.year * 12 + ts.month - 1


def backtest_cutoff(origin: Any, lag_days: int) -> datetime:
    return (month_end(origin) + timedelta(days=lag_days)).to_pydatetime().replace(tzinfo=None)


# --- deterministic tie-break ---------------------------------------------------

def stable_unit_hash(keys: pd.Series, seed: int) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per key (fixed tie-break rule)."""
    out = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys.astype(str)):
        digest = hashlib.blake2b(f"{seed}:{k}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / 2**64
    return out


# --- reports ----------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean_for_json(payload), fh, ensure_ascii=False, indent=2, default=_json_default)
    return path


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def frame_checksum(df: pd.DataFrame, sort_by: list[str], float_digits: int = 8) -> str:
    """Checksum of a table that ignores row order and tiny float noise."""
    d = df.sort_values(sort_by, kind="mergesort").reset_index(drop=True).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(float_digits)
    payload = d.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return sha256_text(payload)
