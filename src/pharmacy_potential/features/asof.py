"""Panels built as of a cutoff, one per training origin.

A training row for origin ``o`` must carry the features that were knowable at ``cutoff(o)``: a delivery or
revision of an earlier month that arrived after ``cutoff(o)`` must not reach it. The expensive part, the
canonical totals, is cached per (cutoff, origin, input hash, code+config fingerprint); contexts that share a
cutoff reuse it. The cache lives under ``<data_root>/cache`` (same protection as the warehouse).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from ..db import input_hash
from ..provenance import source_tree_hash
from ..reconcile.canonical import build_canonical
from ..utils import get_logger, hash_files, month_start, sha256_text
from .build import Panel, panel_frames, panel_from_frames

log = get_logger()


class AsOfPanels:
    def __init__(self, con: duckdb.DuckDBPyConnection, settings, use_cache: bool = True):
        self.con, self.settings, self.use_cache = con, settings, use_cache
        self.input_hash = input_hash(con)
        # Any change to data, config or code changes the key: a stale cache can never be served.
        # Overrides that change what the canonical build contains are part of the key as well (a CLI/test override
        # of the brand or the history start is not visible in the config file hashes).
        scope = json.dumps({k: str(settings.get(k)) for k in ("history_start_month", "brand")}, sort_keys=True)
        self.fingerprint = sha256_text(hash_files(settings.config_files()) + source_tree_hash(settings.root) + scope)
        self.dir: Path = settings.data_root / "cache" / "canonical_totals"
        self.hits = 0
        self.builds = 0
        self.log: list[dict] = []          # (origin, cutoff, panel_last_month) of every panel handed out

    def _key(self, origin: pd.Timestamp, cutoff: datetime) -> str:
        raw = f"{pd.Timestamp(cutoff).isoformat()}|{origin:%Y-%m}|{self.input_hash}|{self.fingerprint}"
        return f"{pd.Timestamp(cutoff):%Y%m%d}_{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

    def _read(self, key: str) -> dict[str, pd.DataFrame] | None:
        if not self.use_cache:
            return None
        try:
            return {n: pd.read_parquet(self.dir / f"{key}_{n}.parquet") for n in ("totals", "forms")}
        except (OSError, ValueError):
            return None

    def _write(self, key: str, frames: dict[str, pd.DataFrame]) -> None:
        if not self.use_cache:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for n, df in frames.items():
                tmp = self.dir / f"{key}_{n}.parquet.tmp"
                df.to_parquet(tmp, index=False)
                tmp.replace(self.dir / f"{key}_{n}.parquet")
        except OSError as exc:             # the cache is an optimisation only
            log.warning("canonical cache not written: %s", exc)

    def panel(self, origin, cutoff: datetime) -> Panel:
        origin = month_start(origin)
        key = self._key(origin, cutoff)
        frames = self._read(key)
        if frames is None:
            canon = build_canonical(self.con, self.settings, cutoff, origin, "asof_panel")
            frames = panel_frames(self.con, canon.table)
            self.con.execute(f"drop table if exists {canon.table}")
            self._write(key, frames)
            self.builds += 1
        else:
            self.hits += 1
        panel = panel_from_frames(self.con, frames, cutoff)
        self.log.append({"origin": origin.strftime("%Y-%m"), "cutoff_utc": pd.Timestamp(cutoff),
                         "panel_last_month": panel.months[-1].strftime("%Y-%m")})
        return panel
