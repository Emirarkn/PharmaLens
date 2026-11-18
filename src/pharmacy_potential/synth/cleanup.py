"""Safe reset of the synthetic demo data root for ``ppi demo --fresh``.

Only what the demo itself creates may be removed. Anything else (an unmanaged file, a second raw
source, a warehouse holding real batches) makes the whole call refuse and nothing is deleted.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb


class UnsafeCleanup(Exception):
    """The data root holds something the synthetic demo does not own; nothing was deleted."""


def _is_synthetic_manifest(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("synthetic") is True
    except (OSError, ValueError):
        return False


def _warehouse_is_synthetic(path: Path) -> bool:
    """True only when the warehouse is a readable DuckDB file in which every batch is synthetic."""
    try:
        con = duckdb.connect(str(path), read_only=True)
    except duckdb.Error:
        return False
    try:
        real = con.execute("select count(*) from ingestion_batch where synthetic is not true").fetchone()[0]
        return real == 0
    except duckdb.Error:
        return False
    finally:
        con.close()


def plan_synthetic_cleanup(data_root: Path, warehouse_file: str = "warehouse.duckdb") -> list[Path]:
    """Top-level paths under ``data_root`` that are safe to delete. Raises UnsafeCleanup otherwise."""
    root = Path(data_root).resolve()
    if not root.exists():
        return []
    if root == Path(root.anchor):
        raise UnsafeCleanup(f"{root} is a filesystem root")
    warehouse_names = {warehouse_file, warehouse_file + ".wal"}
    owned_data = {"raw", "synthetic_truth"}          # written by `ppi synth`
    owned_derived = {"canonical", "cache"}           # rebuilt from the warehouse by every run
    unknown = sorted(p.name for p in root.iterdir() if p.name not in owned_data | owned_derived | warehouse_names)
    if unknown:
        raise UnsafeCleanup(f"unmanaged entries in {root}: {', '.join(unknown)}")
    for p in root.iterdir():
        if p.is_symlink() or any(c.is_symlink() for c in p.rglob("*")):
            raise UnsafeCleanup(f"symbolic link under {p}")

    raw = root / "raw"
    if raw.exists():
        others = sorted(p.name for p in raw.iterdir() if p.name != "synthetic")
        if others:
            raise UnsafeCleanup(f"raw sources other than 'synthetic': {', '.join(others)}")
        synthetic_files = sorted((raw / "synthetic").rglob("*")) if (raw / "synthetic").exists() else []
        for f in synthetic_files:
            if not f.is_file():
                continue
            manifest = f if f.name.endswith(".manifest.json") else f.with_name(f.name + ".manifest.json")
            if not manifest.exists() or not _is_synthetic_manifest(manifest):
                raise UnsafeCleanup(f"{f.relative_to(root)} has no synthetic manifest")

    wh = root / warehouse_file
    derived_present = any((root / n).exists() for n in owned_derived)
    if wh.exists():
        if not _warehouse_is_synthetic(wh):
            raise UnsafeCleanup(f"{warehouse_file} contains non-synthetic batches or is not a readable warehouse")
    elif derived_present:
        raise UnsafeCleanup("derived folders exist but there is no warehouse to prove they are synthetic")
    return [p for p in sorted(root.iterdir())]


def clean_synthetic_root(data_root: Path, warehouse_file: str = "warehouse.duckdb") -> list[Path]:
    """Delete the synthetic demo's files under ``data_root`` (the root folder itself is kept)."""
    targets = plan_synthetic_cleanup(data_root, warehouse_file)
    for p in targets:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return targets
