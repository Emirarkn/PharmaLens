""": `demo --fresh` may only delete what the synthetic demo owns."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pharmacy_potential.cli import app
from pharmacy_potential.db import connect
from pharmacy_potential.synth.cleanup import UnsafeCleanup, clean_synthetic_root

CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml")


def _batch(con, batch_id: str, synthetic: bool) -> None:
    con.execute("insert into ingestion_batch (batch_id, dataset, source, sha256, state, synthetic) "
                "values (?, 'movement_vendor', 'vendor', 'x', 'validated', ?)", [batch_id, synthetic])


def make_root(tmp_path: Path, *, synthetic_manifest: bool = True, real_batch: bool = False,
              warehouse: bool = True) -> Path:
    root = tmp_path / "data"
    f = root / "raw" / "synthetic" / "movement_vendor" / "a.csv"
    f.parent.mkdir(parents=True)
    f.write_text("x\n", encoding="utf-8")
    Path(str(f) + ".manifest.json").write_text(json.dumps({"synthetic": synthetic_manifest}), encoding="utf-8")
    (root / "synthetic_truth").mkdir()
    (root / "synthetic_truth" / "truth.parquet").write_bytes(b"t")
    (root / "canonical" / "run1").mkdir(parents=True)
    (root / "canonical" / "run1" / "c.parquet").write_bytes(b"c")
    if warehouse:
        con = connect(root / "warehouse.duckdb")
        _batch(con, "b1", True)
        if real_batch:
            _batch(con, "b2", False)
        con.close()
    return root


def snapshot(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_removes_everything_the_demo_owns_and_keeps_the_root(tmp_path):
    root = make_root(tmp_path)
    clean_synthetic_root(root)
    assert root.exists() and snapshot(root) == []


def test_missing_root_is_a_noop(tmp_path):
    assert clean_synthetic_root(tmp_path / "nothing") == []


def test_refuses_unmanaged_file_and_deletes_nothing(tmp_path):
    root = make_root(tmp_path)
    (root / "partner_extract.xlsx").write_bytes(b"real")
    before = snapshot(root)
    with pytest.raises(UnsafeCleanup, match="partner_extract.xlsx"):
        clean_synthetic_root(root)
    assert snapshot(root) == before


def test_refuses_second_raw_source(tmp_path):
    root = make_root(tmp_path)
    (root / "raw" / "mutabakat").mkdir()
    (root / "raw" / "mutabakat" / "m.csv").write_text("real", encoding="utf-8")
    before = snapshot(root)
    with pytest.raises(UnsafeCleanup, match="mutabakat"):
        clean_synthetic_root(root)
    assert snapshot(root) == before


def test_refuses_synthetic_folder_file_without_synthetic_manifest(tmp_path):
    root = make_root(tmp_path, synthetic_manifest=False)
    before = snapshot(root)
    with pytest.raises(UnsafeCleanup, match="synthetic manifest"):
        clean_synthetic_root(root)
    assert snapshot(root) == before


def test_refuses_warehouse_with_real_batches(tmp_path):
    root = make_root(tmp_path, real_batch=True)
    before = snapshot(root)
    with pytest.raises(UnsafeCleanup, match="non-synthetic"):
        clean_synthetic_root(root)
    assert snapshot(root) == before


def test_refuses_derived_folders_without_a_warehouse_to_vouch_for_them(tmp_path):
    root = make_root(tmp_path, warehouse=False)
    before = snapshot(root)
    with pytest.raises(UnsafeCleanup, match="no warehouse"):
        clean_synthetic_root(root)
    assert snapshot(root) == before


def test_refuses_a_file_that_is_not_a_duckdb_warehouse(tmp_path):
    root = make_root(tmp_path, warehouse=False)
    (root / "canonical" / "run1" / "c.parquet").unlink()
    (root / "canonical" / "run1").rmdir()
    (root / "canonical").rmdir()
    (root / "warehouse.duckdb").write_bytes(b"not a database")
    with pytest.raises(UnsafeCleanup):
        clean_synthetic_root(root)
    assert (root / "warehouse.duckdb").read_bytes() == b"not a database"


def test_cli_demo_fresh_refuses_and_exits_2_without_deleting(tmp_path):
    root = make_root(tmp_path)
    keep = root / "real_gln_list.csv"
    keep.write_text("868...", encoding="utf-8")
    res = CliRunner().invoke(app, ["demo", "--fresh", "--config", CONFIG, "--data-root", str(root)])
    assert res.exit_code == 2
    assert keep.exists() and (root / "raw" / "synthetic").exists()
