""": real data needs G0 to ingest/reconcile, G0+G1 to score, G0+G1+G2 to backtest/tune."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pharmacy_potential.config import load_settings
from pharmacy_potential.db import connect
from pharmacy_potential.gates import GateBlocked, check_gates, require_gates
from pharmacy_potential.ingest.loader import ingest_file
from pharmacy_potential.ingest.manifest import build_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[2]


class Project:
    """A throw-away project root with its own gate files (the repository's gates are never touched)."""

    def __init__(self, tmp: Path, **gates: str):
        self.root = tmp / "proj"
        shutil.copytree(ROOT / "configs", self.root / "configs")
        (self.root / "docs" / "gates").mkdir(parents=True)
        for g in ("G0", "G1", "G2", "G3", "G4"):
            if gates.get(g) != "missing":
                (self.root / "docs" / "gates" / f"{g}.md").write_text(f"# {g}\n\nstatus: {gates.get(g, 'pending')}\n",
                                                                      encoding="utf-8")
        self.settings = load_settings(self.root / "configs" / "pilot.yaml", data_root=str(tmp / "data"))
        self.con = connect(self.settings.warehouse_path)

    def batch(self, *, synthetic: bool, owner="acme", license_id="lic-1", bid="b1"):
        self.con.execute("insert into ingestion_batch (batch_id, dataset, source, sha256, state, synthetic, owner, "
                         "license_id) values (?, 'movement_vendor', 'vendor', 'x', 'validated', ?, ?, ?)",
                         [bid, synthetic, owner, license_id])
        return self


@pytest.fixture
def mk(tmp_path):
    made = []

    def make(**gates):
        p = Project(tmp_path, **gates)
        made.append(p)
        return p
    yield make
    for p in made:
        p.con.close()


def test_synthetic_data_needs_no_gate(mk):
    p = mk().batch(synthetic=True)
    g = check_gates(p.settings, p.con, False)
    assert g["mode"] == "full" and g["tuning_allowed"]


@pytest.mark.parametrize("exploratory", [False, True])
def test_real_data_without_g0_is_blocked_even_when_exploratory(mk, exploratory):
    p = mk(G0="pending", G1="accepted", G2="accepted").batch(synthetic=False)
    with pytest.raises(GateBlocked, match="G0"):
        check_gates(p.settings, p.con, exploratory)


def test_missing_gate_file_is_not_acceptance(mk):
    p = mk(G0="missing").batch(synthetic=False)
    with pytest.raises(GateBlocked, match="G0"):
        check_gates(p.settings, p.con, True)


def test_unknown_rights_block_even_with_g0_and_exploratory(mk):
    p = mk(G0="accepted").batch(synthetic=False, owner="unknown")
    with pytest.raises(GateBlocked, match="owner/license"):
        check_gates(p.settings, p.con, True)


def test_g1_pending_blocks_scoring_and_exploratory_only_explores(mk):
    p = mk(G0="accepted", G1="pending", G2="pending").batch(synthetic=False)
    with pytest.raises(GateBlocked, match="G1"):
        check_gates(p.settings, p.con, False)
    g = check_gates(p.settings, p.con, True)
    assert g["mode"] == "exploration_only" and not g["tuning_allowed"] and g["exploratory"]


def test_g2_pending_forbids_backtest_and_tuning_but_allows_fixed_scoring(mk):
    p = mk(G0="accepted", G1="accepted", G2="pending").batch(synthetic=False)
    with pytest.raises(GateBlocked, match="G2"):
        check_gates(p.settings, p.con, False, skip_backtest=False)
    g = check_gates(p.settings, p.con, False, skip_backtest=True)
    assert g["mode"] == "untuned" and g["tuning_allowed"] is False


def test_all_three_gates_accepted_allow_the_full_run(mk):
    p = mk(G0="accepted", G1="accepted", G2="accepted").batch(synthetic=False)
    g = check_gates(p.settings, p.con, False)
    assert g["mode"] == "full" and g["tuning_allowed"]


def test_g2_rejected_is_not_accepted(mk):
    p = mk(G0="accepted", G1="accepted", G2="rejected").batch(synthetic=False)
    with pytest.raises(GateBlocked, match="G2"):
        check_gates(p.settings, p.con, False)


def test_ingest_of_a_real_batch_requires_g0_and_stages_nothing_when_blocked(mk, tmp_path):
    p = mk(G0="pending")
    src = p.settings.source_config("pharmacy_master")
    f = tmp_path / "pm.csv"
    f.write_text(";".join(s["source"] for s in src["columns"].values()) + "\n", encoding="utf-8")
    m = build_manifest("pharmacy_master", src["source_name"], f, period_start="2026-01-01", period_end="2026-01-31",
                       owner="acme", license_id="lic-1", synthetic=False)
    write_manifest(m, f)
    with pytest.raises(GateBlocked, match="G0"):
        ingest_file(p.settings, p.con, "pharmacy_master", f)
    assert p.con.execute("select count(*) from ingestion_batch").fetchone()[0] == 0


def test_ingest_of_a_real_batch_proceeds_once_g0_is_accepted(mk, tmp_path):
    p = mk(G0="accepted")
    src = p.settings.source_config("pharmacy_master")
    f = tmp_path / "pm.csv"
    f.write_text(";".join(s["source"] for s in src["columns"].values()) + "\n", encoding="utf-8")
    m = build_manifest("pharmacy_master", src["source_name"], f, period_start="2026-01-01", period_end="2026-01-31",
                       owner="acme", license_id="lic-1", synthetic=False)
    write_manifest(m, f)
    assert ingest_file(p.settings, p.con, "pharmacy_master", f).state in ("validated", "validated_with_quarantine")


def test_require_gates_names_every_missing_gate(mk):
    p = mk(G0="accepted", G1="pending", G2="pending")
    with pytest.raises(GateBlocked) as e:
        require_gates(p.settings, "backtest", "backtest")
    assert "G1" in str(e.value) and "G2" in str(e.value) and "G0" not in str(e.value).split("not accepted:")[1]
