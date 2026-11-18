""": validate_contracts never reports success without something to validate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pharmacy_potential.db import connect
from tests.conftest import G1, GUM
from tests.unit.test_reconcile_contracts import mut

ROOT = Path(__file__).resolve().parents[2]


def run_validator(fx, tmp_name="report.json"):
    fx.con.close()
    report = fx.tmp / tmp_name
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_contracts.py"),
                           "--data-root", str(fx.settings.data_root), "--report", str(report)],
                          capture_output=True, text=True, cwd=str(fx.tmp))
    return proc.returncode, (json.loads(report.read_text(encoding="utf-8")) if report.exists() else None)


def test_empty_warehouse_is_input_unavailable_not_a_pass(tmp_path):
    from tests.conftest import Fixture
    fx = Fixture(tmp_path)                       # warehouse exists, no batch at all
    code, report = run_validator(fx)
    assert code == 2 and report is None


def test_masters_only_passes_but_reports_zero_rows_for_undelivered_movement(fx):
    code, report = run_validator(fx)
    assert code == 0
    assert report["rows_checked"]["stg_movement_mutabakat.gln format"] == 0
    assert report["datasets_without_accepted_batch"] == ["movement_mutabakat", "movement_vendor"]
    assert not any(c["check"] == "required_missing_values_quarantined_not_zeroed" for c in report["checks"])


def test_blank_optional_measure_is_verified_against_the_raw_file(fx):
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "5", free="", ret="")])
    code, report = run_validator(fx)
    chk = next(c for c in report["checks"] if c["check"] == "blank_optional_measures_stay_null")
    assert code == 0 and chk["status"] == "pass" and chk["detail"]["columns_checked"] >= 2


def test_blank_turned_into_zero_is_detected(fx):
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "5", free="", ret="")])
    fx.con.execute("update stg_movement_mutabakat set units_free = 0")        # the failure being guarded against
    code, report = run_validator(fx)
    assert code == 1
    chk = next(c for c in report["checks"] if c["check"] == "blank_optional_measures_stay_null")
    assert chk["status"] == "fail" and chk["detail"]["failing"][0]["field"] == "units_free"


def test_missing_staging_table_for_an_accepted_batch_fails(fx):
    fx.add("movement_vendor", [{"PharmacyGLN": G1, "Month": "202601", "EAN": GUM, "Units": "3"}])
    fx.con.execute("drop table stg_movement_vendor")
    code, report = run_validator(fx)
    assert code == 1
    assert any(c["status"] == "fail" and "missing" in str(c["detail"]) for c in report["checks"])


def test_missing_master_batch_is_input_unavailable(tmp_path):
    from tests.conftest import Fixture
    fx = Fixture(tmp_path)
    fx.add("pharmacy_master", [{"GLN": G1, "EczaneAdi": "A", "IlKodu": "34", "IlceKodu": "34001"}],
           period=("2021-01-01", "2026-01-31"), available="2021-01-01T00:00:00Z")
    code, _ = run_validator(fx)
    assert code == 2
