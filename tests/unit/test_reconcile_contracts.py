"""S01-S06, S11: ingestion, dedup, source selection and observation status invariants."""
from __future__ import annotations

from datetime import datetime

import pytest

from pharmacy_potential.reconcile.canonical import build_canonical
from tests.conftest import G1, G2, G3, GUM, PATCH


def mut(gln, barcode, depot, paid, free="0", ret="0", value="", month="2026-01"):
    return {"GLN": gln, "Donem": month, "Barkod": barcode, "DepoKodu": depot, "SatisAdet": paid, "MFAdet": free,
            "IadeAdet": ret, "NetTutarKDVDahil": value}


def vendor(gln, barcode, units, month="202601"):
    return {"PharmacyGLN": gln, "Month": month, "EAN": barcode, "Units": units}


def cell(fx, table, gln, barcode, month="2026-01-01"):
    return fx.con.execute(f"select * from {table} where gln = ? and barcode = ? and month = ?",
                          [gln, barcode, month]).df().iloc[0]


def canon(fx, cutoff=None, mode="default"):
    return build_canonical(fx.con, fx.settings, cutoff, "2026-01", "t", mode=mode)


def test_s01_primary_selected_never_summed(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    fx.add("movement_vendor", [vendor(G1, GUM, "12")])
    res = canon(fx)
    c = cell(fx, res.table, G1, GUM)
    assert c.net_units == 10 and c.chosen_source == "mutabakat"
    assert c.alt_vendor_net_units == 12          # alternative kept for the difference report
    assert c.net_units != 22
    assert res.report["sources_not_summed"]


def test_s02_fallback_only_when_primary_ineligible(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "UNKNOWN"})
    fx.add("movement_vendor", [vendor(G3, GUM, "12")])
    res = canon(fx)
    c = cell(fx, res.table, G3, GUM)
    assert c.chosen_source == "vendor" and c.net_units == 12
    assert c.selection_reason.startswith("fallback_primary_ineligible")
    c2 = cell(fx, res.table, G3, PATCH)       # neither source observed
    assert c2.observation_status == "unobserved" and c2.net_units != c2.net_units  # NaN


def test_s03_zero_only_with_complete_coverage(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "UNKNOWN"})
    fx.add("movement_mutabakat", [mut(G1, PATCH, "D01", "0")])   # explicit zero row
    res = canon(fx)
    assert cell(fx, res.table, G1, PATCH).observation_status == "observed_zero"
    assert cell(fx, res.table, G1, GUM).observation_status == "observed_zero"      # no row, coverage complete
    assert cell(fx, res.table, G3, PATCH).observation_status == "unobserved"       # no row, coverage unknown
    # Same "no record" situation, different evidence -> different model input.
    assert cell(fx, res.table, G1, GUM).net_units == 0
    assert cell(fx, res.table, G3, PATCH).net_units != cell(fx, res.table, G3, PATCH).net_units


def test_s04_idempotent_reload_and_explicit_revision(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    rows = [mut(G1, GUM, "D01", "5")]
    r1 = fx.add("movement_mutabakat", rows, name="m.csv")
    again = fx.add("movement_mutabakat", rows, name="m_copy.csv")   # same bytes, other name
    assert again.state == "duplicate"
    n = fx.con.execute("select count(*) from stg_movement_mutabakat").fetchone()[0]
    assert n == 1
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "7")], name="m_r02.csv", revision=2,
           supersedes=r1.batch_id, available="2026-02-25T00:00:00Z")
    assert cell(fx, canon(fx).table, G1, GUM).net_units == 7
    # As of a cutoff before the revision arrived, the original value is used (no silent overwrite).
    before = canon(fx, cutoff=datetime(2026, 2, 20))
    assert cell(fx, before.table, G1, GUM).net_units == 5
    states = dict(fx.con.execute("select batch_id, state from ingestion_batch").fetchall())
    assert states[r1.batch_id] == "validated"   # original kept in the archive


def test_s05_conflicting_duplicates_quarantined_depot_parts_aggregated(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [
        mut(G1, GUM, "D01", "4"), mut(G1, GUM, "D01", "6"),           # conflicting repeat of one key
        mut(G2, GUM, "D01", "4"), mut(G2, GUM, "D02", "3"),           # proven depot sub-grain
        mut(G3, GUM, "D02", "2"), mut(G3, GUM, "D02", "2"),           # identical repeat
    ])
    res = canon(fx)
    assert cell(fx, res.table, G1, GUM).observation_status == "quarantined"
    assert cell(fx, res.table, G2, GUM).net_units == 7
    assert cell(fx, res.table, G3, GUM).net_units == 2
    d = res.report["mutabakat_dedup"]
    assert d["identical_repeats_removed"] == 1 and d["conflicting_keys"] == 1


def test_s06_units_contract_net_paid_free_returns(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10", "2", "1", "1234,50")])
    c = cell(fx, canon(fx).table, G1, GUM)
    assert c.net_units == 11
    assert float(c.net_value_vat_inc_try) == pytest.approx(1234.50)   # decimal comma parsed, VAT not re-added


def test_s11_identifiers_and_quarantine(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    r = fx.add("movement_mutabakat", [
        mut(G1, GUM, "D01", "3"),
        mut(G1[1:], GUM, "D01", "3"),                 # leading zero lost (12 digits)
        mut("8.68E+12", GUM, "D01", "3"),             # Excel scientific conversion
        mut(G2, "8690000999999", "D01", "3"),         # unknown barcode
        mut(G2, GUM, "D01", "3", month="2026-13"),    # invalid month
        mut(G2, GUM, "D01", "-2"),                    # negative paid units
    ])
    assert r.rows_read == r.rows_accepted + r.rows_quarantined == 6
    assert r.rows_accepted == 1
    codes = dict(fx.con.execute("select error_code, count(*) from quarantine group by 1").fetchall())
    assert codes["INVALID_GLN"] == 1 and codes["INVALID_GLN_NUMERIC_CONVERSION"] == 1
    assert codes["FK_NOT_FOUND"] == 1 and codes["INVALID_MONTH"] == 1 and codes["NEGATIVE_VALUE"] == 1
    res = canon(fx)
    assert res.report["pk_unique"] and res.report["fk_orphans_gln"] == 0 and res.report["invalid_month_rows"] == 0
    assert fx.con.execute(f"select gln from {res.table} where gln like '0%' limit 1").fetchone()[0] == G1


def test_missing_required_column_rejects_batch_atomically(fx):
    import csv
    p = fx.tmp / "bad.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows([["GLN", "Donem"], [G1, "2026-01"]])
    from pharmacy_potential.ingest.loader import ingest_file
    from pharmacy_potential.ingest.manifest import build_manifest
    m = build_manifest("movement_mutabakat", "mutabakat", p, period_start="2026-01-01", period_end="2026-01-31",
                       synthetic=True, confidentiality="synthetic")
    r = ingest_file(fx.settings, fx.con, "movement_mutabakat", p, m)
    assert r.state == "rejected"
    assert not fx.con.execute("select count(*) from information_schema.tables "
                              "where table_name = 'stg_movement_mutabakat'").fetchone()[0] or \
        fx.con.execute("select count(*) from stg_movement_mutabakat").fetchone()[0] == 0
