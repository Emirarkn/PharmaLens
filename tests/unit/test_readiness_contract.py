"""and : readiness has one schema on every path; the depot
outflow comes from the depot code of the movement rows (no GLN total copied to depots); product scope reaches
stock and sell-in; sell-in is read as of the cutoff."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from pharmacy_potential.readiness.readiness import build_readiness, stock_reconciliation
from pharmacy_potential.reconcile.canonical import build_canonical
from tests.conftest import G1, G2, G3, GUM
from tests.unit.test_reconcile_contracts import mut

TH = {"readiness": {"max_stock_age_days": 45, "sellin_lookback_months": 3}}
BRAND = "DemoBrand"
OTHER = "8690000777001"
SCHEMA = ["gln", "readiness", "readiness_reason", "readiness_score", "age_days"]
FEB = dict(period=("2026-02-01", "2026-02-28"), available="2026-03-02T00:00:00Z")


def stock(snap, sub, units="100", barcode=GUM):
    return {"StokTarihi": snap, "AltDepoKodu": sub, "Barkod": barcode, "EldekiAdet": units}


def add_other_brand(fx):
    fx.add("product_master", [{"Barkod": OTHER, "Marka": "OtherBrand", "Form": "GUM", "BaslangicTarihi": "2019-01-01"}],
           period=("2021-01-01", "2026-01-31"), available="2021-01-02T00:00:00Z")


def test_k12_no_stock_table_still_returns_the_full_schema_with_unknown_score(fx):
    r = build_readiness(fx.con, None, "2026-01", TH, BRAND)
    assert list(r.columns) == SCHEMA
    assert (r["readiness"] == "unknown").all() and (r["readiness_score"] == 0.5).all()


def test_k12_stock_without_sellin_table_is_unknown_not_an_error(fx):
    fx.add("depot_stock", [stock("2026-01-31", "D01S1"), stock("2026-01-31", "D01S2")])
    r = build_readiness(fx.con, None, "2026-01", TH, BRAND)
    assert list(r.columns) == SCHEMA
    g1 = r.set_index("gln").loc[G1]
    assert g1["readiness"] == "unknown" and "sell-in" in g1["readiness_reason"] and g1["readiness_score"] == 0.5


def test_k12_sellin_is_read_as_of_the_cutoff(fx):
    fx.add("depot_stock", [stock("2026-01-31", "D01S1"), stock("2026-01-31", "D01S2")], available="2026-02-01T00:00:00Z")
    fx.add("sellin", [{"Ay": "2026-01", "DepoKodu": "D01", "Barkod": GUM, "GirisAdet": "50"}],
           available="2026-03-01T00:00:00Z")                               # delivered after the cutoff below
    early = build_readiness(fx.con, datetime(2026, 2, 15), "2026-01", TH, BRAND).set_index("gln")
    late = build_readiness(fx.con, datetime(2026, 3, 15), "2026-01", TH, BRAND).set_index("gln")
    assert early.loc[G1, "readiness"] != "high"                            # sell-in not known yet
    assert late.loc[G1, "readiness"] == "high"


# ----------------------------------------------------------------------------- B22: product scope

def test_b22_stock_and_sellin_of_another_brand_do_not_make_demo_brand_ready(fx):
    add_other_brand(fx)
    fx.add("depot_stock", [stock("2026-01-31", "D01S1", barcode=OTHER), stock("2026-01-31", "D01S2", barcode=OTHER)])
    fx.add("sellin", [{"Ay": "2026-01", "DepoKodu": "D01", "Barkod": OTHER, "GirisAdet": "500"}])
    r = build_readiness(fx.con, None, "2026-01", TH, BRAND).set_index("gln")
    assert r.loc[G1, "readiness"] == "unknown" and "no stock snapshot" in r.loc[G1, "readiness_reason"]


def test_b22_demo_brand_evidence_still_counts_when_another_brand_is_present(fx):
    add_other_brand(fx)
    fx.add("depot_stock", [stock("2026-01-31", "D01S1"), stock("2026-01-31", "D01S2"),
                           stock("2026-01-31", "D01S1", barcode=OTHER)])
    fx.add("sellin", [{"Ay": "2026-01", "DepoKodu": "D01", "Barkod": GUM, "GirisAdet": "50"}])
    assert build_readiness(fx.con, None, "2026-01", TH, BRAND).set_index("gln").loc[G1, "readiness"] == "high"


# ----------------------------------------------------------------------------- K12-v2: depot-grain outflow

def balance_fixture(fx, d01_feb_coverage="COMPLETE"):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"}, month="2026-01")
    fx.coverage({"D01": d01_feb_coverage, "D02": "COMPLETE"}, month="2026-02", name="cov2.csv",
                period=("2026-02-01", "2026-02-28"))
    fx.add("depot_stock", [stock("2026-01-31", "D01S1", "60"), stock("2026-01-31", "D01S2", "40"),
                           stock("2026-02-28", "D01S1", "50"), stock("2026-02-28", "D01S2", "28")],
           available="2026-03-01T00:00:00Z", period=("2026-01-01", "2026-02-28"))
    fx.add("sellin", [{"Ay": "2026-02", "DepoKodu": "D01", "Barkod": GUM, "GirisAdet": "0"}],
           period=("2026-02-01", "2026-02-28"), available="2026-03-01T00:00:00Z")


def balance(fx, brand=BRAND):
    canon = build_canonical(fx.con, fx.settings, None, "2026-02", "bal")
    build_readiness(fx.con, None, "2026-02", TH, brand)
    df = stock_reconciliation(fx.con, canon.table, None, brand)
    return df[(df.parent == "D01") & (df.month == "2026-02")].iloc[0]


def outflow(fx, rows):
    fx.add("movement_mutabakat", rows, name="m2.csv", **FEB)


def test_k12v2_outflow_comes_from_the_depot_code_so_a_two_supplier_pharmacy_is_never_double_counted(fx):
    """Regression reproducer: G2 buys from D01 and D02. Its D01 rows are D01's outflow; its D02 rows are D02's."""
    balance_fixture(fx)
    outflow(fx, [mut(G1, GUM, "D01S1", "10", month="2026-02"),            # sub depot of D01
                 mut(G2, GUM, "D01", "12", month="2026-02"),
                 mut(G2, GUM, "D02", "5", month="2026-02")])
    d01 = balance(fx)
    assert d01["status"] == "comparable" and d01["out_units"] == 22      # 10 + 12; G2's D02 box is not on D01
    assert d01["residual"] == 100 + 0 - 22 - 78


def test_k12v2_conflicting_duplicate_rows_are_not_summed(fx):
    balance_fixture(fx)
    outflow(fx, [mut(G1, GUM, "D01S1", "10", month="2026-02"), mut(G1, GUM, "D01S1", "14", month="2026-02")])
    d01 = balance(fx)
    assert d01["status"] == "skipped_with_reason" and "conflicting" in d01["reason"]


def test_k12v2_a_rejected_row_that_could_belong_to_the_depot_makes_the_balance_incomparable(fx):
    balance_fixture(fx)
    outflow(fx, [mut(G1, GUM, "D01", "10", month="2026-02"), mut("868", GUM, "D01", "7", month="2026-02")])
    d01 = balance(fx)
    assert d01["status"] == "skipped_with_reason" and "rejected" in d01["reason"]


def test_k12v2_outflow_coverage_must_be_complete(fx):
    balance_fixture(fx, d01_feb_coverage="PARTIAL")
    outflow(fx, [mut(G1, GUM, "D01", "10", month="2026-02")])
    d01 = balance(fx)
    assert d01["status"] == "skipped_with_reason" and "coverage" in d01["reason"]


def test_k12v2_another_brand_moves_neither_stock_sellin_nor_outflow_of_the_balance(fx):
    add_other_brand(fx)
    balance_fixture(fx)
    outflow(fx, [mut(G1, GUM, "D01", "10", month="2026-02")])
    before = balance(fx)
    fx.add("depot_stock", [stock("2026-01-31", "D01S1", "999", OTHER), stock("2026-02-28", "D01S1", "5", OTHER),
                           stock("2026-01-31", "D01S2", "999", OTHER), stock("2026-02-28", "D01S2", "5", OTHER)],
           available="2026-03-01T00:00:00Z", period=("2026-01-01", "2026-02-28"), name="stock_other.csv")
    fx.add("sellin", [{"Ay": "2026-02", "DepoKodu": "D01", "Barkod": OTHER, "GirisAdet": "777"}],
           period=("2026-02-01", "2026-02-28"), available="2026-03-01T00:00:00Z", name="sellin_other.csv")
    fx.add("movement_mutabakat", [mut(G1, OTHER, "D01", "300", month="2026-02")], name="m_other.csv", **FEB)
    after = balance(fx)
    for col in ("closing", "opening", "received", "out_units", "residual", "status"):
        assert before[col] == after[col], col
