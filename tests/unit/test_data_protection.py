""": brand scope and quarantine never become observed zeros.

The unresolvable-key rule that downgraded a whole depot-month was replaced; see test_zero_evidence.py."""
from __future__ import annotations

import numpy as np
import pytest

from pharmacy_potential.reconcile.canonical import build_canonical, gln_month_totals
from tests.conftest import G1, G2, G3, GUM, PATCH
from tests.unit.test_reconcile_contracts import cell, vendor, mut

OTHER = "8690000777001"


def canon(fx, cutoff=None, mode="default"):
    return build_canonical(fx.con, fx.settings, cutoff, "2026-01", "t", mode=mode)


def add_other_brand(fx):
    fx.add("product_master", [{"Barkod": OTHER, "Marka": "OtherBrand", "Form": "GUM", "BaslangicTarihi": "2019-01-01"}],
           period=("2021-01-01", "2026-01-31"), available="2021-01-02T00:00:00Z")


# ----------------------------------------------------------------------------- K6

def test_k6_other_brand_movement_does_not_change_demo_brand_totals(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10"), mut(G2, PATCH, "D02", "4")])
    before = gln_month_totals(fx.con, canon(fx).table).sort_values(["gln", "month"]).reset_index(drop=True)

    add_other_brand(fx)
    fx.add("movement_mutabakat", [mut(G1, OTHER, "D01", "100"), mut(G3, OTHER, "D02", "50")],
           available="2026-02-16T00:00:00Z")
    fx.add("movement_vendor", [vendor(G1, OTHER, "90")], available="2026-02-16T00:00:00Z")
    res = canon(fx)
    after = gln_month_totals(fx.con, res.table).sort_values(["gln", "month"]).reset_index(drop=True)

    assert before.equals(after)
    assert fx.con.execute(f"select count(*) from {res.table} where barcode = ?", [OTHER]).fetchone()[0] == 0
    assert res.report["out_of_brand_scope_rows"] == {"movement_mutabakat": 2, "movement_vendor": 1}


def test_k6_brand_match_is_case_insensitive(fx):
    fx.settings.overrides["brand"] = "DEMOBRAND"
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    assert cell(fx, canon(fx).table, G1, GUM).net_units == 10


# ----------------------------------------------------------------------------- K7

def test_k7_quarantined_cell_is_not_an_observed_zero(fx):
    """Regression reproducer: SatisAdet=not-a-number under COMPLETE coverage."""
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    r = fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "not-a-number")])
    assert r.rows_quarantined == 1
    res = canon(fx)
    c = cell(fx, res.table, G1, GUM)
    assert c.observation_status == "quarantined"
    assert np.isnan(c.net_units) and c.chosen_source is None
    # An untouched cell in the same pharmacy-month keeps its evidence-based zero.
    assert cell(fx, res.table, G1, PATCH).observation_status == "observed_zero"
    # Totals must not treat the quarantined cell as zero.
    tot = gln_month_totals(fx.con, res.table)
    assert np.isnan(tot[tot.gln == G1].iloc[0].units)


def test_k7_quarantined_cell_falls_back_to_vendor_when_available(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "not-a-number")])
    fx.add("movement_vendor", [vendor(G1, GUM, "12")])
    c = cell(fx, canon(fx).table, G1, GUM)
    assert c.chosen_source == "vendor" and c.net_units == 12
    assert c.selection_reason == "fallback_primary_quarantined_rows"


def test_k7_valid_sibling_row_does_not_hide_a_quarantined_row(fx):
    """One depot's row is fine, the other depot's row is quarantined -> the cell is incomplete."""
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G2, GUM, "D01", "4"), mut(G2, GUM, "D02", "-3")])
    c = cell(fx, canon(fx).table, G2, GUM)
    assert c.observation_status == "quarantined" and np.isnan(c.net_units)








def test_k7_quarantine_is_cutoff_aware(fx):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "not-a-number")], available="2026-03-01T00:00:00Z")
    from datetime import datetime
    early = cell(fx, canon(fx, cutoff=datetime(2026, 2, 20)).table, G1, GUM)
    assert early.observation_status == "observed_zero"      # the bad delivery was not known yet
    late = cell(fx, canon(fx, cutoff=datetime(2026, 3, 2)).table, G1, GUM)
    assert late.observation_status == "quarantined"
