"""acceptance tests 1-8: a zero is produced only from cell-level evidence.

A rejected row limits "no record -> observed zero" for exactly the cells it could belong to, narrowed by the fields
it still shows; recorded cells stay observed; batch quality (a threshold) never changes an observation status."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmacy_potential.config import Settings
from pharmacy_potential.reconcile.canonical import build_canonical, rejection_reports
from tests.conftest import G1, G2, G3, GUM, PATCH
from tests.unit.test_reconcile_contracts import cell, vendor, mut

OTHER = "8690000777001"
UNKNOWN_BC = "8690000999999"
BAD_GLN = "868"                                   # fails the 13-digit rule -> rejected, GLN unresolvable
TWO_MONTHS = ("2026-01-01", "2026-02-28")


def canon(fx, origin="2026-01", cutoff=None, tag="z"):
    return build_canonical(fx.con, fx.settings, cutoff, origin, tag)


def status(fx, res, gln, barcode, month="2026-01-01"):
    return cell(fx, res.table, gln, barcode, month).observation_status


def setup(fx, rows, extra_cov=False, **kw):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    if extra_cov:
        fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"}, month="2026-02", name="cov2.csv", period=("2026-02-01", "2026-02-28"))
    return fx.add("movement_mutabakat", rows, **kw)


def with_policy(monkeypatch, **extra):
    base = Settings.policy.fget
    monkeypatch.setattr(Settings, "policy", property(lambda self: {**base(self), **extra}))


# ----------------------------------------------------------------------------- 1, 2: the counter-example

def test_z1_one_unresolvable_positive_row_below_any_threshold_produces_no_zero_for_the_cells_it_could_belong_to(fx):
    """1.000 accepted rows + 1 unresolvable positive row used to keep 'COMPLETE' and invent zeros."""
    setup(fx, [mut(G3, GUM, "D02", "2"), mut(BAD_GLN, GUM, "D01", "12")])
    res = canon(fx)
    # Cells that could be the owner of the rejected row: D01-linked pharmacies x Jan x GUM, no record.
    assert status(fx, res, G1, GUM) == "unobserved" and np.isnan(cell(fx, res.table, G1, GUM).net_units)
    assert status(fx, res, G2, GUM) == "unobserved"
    assert cell(fx, res.table, G1, GUM).selection_reason == "absence_zero_blocked_by_rejection"
    # Narrowed: other barcode of the same pharmacy, and pharmacies not served by D01, keep their evidence-based zero.
    assert status(fx, res, G1, PATCH) == "observed_zero"
    assert status(fx, res, G3, GUM) == "observed_positive" and status(fx, res, G3, PATCH) == "observed_zero"
    assert res.report["absence_zero_blocked_cells"] == 2


def test_z1_recorded_cells_stay_observed_in_the_same_situation(fx):
    setup(fx, [mut(G1, GUM, "D01", "10"), mut(BAD_GLN, GUM, "D01", "12")])
    res = canon(fx)
    c = cell(fx, res.table, G1, GUM)
    assert c.observation_status == "observed_positive" and c.net_units == 10 and c.chosen_source == "mutabakat"
    assert status(fx, res, G2, GUM) == "unobserved"                                  # G2 has no record there


def test_z1_an_explicit_zero_row_is_a_record_and_stays_observed(fx):
    setup(fx, [mut(G1, GUM, "D01", "0"), mut(BAD_GLN, GUM, "D01", "12")])
    assert status(fx, canon(fx), G1, GUM) == "observed_zero"


def test_z1_blocked_cell_falls_back_to_vendor_when_there_is_a_record(fx):
    setup(fx, [mut(BAD_GLN, GUM, "D01", "12")])
    fx.add("movement_vendor", [vendor(G1, GUM, "9")])
    c = cell(fx, canon(fx).table, G1, GUM)
    assert c.chosen_source == "vendor" and c.net_units == 9
    assert c.selection_reason == "fallback_absence_zero_blocked_by_rejection"


# ----------------------------------------------------------------------------- 3: one test per Z2 row

def test_z2_row1_fully_keyed_row_blocks_only_its_own_cell(fx):
    setup(fx, [mut(G1, GUM, "D01", "not-a-number")])
    res = canon(fx)
    assert status(fx, res, G1, GUM) == "quarantined"
    assert status(fx, res, G1, PATCH) == "observed_zero" and status(fx, res, G2, GUM) == "observed_zero"
    assert res.report["absence_zero_blocked_cells"] == 0                              # the cell is quarantined, not 'blocked'


@pytest.mark.parametrize("barcode", ["12ab", UNKNOWN_BC])
def test_z2_row2_gln_and_month_known_barcode_unknown_blocks_the_pharmacy_month_for_every_barcode(fx, barcode):
    setup(fx, [mut(G1, barcode, "D01", "5")])
    res = canon(fx)
    assert status(fx, res, G1, GUM) == "unobserved" and status(fx, res, G1, PATCH) == "unobserved"
    assert status(fx, res, G2, GUM) == "observed_zero" and status(fx, res, G3, GUM) == "observed_zero"


def test_z2_row3_gln_and_barcode_known_month_unknown_blocks_that_pharmacy_barcode_for_the_batch_period(fx):
    fx.settings.overrides["history_start_month"] = "2026-01"
    setup(fx, [mut(G1, GUM, "D01", "5", month="2026-13")], extra_cov=True, period=TWO_MONTHS)
    res = canon(fx, origin="2026-02")
    for m in ("2026-01-01", "2026-02-01"):
        assert status(fx, res, G1, GUM, m) == "unobserved"
        assert status(fx, res, G1, PATCH, m) == "observed_zero" and status(fx, res, G2, GUM, m) == "observed_zero"


def test_z2_row4_gln_unknown_blocks_the_depot_linked_pharmacies_for_that_month_and_barcode(fx):
    setup(fx, [mut(BAD_GLN, GUM, "D02", "5")])
    res = canon(fx)
    assert status(fx, res, G2, GUM) == "unobserved" and status(fx, res, G3, GUM) == "unobserved"   # both linked to D02
    assert status(fx, res, G1, GUM) == "observed_zero" and status(fx, res, G2, PATCH) == "observed_zero"


def test_z2_a_sub_depot_code_is_resolved_to_its_parent_for_the_pharmacy_link(fx):
    setup(fx, [mut(BAD_GLN, GUM, "D01S1", "5")])                      # D01S1 belongs to parent D01
    res = canon(fx)
    assert status(fx, res, G1, GUM) == "unobserved" and status(fx, res, G3, GUM) == "observed_zero"


def test_z2_row5_only_the_depot_known_blocks_that_depots_pharmacies_for_every_barcode_and_the_batch_period(fx):
    setup(fx, [mut(BAD_GLN, "12ab", "D02", "5", month="2026-13")], extra_cov=True, period=TWO_MONTHS)
    res = canon(fx, origin="2026-02")
    for m in ("2026-01-01", "2026-02-01"):
        for g in (G2, G3):
            assert status(fx, res, g, GUM, m) == "unobserved" and status(fx, res, g, PATCH, m) == "unobserved"
        assert status(fx, res, G1, GUM, m) == "observed_zero"


def test_z2_nothing_known_blocks_every_pharmacy_for_the_batch_period(fx):
    setup(fx, [mut(BAD_GLN, "12ab", "", "5", month="2026-13")])
    res = canon(fx)
    for g in (G1, G2, G3):
        assert status(fx, res, g, GUM) == "unobserved"


def test_z2_rejected_row_of_an_unknown_depot_cannot_be_linked_so_it_blocks_everyone(fx):
    setup(fx, [mut(BAD_GLN, GUM, "D99", "5")])
    res = canon(fx)
    assert all(status(fx, res, g, GUM) == "unobserved" for g in (G1, G2, G3))
    assert all(status(fx, res, g, PATCH) == "observed_zero" for g in (G1, G2, G3))      # barcode known -> narrowed


# ----------------------------------------------------------------------------- 4: product scope first

def test_b21_rejected_row_of_another_brand_does_not_touch_demo_brand_cells(fx):
    fx.add("product_master", [{"Barkod": OTHER, "Marka": "OtherBrand", "Form": "GUM", "BaslangicTarihi": "2019-01-01"}],
           period=("2021-01-01", "2026-01-31"), available="2021-01-02T00:00:00Z")
    setup(fx, [mut(BAD_GLN, OTHER, "D01", "12"), mut(G1, OTHER, "D01", "not-a-number"), mut("868", OTHER, "", "1", month="2026-13")])
    res = canon(fx)
    for g in (G1, G2, G3):
        assert status(fx, res, g, GUM) == "observed_zero" and status(fx, res, g, PATCH) == "observed_zero"
    assert res.report["absence_zero_blocked_cells"] == 0
    # ... while a row whose barcode is NOT known (it might be a DemoBrand product missing from the master) still blocks.
    fx.add("movement_mutabakat", [mut(BAD_GLN, UNKNOWN_BC, "D01", "12")], name="extra.csv", available="2026-02-16T00:00:00Z")
    assert status(fx, canon(fx), G1, GUM) == "unobserved"


# ----------------------------------------------------------------------------- 5: threshold = quality only

def test_z3_the_threshold_changes_only_the_quality_label_never_an_observation_status(fx, monkeypatch):
    setup(fx, [mut(G3, GUM, "D02", "2"), mut(BAD_GLN, GUM, "D01", "12")])
    res_default = canon(fx, tag="q1")
    q1 = rejection_reports(fx.con, res_default, fx.settings)["quarantine_quality"]
    with_policy(monkeypatch, batch_quality={"tau_rel": 0.0, "tau_abs": 0})
    res_tight = canon(fx, tag="q2")
    q2 = rejection_reports(fx.con, res_tight, fx.settings)["quarantine_quality"]
    mut_row = lambda q: q[q.dataset == "movement_mutabakat"].iloc[0]
    assert mut_row(q1).quality == "acceptable_with_rejections" and mut_row(q2).quality == "review_required"
    assert mut_row(q1).unresolved_rows == 1 and mut_row(q1).threshold == 5
    pd.testing.assert_frame_equal(
        fx.con.execute(f"select * from {res_default.table} order by gln, month, barcode").df(),
        fx.con.execute(f"select * from {res_tight.table} order by gln, month, barcode").df())


# ----------------------------------------------------------------------------- 6: a corrected revision brings the zeros back

def test_z4_corrected_revision_restores_the_zeros_from_its_own_availability_time(fx):
    first = setup(fx, [mut(G3, GUM, "D02", "2"), mut(BAD_GLN, GUM, "D01", "12")])
    fx.add("movement_mutabakat", [mut(G3, GUM, "D02", "2"), mut(G1, GUM, "D01", "12")], revision=2,
           supersedes=first.batch_id, available="2026-03-01T00:00:00Z", name="fixed.csv")
    before = canon(fx, cutoff=datetime(2026, 2, 20), tag="r1")
    after = canon(fx, cutoff=datetime(2026, 3, 2), tag="r2")
    assert status(fx, before, G2, GUM) == "unobserved"                                  # still blocked at the earlier cutoff
    assert status(fx, after, G2, GUM) == "observed_zero"                                # the corrected delivery is known
    assert status(fx, after, G1, GUM) == "observed_positive"


# ----------------------------------------------------------------------------- 7: Z5 / Z6 closed

def test_z5_z6_are_closed_by_default(fx):
    pol = fx.settings.policy
    assert pol["uncertainty_bounds_as_model_input"] is False and pol["gln_repair_leading_zero"] is False
    assert pol["absence_zero_evidence"] == "z2_narrowed"


def test_z5_bounds_are_reported_but_change_no_status(fx):
    setup(fx, [mut(G1, GUM, "D01", "10"), mut(BAD_GLN, GUM, "D01", "12"), mut(BAD_GLN, PATCH, "D01", "abc")])
    res = canon(fx)
    snap = fx.con.execute(f"select * from {res.table} order by gln, month, barcode").df()
    b = rejection_reports(fx.con, res, fx.settings)["coverage_bounds"].set_index("gln")
    pd.testing.assert_frame_equal(fx.con.execute(f"select * from {res.table} order by gln, month, barcode").df(), snap)
    assert b.loc[G1, "lower_units"] == 10 and b.loc[G1, "n_possible_rows"] == 2
    assert pd.isna(b.loc[G1, "upper_units"]) and b.loc[G1, "unreadable_rows"] == 1      # 'abc' cannot bound anything


def test_z5_bound_adds_the_readable_units_of_every_row_that_could_belong(fx):
    setup(fx, [mut(G1, GUM, "D01", "10"), mut(BAD_GLN, GUM, "D01", "12", "3")])
    b = rejection_reports(fx.con, canon(fx), fx.settings)["coverage_bounds"].set_index("gln")
    assert b.loc[G1, "upper_units"] == 10 + 15 and b.loc[G2, "lower_units"] == 0 and b.loc[G2, "upper_units"] == 15
    assert G3 not in b.index


def test_zero_blocking_report_lists_each_rejected_row_with_its_narrowing(fx):
    setup(fx, [mut(BAD_GLN, GUM, "D01", "12"), mut(G1, UNKNOWN_BC, "D01", "5")])
    zb = rejection_reports(fx.con, canon(fx), fx.settings)["zero_blocking"]
    assert set(zb.rule_type) == {"gln_unknown_depot_month_barcode", "gln_month_barcode_unknown"}
    row = zb[zb.rule_type == "gln_unknown_depot_month_barcode"].iloc[0]
    assert row.error_code == "INVALID_GLN" and row.pharmacy_months == 2 and row.cells_potentially_affected == 2
    assert zb.cells_blocked.sum() == 4                                                  # G1,G2 x GUM + G1 x GUM,PATCH overlap


# ----------------------------------------------------------------------------- sensitivity mode (superseded)

def test_strict_k7_v1_remains_available_as_a_sensitivity_mode_only(fx, monkeypatch):
    with_policy(monkeypatch, absence_zero_evidence="strict_k7_v1")
    setup(fx, [mut(G1, GUM, "D01", "10"), mut(BAD_GLN, GUM, "D01", "12")])
    res = canon(fx)
    assert status(fx, res, G1, GUM) == "unobserved"                    # the whole depot-month left the primary source
    assert res.report["zero_evidence_mode"] == "strict_k7_v1"
    assert res.report["quarantine_impact"]["movement_mutabakat"]["depot_months_downgraded"] == 1


def test_unknown_evidence_mode_is_refused(fx, monkeypatch):
    with_policy(monkeypatch, absence_zero_evidence="whatever")
    setup(fx, [mut(G1, GUM, "D01", "1")])
    with pytest.raises(ValueError, match="absence_zero_evidence"):
        canon(fx)
    src = (Path(__file__).resolve().parents[2] / "src" / "pharmacy_potential" / "reconcile" / "canonical.py").read_text(encoding="utf-8")
    assert "unpinnable_rejected_rows" not in src            # no policy key that does not name
