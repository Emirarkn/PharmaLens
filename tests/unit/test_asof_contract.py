"""K2, K3: knowledge time per training origin, missing != 0, gross target."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pharmacy_potential.features.asof import AsOfPanels
from pharmacy_potential.features.build import build_panel, features_for_origin, labels_for_origin, training_frame
from pharmacy_potential.reconcile.canonical import build_canonical
from tests.conftest import G1, G2, G3, GUM, PATCH
from tests.unit.test_reconcile_contracts import mut

COVER = {"D01": "COMPLETE", "D02": "COMPLETE"}
FEB20, APR20 = datetime(2026, 2, 20), datetime(2026, 4, 20)


def feats(fx, asof, origin, cutoff):
    p = asof.panel(origin, cutoff)
    return features_for_origin(p, origin, cutoff, fx.settings.thresholds, fx.settings.scenario).set_index("gln")


def visit(gln, day="2026-01-10", rec="2026-01-12", n="V1"):
    return {"ZiyaretID": n, "GLN": gln, "ZiyaretTarihi": day, "KayitTarihi": rec, "Kanal": "FIELD", "Sonuc": "INFO"}


def order(gln, units="4", day="2026-01-10", rec="2026-01-12", n="O1"):
    return {"SiparisNo": n, "SatirNo": "1", "GLN": gln, "Barkod": GUM, "SiparisAdet": units, "IptalAdet": "0",
            "SiparisTarihi": day, "KayitTarihi": rec, "Durum": "COMPLETED"}


# ----------------------------------------------------------------------------- K1

def test_k1_month_delivered_after_the_cutoff_does_not_reach_an_earlier_origin(fx):
    """Regression reproducer: January arrives in April; an origin whose cutoff is 20 Feb must not see it."""
    fx.coverage(COVER, available="2026-04-01T00:00:00Z")
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10"), mut(G1, PATCH, "D01", "5")], available="2026-04-01T00:00:00Z")
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    assert np.isnan(feats(fx, asof, "2026-01", FEB20).loc[G1, "lag0"])
    assert feats(fx, asof, "2026-01", APR20).loc[G1, "lag0"] == 15      # the same history, once it is known
    # What the pre-K1 training path did: one panel built late, features cut only by month -> the leak.
    late = build_panel(fx.con, build_canonical(fx.con, fx.settings, APR20, "2026-01", "late").table, APR20)
    leaky = features_for_origin(late, "2026-01", FEB20, fx.settings.thresholds, fx.settings.scenario).set_index("gln")
    assert leaky.loc[G1, "lag0"] == 15


def test_k1_revision_after_the_cutoff_does_not_change_an_earlier_origin(fx):
    fx.coverage(COVER)
    r1 = fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "15")], revision=2, supersedes=r1.batch_id,
           available="2026-04-01T00:00:00Z", name="rev2.csv")
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    assert feats(fx, asof, "2026-01", FEB20).loc[G1, "lag0"] == 10
    assert feats(fx, asof, "2026-01", APR20).loc[G1, "lag0"] == 15


def test_k1_training_frame_uses_a_panel_as_of_each_origins_own_cutoff(fx):
    fx.settings.overrides["history_start_month"] = "2025-10"
    fx.coverage(COVER, available="2026-04-01T00:00:00Z")
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10"), mut(G1, PATCH, "D01", "5")], available="2026-04-01T00:00:00Z")
    # Matured label panel: everything known by 20 Apr (months through April).
    canon = build_canonical(fx.con, fx.settings, APR20, "2026-04", "lbl")
    label_panel = build_panel(fx.con, canon.table, APR20)
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    # backtest_cutoff(2026-01, 50) = 2026-03-22: the January delivery (1 April) is still unknown.
    tr = training_frame(asof.panel, label_panel, [pd.Timestamp("2026-01-01")], fx.settings.thresholds,
                        fx.settings.scenario, 50)
    assert np.isnan(tr.set_index("gln").loc[G1, "lag0"])
    # ... while the label of that origin does come from the matured panel.
    lab = labels_for_origin(label_panel, "2026-01", fx.settings.thresholds)
    assert lab.set_index("gln").loc[G1, "label_status"] in ("eligible", "censored_base", "censored_future")


def test_k1_panel_cache_is_keyed_on_cutoff_data_and_code(fx, tmp_path):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    a = AsOfPanels(fx.con, fx.settings)
    a.panel("2026-01", FEB20)
    b = AsOfPanels(fx.con, fx.settings)
    b.panel("2026-01", FEB20)                                  # same cutoff + data + code: served from cache
    assert (a.builds, a.hits, b.builds, b.hits) == (1, 0, 0, 1)
    b.panel("2026-01", APR20)                                  # other cutoff: built, not mixed up
    assert b.builds == 1
    fx.add("movement_mutabakat", [mut(G2, GUM, "D01", "3")], name="more.csv", available="2026-02-16T00:00:00Z")
    c = AsOfPanels(fx.con, fx.settings)
    c.panel("2026-01", FEB20)                                  # new data -> new input hash -> rebuilt
    assert c.builds == 1 and c.hits == 0
    assert (fx.settings.data_root / "cache" / "canonical_totals").exists()


def test_k1_training_origins_are_quarterly_and_end_at_origin_minus_horizon():
    from pharmacy_potential.features.build import training_origins
    cfg = {"min_train_origin": "2022-01", "train_origin_step_months": 3}
    o = training_origins(cfg, pd.Timestamp("2026-09-01"), 3)
    assert o[-1] == pd.Timestamp("2026-06-01") and o[0] >= pd.Timestamp("2022-01-01")
    assert all(((x.year - y.year) * 12 + x.month - y.month) == 3 for y, x in zip(o, o[1:]))
    # Validation and test origins share one grid, so their canonical caches are reused (cost control).
    grids = [set(training_origins(cfg, pd.Timestamp(e), 3)) for e in ("2025-09", "2025-12", "2026-03", "2026-06")]
    assert grids[1] >= grids[0] and grids[2] >= grids[1] and grids[3] >= grids[2]


# ----------------------------------------------------------------------------- K2

def test_k2_no_crm_source_means_missing_not_zero(fx):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    f = feats(fx, AsOfPanels(fx.con, fx.settings, use_cache=False), "2026-01", FEB20)
    for col in ("visits_3m", "visits_12m", "order_units_3m", "known_campaign_next3"):
        assert f[col].isna().all(), col
    assert (f["crm_coverage_flag"] == 0).all()


def test_k2_covered_window_gives_counts_and_certified_zero(fx):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    window = ("2025-11-01", "2026-01-31")                       # the 3-month window ending at the origin
    fx.add("visits", [visit(G1), visit(G1, n="V2")], period=window, available="2026-02-10T00:00:00Z")
    fx.add("orders", [order(G1)], period=window, available="2026-02-10T00:00:00Z")
    f = feats(fx, AsOfPanels(fx.con, fx.settings, use_cache=False), "2026-01", FEB20)
    assert f.loc[G1, "visits_3m"] == 2 and f.loc[G1, "order_units_3m"] == 4
    assert f.loc[G2, "visits_3m"] == 0 and f.loc[G2, "order_units_3m"] == 0     # covered, no record -> 0
    assert (f["crm_coverage_flag"] == 1).all()
    assert f["visits_12m"].isna().all()                                         # 12-month window not covered


def test_k2_partially_covered_window_is_missing(fx):
    fx.settings.overrides["history_start_month"] = "2025-12"
    fx.coverage(COVER, month="2026-01")
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    fx.add("visits", [visit(G1)], available="2026-02-10T00:00:00Z")          # period is January only
    fx.add("orders", [order(G1)], available="2026-02-10T00:00:00Z")
    f = feats(fx, AsOfPanels(fx.con, fx.settings, use_cache=False), "2026-01", FEB20)
    assert f["visits_3m"].isna().all() and f["order_units_3m"].isna().all()      # needs Nov-Jan, only Jan is covered
    assert (f["crm_coverage_flag"] == 0).all()


def test_k2_batch_unknown_at_the_cutoff_does_not_cover(fx):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    window = ("2025-11-01", "2026-01-31")
    fx.add("visits", [visit(G1)], period=window, available="2026-03-30T00:00:00Z")
    fx.add("orders", [order(G1)], period=window, available="2026-03-30T00:00:00Z")
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    assert feats(fx, asof, "2026-01", FEB20)["visits_3m"].isna().all()
    assert not feats(fx, asof, "2026-01", datetime(2026, 4, 1))["visits_3m"].isna().any()


def test_k2_campaign_feature_is_missing_without_an_announcement_file(fx):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    assert feats(fx, asof, "2026-01", FEB20)["known_campaign_next3"].isna().all()
    fx.add("commercial_terms", [{"Ay": "2026-03", "Barkod": GUM, "KampanyaID": "C1", "Kapsam": "ALL",
                                 "DuyuruTarihi": "2026-01-20"}], period=("2026-01-01", "2026-03-31"),
           available="2026-01-20T00:00:00Z")
    asof = AsOfPanels(fx.con, fx.settings, use_cache=False)
    assert (feats(fx, asof, "2026-01", FEB20)["known_campaign_next3"] == 1).all()
    assert (feats(fx, asof, "2026-01", datetime(2026, 1, 10))["known_campaign_next3"].isna()).all()


# ----------------------------------------------------------------------------- K3

def test_k3_returns_stay_out_of_the_gross_target(fx):
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "0", "0", "3"),          # net -3, gross 0
                                  mut(G2, GUM, "D01", "6", "2", "1"),          # net 7, gross 8
                                  mut(G3, GUM, "D02", "0", "0", "1")])         # net -1
    canon = build_canonical(fx.con, fx.settings, None, "2026-01", "k3")
    p = build_panel(fx.con, canon.table, None)
    i = {g: int(np.where(p.glns == g)[0][0]) for g in (G1, G2, G3)}
    assert p.Y[i[G1], 0] == -3 and p.Yg[i[G1], 0] == 0
    assert p.Y[i[G2], 0] == 7 and p.Yg[i[G2], 0] == 8
    assert p.Y[i[G3], 0] == -1 and p.Yg[i[G3], 0] == 0
    assert (p.Yg[~np.isnan(p.Yg)] >= 0).all()
    assert canon.report["returns_units_total"] == 5 and canon.report["net_negative_cells"] == 2


def test_k3_vendor_net_negative_cell_is_unobserved_for_the_target(fx):
    fx.coverage({"D01": "UNKNOWN", "D02": "UNKNOWN"})
    fx.add("movement_vendor", [{"PharmacyGLN": G1, "Month": "202601", "EAN": GUM, "Units": "-2"},
                            {"PharmacyGLN": G1, "Month": "202601", "EAN": PATCH, "Units": "4"}])
    canon = build_canonical(fx.con, fx.settings, None, "2026-01", "k3b")
    p = build_panel(fx.con, canon.table, None)
    g = int(np.where(p.glns == G1)[0][0])
    assert p.Yg[g, 0] != p.Yg[g, 0]                                         # NaN: the -2 cell cannot be split
    assert canon.report["vendor_cells_unobserved_for_target"] == 1


def test_k3_model_refuses_a_negative_or_non_finite_target():
    from pharmacy_potential.models.gbm import check_finite_log_target, fit_quantile_models
    with pytest.raises(ValueError, match="non-negative gross outflow"):
        check_finite_log_target(pd.Series([3.0, -1.0, 5.0]))                  # log1p(-1) = -inf
    with pytest.raises(ValueError, match="non-negative gross outflow"):
        check_finite_log_target(pd.Series([3.0, -3.0]))                       # log1p(-3) = NaN
    bad = pd.DataFrame({"target_units_next3": [1.0, -3.0], "origin_month": pd.to_datetime(["2026-01-01"] * 2),
                        "gln": ["a", "b"]})
    with pytest.raises(ValueError, match="negative model target"):
        fit_quantile_models(bad, {}, 1, 1)


def test_k1_cache_key_covers_scope_overrides(fx):
    """A changed history start or brand changes the canonical build; a cached panel from before must not be served."""
    fx.coverage(COVER)
    fx.add("movement_mutabakat", [mut(G1, GUM, "D01", "10")])
    AsOfPanels(fx.con, fx.settings).panel("2026-01", FEB20)
    fx.settings.overrides["history_start_month"] = "2025-12"
    b = AsOfPanels(fx.con, fx.settings)
    assert b.panel("2026-01", FEB20).months[0] == pd.Timestamp("2025-12-01")
    assert (b.builds, b.hits) == (1, 0)
    fx.settings.overrides["brand"] = "SomethingElse"
    c = AsOfPanels(fx.con, fx.settings)
    with pytest.raises(RuntimeError, match="no product of brand"):
        c.panel("2026-01", FEB20)                                   # not served from the DemoBrand cache
