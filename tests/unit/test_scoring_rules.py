"""S08, S12, S13, S14 and S10 (readiness) on small hand-built inputs."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pharmacy_potential.evaluation.metrics import forecast_metrics, precision_at_k
from pharmacy_potential.export.excel import export_workbook, safe_text
from pharmacy_potential.features.build import Panel, labels_for_origin, loo_median
from pharmacy_potential.models.baselines import rank_order
from pharmacy_potential.readiness.readiness import build_readiness
from pharmacy_potential.scoring.score import allocate_total_gap
from pharmacy_potential.utils import RedactingFilter, stable_unit_hash
from tests.conftest import G1, G2, G3, GUM, PATCH


def _panel(Y: np.ndarray) -> Panel:
    G, M = Y.shape
    return Panel(glns=np.array([f"g{i}" for i in range(G)]), city=np.array(["34"] * G),
                 district=np.array(["d1"] * G), months=pd.date_range("2026-01-01", periods=M, freq="MS"),
                 Y=Y, active=np.ones((G, M), bool), fallback=np.zeros((G, M)), Yg=Y.copy())


THRESH = {"proxy_label": {"base_window_months": 3, "future_window_months": 3, "min_excess_ratio": 1.25,
                          "min_abs_increase_per_month": 1.0, "min_future_months_above_base": 2}}


def test_s08_unobserved_future_is_censored_not_negative():
    Y = np.array([[5, 5, 5, 9, 9, 9], [5, 5, 5, np.nan, 5, 5], [5, 5, 5, 5, 5, 5]], float)
    lab = labels_for_origin(_panel(Y), "2026-03", THRESH)
    assert lab.loc[1, "label_status"] == "censored_future"
    assert np.isnan(lab.loc[1, "proxy_positive"])          # censored, not a false negative
    assert lab.loc[0, "proxy_positive"] == 1.0 and lab.loc[2, "proxy_positive"] == 0.0


def test_labels_immature_when_future_not_available():
    lab = labels_for_origin(_panel(np.ones((2, 4))), "2026-03", THRESH)
    assert (lab["label_status"] == "immature").all()


def test_s12_total_gap_allocation_keeps_total_primary():
    raw = pd.DataFrame({"GUM": [8.0, -1.0, np.nan], "PATCH": [-3.0, -2.0, np.nan]})
    total = pd.Series([5.0, 4.0, np.nan])
    a = allocate_total_gap(total, raw)
    assert a.loc[0, "GUM"] == 5 and a.loc[0, "PATCH"] == 0 and a.loc[0, "UNALLOCATED"] == 0
    assert a.loc[1, "UNALLOCATED"] == 4                      # no positive form gap -> residual kept
    assert np.isnan(a.loc[2, "GUM"])                         # unknown total stays unknown
    assert raw.loc[0, "GUM"] == 8                            # raw form gap untouched


def test_s13_undefined_metrics_and_deterministic_ties():
    label = pd.Series([0.0, 0.0, 0.0, 0.0])
    score = pd.Series([1.0, 1.0, 1.0, 0.5])
    tb = pd.Series(stable_unit_hash(pd.Series(["a", "b", "c", "d"]), 7))
    p = precision_at_k(score, label, tb, 2)
    assert p["precision"] == 0.0 and p["lift"] is None      # zero prevalence -> lift undefined
    assert p["ties_at_boundary"] == 3
    assert list(rank_order(score, tb)) == list(rank_order(score, tb))
    fm = forecast_metrics(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))
    assert fm["wape"] is None                                # zero denominator -> null
    assert precision_at_k(score, label.iloc[:0], tb.iloc[:0], 50)["precision"] is None


def test_net_negative_baseline_gives_finite_growth_signal():
    from pharmacy_potential.models.baselines import ml_growth_signal
    f = pd.DataFrame({"baseline_units": [-1.0, 0.0, 4.0], "district_code": ["d", "d", "d"]})
    g = ml_growth_signal(f, pd.Series([3.0, 3.0, 3.0]))
    assert np.isfinite(g).all() and g[0] == g[1]


def test_loo_median_excludes_self():
    vals = np.array([1.0, 2.0, 3.0, 100.0, np.nan])
    med, n = loo_median(vals, np.array(["d"] * 5))
    assert med[3] == 2.0 and n[3] == 3                       # 100 excluded from its own peer set
    assert med[0] == 3.0 and med[4] == 2.5 and n[4] == 4     # unobserved GLN sees all observed peers


def test_s14_formula_injection_and_log_redaction(tmp_path, caplog):
    assert safe_text("=HYPERLINK(\"x\")").startswith("'=")
    assert safe_text("+1") == "'+1" and safe_text("Pharmacy") == "Pharmacy"
    res = pd.DataFrame({"run_id": ["r"], "score_month": ["2026-09"], "gln": [G1], "city_code": ["34"],
                        "district_code": ["34001"], "track": ["B_sales_aware"], "form_or_total": ["TOTAL"],
                        "score_0_100": [50.0], "rank": [1], "baseline_units_per_month": [np.nan],
                        "incremental_gap_units_per_month": [0.0], "forecast_method": ["gbm_p50"],
                        "forecast_next3m_point": [1.0], "forecast_next3m_p10": [0.5], "forecast_next3m_p50": [1.0],
                        "forecast_next3m_p90": [2.0], "confidence": ["low"], "ml_reason_1": ["x"],
                        "ml_reason_2": [""], "ml_reason_3": [""]})
    names = pd.DataFrame({"gln": [G1], "pharmacy_name": ["=HYPERLINK(\"http://evil\")"], "brick_code": ["B1"]})
    path = export_workbook(tmp_path / "x.xlsx", res, names, {
        "run_id": "r", "score_month": "2026-09", "cutoff": "c", "quality_table": pd.DataFrame({"a": [1]}),
        "evaluation_table": pd.DataFrame({"a": [1]})}, synthetic=True, exploratory=False)
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb["Ranking_Total"]
    header = [c.value for c in ws[3]]
    row = {h: c for h, c in zip(header, ws[4])}
    assert row["gln"].value == G1 and row["gln"].data_type == "s"           # leading zero kept, text
    assert row["pharmacy_name"].data_type == "s" and row["pharmacy_name"].value.startswith("'=")
    assert row["baseline_units_per_month"].value is None                   # null stays blank
    assert row["incremental_gap_units_per_month"].value == 0                # zero stays zero
    assert "SYNTHETIC" in ws["A1"].value
    assert all(s.sheet_state == "visible" for s in wb.worksheets)
    rec = logging.LogRecord("ppi", logging.INFO, __file__, 1, "processing %s", (G2,), None)
    RedactingFilter().filter(rec)
    assert G2 not in rec.getMessage() and rec.getMessage().startswith("processing 868")


def test_s10_parent_sellin_not_distributed_and_partial_subdepots(fx):
    fx.add("sellin", [{"Ay": "2026-01", "DepoKodu": "D01", "Barkod": GUM, "GirisAdet": "100"}])
    # Only one of D01's two sub depots reports stock -> evidence is partial; nothing is split 50/50.
    fx.add("depot_stock", [{"StokTarihi": "2026-01-31", "AltDepoKodu": "D01S1", "Barkod": GUM, "EldekiAdet": "100"}])
    r = build_readiness(fx.con, None, "2026-01", {"readiness": {"max_stock_age_days": 45, "sellin_lookback_months": 3}},
                        "DemoBrand")
    r = r.set_index("gln")
    assert r.loc[G1, "readiness"] == "high" and "partial" in r.loc[G1, "readiness_reason"]
    assert r.loc[G3, "readiness"] == "unknown"               # D02 has no stock evidence
    assert not any("stock" in c and "units" in c for c in r.columns)   # no per-pharmacy stock quantity
