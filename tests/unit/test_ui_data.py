"""PharmaLens data layer and a smoke test of the Streamlit app on a tiny hand-made run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmacy_potential.ui import data as D

G1, G2, G3 = "0000000000001", "0000000000002", "0000000000003"
APP = Path(D.__file__).parent / "app.py"


def _scores() -> pd.DataFrame:
    base = dict(run_id="r1", score_month="2026-09", city_code="34", district_code="34001", cohort="x", observed_months_12=12,
                observation_status_last="observed_positive", raw_form_gap=np.nan, scenario_units_per_month_low=1.0,
                scenario_units_per_month_base=2.0, scenario_units_per_month_high=3.0, forecast_method="m",
                readiness="high", readiness_reason="", confidence="high", confidence_score=0.9, reason_1="peer gap",
                reason_2="", reason_3="", ml_reason_1="recent sales raise the forecast", ml_reason_2="", ml_reason_3="",
                source_policy="p", data_cutoff_utc="2026-10-25", model_version="v", tie_group="")
    rows = []
    for gln, rank in ((G1, 1), (G2, 2), (G3, np.nan)):
        rows.append({**base, "gln": gln, "track": "B_sales_aware", "form_or_total": "TOTAL", "score_0_100": 90.0 - (rank if rank == rank else 50),
                     "rank": rank, "eligibility": "eligible" if rank == rank else "no_observed_baseline",
                     "baseline_units_per_month": 5.0 if rank == rank else np.nan, "benchmark_units_per_month": 8.0,
                     "incremental_gap_units_per_month": 3.0 if rank == rank else np.nan,
                     "forecast_next3m_point": 20.0, "forecast_next3m_p10": 10.0, "forecast_next3m_p50": 20.0, "forecast_next3m_p90": 30.0})
        rows.append({**base, "gln": gln, "track": "A_external_only", "form_or_total": "TOTAL", "score_0_100": 50.0, "rank": 1,
                     "eligibility": "eligible", "baseline_units_per_month": np.nan, "benchmark_units_per_month": np.nan,
                     "incremental_gap_units_per_month": np.nan, "forecast_next3m_point": np.nan, "forecast_next3m_p10": np.nan,
                     "forecast_next3m_p50": np.nan, "forecast_next3m_p90": np.nan, "tie_group": "district:34001"})
    rows.append({**base, "gln": G1, "track": "B_sales_aware", "form_or_total": "GUM", "score_0_100": 40.0, "rank": 1,
                 "eligibility": "eligible", "baseline_units_per_month": 1.0, "benchmark_units_per_month": 2.0,
                 "incremental_gap_units_per_month": 1.0, "forecast_next3m_point": np.nan, "forecast_next3m_p10": np.nan,
                 "forecast_next3m_p50": np.nan, "forecast_next3m_p90": np.nan})
    return pd.DataFrame(rows)


@pytest.fixture()
def roots(tmp_path) -> D.Roots:
    r = D.Roots(tmp_path / "reports", tmp_path / "artifacts", tmp_path / "outputs")
    (r.reports / "r1").mkdir(parents=True)
    (r.artifacts / "r1").mkdir(parents=True)
    (r.reports / "r1" / "manifest.json").write_text(json.dumps(
        {"run_id": "r1", "synthetic": True, "exploratory": False, "champion": {"ranking_champion": "peer_gap"},
         "cutoff_utc": "2026-10-25", "gates": {"G0": "pending", "G1": "pending"}}), encoding="utf-8")
    _scores().to_parquet(r.artifacts / "r1" / "score_result.parquet", index=False)
    pd.DataFrame({"mean3": [0.4, -0.1, 0.2], "lag0": [-0.3, 0.2, 0.1], "gln": [G1, G2, G3]}).to_parquet(
        r.artifacts / "r1" / "shap_values.parquet", index=False)
    return r


def test_runs_are_listed_only_when_complete(roots):
    (roots.reports / "half").mkdir()
    (roots.reports / "half" / "manifest.json").write_text("{}", encoding="utf-8")        # no score table
    assert D.list_runs(roots) == ["r1"]


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "a b", "", "x" * 200])
def test_run_id_cannot_escape_the_run_folder(roots, bad):
    with pytest.raises(ValueError):
        D.read_scores(roots, bad)
    with pytest.raises(ValueError):
        D.read_json(roots, bad, "manifest.json")


def test_gln_keeps_its_leading_zeros_and_tracks_split(roots):
    s = D.read_scores(roots, "r1")
    assert set(D.select_track(s, "B_total")["gln"]) == {G1, G2, G3}
    assert set(D.select_track(s, "B_form")["form_or_total"]) == {"GUM"}
    assert len(D.select_track(s, "A_external")) == 3
    with pytest.raises(ValueError):
        D.select_track(s, "nope")


def test_empty_filter_selection_means_no_filter_and_unranked_sort_last(roots):
    tot = D.select_track(D.read_scores(roots, "r1"), "B_total")
    assert len(D.filter_scores(tot, cities=[], eligibility=[], gln_contains="")) == 3
    assert list(D.filter_scores(tot, eligibility=["no_observed_baseline"])["gln"]) == [G3]
    assert list(D.filter_scores(tot, gln_contains="0002")["gln"]) == [G2]
    assert list(D.ranking_view(tot, 10)["gln"]) == [G1, G2, G3]          # the unranked pharmacy is last, not dropped
    assert len(D.ranking_view(tot, 2)) == 2


def test_shap_for_a_pharmacy_is_sorted_by_size_and_missing_gln_is_none(roots):
    sh = D.shap_for(roots, "r1", G1)
    assert list(sh["feature"]) == ["mean3", "lag0"] and sh["shap"].abs().is_monotonic_decreasing
    assert D.shap_for(roots, "r1", "9999999999999") is None
    assert D.pharmacy_rows(D.read_scores(roots, "r1"), f" {G1} ").shape[0] == 3


def test_app_renders_every_page_without_exceptions(roots, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PHARMALENS_REPORTS", str(roots.reports))
    monkeypatch.setenv("PHARMALENS_ARTIFACTS", str(roots.artifacts))
    monkeypatch.setenv("PHARMALENS_OUTPUTS", str(roots.outputs))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    for page in ("Rankings", "Pharmacy", "Data quality", "Evaluation", "Overview"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    at.sidebar.radio[0].set_value("Pharmacy").run()
    at.text_input[0].set_value(G1).run()
    assert not at.exception and any("peer gap" in m.value for m in at.markdown)
    at.text_input[0].set_value("9999999999999").run()
    assert not at.exception and any("No rows for GLN" in i.value for i in at.info)


def test_app_without_any_run_shows_an_empty_state(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    for k in ("REPORTS", "ARTIFACTS", "OUTPUTS"):
        monkeypatch.setenv(f"PHARMALENS_{k}", str(tmp_path / k.lower()))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception and any("No finished run found" in i.value for i in at.info)
