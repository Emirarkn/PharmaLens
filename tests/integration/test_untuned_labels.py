""": a run on fixed engineering defaults is labelled everywhere and is never G3 evidence."""
from __future__ import annotations

import openpyxl
import pandas as pd
import pytest

from pharmacy_potential import pipeline
from pharmacy_potential.utils import read_json

pytestmark = pytest.mark.integration


def test_untuned_run_is_labelled_in_scores_manifest_model_card_and_excel(small_universe, monkeypatch):
    s = small_universe
    monkeypatch.setattr(pipeline, "check_gates", lambda *a, **k: {
        "synthetic": True, "exploratory": False, "mode": "untuned", "tuning_allowed": False,
        "gates": {"G0": "pending", "G1": "pending", "G2": "pending", "G3": "pending", "G4": "pending"}})
    res = pipeline.run_scoring(s, skip_backtest=True, run_label="untuned_1")
    sr = pd.read_parquet(res["artifacts"] / "score_result.parquet")
    assert set(sr["model_tuning"]) == {"untuned_engineering_default"}
    man = read_json(res["reports"] / "manifest.json")
    assert man["model_tuning"] == "untuned_engineering_default" and man["usable_as_g3_evidence"] is False
    card = read_json(res["artifacts"] / "model_card.json")
    assert card["model_tuning"] == "untuned_engineering_default" and card["tuned"] is False
    assert card["params"]["num_leaves"] == s.get("model")["gbm"]["untuned"]["num_leaves"]    # no grid selection
    wb = openpyxl.load_workbook(res["excel"], read_only=True)
    readme = " ".join(str(c) for row in wb["READ_ME"].iter_rows(values_only=True) for c in row if c)
    assert "UNCALIBRATED FORECAST" in readme
    header = [c.value for c in next(wb["Ranking_Total"].iter_rows(min_row=3, max_row=3))]
    assert "model_tuning" in header


def test_tuned_synthetic_run_says_tuned(small_universe):
    res = pipeline.run_scoring(small_universe, skip_backtest=True, run_label="tuned_1")
    assert set(pd.read_parquet(res["artifacts"] / "score_result.parquet")["model_tuning"]) == {"tuned"}
    assert read_json(res["reports"] / "manifest.json")["usable_as_g3_evidence"] is False      # no backtest was run
