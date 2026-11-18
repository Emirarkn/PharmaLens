"""(B15, B16 + champion rule): one comparison universe, empty samples give 'inconclusive', not a crash."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pharmacy_potential.config import load_settings
from pharmacy_potential.evaluation.backtest import acceptance, choose_champion, forecast_block
from pharmacy_potential.evaluation.metrics import cluster_bootstrap_diff, evaluate_ranking, forecast_metrics
from pharmacy_potential.models.baselines import SALES_AWARE, SIMPLE_BASELINES

ROOT = Path(__file__).resolve().parents[2]
S = load_settings(ROOT / "configs" / "pilot.yaml")
METHODS = ("seasonal_naive", "moving_average_3", "holt_winters", "gbm_p50")


def fc_frame(actual, **preds) -> pd.DataFrame:
    n = len(actual)
    df = pd.DataFrame({"target_units_next3": actual, "city_code": "34", "fc_gbm_p10": 0.0, "fc_gbm_p90": 1e9})
    for m in METHODS:
        df[f"fc_{m}"] = preds.get(m, [50.0] * n)
    return df


def test_k10_forecast_candidates_share_one_universe_B15():
    # The sparse candidate predicts only the easy row; it used to be scored on n=1 with WAPE 0.
    assert forecast_metrics(pd.Series([10.0, 100.0]), pd.Series([10.0, np.nan]))["wape"] == 0
    df = fc_frame([10.0, 100.0], seasonal_naive=[10.0, np.nan], moving_average_3=[12.0, 90.0],
                  holt_winters=[9.0, 99.0], gbm_p50=[11.0, 101.0])
    out, universe = forecast_block(df, holdout="21")
    assert {out[m]["n"] for m in METHODS} == {1}                              # same GLN list for every method
    assert out["moving_average_3"]["wape"] == 2 / 10                          # evaluated on the same easy row only
    assert out["seasonal_naive"]["wape"] == 0
    assert out["seasonal_naive"]["coverage"] == 0.5 and out["gbm_p50"]["coverage"] == 1.0
    assert universe == {"n_common": 1, "n_actual_observed": 2}


def test_k10_holdout_city_is_outside_the_forecast_universe():
    df = fc_frame([10.0, 20.0])
    df["city_code"] = ["34", "21"]
    assert forecast_block(df, holdout="21")[0]["gbm_p50"]["n"] == 1


def test_k10_empty_bootstrap_is_null_with_a_reason_B16():
    empty = pd.DataFrame({"district_code": [], "label": [], "_tiebreak": [], "peer_gap": [], "current_sales": []})
    out = cluster_bootstrap_diff(empty, "peer_gap", "current_sales", 100, 5, 1)
    assert out["reps"] == 0 and out["diff_mean"] is None and out["diff_ci95"] is None and out["note"]


def empty_subset(cands):
    nothing = pd.Series([], dtype=float)
    return {"n": 0, "positives": 0, "prevalence": None,
            "candidates": {c: evaluate_ranking(nothing, nothing, nothing, nothing, [50, 100, 250]) for c in cands}}


def test_k10_acceptance_with_no_eligible_test_gln_is_inconclusive_not_an_exception():
    cands = list(dict.fromkeys(SIMPLE_BASELINES + SALES_AWARE))
    test = {"subsets": {k: empty_subset(cands) for k in ("overall_excl_holdout", "holdout_city", "low_sales_excl_holdout")}}
    out = acceptance(test, [test], "b_composite", S)
    assert out["decision_proposal"] == "inconclusive"
    assert not out["checks"]["min_sample_overall"]["pass"]


def val(precisions: dict[str, float], wape: dict[str, float] | None = None) -> dict:
    wape = wape or {m: 0.5 for m in METHODS}
    return {"subsets": {"overall_excl_holdout": {"candidates": {c: {"P@100": {"precision": p}} for c, p in precisions.items()}}},
            "forecast": {m: {"wape": w} for m, w in wape.items()}}


def test_k10_complex_candidate_must_beat_the_best_simpler_one_by_two_percent():
    # Reviewer example .500 / .509 / .515: hybrid is +3% over peer_gap but only +1.2% over b_composite.
    r = val({"peer_gap": 0.500, "b_composite": 0.509, "b_hybrid": 0.515, "ml_growth": 0.40})
    assert choose_champion([r])["ranking_champion"] == "peer_gap"
    r = val({"peer_gap": 0.500, "b_composite": 0.520, "b_hybrid": 0.530, "ml_growth": 0.40})
    assert choose_champion([r])["ranking_champion"] == "b_composite"           # +4% ok; hybrid only +1.9% over it
    r = val({"peer_gap": 0.500, "b_composite": 0.520, "b_hybrid": 0.560, "ml_growth": 0.40})
    assert choose_champion([r])["ranking_champion"] == "b_hybrid"


def test_k10_champion_without_any_validation_evidence_defaults_to_the_simplest():
    r = val({c: None for c in SALES_AWARE})
    assert choose_champion([r])["ranking_champion"] == "peer_gap"


def test_k10_immature_origin_still_has_a_graded_label_column():
    from pharmacy_potential.features.build import Panel, labels_for_origin
    Y = np.ones((2, 4))
    p = Panel(glns=np.array(["a", "b"]), city=np.array(["34"] * 2), district=np.array(["d"] * 2),
              months=pd.date_range("2026-01-01", periods=4, freq="MS"), Y=Y, active=np.ones((2, 4), bool),
              fallback=np.zeros((2, 4)), Yg=Y.copy())
    lab = labels_for_origin(p, "2026-03", S.thresholds)
    assert (lab["label_status"] == "immature").all() and "graded_label" in lab and lab["graded_label"].isna().all()
