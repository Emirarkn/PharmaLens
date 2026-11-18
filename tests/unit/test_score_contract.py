"""readiness never changes opportunity rank; K5 null contract of form rows."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pharmacy_potential.config import load_settings
from pharmacy_potential.features.build import Panel
from pharmacy_potential.models.baselines import candidate_scores
from pharmacy_potential.scoring.score import allocate_total_gap, form_eligibility, form_allocation

ROOT = Path(__file__).resolve().parents[2]
S = load_settings(ROOT / "configs" / "pilot.yaml")


# ----------------------------------------------------------------------------- K4

def opportunity_frame(readiness: list[float]) -> pd.DataFrame:
    n = len(readiness)
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "gln": [f"g{i}" for i in range(n)], "city_code": "34", "district_code": ["d1", "d2"] * (n // 2),
        "n_obs12": 12, "baseline_units": rng.uniform(1, 30, n), "baseline_gross": rng.uniform(1, 30, n),
        "ext_base_units": rng.uniform(5, 40, n), "peer_gap_rel": rng.uniform(0, 1, n),
        "peer_gap_units": rng.uniform(0, 20, n), "readiness_score": readiness})


def test_k4_readiness_does_not_change_any_opportunity_score():
    n = 40
    rng = np.random.default_rng(1)
    p50 = pd.Series(rng.uniform(3, 60, n))
    high = candidate_scores(opportunity_frame([1.0] * n), 7, S.get("score_weights"), p50)
    low = candidate_scores(opportunity_frame([0.25] * n), 7, S.get("score_weights"), p50)
    pd.testing.assert_frame_equal(high, low)                                   # every candidate, every row
    mixed = candidate_scores(opportunity_frame(list(rng.choice([0.25, 0.5, 1.0], n))), 7, S.get("score_weights"), p50)
    pd.testing.assert_frame_equal(high, mixed)


def test_k4_scoring_works_without_any_readiness_column():
    f = opportunity_frame([1.0] * 10).drop(columns="readiness_score")
    assert "b_composite" in candidate_scores(f, 7, S.get("score_weights"))


def test_k4_composite_weights_are_v002_without_continuity():
    w = S.get("score_weights")
    assert S.get("score_weights_version") == "v002"
    assert set(w["b_composite"]) == {"peer_gap", "external"}
    assert w["b_composite"] == {"peer_gap": 0.7, "external": 0.3} and abs(sum(w["b_composite"].values()) - 1) < 1e-12


def test_k4_a_config_that_puts_readiness_back_in_the_score_is_refused():
    import pytest
    bad = {"b_composite": {"peer_gap": 0.6, "external": 0.25, "continuity": 0.15}, "b_hybrid": S.get("score_weights")["b_hybrid"]}
    with pytest.raises(ValueError, match="readiness"):
        candidate_scores(opportunity_frame([1.0] * 4), 7, bad)


# ----------------------------------------------------------------------------- K5

def form_ctx(total_unknown_for: str | None = None):
    glns = ["a", "b", "c", "d"]
    months = pd.date_range("2025-01-01", periods=14, freq="MS")
    G, M = 4, 14
    gum = np.tile(np.array([[10.0], [2.0], [6.0], [4.0]]), (1, M))
    patch = np.tile(np.array([[1.0], [1.0], [1.0], [1.0]]), (1, M))
    unknown = np.full((G, M), 2.0)
    Y = gum + patch + unknown
    panel = Panel(glns=np.array(glns), city=np.array(["34"] * G), district=np.array(["d1"] * G), months=months, Y=Y,
                  active=np.ones((G, M), bool), fallback=np.zeros((G, M)),
                  form_Y={"GUM": gum, "PATCH": patch, "UNKNOWN": unknown}, Yg=Y.copy())
    f = pd.DataFrame({"gln": glns, "district_code": "d1", "city_code": "34", "ext_base_units": [8.0, 8.0, 8.0, 8.0]})
    return SimpleNamespace(panel=panel, features=f, origin=months[-1])


def test_k5_unknown_total_gap_nulls_every_form_row_and_unknown_form_is_never_zero():
    ctx = form_ctx()
    total = pd.Series([5.0, np.nan, 3.0, 0.0])                                # GLN "b" has an unknown TOTAL gap
    fa = form_allocation(ctx, {"peer": {"shrink_k": 20}}, total)
    fa["eligibility"] = form_eligibility(fa["form"], fa["allocated_gap"])
    b = fa[fa.gln == "b"]
    assert b["allocated_gap"].isna().all()                                    # GUM, PATCH, UNKNOWN, UNALLOCATED
    assert (b["eligibility"].isin(["no_observed_baseline", "form_unmapped"])).all()
    unk = fa[fa.form == "UNKNOWN"]
    assert unk["allocated_gap"].isna().all()                                  # never 0, even where the total is known
    assert (unk["eligibility"] == "form_unmapped").all()
    known = fa[(fa.gln != "b") & (fa.form != "UNKNOWN")]
    assert known["allocated_gap"].notna().all() and (known["eligibility"] == "eligible").all()


def test_k5_eligible_rows_always_have_a_gap_and_null_gaps_are_never_eligible():
    forms = pd.Series(["GUM", "PATCH", "UNKNOWN", "UNALLOCATED", "GUM"])
    gap = pd.Series([2.0, 0.0, np.nan, 1.5, np.nan])
    el = form_eligibility(forms, gap)
    assert list(el) == ["eligible", "eligible", "form_unmapped", "eligible", "no_observed_baseline"]


def test_k5_allocation_keeps_total_primary_and_unknown_total_unknown():
    raw = pd.DataFrame({"GUM": [8.0, np.nan], "PATCH": [-3.0, np.nan]})
    a = allocate_total_gap(pd.Series([5.0, np.nan]), raw)
    assert a.loc[0, "GUM"] == 5 and a.loc[0, "PATCH"] == 0
    assert a.loc[1].isna().all()
