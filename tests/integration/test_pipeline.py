"""S07, S09, S15, S16 and an end-to-end run on a small synthetic universe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmacy_potential.db import connect
from pharmacy_potential.demand.external import build_external_opportunity
from pharmacy_potential.features.asof import AsOfPanels
from pharmacy_potential.features.build import build_panel, features_for_origin, training_frame
from pharmacy_potential.reconcile.canonical import build_canonical
from pharmacy_potential.utils import backtest_cutoff, month_range, to_naive_utc

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ctx(small_universe):
    s = small_universe
    con = connect(s.warehouse_path)
    cutoff = backtest_cutoff("2025-06", 50)
    canon = build_canonical(con, s, cutoff, "2025-06", "t_feat")
    panel = build_panel(con, canon.table, cutoff)
    yield s, con, cutoff, panel
    con.close()


def test_s07_future_side_data_does_not_change_features(ctx):
    s, con, cutoff, panel = ctx
    th, sc = s.thresholds, s.scenario
    f1 = features_for_origin(panel, "2025-06", cutoff, th, sc)
    # Inject knowledge that only became available after the cutoff.
    late = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    g = panel.glns[0]
    panel.side["visits"] = pd.concat([panel.side["visits"], pd.DataFrame(
        {"gln": [g] * 50, "month": [pd.Timestamp("2025-06-01")] * 50, "avail": [late] * 50})], ignore_index=True)
    panel.side["orders"] = pd.concat([panel.side["orders"], pd.DataFrame(
        {"gln": [g], "month": [pd.Timestamp("2025-06-01")], "units": [999], "avail": [late]})], ignore_index=True)
    panel.side["campaigns"] = pd.concat([panel.side["campaigns"], pd.DataFrame(
        {"month": [pd.Timestamp("2025-07-01")], "barcode": ["x"], "avail": [late]})], ignore_index=True)
    f2 = features_for_origin(panel, "2025-06", cutoff, th, sc)
    pd.testing.assert_frame_equal(f1, f2)


def test_s16_features_ignore_months_after_origin_and_training_is_purged(ctx):
    s, con, cutoff, panel = ctx
    th, sc = s.thresholds, s.scenario
    f1 = features_for_origin(panel, "2025-03", backtest_cutoff("2025-03", 50), th, sc)
    saved = panel.Y.copy()
    o = panel.month_idx("2025-03")
    panel.Y[:, o + 1:] = 1e6                       # poison the future
    f2 = features_for_origin(panel, "2025-03", backtest_cutoff("2025-03", 50), th, sc)
    panel.Y = saved
    pd.testing.assert_frame_equal(f1, f2)
    # Training origins for evaluation origin 2025-06 stop at 2025-03: label windows never pass the origin.
    origins = month_range("2024-10", "2025-03")
    tr = training_frame(AsOfPanels(con, s, use_cache=False).panel, panel, origins, th, sc, 50)
    assert tr["origin_month"].max() == pd.Timestamp("2025-03-01")
    assert tr.loc[tr["origin_month"] == pd.Timestamp("2025-03-01"), "target_units_next3"].notna().any()


def test_s09_external_track_ignores_sales(small_universe):
    s = small_universe
    con = connect(s.warehouse_path)
    cutoff = to_naive_utc(s.get("score_cutoff_utc"))
    a1, checks = build_external_opportunity(con, cutoff, "2026-09", s.scenario)
    con.execute("create temp table _backup as select * from stg_movement_mutabakat")
    con.execute("update stg_movement_mutabakat set units_paid = units_paid * 7 + 3")   # perturb/reverse sales
    a2, _ = build_external_opportunity(con, cutoff, "2026-09", s.scenario)
    con.execute("delete from stg_movement_mutabakat")
    con.execute("insert into stg_movement_mutabakat select * from _backup")
    con.close()
    pd.testing.assert_frame_equal(a1, a2)
    assert checks["share_sum_per_district_max_abs_error"] < 1e-9 and checks["scenario_ordering_ok"]
    assert checks["district_box_total_matches_scenario"]
    tied = a1.groupby("district_code")["a_score_0_100"].nunique()
    assert (tied == 1).all()                         # same district -> tied, no fake precision


def test_s15_reproducible_scores_and_revision_creates_new_run(small_universe):
    from pharmacy_potential.pipeline import run_scoring
    s = small_universe
    r1 = run_scoring(s, skip_backtest=True, write_outputs=False, run_label="rep_a")
    r2 = run_scoring(s, skip_backtest=True, write_outputs=False, run_label="rep_b")
    if r1["score_checksum"] != r2["score_checksum"]:
        key = ["track", "form_or_total", "gln"]
        a = r1["score_result"].drop(columns="run_id").sort_values(key).reset_index(drop=True)
        b = r2["score_result"].drop(columns="run_id").sort_values(key).reset_index(drop=True)
        diff = [c for c in a.columns if not a[c].equals(b[c])]
        pytest.fail(f"non-deterministic columns: {diff}")


def test_end_to_end_run_with_backtest(small_universe):
    from pharmacy_potential.pipeline import run_scoring
    import openpyxl
    s = small_universe
    res = run_scoring(s, include_names=True)
    rep = res["reports"]
    for f in ("quality.json", "reconciliation.csv", "coverage.csv", "asof_audit.json", "evaluation.json",
              "evaluation.md", "acceptance.json", "synthetic_truth_check.json", "manifest.json", "cost.json"):
        assert (rep / f).exists(), f
    assert res["evaluation"]["decision_proposal"] in ("pass", "fail", "inconclusive")
    from pharmacy_potential.utils import read_json
    ev = read_json(rep / "evaluation.json")
    for o in ev["origins"]:      # the draft label is validation-only; the locked test origin never sees it
        assert ("alt_label_v002" in o) == (o["role"] == "validation")
        assert o["forecast_universe"]["n_common"] <= o["forecast_universe"]["n_actual_observed"]
    truth = read_json(rep / "synthetic_truth_check.json")
    assert {"peer_gap", "b_composite", "b_hybrid", "ml_growth", "current_sales"} <= set(truth["candidates"])
    wb = openpyxl.load_workbook(res["excel"], read_only=True)
    assert {"READ_ME", "Ranking_Total", "Ranking_Form", "External_Only", "Forecast", "Explanations",
            "Coverage_Quality", "Evaluation", "Data_Dictionary"} <= set(wb.sheetnames)
    sr = pd.read_parquet(res["artifacts"] / "score_result.parquet")
    total_b = sr[(sr["track"] == "B_sales_aware") & (sr["form_or_total"] == "TOTAL")]
    assert total_b["gln"].is_unique
    assert ((total_b["score_0_100"].dropna() >= 0) & (total_b["score_0_100"].dropna() <= 100)).all()
    no_base = total_b[total_b["baseline_units_per_month"].isna()]
    assert no_base["incremental_gap_units_per_month"].isna().all() and no_base["rank"].isna().all()
    a = sr[sr["track"] == "A_external_only"]
    assert a["gln"].is_unique and len(a) == len(total_b)
