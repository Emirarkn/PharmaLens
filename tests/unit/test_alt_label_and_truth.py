"""Draft alternative label (validation only, never a gate input) and per-candidate truth check."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pharmacy_potential.config import load_settings
from pharmacy_potential.features.build import Panel, labels_peer_closure
from pharmacy_potential.pipeline import truth_topk

ROOT = Path(__file__).resolve().parents[2]
S = load_settings(ROOT / "configs" / "pilot.yaml")
ALT = S.alt_thresholds


def panel_with(rows: list[list[float]]) -> Panel:
    Yg = np.array(rows, float)
    G, M = Yg.shape
    return Panel(glns=np.array([f"g{i}" for i in range(G)]), city=np.array(["34"] * G), district=np.array(["d"] * G),
                 months=pd.date_range("2026-01-01", periods=M, freq="MS"), Y=np.full((G, M), 999.0),   # net: must be ignored
                 active=np.ones((G, M), bool), fallback=np.zeros((G, M)), Yg=Yg)


def test_draft_config_exists_and_the_default_label_is_untouched():
    assert S.thresholds["thresholds_version"] == "thresholds_v001"
    assert ALT["thresholds_version"] == "thresholds_v002" and ALT["approval_status"].startswith("draft_unapproved")
    assert any(p.name == "thresholds_v002.yaml" for p in S.config_files())          # hashed into every run manifest
    assert S.thresholds["proxy_label"]["min_excess_ratio"] == 1.25                  # v001 values unchanged


def test_peer_closure_label_on_the_gross_series():
    #            Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep   (origin = Jun, base Apr-Jun, future Jul-Sep)
    rows = [[0, 0, 0, 4, 4, 4, 7, 7, 7],           # g0 closes 3 of a 6-box gap (0.5) in all 3 months -> positive
            [0, 0, 0, 4, 4, 4, 5, 5, 5],           # g1 closes 1 of 6 -> negative
            [0, 0, 0, 9, 9, 9, 12, 12, 12],        # g2 gap to benchmark only 1 box -> ineligible
            [0, 0, 0, 4, 4, 4, 7, np.nan, 7],      # g3 future not fully observed -> censored, not negative
            [0, 0, 0, 4, 4, 4, 2, 2, 2]]           # g4 shrinks -> negative (closure < 0), not an error
    p = panel_with(rows)
    feats = pd.DataFrame({"gln": p.glns, "peer_benchmark": 10.0})
    lab = labels_peer_closure(p, "2026-06", feats, ALT).set_index("gln")
    assert lab.loc["g0", "alt_positive"] == 1.0 and lab.loc["g0", "alt_closure"] == 0.5
    assert lab.loc["g1", "alt_positive"] == 0.0
    assert np.isnan(lab.loc["g2", "alt_positive"]) and lab.loc["g2", "alt_status"] == "censored_or_no_gap"
    assert np.isnan(lab.loc["g3", "alt_positive"])
    assert lab.loc["g4", "alt_positive"] == 0.0 and lab.loc["g4", "alt_closure"] < 0


def test_peer_closure_label_is_immature_without_a_matured_future():
    p = panel_with([[1, 1, 1, 1, 1, 1, 1]])
    lab = labels_peer_closure(p, "2026-06", pd.DataFrame({"gln": p.glns, "peer_benchmark": 10.0}), ALT)
    assert (lab["alt_status"] == "immature").all() and lab["alt_positive"].isna().all()


def test_truth_topk_ranks_by_score_skips_unscored_and_breaks_ties_deterministically():
    df = pd.DataFrame({"gln": list("abcdef"), "score": [0.9, 0.8, np.nan, 0.8, 0.1, 0.5],
                       "_tiebreak": [0.1, 0.9, 0.5, 0.2, 0.3, 0.4],
                       "underserved": [True, False, True, True, False, False],
                       "realizable_gap": [10.0, 1.0, 99.0, 6.0, 0.0, 2.0]})
    out = truth_topk(df, "score", prevalence=0.5, ks=(2, 3))
    assert out["n_ranked"] == 5                                            # the unscored GLN "c" is never ranked
    assert out["top2"]["underserved_precision"] == 0.5                     # a (0.9) then b (0.8, higher tiebreak)
    assert out["top3"]["underserved_precision"] == 2 / 3                   # + d (0.8, lower tiebreak)
    assert out["top2"]["lift"] == 1.0 and out["top3"]["mean_realizable_gap"] == (10 + 1 + 6) / 3
    assert truth_topk(df, "score", prevalence=None, ks=(2,))["top2"]["lift"] is None
