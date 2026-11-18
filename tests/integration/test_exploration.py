""": an exploration-only run (real data, G1 pending, --exploratory) trains and scores nothing."""
from __future__ import annotations

import pandas as pd
import pytest

from pharmacy_potential import pipeline
from pharmacy_potential.db import connect
from pharmacy_potential.utils import read_json

pytestmark = pytest.mark.integration


def test_exploration_run_produces_quality_and_track_a_but_trains_no_model(small_universe, monkeypatch):
    s = small_universe
    trained = []
    monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: trained.append("context"))
    monkeypatch.setattr(pipeline, "run_backtest", lambda *a, **k: trained.append("backtest"))
    monkeypatch.setattr(pipeline, "check_gates", lambda *a, **k: {
        "synthetic": False, "exploratory": True, "mode": "exploration_only", "tuning_allowed": False,
        "gates": {"G0": "accepted", "G1": "pending", "G2": "pending", "G3": "pending", "G4": "pending"}})
    res = pipeline.run_scoring(s, exploratory=True, run_label="expl_1")
    assert trained == [] and res["mode"] == "exploration_only" and res["excel"] is None
    rep, art = res["reports"], res["artifacts"]
    man = read_json(rep / "manifest.json")
    assert man["run_kind"] == "exploration" and man["models_trained"] is False and man["exploratory"] is True
    assert "EXPLORATORY" in read_json(rep / "quality.json")["label"]
    assert (rep / "reconciliation.csv").exists() and (rep / "coverage.csv").exists()
    ext = pd.read_parquet(art / "external_track_a.parquet")
    assert "a_score_0_100" in ext and ext["gln"].is_unique
    # No sales-aware artefact of any kind.
    assert not list(art.glob("gbm_q*.txt")) and not (art / "score_result.parquet").exists()
    assert not list((s.outputs_root / "expl_1").glob("*.xlsx"))
    con = connect(s.warehouse_path)
    try:
        assert con.execute("select run_kind, status from score_run where run_id = 'expl_1'").fetchone() == \
            ("exploration", "succeeded")
    finally:
        con.close()
