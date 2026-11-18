"""Gate enforcement for real (non-synthetic) data.

Gate files live in ``docs/gates``. No code path ever writes ``accepted``: acceptance is the owner's.

| Step                                             | Gates that must be ``accepted`` (real data) |
|--------------------------------------------------|---------------------------------------------|
| profile / ingest / reconcile / quality reports   | G0                                          |
| score + model training (fixed settings)          | G0, G1                                      |
| backtest / hyper-parameter selection / final test| G0, G1, G2                                  |

``--exploratory`` waives G1 only and never trains a model: it yields data quality, reconciliation and the
external (track A) output.
"""
from __future__ import annotations

import re
from typing import Any

import duckdb

from .config import Settings

STEP_GATES = {"data": ("G0",), "score": ("G0", "G1"), "backtest": ("G0", "G1", "G2")}


class GateBlocked(Exception):
    pass


def gate_status(settings: Settings, gate: str) -> str:
    path = settings.root / "docs" / "gates" / f"{gate}.md"
    if not path.exists():
        return "missing"
    m = re.search(r"^status:\s*(\w+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1).lower() if m else "unknown"


def require_gates(settings: Settings, step: str, what: str) -> None:
    """Raise GateBlocked unless every gate the step needs is accepted (callers skip this for synthetic data)."""
    missing = {g: gate_status(settings, g) for g in STEP_GATES[step] if gate_status(settings, g) != "accepted"}
    if missing:
        raise GateBlocked(f"{what} on real (non-synthetic) data needs {', '.join(STEP_GATES[step])} accepted in "
                          f"docs/gates; not accepted: {missing}. Only the owner can accept a gate.")


def data_is_synthetic(con: duckdb.DuckDBPyConnection) -> bool:
    row = con.execute("select count(*), count(case when synthetic then 1 end) from ingestion_batch "
                      "where state in ('validated', 'validated_with_quarantine')").fetchone()
    return row[0] > 0 and row[0] == row[1]


def has_real_batches(con: duckdb.DuckDBPyConnection) -> bool:
    return bool(con.execute("select count(*) from ingestion_batch where state <> 'rejected' "
                            "and synthetic is not true").fetchone()[0])


def check_gates(settings: Settings, con: duckdb.DuckDBPyConnection, exploratory: bool,
                skip_backtest: bool = False) -> dict[str, Any]:
    """Decide what a scoring run may do. ``mode``: full | untuned | exploration_only."""
    synthetic = data_is_synthetic(con)
    status = {g: gate_status(settings, g) for g in ("G0", "G1", "G2", "G3", "G4")}
    if synthetic:
        return {"synthetic": True, "exploratory": bool(exploratory), "gates": status, "mode": "full",
                "tuning_allowed": True}
    if status["G0"] != "accepted":
        raise GateBlocked(f"Real (non-synthetic) batches present but G0 is {status['G0']}, not accepted "
                          "(access and rights). --exploratory does not waive G0.")
    unknown_rights = con.execute("""select count(*) from ingestion_batch where state <> 'rejected'
                                    and (owner = 'unknown' or license_id = 'unknown')""").fetchone()[0]
    if unknown_rights:
        raise GateBlocked(f"{unknown_rights} batch(es) have unknown owner/license although G0 is accepted; "
                          "fix the manifests. --exploratory does not waive this.")
    if status["G1"] != "accepted":
        if not exploratory:
            raise GateBlocked("Real batches present but G1 is not accepted. --exploratory produces data-quality, "
                              "reconciliation and external (track A) output only; no model is trained.")
        return {"synthetic": False, "exploratory": True, "gates": status, "mode": "exploration_only",
                "tuning_allowed": False}
    tuning = status["G2"] == "accepted"
    if not tuning and not skip_backtest:
        raise GateBlocked("Backtest and hyper-parameter selection on real data need G2 accepted (pilot protocol). "
                          "Run with --skip-backtest to score with fixed, untuned model settings.")
    return {"synthetic": False, "exploratory": bool(exploratory), "gates": status,
            "mode": "full" if tuning else "untuned", "tuning_allowed": tuning}
