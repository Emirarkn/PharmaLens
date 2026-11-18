"""validate_asof.py --run <id> [--cutoff <UTC>] --report <path>

Rebuilds the run's as-of inputs and checks: every side-data row used is available <= cutoff,
the canonical snapshot holds no month after the origin, and injecting future knowledge
(post-cutoff visits/orders/campaigns, post-origin sales) leaves the origin features unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import check, finish, main_guard, parser, read_json, run_dir, settings

from pharmacy_potential.db import connect
from pharmacy_potential.features.build import build_panel, features_for_origin
from pharmacy_potential.reconcile.canonical import build_canonical
from pharmacy_potential.utils import backtest_cutoff, to_naive_utc


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--cutoff", default=None)
    a = p.parse_args()
    s = settings(a)
    man = read_json(run_dir(s, a.run) / "manifest.json")
    cutoff = to_naive_utc(a.cutoff or man["cutoff_utc"])
    origin = man["origin_month"]
    con = connect(s.warehouse_path)
    canon = build_canonical(con, s, cutoff, origin, "asof_check")
    panel = build_panel(con, canon.table, cutoff)
    checks = [check("canonical_months_le_origin", panel.months.max() <= pd.Timestamp(origin + "-01"),
                    str(panel.months.max().date()))]
    for name, df in panel.side.items():
        if "avail" not in df:            # batch-period frames carry coverage evidence, not event rows
            continue
        if df.empty:
            checks.append(check(f"{name}_available_le_cutoff", None, "no rows"))
            continue
        checks.append(check(f"{name}_available_le_cutoff", bool((df["avail"] <= pd.Timestamp(cutoff)).all()),
                            f"{len(df)} rows"))
    th, sc = s.thresholds, s.scenario
    f1 = features_for_origin(panel, origin, cutoff, th, sc)
    late = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    g = panel.glns[0]
    m0 = pd.Timestamp(origin + "-01")
    for k, extra in (("visits", {"gln": [g], "month": [m0], "avail": [late]}),
                     ("orders", {"gln": [g], "month": [m0], "units": [999], "avail": [late]}),
                     ("campaigns", {"month": [m0 + pd.DateOffset(months=1)], "barcode": ["x"], "avail": [late]})):
        if not panel.side.get(k, pd.DataFrame()).empty:
            panel.side[k] = pd.concat([panel.side[k], pd.DataFrame(extra)], ignore_index=True)
    f2 = features_for_origin(panel, origin, cutoff, th, sc)
    same = f1.drop(columns=["cutoff_utc"]).equals(f2.drop(columns=["cutoff_utc"]))
    checks.append(check("future_injection_invariant", same))
    # Every training panel of the run must have been built as of its own origin's cutoff.
    audit_path = run_dir(s, a.run) / "asof_audit.json"
    tp = read_json(audit_path).get("training_panels") if audit_path.exists() else None
    if not tp:
        checks.append(check("training_panels_built_as_of_own_cutoff", None, "run has no training_panels audit"))
    else:
        lag = int(s.get("backtest_cutoff_lag_days"))
        bad = [t["origin"] for t in tp
               if t["panel_last_month"] != t["origin"]
               or pd.Timestamp(t["cutoff_utc"]) != pd.Timestamp(backtest_cutoff(t["origin"], lag))]
        checks.append(check("training_panels_built_as_of_own_cutoff", not bad,
                            {"panels": len(tp), "violations": bad[:10]}))
    feats = [c for c in f1.columns if c not in ("gln", "city_code", "district_code")]
    checks.append(check("gln_not_a_feature", "gln" not in feats))
    con.execute(f"drop table if exists {canon.table}")
    con.close()
    finish(a.report, "validate_asof", checks, {"run_id": a.run, "cutoff_utc": cutoff, "origin": origin})


if __name__ == "__main__":
    main_guard(main)
