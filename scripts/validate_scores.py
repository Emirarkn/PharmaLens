"""validate_scores.py --run <id> --report <path>

Unique GLN x form x track, bounded scores, eligibility vs null gaps, total/form allocation
consistency, track A ties per district, no coordinates in outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import check, finish, main_guard, parser, run_dir, settings


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    s = settings(a)
    run_dir(s, a.run)
    path = s.artifacts_root / a.run / "score_result.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    r = pd.read_parquet(path)
    checks = [check("unique_gln_form_track", not r.duplicated(["gln", "form_or_total", "track"]).any())]
    num = r.select_dtypes("number")
    checks.append(check("no_infinite_values", not np.isinf(num.to_numpy(dtype=float)).any()))
    sc = r["score_0_100"].dropna()
    checks.append(check("score_bounded_0_100", bool(((sc >= 0) & (sc <= 100)).all())))
    tb = r[(r["track"] == "B_sales_aware") & (r["form_or_total"] == "TOTAL")]
    nob = tb[tb["baseline_units_per_month"].isna()]
    checks.append(check("no_gap_without_observed_baseline", bool(nob["incremental_gap_units_per_month"].isna().all()
                                                                  and nob["rank"].isna().all()), f"{len(nob)} GLN"))
    checks.append(check("gap_nonnegative", bool((tb["incremental_gap_units_per_month"].dropna() >= -1e-9).all())))
    fr = r[(r["track"] == "B_sales_aware") & (r["form_or_total"] != "TOTAL")]
    alloc = fr.groupby("gln")["incremental_gap_units_per_month"].sum(min_count=1)
    tot = tb.set_index("gln")["incremental_gap_units_per_month"]
    both = pd.concat([tot.rename("total"), alloc.rename("alloc")], axis=1).dropna()
    checks.append(check("form_allocation_sums_to_total_gap", bool(np.allclose(both["total"], both["alloc"], atol=1e-6)),
                        f"{len(both)} GLN compared"))
    # Null / eligibility contract on EVERY track-B row, TOTAL and all forms.
    b = r[r["track"] == "B_sales_aware"]
    null_gap = b["incremental_gap_units_per_month"].isna()
    leak = b[null_gap & (b["eligibility"].eq("eligible") | b["rank"].notna() | b["score_0_100"].notna())]
    checks.append(check("null_gap_rows_are_not_eligible_and_unranked", leak.empty, f"{len(leak)} rows"))
    nogap = b[~null_gap & ~b["eligibility"].eq("eligible")]
    checks.append(check("non_null_gap_rows_are_eligible", nogap.empty, f"{len(nogap)} rows"))
    unknown_total = set(tb.loc[tb["incremental_gap_units_per_month"].isna(), "gln"])
    cascade = fr[fr["gln"].isin(unknown_total) & (fr["incremental_gap_units_per_month"].notna()
                                                  | fr["rank"].notna() | fr["score_0_100"].notna())]
    checks.append(check("unknown_total_gap_nulls_all_form_rows", cascade.empty, f"{len(cascade)} rows"))
    unk = fr[fr["form_or_total"] == "UNKNOWN"]
    checks.append(check("unknown_form_never_numeric",
                        bool(unk["incremental_gap_units_per_month"].isna().all() and unk["rank"].isna().all()
                             and unk["eligibility"].eq("form_unmapped").all()), f"{len(unk)} rows"))
    ta = r[r["track"] == "A_external_only"]
    ties = ta.groupby("district_code")["score_0_100"].nunique(dropna=True)
    checks.append(check("track_A_tied_within_district", bool((ties <= 1).all())))
    checks.append(check("no_coordinates", not any(c in r.columns for c in ("lat", "lon", "latitude", "longitude"))))
    sc_ok = (r["scenario_units_per_month_low"] <= r["scenario_units_per_month_base"]) & \
            (r["scenario_units_per_month_base"] <= r["scenario_units_per_month_high"])
    checks.append(check("scenario_low_le_base_le_high", bool(sc_ok[r["scenario_units_per_month_base"].notna()].all())))
    q = r[["forecast_next3m_p10", "forecast_next3m_p50", "forecast_next3m_p90"]].dropna()
    checks.append(check("forecast_quantiles_ordered", bool(((q.iloc[:, 0] <= q.iloc[:, 1]) & (q.iloc[:, 1] <= q.iloc[:, 2])).all())))
    finish(a.report, "validate_scores", checks, {"run_id": a.run, "rows": int(len(r))})


if __name__ == "__main__":
    main_guard(main)
