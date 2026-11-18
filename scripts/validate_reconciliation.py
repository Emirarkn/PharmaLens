"""validate_reconciliation.py --run <id> --report <path>

On the run's canonical snapshot: exactly one chosen source per observed cell, value equals that
source's own measurement (no sums/averages of sources), lineage present, PK unique, zero only with
complete coverage, policy version recorded."""
from __future__ import annotations

import duckdb

from _common import check, finish, main_guard, parser, run_dir, settings


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--policy", default=None, help="Expected policy version (defaults to config)")
    a = p.parse_args()
    s = settings(a)
    run_dir(s, a.run)
    path = s.data_root / "canonical" / a.run / "fact_canonical_movement.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    con = duckdb.connect()
    con.execute(f"create view c as select * from read_parquet('{path.as_posix()}')")
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    expected_policy = a.policy or s.policy["policy_version"]
    obs = "observation_status in ('observed_positive','observed_zero','observed_net_negative')"
    checks = [
        check("pk_unique", q("select count(*) - count(distinct (gln, month, barcode)) from c") == 0),
        check("observed_cells_have_one_source", q(f"select count(*) from c where {obs} and chosen_source is null") == 0),
        check("value_equals_chosen_source_measurement", q("""
            select count(*) from c where (chosen_source = 'mutabakat' and net_units <> coalesce(alt_mutabakat_net_units, 0))
                                      or (chosen_source = 'vendor' and net_units <> alt_vendor_net_units)""") == 0),
        check("never_sum_of_sources", q("""select count(*) from c where alt_mutabakat_net_units is not null
              and alt_vendor_net_units is not null and alt_vendor_net_units <> 0 and alt_mutabakat_net_units <> 0
              and net_units = alt_mutabakat_net_units + alt_vendor_net_units""") == 0),
        check("lineage_present", q(f"""select count(*) from c where {obs} and selected_source_row_ids is null
              and selection_reason <> 'primary_complete_coverage_no_record_zero'""") == 0),
        check("zero_without_record_only_with_complete_coverage", q("""select count(*) from c
              where selection_reason = 'primary_complete_coverage_no_record_zero' and primary_coverage <> 'COMPLETE'""") == 0),
        check("unobserved_has_null_units", q("select count(*) from c where observation_status = 'unobserved' "
                                             "and net_units is not null") == 0),
        check("policy_version_recorded", q(f"select count(*) from c where selection_policy_version <> '{expected_policy}'") == 0),
    ]
    stats = dict(con.execute("select observation_status, count(*) from c group by 1").fetchall())
    finish(a.report, "validate_reconciliation", checks, {"run_id": a.run, "status_counts": stats})


if __name__ == "__main__":
    main_guard(main)
