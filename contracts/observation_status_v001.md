# Observation status and source selection — v001 (DRAFT)

Policy file: `configs/selection_policy_v001.yaml` (`approval_status: draft_unapproved`).

## Cell (GLN × month × barcode) statuses

| Status | When | `net_units` |
|---|---|---|
| `observed_positive` | Selected source > 0 | value |
| `observed_zero` | Selected source = 0 **or** reconciliation coverage is COMPLETE and there is no record | 0 |
| `observed_net_negative` | Net returns > sales | negative value |
| `unobserved` | No eligible source (coverage unknown/incomplete, and absent from VENDOR too) | NULL |
| `out_of_scope` | Pharmacy or product is not active that month | NULL |
| `quarantined` | Conflicting duplicate records exist and there is no fallback | NULL |

## Selection order
1. If **all** depots linked to the pharmacy in that month declare reconciliation coverage `COMPLETE` and the cell has no conflict → **reconciliation** (no record means 0).
2. Otherwise, if VENDOR has a conflict-free record → **VENDOR fallback**. A missing VENDOR record is never counted as zero.
3. Otherwise → `unobserved` (or `quarantined` if there is a conflict).

Sources are **never summed or averaged**. A large difference (`|m−i| > 3` boxes and > 30%) is reported through `conflict_flag` and does not change the decision.

## Duplicates within a source
- Duplicates whose measurements are all identical → a single row (counted in `identical_repeats_removed`).
- Same key with different measurements → all rows are quarantined (`conflicting_keys`); the cell cannot be selected from that source.
- A proven finer grain (reconciliation `depot_code`) → aggregated within the source.
- Revisions arrive only through `supersedes_batch_id` in the manifest; a new file never silently overwrites an old one. If the revision had not yet arrived at the as-of cutoff, the old value is used.

## Availability (as-of)
- `batch` mode: a row is known at its delivery time (`available_at`).
- `retrospective_lag` mode (historical backfill): a row is **assumed** known at period end + `retrospective_lag_days` (demo: 45 days); such backtests are labelled "retrospective".
- If an explicit `available_at` column exists (visit record date, campaign announcement date, population publication date), it is used.
- Master files have no history, so the current snapshot is used retrospectively (flagged in `asof_audit.json`).

## GLN × month total status (for features)
`observed` if every in-scope barcode cell is observed; `partial` if only some are (a lower bound only, counted as missing in features); `unobserved` if none are.
