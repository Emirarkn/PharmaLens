"""validate_contracts.py [--batch <id>] --report <path>

Row accounting per batch (read = accepted + quarantined), staging key uniqueness of masters,
identifier formats, and that a blank optional numeric input stayed NULL between the raw file and
staging (compared on the stored raw file, per batch). Exit 2 when there is nothing to validate;
a check that examined zero rows says so in its detail and in ``rows_checked``."""
from __future__ import annotations

import json
from pathlib import Path

from _common import check, finish, main_guard, parser, settings

from pharmacy_potential.db import ACCEPTED_STATES, connect, table_exists
from pharmacy_potential.ingest.loader import read_raw

REQUIRED_DATASETS = ("pharmacy_master", "product_master")
FORMAT_CHECKS = (("movement_mutabakat", "stg_movement_mutabakat", "gln", r"^\d{13}$"),
                 ("movement_vendor", "stg_movement_vendor", "gln", r"^\d{13}$"),
                 ("pharmacy_master", "stg_pharmacy_master", "gln", r"^\d{13}$"),
                 ("product_master", "stg_product_master", "barcode", r"^\d{8,14}$"))


def blank_preservation(con, s, batch_id: str, dataset: str, stored_path: str) -> list[dict]:
    """raw blanks == staging NULLs + blanks among quarantined rows, for every optional numeric column."""
    src = s.source_config(dataset)
    raw = read_raw(Path(stored_path), src)
    quarantined = [json.loads(r[0]) for r in con.execute(
        "select raw_record from quarantine where batch_id = ?", [batch_id]).fetchall()]
    out = []
    for field, spec in src["columns"].items():
        if spec.get("required") or spec["type"] not in ("int", "decimal"):
            continue
        col = spec["source"]
        raw_blank = int((raw[col].astype("string").str.strip().fillna("") == "").sum()) if col in raw else len(raw)
        q_blank = sum(1 for r in quarantined if str(r.get(col, "")).strip() == "")
        staged_null = con.execute(f"select count(*) from {src['stage_table']} where _batch_id = ? and {field} is null",
                                  [batch_id]).fetchone()[0]
        out.append({"batch": batch_id, "field": field, "raw_blank": raw_blank, "quarantined_blank": q_blank,
                    "staged_null": int(staged_null), "ok": raw_blank == staged_null + q_blank})
    return out


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--batch", default=None)
    a = p.parse_args()
    s = settings(a)
    if not s.warehouse_path.exists():
        raise FileNotFoundError(s.warehouse_path)
    con = connect(s.warehouse_path)
    where = "where batch_id = ?" if a.batch else ""
    params = [a.batch] if a.batch else []
    rows = con.execute(f"select batch_id, dataset, state, rows_read, rows_accepted, rows_quarantined, "
                       f"row_count_declared, stored_path from ingestion_batch {where}", params).fetchall()
    if a.batch and not rows:
        raise FileNotFoundError(f"batch {a.batch}")
    if not rows:
        raise FileNotFoundError("no ingestion batches in the warehouse: nothing to validate")
    accepted = [r for r in rows if r[2] in ACCEPTED_STATES]
    delivered = {r[1] for r in accepted}
    if not a.batch:
        missing = [d for d in REQUIRED_DATASETS if d not in delivered]
        if missing:
            raise FileNotFoundError(f"required master dataset(s) without an accepted batch: {missing}")
    checks, rows_checked = [], {}

    bad_acc = [r[0] for r in rows if r[2] not in ("rejected", "duplicate") and r[3] != (r[4] or 0) + (r[5] or 0)]
    checks.append(check("row_accounting read=accepted+quarantined", not bad_acc,
                        {"batches_checked": len(rows), "failing": bad_acc[:10]}))
    bad_decl = [r[0] for r in rows if r[6] is not None and r[3] is not None and r[6] != r[3]]
    checks.append(check("row_count_matches_manifest", not bad_decl, {"batches_checked": len(rows), "failing": bad_decl[:10]}))

    for dataset, table, col, rx in FORMAT_CHECKS:
        name = f"{table}.{col} format"
        if dataset not in delivered:
            rows_checked[name] = 0
            if not a.batch:
                checks.append(check(name, True, "0 rows checked: dataset not delivered"))
            continue
        if not table_exists(con, table):
            checks.append(check(name, False, f"accepted batch exists but staging table {table} is missing"))
            continue
        n, bad = con.execute(f"select count(*), count(case when not regexp_full_match({col}, '{rx}') then 1 end) "
                             f"from {table}").fetchone()
        rows_checked[name] = int(n)
        checks.append(check(name, bad == 0, f"{n} rows checked, {bad} invalid"))
    if table_exists(con, "dim_pharmacy"):
        n, u = con.execute("select count(*), count(distinct gln) from dim_pharmacy").fetchone()
        checks.append(check("dim_pharmacy one valid row per GLN", n == u, f"{n} rows / {u} GLN"))

    # Null vs zero, evidenced on the raw file: a blank optional measure must reach staging as NULL.
    preservation, unreadable = [], []
    for r in accepted:
        try:
            preservation += blank_preservation(con, s, r[0], r[1], r[7])
        except OSError:
            unreadable.append(r[0])
    bad_blank = [x for x in preservation if not x["ok"]]
    rows_checked["blank_optional_measures_stay_null"] = len(preservation)
    if unreadable:
        checks.append(check("blank_optional_measures_stay_null", None, f"raw file missing for {unreadable[:10]}"))
    elif not preservation:
        checks.append(check("blank_optional_measures_stay_null", True, "0 optional numeric columns checked"))
    else:
        checks.append(check("blank_optional_measures_stay_null", not bad_blank,
                            {"columns_checked": len(preservation), "failing": bad_blank[:10]}))
    finish(a.report, "validate_contracts", checks,
           {"batches_checked": len(rows), "rows_checked": rows_checked,
            "datasets_without_accepted_batch": sorted({"movement_mutabakat", "movement_vendor"} - delivered)})


if __name__ == "__main__":
    main_guard(main)
