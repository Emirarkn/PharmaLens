"""validate_manifest.py --manifest <batch.json> --report <quality.json>

Checks schema, hash against the data file, period, source/revision and owner/license state."""
from __future__ import annotations

from pathlib import Path

from _common import check, finish, main_guard, parser, read_json

from pharmacy_potential.ingest.manifest import validate_manifest
from pharmacy_potential.utils import sha256_file


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    mpath = Path(a.manifest)
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    m = read_json(mpath)
    data = mpath.with_name(mpath.name.replace(".manifest.json", ""))
    errs = validate_manifest(m)
    checks = [check("schema", not errs, errs),
              check("data_file_present", data.exists(), str(data.name))]
    checks.append(check("sha256_matches", sha256_file(data) == m.get("sha256") if data.exists() else None))
    checks.append(check("period_order", m.get("period_start", "") <= m.get("period_end", "")))
    checks.append(check("revision_supersedes_consistent",
                        (m.get("revision", 1) == 1) == (m.get("supersedes_batch_id") in (None, "")),
                        "revision >1 must name the batch it supersedes"))
    rights_known = m.get("owner") not in (None, "", "unknown") and m.get("license_id") not in (None, "", "unknown")
    checks.append(check("owner_and_license_known (G0)", rights_known, f"owner={m.get('owner')} license={m.get('license_id')}"))
    finish(a.report, "validate_manifest", checks, {"batch_id": m.get("batch_id")})


if __name__ == "__main__":
    main_guard(main)
