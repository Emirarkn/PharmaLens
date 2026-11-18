"""Batch manifests: build, validate, locate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from ..utils import parse_utc, read_json, sha256_file, utc_now

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "contracts" / "manifest.schema.json"


def load_schema() -> dict:
    return read_json(SCHEMA_PATH)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(load_schema())
    errors = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in validator.iter_errors(manifest)]
    for key in ("received_at_utc", "available_at"):
        try:
            parse_utc(manifest.get(key, ""))
        except (ValueError, TypeError):
            errors.append(f"{key}: not an ISO-8601 timestamp")
    if manifest.get("period_start", "") > manifest.get("period_end", ""):
        errors.append("period_start after period_end")
    if manifest.get("availability_mode") == "retrospective_lag" and manifest.get("retrospective_lag_days") is None:
        errors.append("retrospective_lag requires retrospective_lag_days")
    return errors


def make_batch_id(dataset: str, period_start: str, period_end: str, revision: int, sha: str) -> str:
    return f"{dataset}__{period_start[:7]}_{period_end[:7]}__r{revision:02d}__{sha[:10]}"


def sidecar_path(data_file: Path) -> Path:
    return data_file.with_name(data_file.name + ".manifest.json")


def build_manifest(dataset: str, source: str, data_file: Path, *, period_start: str, period_end: str,
                   revision: int = 1, supersedes_batch_id: str | None = None,
                   confidentiality: str = "restricted", license_id: str = "unknown", owner: str = "unknown",
                   received_at_utc: str | None = None, available_at: str | None = None,
                   availability_mode: str = "batch", retrospective_lag_days: int | None = None,
                   synthetic: bool = False, schema_version: str = "v001", row_count: int | None = None,
                   notes: str | None = None) -> dict[str, Any]:
    sha = sha256_file(data_file)
    received = received_at_utc or utc_now().isoformat().replace("+00:00", "Z")
    return {
        "batch_id": make_batch_id(dataset, period_start, period_end, revision, sha),
        "dataset": dataset,
        "source": source,
        "file_path": data_file.name,
        "sha256": sha,
        "received_at_utc": received,
        "period_start": period_start,
        "period_end": period_end,
        "schema_version": schema_version,
        "revision": revision,
        "supersedes_batch_id": supersedes_batch_id,
        "confidentiality": confidentiality,
        "license_id": license_id,
        "owner": owner,
        "row_count": row_count,
        "available_at": available_at or received,
        "availability_mode": availability_mode,
        "retrospective_lag_days": retrospective_lag_days,
        "synthetic": synthetic,
        "notes": notes,
    }


def effective_available_from(manifest: dict[str, Any]):
    """Earliest time any row of the batch may be considered known.

    Retrospective backfills are assumed known row by row (see row-level ``_available_at``);
    the batch itself is therefore usable from its period start. Row filters do the precise cut.
    """
    if manifest["availability_mode"] == "retrospective_lag":
        return min(parse_utc(manifest["period_start"] + "T00:00:00Z"), parse_utc(manifest["available_at"]))
    return parse_utc(manifest["available_at"])


def write_manifest(manifest: dict[str, Any], data_file: Path) -> Path:
    path = sidecar_path(data_file)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return path
