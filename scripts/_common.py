"""Shared helpers for validation scripts. Exit codes: 0 pass, 1 validation fail, 2 config/input unavailable.
A skipped critical check is never reported as pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharmacy_potential.config import load_settings  # noqa: E402
from pharmacy_potential.utils import read_json, utc_now, write_json  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--report", required=True, help="Machine-readable JSON report path")
    return p


def settings(args):
    return load_settings(args.config, data_root=args.data_root)


def run_dir(s, run_id: str) -> Path:
    d = s.reports_root / run_id
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"run {run_id} has no manifest at {d}")
    return d


def finish(report_path: str, name: str, checks: list[dict[str, Any]], extra: dict | None = None) -> None:
    failed = [c for c in checks if c["status"] == "fail"]
    skipped = [c for c in checks if c["status"] == "skipped"]
    status = "fail" if failed else ("incomplete" if skipped else "pass")
    payload = {"validator": name, "status": status, "checked_at": utc_now(), "checks": checks, **(extra or {})}
    write_json(Path(report_path), payload)
    print(f"{name}: {status} ({len(checks) - len(failed) - len(skipped)} pass, {len(failed)} fail, {len(skipped)} skipped)"
          f" -> {report_path}")
    for c in failed:
        print(f"  FAIL {c['check']}: {c.get('detail', '')}")
    sys.exit(1 if failed or skipped else 0)


def check(name: str, ok: bool | None, detail: Any = "") -> dict[str, Any]:
    return {"check": name, "status": "skipped" if ok is None else ("pass" if ok else "fail"), "detail": detail}


def main_guard(fn: Callable[[], None]) -> None:
    try:
        fn()
    except FileNotFoundError as e:
        print(f"input unavailable: {e}", file=sys.stderr)
        sys.exit(2)


__all__ = ["parser", "settings", "run_dir", "finish", "check", "main_guard", "read_json", "ROOT"]
