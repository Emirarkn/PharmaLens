"""Evidence that binds a run to the exact code, config, environment and models that produced it
. Every function degrades to an explicit ``unavailable`` marker, never to a guess."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import Settings
from .utils import sha256_file, sha256_text

KEY_PACKAGES = ("numpy", "pandas", "duckdb", "lightgbm", "scikit-learn", "shap", "pyarrow", "pyyaml")


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(root: Path) -> dict[str, Any]:
    """HEAD commit and a hash of every uncommitted tracked change plus untracked file names."""
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return {"commit": None, "dirty": None, "reason": "git unavailable or not a repository"}
    diff = _git(root, "diff", "HEAD") or ""
    untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    dirty = bool(diff.strip() or untracked.strip())
    return {"commit": head.strip(), "dirty": dirty,
            "dirty_diff_sha256": sha256_text(diff + "\n--untracked--\n" + untracked) if dirty else None}


def source_tree_hash(root: Path) -> str:
    """Hash of every Python source under ``src/`` (works without git, detects any local edit)."""
    h = hashlib.sha256()
    for p in sorted((root / "src").rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def environment_snapshot(root: Path) -> dict[str, Any]:
    dists = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions()
                   if d.metadata["Name"])
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    lock = root / "requirements.lock"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "key_packages": versions,
            "installed_packages_sha256": sha256_text("\n".join(dists)),
            "requirements_lock_sha256": sha256_file(lock) if lock.exists() else None}


def file_hashes(paths: list[Path], base: Path) -> dict[str, str]:
    return {p.relative_to(base).as_posix(): sha256_file(p) for p in sorted(paths) if p.is_file()}


def run_provenance(settings: Settings) -> dict[str, Any]:
    """Everything a later reader needs to decide whether a run can be rebuilt on this checkout."""
    root = settings.root
    effective = settings.effective_config()
    return {
        "git": git_state(root),
        "source_tree_sha256": source_tree_hash(root),
        "environment": environment_snapshot(root),
        "config_file_sha256": file_hashes(settings.config_files(), root),
        "effective_config": effective,
        "effective_config_sha256": sha256_text(json.dumps(effective, sort_keys=True, default=str)),
    }
