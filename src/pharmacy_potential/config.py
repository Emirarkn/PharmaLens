"""Configuration loading and path resolution.

All paths in configs are relative to the project root (the parent of ``configs/``),
so no user-specific path is ever embedded in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    root: Path
    config_path: Path
    raw: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        return self.raw.get(key, default)

    def _path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def data_root(self) -> Path:
        return self._path(self.overrides.get("data_root") or self.raw["paths"]["data_root"])

    @property
    def reports_root(self) -> Path:
        return self._path(self.overrides.get("reports_root") or self.raw["paths"]["reports_root"])

    @property
    def outputs_root(self) -> Path:
        return self._path(self.overrides.get("outputs_root") or self.raw["paths"]["outputs_root"])

    @property
    def artifacts_root(self) -> Path:
        return self._path(self.overrides.get("artifacts_root") or self.raw["paths"]["artifacts_root"])

    @property
    def warehouse_path(self) -> Path:
        return self.data_root / self.raw["paths"]["warehouse_file"]

    @property
    def sources_dir(self) -> Path:
        return self._path(self.raw["sources_dir"])

    def source_config(self, dataset: str) -> dict[str, Any]:
        path = self.sources_dir / f"{dataset}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No source mapping for dataset '{dataset}' at {path}")
        return load_yaml(path)

    @property
    def policy(self) -> dict[str, Any]:
        return load_yaml(self._path(self.raw["selection_policy"]))

    @property
    def thresholds(self) -> dict[str, Any]:
        return load_yaml(self._path(self.raw["thresholds"]))

    @property
    def alt_thresholds(self) -> dict[str, Any] | None:
        return load_yaml(self._path(self.raw["alt_thresholds"])) if self.raw.get("alt_thresholds") else None

    @property
    def scenario(self) -> dict[str, Any]:
        return load_yaml(self._path(self.raw["scenario"]))

    def effective_config(self) -> dict[str, Any]:
        """Parsed content of every config file plus the overrides in force (frozen into run manifests)."""
        files = {p.relative_to(self.root).as_posix(): load_yaml(p) for p in self.config_files()}
        return {"files": files, "overrides": {k: str(v) if isinstance(v, Path) else v
                                              for k, v in sorted(self.overrides.items())}}

    def config_files(self) -> list[Path]:
        """All config files that define a run (hashed into run manifests)."""
        files = [self.config_path, self._path(self.raw["selection_policy"]),
                 self._path(self.raw["thresholds"]), self._path(self.raw["scenario"])]
        if self.raw.get("alt_thresholds"):
            files.append(self._path(self.raw["alt_thresholds"]))
        files += sorted(self.sources_dir.glob("*.yaml"))
        return files


def load_settings(config: str | Path = "configs/pilot.yaml", **overrides: Any) -> Settings:
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    root = config_path.parent.parent
    return Settings(root=root, config_path=config_path, raw=load_yaml(config_path),
                    overrides={k: v for k, v in overrides.items() if v is not None})
