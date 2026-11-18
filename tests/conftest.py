"""Shared fixtures. Hand-made fixtures (S01-S06, S10-S12) are tiny CSVs written in the exact
source formats; scenario tests on dynamics (S07-S09, S13, S15, S16) use a small synthetic universe."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pharmacy_potential.config import load_settings
from pharmacy_potential.db import connect
from pharmacy_potential.ingest.loader import ingest_file
from pharmacy_potential.ingest.manifest import build_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]

G1, G2, G3 = "0123456789012", "8680000000002", "8680000000003"   # G1 has a leading zero on purpose
GUM, PATCH = "8690000123101", "8690000123201"


def make_settings(tmp: Path, **extra):
    return load_settings(ROOT / "configs" / "pilot.yaml", data_root=str(tmp / "data"),
                         reports_root=str(tmp / "reports"), outputs_root=str(tmp / "outputs"),
                         artifacts_root=str(tmp / "artifacts"), **extra)


class Fixture:
    """Writes source-format CSV + manifest and ingests it through the real loader."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.settings = make_settings(tmp, history_start_month="2026-01")
        self.con = connect(self.settings.warehouse_path)
        self.n = 0

    def add(self, dataset: str, rows: list[dict], *, period=("2026-01-01", "2026-01-31"),
            available="2026-02-15T00:00:00Z", revision=1, supersedes=None, name: str | None = None):
        src = self.settings.source_config(dataset)
        sep = src["file"].get("delimiter", ",")
        self.n += 1
        path = self.tmp / "in" / dataset / (name or f"{dataset}_{self.n:03d}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = [spec["source"] for spec in src["columns"].values()]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter=sep, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        m = build_manifest(dataset, src["source_name"], path, period_start=period[0], period_end=period[1],
                           revision=revision, supersedes_batch_id=supersedes, confidentiality="synthetic",
                           license_id="fixture", owner="tests", received_at_utc=available, available_at=available,
                           synthetic=True)
        write_manifest(m, path)
        return ingest_file(self.settings, self.con, dataset, path)

    def masters(self):
        self.add("pharmacy_master", [
            {"GLN": G1, "EczaneAdi": "=HYPERLINK(\"http://x\")", "IlKodu": "34", "IlceKodu": "34001", "AcilisTarihi": "01.01.2015"},
            {"GLN": G2, "EczaneAdi": "Pharmacy B", "IlKodu": "34", "IlceKodu": "34001", "AcilisTarihi": "01.01.2015"},
            {"GLN": G3, "EczaneAdi": "Pharmacy C", "IlKodu": "34", "IlceKodu": "34002", "AcilisTarihi": "01.01.2015"},
        ], period=("2021-01-01", "2026-01-31"), available="2021-01-01T00:00:00Z")
        self.add("product_master", [
            {"Barkod": GUM, "Marka": "DemoBrand", "Form": "GUM", "BaslangicTarihi": "2019-01-01"},
            {"Barkod": PATCH, "Marka": "DemoBrand", "Form": "PATCH", "BaslangicTarihi": "2019-01-01"},
        ], period=("2021-01-01", "2026-01-31"), available="2021-01-01T00:00:00Z")
        self.add("depot_master", [
            {"depot_code": "D01", "level": "PARENT"}, {"depot_code": "D02", "level": "PARENT"},
            {"depot_code": "D01S1", "parent_depot_code": "D01", "level": "SUB"},
            {"depot_code": "D01S2", "parent_depot_code": "D01", "level": "SUB"},
        ], period=("2021-01-01", "2026-01-31"), available="2021-01-01T00:00:00Z")
        self.add("pharmacy_depot", [
            {"GLN": G1, "DepoKodu": "D01"}, {"GLN": G2, "DepoKodu": "D01"}, {"GLN": G2, "DepoKodu": "D02"},
            {"GLN": G3, "DepoKodu": "D02"},
        ], period=("2021-01-01", "2026-01-31"), available="2021-01-01T00:00:00Z")
        return self

    def coverage(self, statuses: dict[str, str], month="2026-01", **kw):
        rows = [{"source": "MUTABAKAT", "month": month, "scope_type": "DEPOT", "scope_id": d, "status": s}
                for d, s in statuses.items()]
        rows.append({"source": "VENDOR", "month": month, "scope_type": "PANEL", "scope_id": "ALL", "status": "UNKNOWN"})
        return self.add("coverage_register", rows, **kw)


@pytest.fixture
def fx(tmp_path):
    f = Fixture(tmp_path).masters()
    yield f
    f.con.close()


@pytest.fixture(scope="session")
def small_universe(tmp_path_factory):
    """~200 GLN synthetic universe, generated and ingested once per test session."""
    from pharmacy_potential.pipeline import ingest_all
    from pharmacy_potential.synth.generator import generate
    tmp = tmp_path_factory.mktemp("universe")
    s = make_settings(tmp)
    generate(s, scale=0.05)
    con = connect(s.warehouse_path)
    ingest_all(s, con, s.data_root / "raw" / "synthetic")
    con.close()
    return s
