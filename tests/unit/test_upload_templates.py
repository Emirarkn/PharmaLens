"""Versioned templates, Excel upload through the ONE ingestion chain, preview without writes,
error report. Acceptance examples: leading-zero GLN stays text, bad file gives a clear error, same file twice adds
nothing, revision keeps history, synthetic-only example sheet."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import openpyxl
import pytest
import yaml

from pharmacy_potential.ingest.loader import ingest_file
from pharmacy_potential.ingest.manifest import build_manifest, write_manifest
from pharmacy_potential.ingest.preview import preview_file, write_error_report
from pharmacy_potential.ingest.templates import available_templates, build_template, template_version
from tests.conftest import G1, G2, GUM, PATCH

MUT_COLS = ["GLN", "Donem", "Barkod", "DepoKodu", "SatisAdet", "MFAdet", "IadeAdet", "NetTutarKDVDahil"]


def fill(template: Path, rows: list[list], out: Path) -> Path:
    """What a user does: open the template, type rows into 'Data' (GLN/barcode as text), save as another file."""
    wb = openpyxl.load_workbook(template)
    ws = wb["Data"]
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            cell = ws.cell(r, c, v)
            if isinstance(v, str):
                cell.data_type = "s"            # a typed text cell, even when it starts with '=' (not a formula)
            if c in (1, 3):
                cell.number_format = "@"
    wb.save(out)
    return out


def manifest_for(fx, dataset, path, **kw):
    src = fx.settings.source_config(dataset)
    m = build_manifest(dataset, src["source_name"], path, period_start="2026-01-01", period_end="2026-01-31",
                       confidentiality="synthetic", license_id="fixture", owner="tests", synthetic=True,
                       received_at_utc="2026-02-15T00:00:00Z", available_at="2026-02-15T00:00:00Z", **kw)
    write_manifest(m, path)
    return m


def test_template_for_every_mapped_dataset_has_the_mapping_columns_and_a_labelled_example(fx, tmp_path):
    names = available_templates(fx.settings)
    assert {"movement_mutabakat", "movement_vendor", "pharmacy_master", "visits"} <= set(names)
    for name in names:
        wb = openpyxl.load_workbook(build_template(fx.settings, name, tmp_path / "t"))
        assert wb.sheetnames == ["Data", "Description", "Example_SYNTHETIC"]            # 'Data' first: the loader's default
        cols = [s["source"] for s in fx.settings.source_config(name)["columns"].values()]
        assert [c.value for c in wb["Data"][1]] == cols
        assert wb["Data"].max_row == 1                                              # blank: no invented values
        assert "SYNTHETIC" in wb["Example_SYNTHETIC"]["A1"].value
        assert wb["Data"]["A2"].number_format == "@" or wb["Data"].column_dimensions["A"].width >= 14


def test_template_version_follows_the_mapping(fx, tmp_path):
    v1 = template_version(fx.settings, "movement_mutabakat")
    assert v1.startswith("v001-") and v1 == template_version(fx.settings, "movement_mutabakat")
    other = tmp_path / "cfg"
    shutil.copytree(fx.settings.sources_dir, other)
    y = yaml.safe_load((other / "movement_mutabakat.yaml").read_text(encoding="utf-8"))
    y["columns"]["units_paid"]["source"] = "SatisAdet2"
    (other / "movement_mutabakat.yaml").write_text(yaml.safe_dump(y), encoding="utf-8")
    fx.settings.raw["sources_dir"] = str(other)
    assert template_version(fx.settings, "movement_mutabakat") != v1


def test_filled_template_loads_through_the_normal_chain_and_keeps_leading_zero_text(fx, tmp_path):
    fx.coverage({"D01": "COMPLETE", "D02": "COMPLETE"})
    tpl = build_template(fx.settings, "movement_mutabakat", tmp_path / "t")
    f = fill(tpl, [[G1, "2026-01", GUM, "D01", 10, 2, 1, "1234,50"], [G2, "2026-01", PATCH, "D02", 4, None, None, None]],
             tmp_path / "upload.xlsx")
    r = ingest_file(fx.settings, fx.con, "movement_mutabakat", f, manifest_for(fx, "movement_mutabakat", f))
    assert r.state == "validated" and (r.rows_read, r.rows_accepted, r.rows_quarantined) == (2, 2, 0)
    assert fx.con.execute("select gln from stg_movement_mutabakat where barcode = ?", [GUM]).fetchone()[0] == G1  # '0123...'
    # Blank optional cells stay NULL, never 0 (acceptance: missing is not zero).
    row = fx.con.execute("select units_free, returns_units, net_value_vat_inc_try from stg_movement_mutabakat "
                         "where barcode = ?", [PATCH]).fetchone()
    assert row == (None, None, None)


def test_same_excel_file_twice_adds_nothing_and_a_revision_keeps_the_original(fx, tmp_path):
    tpl = build_template(fx.settings, "movement_mutabakat", tmp_path / "t")
    f1 = fill(tpl, [[G1, "2026-01", GUM, "D01", 10, 0, 0, None]], tmp_path / "a.xlsx")
    r1 = ingest_file(fx.settings, fx.con, "movement_mutabakat", f1, manifest_for(fx, "movement_mutabakat", f1))
    copy = tmp_path / "a_copy.xlsx"
    shutil.copy(f1, copy)
    again = ingest_file(fx.settings, fx.con, "movement_mutabakat", copy, manifest_for(fx, "movement_mutabakat", copy))
    assert again.state == "duplicate"
    assert fx.con.execute("select count(*) from stg_movement_mutabakat").fetchone()[0] == 1
    f2 = fill(tpl, [[G1, "2026-01", GUM, "D01", 12, 0, 0, None]], tmp_path / "b.xlsx")
    r2 = ingest_file(fx.settings, fx.con, "movement_mutabakat", f2,
                     manifest_for(fx, "movement_mutabakat", f2, revision=2, supersedes_batch_id=r1.batch_id))
    assert r2.state == "validated"
    states = dict(fx.con.execute("select batch_id, state from ingestion_batch").fetchall())
    assert states[r1.batch_id] == "validated"                                      # original kept, not overwritten


def test_preview_matches_the_real_load_and_writes_nothing(fx, tmp_path):
    tpl = build_template(fx.settings, "movement_mutabakat", tmp_path / "t")
    f = fill(tpl, [[G1, "2026-01", GUM, "D01", 10, 0, 0, None],
                   ["868", "2026-01", GUM, "D01", 3, 0, 0, None],                 # bad GLN
                   [G2, "2026-13", GUM, "D01", 3, 0, 0, None],                    # bad month
                   [G2, "2026-01", GUM, "D01", "abc", 0, 0, None],                # bad number
                   [G1, "2026-01", GUM, "D01", 10, 0, 0, None]],                  # same key + same values as row 1
             tmp_path / "mixed.xlsx")
    before = fx.con.execute("select (select count(*) from ingestion_batch), (select count(*) from quarantine)").fetchone()
    pv = preview_file(fx.settings, fx.con, "movement_mutabakat", f)
    assert fx.con.execute("select (select count(*) from ingestion_batch), (select count(*) from quarantine)").fetchone() == before
    assert (pv["rows_read"], pv["rows_accepted"], pv["rows_rejected"]) == (5, 2, 3)
    assert set(pv["error_counts"]) == {"INVALID_GLN", "INVALID_MONTH", "INVALID_NUMBER"}
    assert pv["duplicate_key_rows"] == 2 and pv["already_ingested_as"] is None and pv["ok"]
    r = ingest_file(fx.settings, fx.con, "movement_mutabakat", f, manifest_for(fx, "movement_mutabakat", f))
    assert (r.rows_read, r.rows_accepted, r.rows_quarantined) == (pv["rows_read"], pv["rows_accepted"], pv["rows_rejected"])
    assert preview_file(fx.settings, fx.con, "movement_mutabakat", f)["already_ingested_as"] == r.batch_id


def test_error_report_lists_rejected_rows_with_reasons_as_plain_text(fx, tmp_path):
    tpl = build_template(fx.settings, "movement_mutabakat", tmp_path / "t")
    f = fill(tpl, [[G1, "2026-01", GUM, "D01", 10, 0, 0, None], ["868", "2026-01", GUM, "=HYPERLINK(\"http://x\")", 3, 0, 0, None]],
             tmp_path / "bad.xlsx")
    pv = preview_file(fx.settings, fx.con, "movement_mutabakat", f)
    rep = write_error_report(pv, tmp_path / "errors.xlsx")
    ws = openpyxl.load_workbook(rep)["Hatali_satirlar"]
    header = [c.value for c in ws[1]]
    assert header[:2] == ["source_row", "error"] and ws.max_row == 2
    row = dict(zip(header, [c.value for c in ws[2]]))
    assert row["source_row"] == "2" and "INVALID_GLN" in row["error"]
    assert row["DepoKodu"].startswith("=") and ws.cell(2, header.index("DepoKodu") + 1).data_type == "s"   # text, not a formula


def test_missing_required_column_is_a_clear_fatal_error(fx, tmp_path):
    f = tmp_path / "wrong.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Data"
    wb.active.append(["GLN", "Donem"])
    wb.save(f)
    pv = preview_file(fx.settings, fx.con, "movement_mutabakat", f)
    assert not pv["ok"] and "Missing required columns" in pv["fatal"]


def test_csv_delivery_still_loads_the_same_way(fx, tmp_path):
    """The Excel path is an addition: a CSV delivery is untouched (and read as before)."""
    f = tmp_path / "d.csv"
    with open(f, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";", lineterminator="\n")
        w.writerow(MUT_COLS)
        w.writerow([G1, "2026-01", GUM, "D01", "10", "0", "0", ""])
    r = ingest_file(fx.settings, fx.con, "movement_mutabakat", f, manifest_for(fx, "movement_mutabakat", f))
    assert r.rows_accepted == 1


def test_cli_template_writes_files_and_preview_of_a_real_file_needs_g0(fx, tmp_path):
    from typer.testing import CliRunner

    from pharmacy_potential.cli import app
    cfg = str(Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml")
    res = CliRunner().invoke(app, ["template", "--dataset", "pharmacy_master", "--out", str(tmp_path / "o"), "--config", cfg])
    assert res.exit_code == 0 and list((tmp_path / "o").glob("template_pharmacy_master_v001-*.xlsx"))
    assert CliRunner().invoke(app, ["template", "--config", cfg]).exit_code == 2           # nothing asked for
    f = fill(build_template(fx.settings, "pharmacy_master", tmp_path / "t"), [[G1, "A", "34", "34001", "01.01.2015"]],
             tmp_path / "pm.xlsx")
    # The repository's G0 is pending, and this file is not under the synthetic demo folder.
    res = CliRunner().invoke(app, ["preview", "--dataset", "pharmacy_master", "--path", str(f), "--config", cfg,
                                   "--data-root", str(fx.settings.data_root)])
    assert res.exit_code == 2 and "G0" in res.output
