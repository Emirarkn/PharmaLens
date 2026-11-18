"""Versioned Excel upload templates, generated from the source mappings.

A template is not a second contract: its columns ARE ``configs/sources/<dataset>.yaml``. The template version is a
hash of that mapping, so any change to the mapping yields a new version and an old template is recognisable as old.
Identifier and month/date columns are formatted as text (a GLN keeps its leading zero). The example sheet is
labelled synthetic and is never read by ingestion; the data sheet ("Data") is read by the normal load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import xlsxwriter

from ..config import Settings
from ..utils import sha256_text

TEMPLATE_SCHEMA = "v001"
TEXT_TYPES = {"gln", "barcode", "code", "string", "month", "date"}
FORMAT_NOTE = {
    "gln": "13-digit number, as TEXT (leading zero preserved; scientific notation rejected)",
    "barcode": "8-14 digit number, as TEXT",
    "month": "month",
    "date": "date",
    "int": "integer",
    "decimal": "decimal number",
    "code": "code (text)",
    "string": "text",
}


def template_version(settings: Settings, dataset: str) -> str:
    text = (settings.sources_dir / f"{dataset}.yaml").read_text(encoding="utf-8")
    return f"{TEMPLATE_SCHEMA}-{sha256_text(text)[:8]}"


def _example_value(ctype: str, fmt: dict[str, Any], i: int) -> str:
    dec = fmt.get("decimal", ".")
    return {
        "gln": f"0{i:012d}", "barcode": f"869000000{i:04d}", "code": f"CODE{i}", "string": f"EXAMPLE {i}",
        "month": (fmt.get("month_formats") or ["%Y-%m"])[0].replace("%Y", "2026").replace("%m", "01"),
        "date": (fmt.get("date_format") or "%Y-%m-%d").replace("%Y", "2026").replace("%m", "01").replace("%d", "15"),
        "int": str(3 + i), "decimal": f"{10 + i}{dec}50",
    }[ctype]


def build_template(settings: Settings, dataset: str, out_dir: Path) -> Path:
    src = settings.source_config(dataset)
    ver = template_version(settings, dataset)
    fmt = src.get("file", {})
    cols = list(src["columns"].values())
    out = Path(out_dir) / f"template_{dataset}_{ver}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(out), {"strings_to_numbers": False, "strings_to_formulas": False,
                                        "strings_to_urls": False})
    head = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    req = wb.add_format({"bold": True, "bg_color": "#FCE4D6", "border": 1})
    text = wb.add_format({"num_format": "@"})
    warn = wb.add_format({"bold": True, "font_color": "#9C0006", "text_wrap": True})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})

    data = wb.add_worksheet("Data")                          # first sheet: the default sheet of the loader
    for j, spec in enumerate(cols):
        data.write_string(0, j, spec["source"], req if spec.get("required") else head)
        data.set_column(j, j, max(14, len(spec["source"]) + 2), text if spec["type"] in TEXT_TYPES else None)
    data.freeze_panes(1, 0)

    readme = wb.add_worksheet("Description")
    readme.set_column(0, 0, 28)
    readme.set_column(1, 4, 34)
    lines = [("Template", f"{dataset}  |  version {ver}"), ("Source name", src["source_name"]),
             ("Usage", "Only the 'Data' sheet is read. Column names must NOT be changed. An empty cell is NOT zero (it stays missing)."),
             ("Revision", "A correction of an earlier delivery is uploaded as a new file and points to the old one in the manifest "
                          "via `supersedes_batch_id`; the old record is never deleted or overwritten."),
             ("Re-upload", "A file with identical content uploaded a second time has no effect (counted as a duplicate)."),
             ("Decimal separator / date", f"decimal '{fmt.get('decimal', '.')}'; date format {fmt.get('date_format', '%Y-%m-%d')}; "
                                        f"month formats {fmt.get('month_formats', ['%Y-%m'])}"),
             ("Key", ", ".join(src.get("key", []))),
             ("Unknowns", "Required fields, enum/status values and the freshness contract are finalized with a real sample and "
                               "the data owner. This template only reflects the current mapping.")]
    readme.write_string(0, 0, "WARNING: uploading real data requires G0 (access and authorization) to be accepted.", warn)
    for i, (k, v) in enumerate(lines, start=2):
        readme.write_string(i, 0, k, head)
        readme.write_string(i, 1, v, wrap)
    r0 = len(lines) + 4
    for j, h in enumerate(("Column (on the Data sheet)", "Field", "Type / format", "Required")):
        readme.write_string(r0, j, h, head)
    for i, (field, spec) in enumerate(src["columns"].items(), start=r0 + 1):
        readme.write_string(i, 0, spec["source"])
        readme.write_string(i, 1, field)
        readme.write_string(i, 2, FORMAT_NOTE[spec["type"]])
        readme.write_string(i, 3, "yes" if spec.get("required") else "no")

    ex = wb.add_worksheet("Example_SYNTHETIC")
    ex.write_string(0, 0, "SYNTHETIC EXAMPLE: made-up values, not imported (this sheet is not read) and does not represent a real pharmacy.", warn)
    for j, spec in enumerate(cols):
        ex.write_string(1, j, spec["source"], head)
        ex.set_column(j, j, max(14, len(spec["source"]) + 2), text)
        for i in (1, 2):
            ex.write_string(1 + i, j, _example_value(spec["type"], fmt, i))
    wb.close()
    return out


def available_templates(settings: Settings) -> list[str]:
    return sorted(p.stem for p in settings.sources_dir.glob("*.yaml"))
