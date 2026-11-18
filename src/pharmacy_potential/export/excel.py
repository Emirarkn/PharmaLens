"""Excel workbook for channel managers.

Safety: identifiers are written as text (leading zeros kept), text cells that could be read as
formulas are neutralised, null stays blank while zero stays 0, no hidden sheets, row limit checked.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xlsxwriter

EXCEL_MAX_ROWS = 1_048_576
TEXT_COLUMNS = {"gln", "barcode", "district_code", "city_code", "brick_code", "run_id", "score_month"}
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_text(value: Any) -> str:
    """Neutralise spreadsheet formula injection (OWASP CSV/Excel injection)."""
    s = str(value)
    return "'" + s if s.startswith(FORMULA_PREFIXES) else s


def _is_null(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    try:
        return isinstance(v, (float, np.floating)) and not math.isfinite(v)   # NaN/inf -> blank, never an error cell
    except TypeError:
        return False


class Book:
    def __init__(self, path: Path, banner: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.wb = xlsxwriter.Workbook(str(path), {"strings_to_formulas": False, "strings_to_urls": False,
                                                  "strings_to_numbers": False, "nan_inf_to_errors": False})
        self.banner = banner
        self.f_banner = self.wb.add_format({"bold": True, "font_color": "#B00020", "font_size": 14})
        self.f_head = self.wb.add_format({"bold": True, "bg_color": "#DDE6F0", "border": 1, "text_wrap": True,
                                          "valign": "top"})
        self.f_num = self.wb.add_format({"num_format": "0.00"})
        self.f_int = self.wb.add_format({"num_format": "0"})
        self.f_text = self.wb.add_format({"num_format": "@"})
        self.f_wrap = self.wb.add_format({"text_wrap": True, "valign": "top"})
        self.f_bold = self.wb.add_format({"bold": True})

    def sheet(self, name: str):
        ws = self.wb.add_worksheet(name)
        ws.write_string(0, 0, self.banner, self.f_banner)
        ws.set_header('&C&"Calibri,Bold"&14&KB00020' + self.banner.replace("&", "&&"))
        return ws

    def table(self, name: str, df: pd.DataFrame, widths: dict[str, int] | None = None, start_row: int = 2) -> None:
        if len(df) + start_row + 1 > EXCEL_MAX_ROWS:
            raise ValueError(f"Sheet {name}: {len(df)} rows exceed the Excel row limit; split the export")
        ws = self.sheet(name)
        cols = list(df.columns)
        for j, c in enumerate(cols):
            ws.write_string(start_row, j, c, self.f_head)
            ws.set_column(j, j, (widths or {}).get(c, max(10, min(40, len(c) + 2))))
        values = df.to_numpy(dtype=object)
        for i, row in enumerate(values, start=start_row + 1):
            for j, v in enumerate(row):
                c = cols[j]
                if _is_null(v):
                    ws.write_blank(i, j, None)
                elif c in TEXT_COLUMNS:
                    ws.write_string(i, j, safe_text(v), self.f_text)
                elif isinstance(v, (bool, np.bool_)):
                    ws.write_boolean(i, j, bool(v))
                elif isinstance(v, (int, np.integer)):
                    ws.write_number(i, j, int(v), self.f_int)
                elif isinstance(v, (float, np.floating)):
                    ws.write_number(i, j, float(v), self.f_num)
                else:
                    ws.write_string(i, j, safe_text(v))
        ws.freeze_panes(start_row + 1, 0)
        if len(df):
            ws.autofilter(start_row, 0, start_row + len(df), len(cols) - 1)

    def text_sheet(self, name: str, lines: list[tuple[str, str]]) -> None:
        ws = self.sheet(name)
        ws.set_column(0, 0, 34)
        ws.set_column(1, 1, 120)
        for i, (k, v) in enumerate(lines, start=2):
            ws.write_string(i, 0, safe_text(k), self.f_bold if v == "" else None)
            ws.write_string(i, 1, safe_text(v), self.f_wrap)

    def close(self) -> None:
        self.wb.close()


RANKING_COLUMNS = ["run_id", "score_month", "gln", "pharmacy_name", "city_code", "district_code", "brick_code", "track",
                   "form_or_total", "score_0_100", "rank", "tie_group", "eligibility", "cohort", "observed_months_12",
                   "observation_status_last", "baseline_units_per_month", "benchmark_units_per_month",
                   "incremental_gap_units_per_month", "scenario_units_per_month_low", "scenario_units_per_month_base",
                   "scenario_units_per_month_high", "readiness", "confidence", "confidence_score", "reason_1", "reason_2",
                   "reason_3", "source_policy", "data_cutoff_utc", "model_tuning"]

COLUMN_DOCS = {
    "model_tuning": "tuned = hyperparameters selected on validation; untuned_engineering_default = fixed engineering "
                    "default, not calibrated (not G3 evidence).",
    "score_0_100": "Relative percentile index (0-100). NOT a probability.",
    "rank": "1 = highest priority. In track A, equal scores within a district share the same (min) rank.",
    "tie_group": "Group sharing the same external score (track A: district). Order within a group does not indicate an opportunity difference.",
    "eligibility": "eligible / no_observed_baseline / no_external_data. No fake sales gap is produced for an unobserved pharmacy.",
    "cohort": "low_sales = baseline below the province 25th percentile at the cutoff; no_observed_baseline = fewer than 2 observations in the last 3 months.",
    "baseline_units_per_month": "Observed mean of the last 3 months of depot->pharmacy net boxes/month (not consumer sales). Empty = not observed.",
    "benchmark_units_per_month": "District peer median (excluding itself), shrunk to province level, scaled by external demand; boxes/month.",
    "incremental_gap_units_per_month": "max(0, benchmark - baseline) boxes/month. A comparative gap, not the return of an intervention.",
    "raw_form_gap": "Raw gap by form (can be negative). The reconciled allocation is in the incremental_gap column.",
    "scenario_units_per_month_low": "External scenario demand (low) boxes/month/pharmacy - not a measurement, an uncalibrated scenario range.",
    "scenario_units_per_month_base": "External scenario demand (base) boxes/month/pharmacy.",
    "scenario_units_per_month_high": "External scenario demand (high) boxes/month/pharmacy.",
    "forecast_next3m_point": "Next-3-month TOTAL boxes forecast by the champion method (not monthly).",
    "forecast_next3m_p10": "LightGBM quantile p10, next-3-month total boxes (lower prediction bound).",
    "forecast_next3m_p50": "LightGBM quantile p50, next-3-month total boxes.",
    "forecast_next3m_p90": "LightGBM quantile p90, next-3-month total boxes (upper prediction bound).",
    "readiness": "Supply-network evidence: high / limited / unknown. NOT a claim that the pharmacy has stock; it does not suppress the score.",
    "confidence": "Coverage, freshness, source consistency, peer size, interval width, similarity to the training domain. Low confidence does not mean low opportunity.",
    "reason_1": "Rule-based rationale (A: external; B: peer/external/model/observation).",
    "ml_reason_1": "SHAP: largest contributions in the p50 model (log scale). A model contribution, not a causal explanation.",
    "observed_months_12": "Number of months in the last 12 in which all cells were observed.",
    "source_policy": "Source selection policy version (reconciliation first, VENDOR fallback; never summed).",
    "data_cutoff_utc": "Data that became available after this date was not used.",
}


def export_workbook(path: Path, result: pd.DataFrame, names: pd.DataFrame | None, context: dict[str, Any],
                    synthetic: bool, exploratory: bool) -> Path:
    banner = ("SYNTHETIC DEMO - not real data" if synthetic else "INTERNAL - AUTHORIZED USE") + \
             (" | EXPLORATORY: G1 not accepted" if exploratory else "") + \
             (" | G3 not accepted: awaiting exploration/validation" if context.get("g3_decision") != "accepted" else "") + \
             f" | run {context['run_id']}"
    r = result.copy()
    if names is not None:
        r = r.merge(names, on="gln", how="left")
    else:
        r["pharmacy_name"] = np.nan
        r = r.merge(context["brick"], on="gln", how="left") if "brick" in context else r
    if "brick_code" not in r:
        r["brick_code"] = np.nan

    book = Book(path, banner)
    readme = [
        ("PHARMALENS", ""),
        ("Run", context["run_id"]),
        ("Score month / data cutoff", f"{context['score_month']} / {context['cutoff']}"),
        ("Data type", "SYNTHETIC DEMO - no row is a real pharmacy or sale." if synthetic else "Real data (authorized)"),
        ("Gate status", context.get("gates_text", "")),
        ("Main question", "Which pharmacy, in which form, and for which reasons holds an unmet opportunity worth reviewing?"),
        ("DEFINITIONS", ""),
        ("Movement (sell-out)", "The flow the sources call 'sell-out' is depot->pharmacy outflow; it is not consumer sales (sell-through)."),
        ("Sources", "VENDOR and reconciliation are alternative measurements of the same flow; they are NEVER summed. A per-cell policy selects a single source."),
        ("Empty cell vs 0", "Empty = not observed/unknown. 0 = observed zero with verified coverage. They are different."),
        ("Track A (External_Only)", "External population/pharmacy density + scenario only. Reads no sales, visits, orders or supply; equal within a district (tie)."),
        ("Track B (Ranking_Total)", f"Sales-aware unmet opportunity. Champion method: {context.get('champion')}. "
                                    "It does not copy existing large customers; peer gap + external demand + (if selected) model growth signal."),
        ("Extra boxes", "incremental_gap = max(0, benchmark - baseline) boxes/month: a comparative gap, not the certain return of an intervention."),
        ("Scenario vs forecast", "The scenario range (boxes/month, external, uncalibrated) and the model forecast interval (next-3-month total) are different columns."),
        ("Readiness", "Evidence that the connected depot carries the product; it is not pharmacy stock. It does not silently suppress the score."),
        ("Confidence", "Low confidence does not mean low opportunity; use it for the exploration queue."),
        ("SHAP rationales", "Model contributions (log scale), not a causal explanation."),
        ("Evaluation", "Backtest labels are PROXIES (sustained excess growth versus the district). Not evidence of a causal effect or ROI (needs G5)."),
        *([("Model calibration", "UNCALIBRATED FORECAST: the model was trained with fixed engineering defaults (no hyperparameter "
                                  "selection, G2 not accepted). This run is not used for champion selection or as G3 evidence.")]
          if context.get("model_tuning", "tuned") != "tuned" else []),
        ("Location", "There are no pharmacy coordinates; any map would be at province/district level only. Brick location is not provided."),
        ("Privacy", "Pharmacy name is a restricted field; it appears only in an authorized export. No external sharing."),
        ("Sheets", "Ranking_Total, Ranking_Form, External_Only, Forecast, Explanations, Coverage_Quality, Evaluation, Data_Dictionary"),
    ]
    book.text_sheet("READ_ME", readme)

    def ranking(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in RANKING_COLUMNS if c in df.columns]
        return df[cols]

    tot_b = r[(r["track"] == "B_sales_aware") & (r["form_or_total"] == "TOTAL")].sort_values(
        ["rank", "gln"], na_position="last", kind="mergesort")
    book.table("Ranking_Total", ranking(tot_b), {"reason_1": 50, "reason_2": 45, "reason_3": 40, "gln": 16})
    form_b = r[(r["track"] == "B_sales_aware") & (r["form_or_total"] != "TOTAL")].sort_values(
        ["form_or_total", "rank", "gln"], na_position="last", kind="mergesort")
    fcols = ranking(form_b).columns.tolist()
    if "raw_form_gap" in form_b and "incremental_gap_units_per_month" in fcols:
        fcols.insert(fcols.index("incremental_gap_units_per_month") + 1, "raw_form_gap")
    book.table("Ranking_Form", form_b[fcols], {"reason_1": 55})
    ext = r[r["track"] == "A_external_only"].sort_values(["rank", "district_code", "gln"], kind="mergesort")
    book.table("External_Only", ranking(ext), {"reason_1": 55, "reason_2": 40, "reason_3": 50})
    fc = tot_b[["gln", "pharmacy_name", "city_code", "district_code", "baseline_units_per_month", "forecast_method",
                "forecast_next3m_point", "forecast_next3m_p10", "forecast_next3m_p50", "forecast_next3m_p90",
                "confidence"]].sort_values("gln")
    book.table("Forecast", fc)
    ex = tot_b[["gln", "pharmacy_name", "rank", "ml_reason_1", "ml_reason_2", "ml_reason_3"]]
    book.table("Explanations", ex, {"ml_reason_1": 60, "ml_reason_2": 60, "ml_reason_3": 60})
    if "shap_importance" in context:
        ws = book.wb.get_worksheet_by_name("Explanations")
        col0 = 8
        ws.write_string(2, col0, "Global SHAP importance ranking (mean |SHAP|, log scale)", book.f_head)
        for i, row in enumerate(context["shap_importance"].head(20).itertuples(), start=3):
            ws.write_string(i, col0, row.feature)
            ws.write_number(i, col0 + 1, float(row.mean_abs_shap), book.f_num)
            ws.write_string(i, col0 + 2, row.description or "")
        ws.set_column(col0, col0, 26)
        ws.set_column(col0 + 2, col0 + 2, 70)
        if context.get("shap_png") and Path(context["shap_png"]).exists():
            ws.insert_image(25, col0, str(context["shap_png"]), {"x_scale": 0.8, "y_scale": 0.8})
    book.table("Coverage_Quality", context["quality_table"], {"metric": 50, "value": 40})
    book.table("Evaluation", context["evaluation_table"], {"candidate": 22, "metric": 18})
    dd = [{"column": c, "definition": COLUMN_DOCS.get(c, "")} for c in RANKING_COLUMNS + [
        "raw_form_gap", "forecast_next3m_point", "forecast_next3m_p10", "forecast_next3m_p50", "forecast_next3m_p90",
        "ml_reason_1"]]
    dd += [{"column": f"feature:{k}", "definition": f"{v[1]} [{v[0]}]"} for k, v in context.get("feature_dictionary", {}).items()]
    book.table("Data_Dictionary", pd.DataFrame(dd), {"column": 34, "definition": 110})
    book.close()
    return path
