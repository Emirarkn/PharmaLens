"""PharmaLens - local, read-only viewer of finished runs. Start it with ``ppi ui`` (binds to 127.0.0.1 only).

It reads the artifacts a run already wrote (scores, reports, SHAP); it never ingests, trains or writes data.
Paths arrive through environment variables set by the ``ppi ui`` command.
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pharmacy_potential.ui import data as D

ACCENT = "#0F766E"
FOOTNOTE = ("Scores are percentile indexes, not probabilities. Depot-to-pharmacy outflow is a supply-side signal, not "
            "consumer sales. The output shows association, not causation.")
RANK_COLUMNS = ["rank", "gln", "city_code", "district_code", "form_or_total", "score_0_100", "baseline_units_per_month",
                "benchmark_units_per_month", "incremental_gap_units_per_month", "forecast_next3m_p50", "readiness",
                "confidence", "eligibility", "reason_1"]
PAGES = ["Overview", "Rankings", "Pharmacy", "Data quality", "Evaluation"]


def roots() -> D.Roots:
    base = Path(os.environ.get("PHARMALENS_ROOT", ".")).resolve()
    return D.Roots(reports=Path(os.environ.get("PHARMALENS_REPORTS", base / "reports")),
                   artifacts=Path(os.environ.get("PHARMALENS_ARTIFACTS", base / "artifacts")),
                   outputs=Path(os.environ.get("PHARMALENS_OUTPUTS", base / "outputs")))


@st.cache_data(show_spinner="Loading run...")
def load_scores(reports: str, artifacts: str, outputs: str, run_id: str) -> pd.DataFrame:
    return D.read_scores(D.Roots(Path(reports), Path(artifacts), Path(outputs)), run_id)


def empty(message: str, hint: str = "") -> None:
    st.info(message + (f"\n\n{hint}" if hint else ""))


def banner(manifest: dict) -> None:
    gates = manifest.get("gates") or {}
    if manifest.get("synthetic", gates.get("synthetic", True)):
        st.warning("**Synthetic demo.** No row is a real pharmacy or sale. Results make no performance claim.", icon=":material/science:")
    if manifest.get("exploratory"):
        st.warning("**Exploratory run.** No model was trained; there is no track-B score.", icon=":material/explore:")


def page_overview(r: D.Roots, run_id: str, scores: pd.DataFrame, manifest: dict) -> None:
    st.header("Overview")
    banner(manifest)
    tot = D.select_track(scores, "B_total")
    ext = D.select_track(scores, "A_external")
    eligible = int((tot["eligibility"] == "eligible").sum()) if len(tot) else 0
    c = st.columns(4)
    c[0].metric("Score month", str(scores["score_month"].iloc[0]))
    c[1].metric("Pharmacies (track B)", f"{len(tot):,}")
    c[2].metric("Eligible for a sales gap", f"{eligible:,}", help="Pharmacies with enough observed history. Others get no invented sales gap.")
    c[3].metric("Pharmacies (track A)", f"{ext['gln'].nunique():,}")
    acc = D.read_json(r, run_id, "acceptance.json") or {}
    champ = manifest.get("champion")
    ranking_champ = champ.get("ranking_champion", "-") if isinstance(champ, dict) else (champ or "-")
    c = st.columns(3)
    c[0].metric("Ranking champion", str(ranking_champ), help="Chosen on validation origins only, and it must beat the simple baselines.")
    c[1].metric("G3 proposal", str(acc.get("decision_proposal", "-")), help="A machine-readable proposal. Only the owner accepts a gate.")
    c[2].metric("Data cutoff (UTC)", str(manifest.get("cutoff_utc", "-"))[:10])
    st.subheader("Gates")
    raw_gates = manifest.get("gates") or {}
    gate_map = raw_gates.get("gates", raw_gates)          # flat {"G0": "pending"} (older nested form also accepted)
    gate_map = {k: v for k, v in gate_map.items() if isinstance(v, str)}
    if gate_map:
        st.dataframe(pd.DataFrame({"gate": list(gate_map), "status": list(gate_map.values())}), hide_index=True, width="content")
    else:
        empty("No gate information in this run's manifest.")
    st.subheader("How to read this")
    st.markdown(
        "- **Track B** ranks pharmacies by the gap between their observed sales and what comparable pharmacies sell, "
        "combined with external demand and a model growth signal.\n"
        "- **Track A** uses external demand only (population, pharmacy density, scenario); it reads no sales, so pharmacies "
        "in the same district tie.\n"
        "- An **empty** value means *not observed*, never zero.")
    wb = D.workbook_path(r, run_id)
    if wb:
        st.download_button("Download Excel workbook", wb.read_bytes(), file_name=wb.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Run manifest"):
        st.json({k: manifest[k] for k in ("run_id", "code_version", "origin_month", "cutoff_utc", "selection_policy",
                                          "seed", "champion", "synthetic", "exploratory") if k in manifest})


def page_rankings(scores: pd.DataFrame) -> None:
    st.header("Rankings")
    track = st.radio("Track", list(D.TRACKS), format_func=D.TRACKS.get, horizontal=True)
    df = D.select_track(scores, track)
    if df.empty:
        empty("This run has no rows for this track.", "An exploratory run has no track-B rows.")
        return
    with st.expander("Filters", expanded=True):
        a, b, c = st.columns(3)
        cities = a.multiselect("City code", sorted(df["city_code"].astype(str).unique()))
        pool = df[df["city_code"].astype(str).isin(cities)] if cities else df
        districts = b.multiselect("District code", sorted(pool["district_code"].astype(str).unique()))
        elig = c.multiselect("Eligibility", sorted(df["eligibility"].dropna().unique()))
        a, b, c = st.columns(3)
        ready = a.multiselect("Readiness", sorted(df["readiness"].dropna().unique()),
                              help="Supply-network evidence. It never changes the rank.")
        forms = b.multiselect("Form", sorted(df["form_or_total"].unique())) if track == "B_form" else []
        gln_q = c.text_input("GLN contains")
        top_n = st.slider("Rows shown", 10, 500, 100, 10)
    view = D.ranking_view(D.filter_scores(df, cities=cities, districts=districts, eligibility=elig, readiness=ready,
                                          forms=forms, gln_contains=gln_q), top_n)
    st.caption(f"{len(view):,} of {len(df):,} rows. Rank 1 = highest priority; an empty rank means the pharmacy could not be ranked.")
    if view.empty:
        empty("No rows match these filters.", "Clear a filter to see more.")
        return
    cols = [c for c in RANK_COLUMNS if c in view.columns and not (c == "form_or_total" and track != "B_form")]
    ev = st.dataframe(view[cols], hide_index=True, on_select="rerun", selection_mode="single-row",
                      column_config={"score_0_100": st.column_config.ProgressColumn("score (0-100)", min_value=0, max_value=100, format="%.1f"),
                                     "rank": st.column_config.NumberColumn("rank", format="%d"),
                                     "reason_1": st.column_config.TextColumn("main reason", width="large")})
    sel = ev.selection.rows
    if sel:
        st.session_state["gln"] = str(view.iloc[sel[0]]["gln"])
        st.success(f"Selected GLN {st.session_state['gln']}. Open the **Pharmacy** page for details.")
    st.download_button("Download shown rows (CSV)", view[cols].to_csv(index=False).encode("utf-8"),
                       file_name=f"pharmalens_{track}.csv", mime="text/csv")


def page_pharmacy(r: D.Roots, run_id: str, scores: pd.DataFrame) -> None:
    st.header("Pharmacy")
    gln = st.text_input("GLN", value=st.session_state.get("gln", ""), help="13 digits, kept as text (leading zeros matter).")
    if not gln.strip():
        empty("Enter a GLN, or select a row on the Rankings page.")
        return
    rows = D.pharmacy_rows(scores, gln)
    if rows.empty:
        empty(f"No rows for GLN {gln.strip()} in this run.", "Check the number and that it is the same run.")
        return
    tot = rows[(rows["track"] == "B_sales_aware") & (rows["form_or_total"] == "TOTAL")]
    ext = rows[rows["track"] == "A_external_only"]
    main = tot.iloc[0] if len(tot) else None
    st.subheader(f"GLN {gln.strip()}")
    if main is not None:
        c = st.columns(4)
        c[0].metric("Rank (track B)", "-" if pd.isna(main["rank"]) else f"{int(main['rank']):,}")
        c[1].metric("Score", "-" if pd.isna(main["score_0_100"]) else f"{main['score_0_100']:.1f}")
        c[2].metric("Confidence", str(main["confidence"]), help="Low confidence does not mean low opportunity.")
        c[3].metric("Readiness", str(main["readiness"]), help="Supply evidence only; it never changes the rank.")
        c = st.columns(3)
        fmt = lambda v: "not observed" if pd.isna(v) else f"{v:,.1f}"
        c[0].metric("Observed baseline (boxes/month)", fmt(main["baseline_units_per_month"]))
        c[1].metric("Peer benchmark (boxes/month)", fmt(main["benchmark_units_per_month"]))
        c[2].metric("Comparative gap (boxes/month)", fmt(main["incremental_gap_units_per_month"]), help="Not the return of an intervention.")
        st.markdown("**Why this rank**")
        for col in ("reason_1", "reason_2", "reason_3"):
            if isinstance(main.get(col), str) and main[col]:
                st.markdown(f"- {main[col]}")
        ml = [main.get(c) for c in ("ml_reason_1", "ml_reason_2", "ml_reason_3") if isinstance(main.get(c), str) and main.get(c)]
        if ml:
            st.markdown("**Model contributions (SHAP)**")
            for t in ml:
                st.markdown(f"- {t}")
        fc = pd.DataFrame({"quantile": ["p10", "p50", "p90"],
                           "boxes": [main["forecast_next3m_p10"], main["forecast_next3m_p50"], main["forecast_next3m_p90"]]}).dropna()
        if len(fc):
            st.markdown("**Forecast, next 3 months total**")
            st.altair_chart(alt.Chart(fc).mark_bar(color=ACCENT).encode(
                x=alt.X("quantile:N", title=None, sort=["p10", "p50", "p90"], axis=alt.Axis(labelAngle=0)), y=alt.Y("boxes:Q", title="boxes (3-month total)"),
                tooltip=["quantile", alt.Tooltip("boxes:Q", format=",.0f")]).properties(height=220), width="stretch")
        sh = D.shap_for(r, run_id, gln)
        if sh is not None:
            st.markdown("**Largest feature contributions (log scale)**")
            st.altair_chart(alt.Chart(sh).mark_bar().encode(
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
                x=alt.X("shap:Q", title="contribution to the p50 forecast"),
                color=alt.condition(alt.datum.shap > 0, alt.value(ACCENT), alt.value("#B45309")),
                tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")]).properties(height=30 * len(sh)), width="stretch")
    else:
        empty("This pharmacy has no track-B row (not eligible or exploratory run).")
    if len(ext):
        st.markdown("**Track A (external demand only)**")
        e = ext.iloc[0]
        st.write(f"Score {e['score_0_100']:.1f}, tie group `{e['tie_group']}` "
                 f"(pharmacies in the same group share this score; order inside a group is not an opportunity difference).")
    with st.expander("All rows for this GLN"):
        st.dataframe(rows.drop(columns=["run_id"], errors="ignore"), hide_index=True)


def page_quality(r: D.Roots, run_id: str) -> None:
    st.header("Data quality")
    q = D.read_json(r, run_id, "quality.json")
    if q is None:
        empty("No quality report for this run.")
        return
    if q.get("label"):
        st.warning(q["label"])
    if q.get("ingest"):
        st.subheader("Ingestion by dataset")
        st.dataframe(pd.DataFrame(q["ingest"]), hide_index=True)
    if q.get("quarantine"):
        st.subheader("Quarantine by error code")
        qa = pd.DataFrame({"error_code": list(q["quarantine"]), "rows": list(q["quarantine"].values())})
        st.bar_chart(qa, x="error_code", y="rows", color=ACCENT)
        st.caption("Rejected rows are kept, never silently dropped. They limit where a zero may be inferred.")
    cov = D.read_csv(r, run_id, "coverage.csv")
    if cov is not None and len(cov):
        st.subheader("Source coverage")
        st.dataframe(cov.groupby(["source", "coverage_status"], as_index=False)["scopes"].sum(), hide_index=True)
    rec = D.read_csv(r, run_id, "reconciliation.csv")
    if rec is not None and len(rec):
        st.subheader("Reconciliation of the two sources")
        st.dataframe(rec, hide_index=True)
    asof = D.read_json(r, run_id, "asof_audit.json")
    if asof:
        with st.expander("As-of audit (what was known at each cutoff)"):
            st.json(asof)
    rest = {k: v for k, v in q.items() if k not in ("ingest", "quarantine", "label")}
    if rest:
        with st.expander("Other quality metrics"):
            st.json(rest)


def page_evaluation(r: D.Roots, run_id: str) -> None:
    st.header("Evaluation")
    acc = D.read_json(r, run_id, "acceptance.json")
    md = D.read_text(r, run_id, "evaluation.md")
    if acc is None and md is None:
        empty("No evaluation for this run.", "Exploratory runs and runs with --skip-backtest do not produce one.")
        return
    if acc:
        st.metric("G3 decision proposal", str(acc.get("decision_proposal", "-")))
        st.caption("Backtest labels are proxies (sustained excess growth versus the district). This is not evidence of a causal effect or ROI.")
        checks = acc.get("checks") or {}
        if checks:
            st.dataframe(pd.DataFrame([{"check": k, "pass": v.get("pass"), "value": v.get("value"), "target": v.get("target")}
                                       for k, v in checks.items()]), hide_index=True)
    if md:
        with st.expander("Full evaluation report", expanded=False):
            st.markdown(md)
    png = D.image_path(r, run_id, "shap_global.png")
    imp = D.read_csv(r, run_id, "shap_global_importance.csv")
    if imp is not None and len(imp):
        st.subheader("What drives the model overall")
        top = imp.head(15)
        st.altair_chart(alt.Chart(top).mark_bar(color=ACCENT).encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)), x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log scale)"),
            tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")]).properties(height=26 * len(top)), width="stretch")
    elif png:
        st.image(str(png), caption="Global SHAP importance")
    truth = D.read_json(r, run_id, "synthetic_truth_check.json")
    if truth:
        with st.expander("Synthetic truth check (possible only because the data is synthetic)"):
            st.json(truth)


def main() -> None:
    st.set_page_config(page_title="PharmaLens", page_icon=":material/search_insights:", layout="wide")
    r = roots()
    st.sidebar.title("PharmaLens")
    st.sidebar.caption("Explainable pharmacy opportunity viewer")
    runs = D.list_runs(r)
    if not runs:
        st.title("PharmaLens")
        empty("No finished run found.", f"Looked in `{r.reports}` and `{r.artifacts}`. Create one with `ppi demo --fresh`, then reload.")
        return
    run_id = st.sidebar.selectbox("Run", runs, help="Newest first.")
    page = st.sidebar.radio("Page", PAGES)
    manifest = D.read_json(r, run_id, "manifest.json") or {}
    try:
        scores = load_scores(str(r.reports), str(r.artifacts), str(r.outputs), run_id)
    except Exception as e:  # unreadable run folder: say what and how to recover, keep the sidebar usable
        st.error(f"Could not read run `{run_id}`: {e}. Pick another run or re-run the pipeline.")
        return
    if page == "Overview":
        page_overview(r, run_id, scores, manifest)
    elif page == "Rankings":
        page_rankings(scores)
    elif page == "Pharmacy":
        page_pharmacy(r, run_id, scores)
    elif page == "Data quality":
        page_quality(r, run_id)
    else:
        page_evaluation(r, run_id)
    st.divider()
    st.caption(FOOTNOTE)


main()
