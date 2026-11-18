# PharmaLens

**Which pharmacies hold an unmet opportunity for an OTC product, and can we explain why, without trusting dirty data?**

PharmaLens is an explainable decision-support pipeline that ranks pharmacies by unmet sales opportunity. It is built around one idea: **a ranking is only as good as the discipline applied to missing, late and conflicting data.**

> **Status: synthetic-data demo.** No real data is included or processed. Every gate for real data (G0–G5) is `pending`. Demo outputs make no commercial accuracy claim.

## What makes it different

| Principle | What the code does |
|---|---|
| **Missing ≠ 0** | An empty cell is "unknown". A zero is only produced from cell-level evidence (verified coverage, no rejected row that could belong to that cell). |
| **As-of correctness** | Every training origin gets its own panel built from what was known at its own cutoff. Late-arriving data cannot leak into earlier training rows. |
| **Sources are never summed** | A vendor panel and the supply reconciliation measure the same flow. A per-cell policy picks one source and keeps the other as an alternative. |
| **Quarantine, don't drop** | Bad rows go to a quarantine table with an error code. Revisions are explicit (`supersedes`), never silent overwrites. |
| **Explain every rank** | Rule-based reasons plus SHAP contributions per pharmacy. Scores are percentile indexes, not probabilities. |
| **Honest evaluation** | Rolling validation origins, a locked test origin, a holdout city, and a baseline the model must beat before it can be champion. Labels are proxies and say so. |
| **Gated real-data use** | Real data needs owner-accepted gates before ingest, scoring or tuning. No code path ever writes `accepted`. |

## Pipeline

```
raw files ──► ingest ──► reconcile ──► as-of features ──► models ──► backtest ──► score ──► Excel
 (CSV/XLSX)   typed       as-of         peers, lags,       LightGBM    rolling +   two tracks   channel-
 + manifest   staging,    canonical     external demand    quantile,   locked +    (external,   manager
              quarantine, movement,                        Holt-       holdout     sales-aware) workbook
              batch state quality                          Winters,
                          reports                          baselines
```

- **Track A (external only):** district adult population and pharmacy density × scenario demand. Reads no sales.
- **Track B (sales-aware):** peer gap + external demand + model growth signal. Champion chosen on validation origins only.
- **Forecast:** LightGBM quantiles (p10/p50/p90), vectorised Holt-Winters, seasonal-naive and moving-average baselines.
- **Outputs:** Excel workbook (README, rankings by total and form, forecast, SHAP explanations, coverage/quality, evaluation, data dictionary) plus JSON/CSV reports and a run manifest with full provenance.

## Quick start

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e ".[dev]" --no-deps

ppi demo --fresh      # synthetic universe -> ingest -> reconcile -> model -> backtest -> score -> Excel (about 10 min)
pytest                # unit + integration tests
```

Outputs land in `outputs/<run_id>/` (workbook), `reports/<run_id>/` (quality, reconciliation, evaluation, manifest) and `artifacts/<run_id>/` (scores, models, model card). All of these are git-ignored.

### Main commands

| Command | Purpose |
|---|---|
| `ppi synth [--scale 0.05]` | Generate the synthetic universe in real source-file formats |
| `ppi template --dataset D` | Versioned Excel upload template generated from a source mapping |
| `ppi preview --dataset D --path F` | Show what a file would do (rows, periods, error codes, duplicates) without writing anything |
| `ppi profile --path F` | Profile a new file's structure (no raw values shown) and draft a mapping |
| `ppi ingest --dataset D --path F` | Immutable raw copy, type checks, quarantine, atomic load |
| `ppi reconcile --origin YYYY-MM` | As-of canonical movement + quality and reconciliation reports |
| `ppi run [--exploratory]` | Monthly run; real data enforces gates, `--exploratory` never trains a model |
| `ppi status` | Batches, gates and latest runs |
| `ppi ui [--port 8501]` | Open the local PharmaLens viewer |

Validation scripts (exit code 0 = pass, 1 = fail, 2 = no input) live in `scripts/`: reconciliation, as-of, scores, contracts, manifest and reproducibility.

## PharmaLens viewer (local, read-only)

After a run has finished, open it in the browser:

```bash
pip install -e ".[ui]"
ppi ui                 # http://127.0.0.1:8501 (binds to localhost only)
```

| Page | What you see |
|---|---|
| Overview | Run, score month, champion, G3 proposal, gate status, Excel download |
| Rankings | Track B total / by form and track A; filters; select a row to open it |
| Pharmacy | One GLN: rank, baseline vs peer benchmark, reasons, forecast interval, SHAP contributions |
| Data quality | Ingestion by dataset, quarantine by error code, source coverage, reconciliation |
| Evaluation | G3 proposal and checks, evaluation report, global SHAP importance |

The viewer never ingests, trains or writes data. It only reads the files a run already produced.

## Repository layout

```
src/pharmacy_potential/   ingest · reconcile · features · models · evaluation · scoring · readiness · export · synth · ui
configs/                  pilot config, source mappings, selection policy, thresholds, scenarios, synthetic universe
contracts/                manifest schema, per-table JSON schemas, units/value and observation-status contracts
scripts/                  validators and the reproducibility verifier
tests/                    unit and integration tests (as-of leakage, zero evidence, gates, reproducibility, ...)
docs/gates/               gate stubs (all `pending`)
```

The Python package keeps the name `pharmacy_potential` and the CLI is `ppi`. Source-file column names in the mappings (for example `Barkod`, `Marka`) are the input schema of the original delivery format and are kept as is.

## Limitations (read before drawing conclusions)

- **Synthetic data only.** The generator embeds a learnable signal by design, so metrics say nothing about real-world performance.
- **Proxy labels.** "Sustained excess growth versus the district" is a proxy for unmet opportunity, not a measured one. Ranking quality against the true target is explicitly not claimed.
- **Untuned defaults.** Until a gate accepts tuning, models use fixed engineering defaults and the workbook says so.
- **Depot→pharmacy outflow is not consumer sales.** It is a supply-side signal.
- **Association, not causation.** Nothing here estimates intervention effect or ROI.

## License

MIT, see [LICENSE](LICENSE).
