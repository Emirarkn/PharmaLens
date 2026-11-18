"""Synthetic universe generator.

Writes raw source files in the *same format real deliveries are expected to have*
(CSV + sidecar manifest), so the demo exercises exactly the ingest -> reconcile ->
features -> score -> export path real data will take later.

A hidden ground truth (latent potential, realization rate, activation events) is written
separately to ``<data_root>/synthetic_truth`` and is never read by the pipeline; only the
demo-only ``synthetic_truth_check`` report uses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings, load_yaml
from ..ingest.manifest import build_manifest, write_manifest
from ..utils import get_logger, month_range, write_json

log = get_logger()

SEASON = np.array([1.30, 1.10, 1.00, 0.95, 0.92, 0.88, 0.85, 0.88, 1.00, 1.02, 1.00, 1.10])
CITY_PARENTS = {"34": [1, 2, 3, 4, 5], "06": [4, 5, 6, 7], "21": [8, 9], "47": [8, 9, 10],
                "61": [11, 12], "44": [9, 10, 12]}
UNKNOWN_BARCODE = "8690000999999"
REVISED_PARENT = "D03"


@dataclass
class Universe:
    months: list[pd.Timestamp]
    geo: pd.DataFrame
    pharm: pd.DataFrame
    products: pd.DataFrame
    depots: pd.DataFrame
    links: pd.DataFrame
    units: np.ndarray        # [P, T, B] true depot->pharmacy units (paid+free)
    free: np.ndarray         # [P, T, B]
    returns: np.ndarray      # [P, T, B]
    visits: np.ndarray       # [P, T]
    realization: np.ndarray  # [P, T]
    potential: np.ndarray    # [P, T]
    listed: np.ndarray       # [P, B]
    campaigns: pd.DataFrame


# --------------------------------------------------------------------------- universe

def _unique_glns(rng: np.random.Generator, n: int) -> list[str]:
    out: set[str] = set()
    res: list[str] = []
    while len(res) < n:
        if rng.random() < 0.05:
            g = "0" + "".join(map(str, rng.integers(0, 10, 12)))
        else:
            g = "868" + "".join(map(str, rng.integers(0, 10, 10)))
        if g not in out:
            out.add(g)
            res.append(g)
    return res


def build_universe(cfg: dict[str, Any], scale: float = 1.0, end_month: str | None = None) -> Universe:
    rng = np.random.default_rng(cfg["seed"])
    months = month_range(cfg["start_month"], end_month or cfg["end_month"])
    T = len(months)
    truth = cfg["truth"]

    # Geography and pharmacies
    geo_rows, pharm_rows = [], []
    for city, (name, n_ph, n_d, apr) in cfg["cities"].items():
        n_ph = max(30, int(round(n_ph * scale)))
        n_d = max(2, min(n_d, n_ph // 12))
        shares = rng.dirichlet(np.full(n_d, 2.5))
        counts = np.maximum(1, rng.multinomial(n_ph - n_d, shares) + 1)
        for i, cnt in enumerate(counts):
            dcode = f"{city}{i + 1:03d}"
            adults = int(cnt * rng.uniform(*apr) * rng.uniform(0.9, 1.1))
            smoke = float(np.clip(rng.normal(1.0, truth["district_smoking_sd"]), 0.6, 1.5))
            geo_rows.append((city, name, dcode, f"{name}-Ilce-{i + 1:02d}", int(cnt), adults, smoke))
            for _ in range(cnt):
                pharm_rows.append((city, dcode))
    geo = pd.DataFrame(geo_rows, columns=["city_code", "city_name", "district_code", "district_name",
                                          "n_pharm", "adult_pop", "smoke_factor"])
    pharm = pd.DataFrame(pharm_rows, columns=["city_code", "district_code"])
    P = len(pharm)
    pharm["gln"] = _unique_glns(rng, P)
    pharm["pharmacy_name"] = [f"SYNTHETIC PHARMACY {c}-{i + 1:05d}" for i, c in enumerate(pharm["city_code"])]
    pharm["brick_code"] = [f"B{d}{k}" for d, k in zip(pharm["district_code"], rng.integers(1, 4, P))]
    t_idx = np.arange(T)
    open_idx = np.zeros(P, dtype=int)
    close_idx = np.full(P, T, dtype=int)  # exclusive
    opens = rng.random(P) < 0.04
    open_idx[opens] = rng.integers(12, T - 6, opens.sum())
    closes = (~opens) & (rng.random(P) < 0.02)
    close_idx[closes] = rng.integers(24, T - 3, closes.sum())
    active = (t_idx[None, :] >= open_idx[:, None]) & (t_idx[None, :] < close_idx[:, None])
    pharm["valid_from"] = [months[i].strftime("%d.%m.%Y") if o else "01.01.2015" for i, o in zip(open_idx, opens)]
    pharm["valid_to"] = [(months[c] - timedelta(days=1)).strftime("%d.%m.%Y") if cl else ""
                         for c, cl in zip(np.minimum(close_idx, T - 1), closes)]
    pharm["open_idx"], pharm["close_idx"] = open_idx, close_idx

    # Products
    prod_rows = []
    for suffix, form, strength, pack, price in cfg["products"]:
        barcode = f"8690000123{suffix}"
        valid_from = "2022-06-01" if suffix == "203" else "2019-01-01"
        prod_rows.append((barcode, "DemoBrand", form, strength, pack, "box", valid_from, "", price))
    products = pd.DataFrame(prod_rows, columns=["barcode", "brand", "form", "strength", "pack_size",
                                                "unit_definition", "valid_from", "valid_to", "price"])
    B = len(products)
    prod_active = np.ones((T, B), dtype=bool)
    for b, vf in enumerate(products["valid_from"]):
        prod_active[:, b] = np.array([m >= pd.Timestamp(vf) - pd.offsets.MonthBegin(1) for m in months])

    # Depots
    dep_rows = []
    for k in range(1, cfg["depots"]["n_parent"] + 1):
        pcode = f"D{k:02d}"
        cities = [c for c, ps in CITY_PARENTS.items() if k in ps] or ["34"]
        dep_rows.append((pcode, "", "PARENT", f"Synthetic Wholesaler {k:02d}", cities[0]))
        for j in range(1, rng.integers(*cfg["depots"]["sub_per_parent"]) + 1):
            dep_rows.append((f"{pcode}S{j}", pcode, "SUB", f"Synthetic Wholesaler {k:02d} branch {j}",
                             cities[(j - 1) % len(cities)]))
    depots = pd.DataFrame(dep_rows, columns=["depot_code", "parent_depot_code", "level", "depot_name", "city_code"])

    link_rows = []
    primary = []
    for i, city in enumerate(pharm["city_code"]):
        parents = CITY_PARENTS[city]
        p1 = int(rng.choice(parents))
        primary.append(f"D{p1:02d}")
        link_rows.append((pharm.at[i, "gln"], f"D{p1:02d}", 0))
        if len(parents) > 1 and rng.random() < cfg["depots"]["second_supplier_share"]:
            p2 = int(rng.choice([p for p in parents if p != p1]))
            link_rows.append((pharm.at[i, "gln"], f"D{p2:02d}", 1))
    links = pd.DataFrame(link_rows, columns=["gln", "depot_code", "slot"])
    pharm["primary_depot"] = primary

    # Hidden truth: potential, realization dynamics, visits
    geo_idx = pharm["district_code"].map({d: i for i, d in enumerate(geo["district_code"])}).to_numpy()
    attract = rng.lognormal(0.0, truth["attractiveness_sigma"], P)
    dist_sum = np.bincount(geo_idx, weights=attract, minlength=len(geo))
    share = attract / dist_sum[geo_idx]
    years = np.array([(m.year - months[0].year) + (m.month - 1) / 12 for m in months])
    trend = (1 + truth["yearly_trend"]) ** years
    district_demand = (geo["adult_pop"].to_numpy() * truth["boxes_per_adult_month"] * geo["smoke_factor"].to_numpy())
    potential = district_demand[geo_idx][:, None] * share[:, None] * trend[None, :]

    u = rng.random(P)
    kind = np.where(u < truth["never_stock_share"], "never",
                    np.where(u < truth["never_stock_share"] + truth["underserved_share"], "underserved", "normal"))
    r = np.where(kind == "never", 0.0,
                 np.where(kind == "underserved", rng.uniform(0.10, 0.45, P), rng.beta(8, 2.5, P)))
    target = r.copy()
    realization = np.zeros((P, T))
    visits = np.zeros((P, T), dtype=int)
    crm = cfg["crm"]
    for t in range(T):
        lam = crm["visit_rate_base"] + crm["visit_rate_by_realization"] * r
        visits[:, t] = rng.poisson(lam)
        live = kind != "never"
        prev_visits = visits[:, t - 1] if t else visits[:, t]
        hazard = truth["activation_base_hazard"] + truth["activation_visit_hazard"] * prev_visits
        activate = live & (r < 0.55) & (target <= r + 1e-9) & (rng.random(P) < hazard)
        target = np.where(activate, rng.uniform(0.65, 0.95, P), target)
        deact = live & (r > 0.5) & (rng.random(P) < truth["deactivation_hazard"])
        target = np.where(deact, r * 0.65, target)
        moving = np.abs(target - r) > 1e-9
        r = np.where(moving, r + (target - r) / 3.0, r + 0.01 * (0.75 - r) + rng.normal(0, 0.015, P))
        r = np.where(live, np.clip(r, 0.02, 1.0), 0.0)
        target = np.where(moving, target, r)
        realization[:, t] = r

    # Listing / form mix
    form_w = cfg["form_weights"]
    forms = products["form"].tolist()
    n_by_form = pd.Series(forms).value_counts().to_dict()
    base_w = np.array([form_w[f] / n_by_form[f] for f in forms])
    listed = rng.random((P, B)) < 0.65
    listed[:, 0] = True
    mix = rng.dirichlet(np.full(B, 1.5), P) * base_w[None, :] * listed

    # Campaigns: Jan & Sep on GUM (MF 10+1), May on PATCH (8% discount)
    camp_rows = []
    gum_b = [b for b, f in enumerate(forms) if f == "GUM"]
    patch_b = [b for b, f in enumerate(forms) if f == "PATCH"]
    camp = np.zeros((T, B))
    disc = np.zeros((T, B))
    mf = np.zeros((T, B), dtype=bool)
    for t, m in enumerate(months):
        if m.month in (1, 9):
            for b in gum_b:
                camp[t, b], mf[t, b] = 1, True
                camp_rows.append((m, products.at[b, "barcode"], f"C{m:%Y%m}GUM", "ALL_PHARMACIES", "10+1", 0.0,
                                  "ISF_NET_VAT_INC", m - timedelta(days=20)))
        if m.month == 5:
            for b in patch_b:
                camp[t, b], disc[t, b] = 1, 0.08
                camp_rows.append((m, products.at[b, "barcode"], f"C{m:%Y%m}PAT", "ALL_PHARMACIES", "", 0.08,
                                  "ISF_NET_VAT_INC", m - timedelta(days=20)))
    campaigns = pd.DataFrame(camp_rows, columns=["month", "barcode", "campaign_id", "scope", "mf", "discount",
                                                 "price_basis", "announced"])

    units = np.zeros((P, T, B), dtype=np.int64)
    season = SEASON[[m.month - 1 for m in months]]
    for t in range(T):
        pm = mix * prod_active[t][None, :]
        tot_w = pm.sum(axis=1, keepdims=True)
        pm = np.divide(pm, tot_w, out=np.full_like(pm, 1.0 / B), where=tot_w > 0)
        camp_factor = 1 + truth["campaign_lift"] * (pm * camp[t][None, :]).sum(axis=1)
        expected = potential[:, t] * realization[:, t] * season[t] * camp_factor
        noise = rng.gamma(truth["noise_shape"], 1.0 / truth["noise_shape"], P)
        n = rng.poisson(expected * noise) * active[:, t]
        units[:, t, :] = rng.multinomial(n, pm)

    free = np.where(mf[None, :, :], units // 11, 0)
    ret_mask = rng.random(units.shape) < cfg["measurement"]["return_cell_prob"]
    returns = np.where(ret_mask & listed[:, None, :] & active[:, :, None], rng.integers(1, 3, units.shape), 0)

    pharm["kind"] = kind
    products["camp_disc"] = 0.0
    u_obj = Universe(months, geo, pharm, products, depots, links, units, free, returns, visits,
                     realization, potential, listed, campaigns)
    u_obj.active = active  # type: ignore[attr-defined]
    u_obj.prod_active = prod_active  # type: ignore[attr-defined]
    u_obj.discount = disc  # type: ignore[attr-defined]
    return u_obj


# --------------------------------------------------------------------------- writers

class _Writer:
    def __init__(self, out_root: Path, cfg: dict[str, Any]):
        self.root = out_root
        self.cfg = cfg
        self.files: list[dict[str, Any]] = []

    def write(self, dataset: str, source: str, df: pd.DataFrame, name: str, *, sep: str = ",",
              decimal: str = ".", period: tuple[pd.Timestamp, pd.Timestamp], available_at: pd.Timestamp,
              mode: str, revision: int = 1, supersedes: str | None = None, notes: str | None = None) -> str:
        folder = self.root / dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        df.to_csv(path, sep=sep, decimal=decimal, index=False, encoding="utf-8", lineterminator="\n")
        stamp = available_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        m = build_manifest(
            dataset, source, path, period_start=period[0].strftime("%Y-%m-%d"),
            period_end=(period[1] + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"), revision=revision,
            supersedes_batch_id=supersedes, confidentiality="synthetic", license_id="synthetic-demo",
            owner="synthetic-generator", received_at_utc=stamp, available_at=stamp, availability_mode=mode,
            retrospective_lag_days=self.cfg["retrospective_lag_days"] if mode == "retrospective_lag" else None,
            synthetic=True, row_count=len(df), notes=notes)
        write_manifest(m, path)
        self.files.append({"dataset": dataset, "file": path.name, "rows": len(df), "batch_id": m["batch_id"]})
        return m["batch_id"]


def _deliveries(cfg: dict[str, Any], months: list[pd.Timestamp], rng: np.random.Generator):
    """Yearly backfill files up to backfill_until_year, then monthly live deliveries."""
    out = []
    by_year: dict[int, list[pd.Timestamp]] = {}
    for m in months:
        by_year.setdefault(m.year, []).append(m)
    backfill_at = pd.Timestamp(f"{cfg['backfill_until_year'] + 1}-01-20")
    for year, ms in sorted(by_year.items()):
        if year <= cfg["backfill_until_year"]:
            out.append((ms, f"{year}", backfill_at, "retrospective_lag"))
        else:
            for m in ms:
                lag = int(rng.integers(*cfg["monthly_delivery_lag_days"]))
                out.append(([m], f"{m:%Y-%m}", m + pd.offsets.MonthEnd(0) + timedelta(days=lag), "batch"))
    return out


def _inject_problems(df: pd.DataFrame, rng: np.random.Generator, meas: dict[str, Any], cols: dict[str, str],
                     n_bad: int, zero_glns: list[str]) -> pd.DataFrame:
    """Exact duplicates, conflicting duplicates and malformed rows (all must end in dedup/quarantine)."""
    if df.empty:
        return df
    df = df.copy()
    for c in df.columns:  # raw files are text; keep the source decimal convention
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(2).astype(str).str.replace(".", cols.get("decimal", "."), regex=False)
        else:
            df[c] = df[c].astype(str)
    n = len(df)
    dup = df.sample(n=max(1, int(n * meas["exact_duplicate_share"])), random_state=int(rng.integers(1e9)))
    conf = df.sample(n=max(1, int(n * meas["conflicting_duplicate_share"])), random_state=int(rng.integers(1e9))).copy()
    conf[cols["units"]] = (conf[cols["units"]].astype(int) + 1).astype(str)
    bad = df.sample(n=n_bad, random_state=int(rng.integers(1e9)), replace=True).copy().reset_index(drop=True)
    kinds = np.arange(len(bad)) % 6
    bad.loc[kinds == 0, cols["month"]] = "2023-13"
    bad.loc[kinds == 1, cols["barcode"]] = UNKNOWN_BARCODE
    if zero_glns:
        bad.loc[kinds == 2, cols["gln"]] = [g[1:] for g in rng.choice(zero_glns, (kinds == 2).sum())]
    bad.loc[kinds == 3, cols["gln"]] = "8.68123E+12"
    bad.loc[kinds == 4, cols["units"]] = "-5" if cols.get("nonneg") else "abc"
    bad.loc[kinds == 5, cols["units"]] = "abc"
    out = pd.concat([df, dup, conf, bad], ignore_index=True)
    return out.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)


def generate(settings: Settings, config_path: Path | None = None, scale: float = 1.0,
             end_month: str | None = None, out_root: Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path or settings.root / "configs" / "synthetic_v001.yaml")
    uni = build_universe(cfg, scale=scale, end_month=end_month)
    rng = np.random.default_rng(cfg["seed"] + 1)
    out_root = out_root or settings.data_root / "raw" / "synthetic"
    w = _Writer(out_root, cfg)
    months = uni.months
    P, T, B = uni.units.shape
    meas = cfg["measurement"]
    glns = uni.pharm["gln"].to_numpy()
    barcodes = uni.products["barcode"].to_numpy()
    month_s = np.array([m.strftime("%Y-%m") for m in months])
    master_at = pd.Timestamp(months[-1] + pd.offsets.MonthEnd(0) + timedelta(days=1))
    span = (months[0], months[-1])
    zero_glns = [g for g in glns if g.startswith("0")]

    # ---- masters (single retrospective snapshots) ----------------------------------------
    ph = uni.pharm
    w.write("pharmacy_master", "master", pd.DataFrame({
        "GLN": ph["gln"], "EczaneAdi": ph["pharmacy_name"], "IlKodu": ph["city_code"], "IlceKodu": ph["district_code"],
        "BrickKodu": ph["brick_code"], "AcilisTarihi": ph["valid_from"], "KapanisTarihi": ph["valid_to"]}),
        "pharmacy_master_snapshot.csv", sep=";", decimal=",", period=span, available_at=master_at,
        mode="retrospective_lag", notes="Current master snapshot; no as-of history available.")
    pr = uni.products
    w.write("product_master", "master", pd.DataFrame({
        "Barkod": pr["barcode"], "Marka": pr["brand"], "Form": pr["form"], "Guc": pr["strength"],
        "PaketAdet": pr["pack_size"], "BirimTanimi": pr["unit_definition"], "BaslangicTarihi": pr["valid_from"],
        "BitisTarihi": pr["valid_to"], "ListeFiyatiKDVDahil": pr["price"]}),
        "product_master_snapshot.csv", period=span, available_at=master_at, mode="retrospective_lag")
    dp = uni.depots
    w.write("depot_master", "master", pd.DataFrame({
        "depot_code": dp["depot_code"], "parent_depot_code": dp["parent_depot_code"], "level": dp["level"],
        "depot_name": dp["depot_name"], "city_code": dp["city_code"], "valid_from": "2015-01-01", "valid_to": ""}),
        "depot_master_snapshot.csv", period=span, available_at=master_at, mode="retrospective_lag")
    w.write("pharmacy_depot", "master", pd.DataFrame({
        "GLN": uni.links["gln"], "DepoKodu": uni.links["depot_code"], "BaslangicTarihi": "01.01.2020",
        "BitisTarihi": ""}), "pharmacy_depot_snapshot.csv", sep=";", decimal=",", period=span,
        available_at=master_at, mode="retrospective_lag")

    # ---- external geography (one vintage per reference year) -----------------------------
    geo = uni.geo
    for ref in range(2020, 2026):
        growth = 1.01 ** (ref - 2024)
        pub = pd.Timestamp(f"{ref + 1}-03-15")
        rows = []
        for _, g in geo.iterrows():
            rows.append((g.district_code, g.city_code, g.district_code, "DISTRICT", g.district_name, ref,
                         "ADULT_POPULATION", round(g.adult_pop * growth), "persons", "15+"))
            off = int(rng.integers(-2, 3))
            rows.append((g.district_code, g.city_code, g.district_code, "DISTRICT", g.district_name, ref,
                         "PHARMACY_COUNT_OFFICIAL", max(1, g.n_pharm + off), "pharmacies", ""))
        df = pd.DataFrame(rows, columns=["geo_id", "city_code", "district_code", "geo_level", "name",
                                         "reference_period", "metric_id", "value", "unit", "age_scope"])
        df["publication_date"] = pub.strftime("%Y-%m-%d")
        df["source_url"] = "synthetic://population"
        df["license_id"] = "SYNTHETIC"
        w.write("external_geo", "external", df, f"external_geo_ref{ref}.csv",
                period=(pd.Timestamp(f"{ref}-01-01"), pd.Timestamp(f"{ref}-12-01")), available_at=pub, mode="batch")

    # ---- depot split of true flows -------------------------------------------------------
    link_slots = uni.links.pivot_table(index="gln", columns="slot", values="depot_code", aggfunc="first")
    dep1 = link_slots[0].reindex(glns).to_numpy()
    dep2 = link_slots[1].reindex(glns).to_numpy() if 1 in link_slots else np.full(P, np.nan, dtype=object)
    has2 = pd.notna(dep2)
    share1 = np.where(has2, 0.7, 1.0)
    paid = uni.units - uni.free
    paid1 = rng.binomial(paid, share1[:, None, None])
    free1 = rng.binomial(uni.free, share1[:, None, None])
    flows = [(dep1, paid1, free1, uni.returns), (dep2, paid - paid1, uni.free - free1, np.zeros_like(uni.returns))]

    parents = sorted(uni.depots.loc[uni.depots["level"] == "PARENT", "depot_code"])
    cov_probs = meas["mutabakat_coverage"]
    status = rng.choice(["COMPLETE", "PARTIAL", "MISSING"], size=(len(parents), T),
                        p=[cov_probs["complete"], cov_probs["partial"], cov_probs["missing"]])
    pidx = {p: i for i, p in enumerate(parents)}
    t_rev = len(months) - 3 if len(months) >= 3 and months[-3].year > cfg["backfill_until_year"] else None
    if t_rev is not None and REVISED_PARENT in pidx:
        status[pidx[REVISED_PARENT], t_rev] = "COMPLETE"   # true state; r01 delivers it as PARTIAL

    def mut_rows(t: int, revision_partial: bool) -> pd.DataFrame:
        frames = []
        for dep_arr, pd_arr, fr_arr, rt_arr in flows:
            ok = pd.notna(dep_arr)
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            dep_codes = dep_arr[idx].astype(str)
            st = np.array([status[pidx[d], t] for d in dep_codes])
            u_tot = pd_arr[idx, t, :] + fr_arr[idx, t, :]
            emit = (u_tot > 0) | (rt_arr[idx, t, :] > 0)
            zero_listed = (~emit) & uni.listed[idx] & uni.active[idx, t][:, None] & uni.prod_active[t][None, :]
            emit |= zero_listed & (rng.random(emit.shape) < meas["mutabakat_explicit_zero_share"])
            emit &= (st != "MISSING")[:, None]
            drop = ((st == "PARTIAL")[:, None]) & (rng.random(emit.shape) < meas["mutabakat_partial_drop"])
            if revision_partial:
                drop |= (dep_codes == REVISED_PARENT)[:, None] & (rng.random(emit.shape) < meas["mutabakat_partial_drop"])
            emit &= ~drop
            ii, bb = np.nonzero(emit)
            pi = idx[ii]
            price = uni.products["price"].to_numpy()[bb]
            disc = uni.discount[t, bb]  # type: ignore[attr-defined]
            p_units = pd_arr[pi, t, bb]
            frames.append(pd.DataFrame({
                "GLN": glns[pi], "Donem": month_s[t], "Barkod": barcodes[bb], "DepoKodu": dep_codes[ii],
                "SatisAdet": p_units, "MFAdet": fr_arr[pi, t, bb], "IadeAdet": rt_arr[pi, t, bb],
                "NetTutarKDVDahil": np.round(p_units * price * (1 - disc), 2)}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def cov_rows(t: int, revision_partial: bool) -> pd.DataFrame:
        rows = []
        for p in parents:
            st = status[pidx[p], t]
            if revision_partial and p == REVISED_PARENT:
                st = "PARTIAL"
            if st == "MISSING":
                if rng.random() < 0.5:
                    continue          # no declaration at all
                st = "UNKNOWN"
            rows.append(("MUTABAKAT", month_s[t], "DEPOT", p, st, "depot delivery log (synthetic)"))
        rows.append(("VENDOR", month_s[t], "PANEL", "ALL", "UNKNOWN", "panel completeness not declared"))
        return pd.DataFrame(rows, columns=["source", "month", "scope_type", "scope_id", "status", "evidence"])

    # VENDOR panel measurement
    panel = rng.random(P) < meas["vendor_panel_share"]
    vendor_scale = rng.uniform(*meas["vendor_scale_range"], P)
    net_true = uni.units - uni.returns

    def vendor_rows(t: int) -> pd.DataFrame:
        miss = rng.random(P) < meas["vendor_missing_month_share"]
        val = np.round(np.clip(net_true[:, t, :], 0, None) * vendor_scale[:, None] * rng.uniform(0.97, 1.03, (P, B)))
        emit = (val > 0) & (panel & ~miss)[:, None]
        ii, bb = np.nonzero(emit)
        return pd.DataFrame({"PharmacyGLN": glns[ii], "Month": [months[t].strftime("%Y%m")] * len(ii),
                             "EAN": barcodes[bb], "Units": val[ii, bb].astype(int)})

    mut_cols = {"units": "SatisAdet", "month": "Donem", "barcode": "Barkod", "gln": "GLN", "nonneg": True,
                "decimal": ","}
    vendor_cols = {"units": "Units", "month": "Month", "barcode": "EAN", "gln": "PharmacyGLN"}
    deliveries = _deliveries(cfg, months, rng)
    m_index = {m: i for i, m in enumerate(months)}
    # Malformed rows exist to exercise quarantine, so their count is the full-universe figure scaled with the
    # universe (>= 6 keeps every defect kind present). A fixed count would make a small universe ~20x dirtier,
    # and every unpinnable rejected row withdraws a depot-month's completeness claim.
    bad_scale = min(1.0, float(scale))
    n_bad_mut = max(6, round(cfg["measurement"]["bad_rows_per_file"] * bad_scale))
    n_bad_vendor = max(6, round(max(6, cfg["measurement"]["bad_rows_per_file"] // 3) * bad_scale))
    for ms, tag, avail, mode in deliveries:
        ts = [m_index[m] for m in ms]
        per = (ms[0], ms[-1])
        is_rev_month = t_rev is not None and ts == [t_rev]
        mut = pd.concat([mut_rows(t, is_rev_month) for t in ts], ignore_index=True)
        mut = _inject_problems(mut, rng, meas, mut_cols, n_bad_mut, zero_glns)
        r01 = w.write("movement_mutabakat", "mutabakat", mut, f"mutabakat__gln-month-barcode__{tag}__r01.csv",
                      sep=";", decimal=",", period=per, available_at=avail, mode=mode)
        cov = pd.concat([cov_rows(t, is_rev_month) for t in ts], ignore_index=True)
        c01 = w.write("coverage_register", "coverage", cov, f"coverage__source-month-scope__{tag}__r01.csv",
                      period=per, available_at=avail, mode=mode)
        if is_rev_month:
            rev_at = avail + timedelta(days=12)
            mut2 = mut_rows(ts[0], False)
            w.write("movement_mutabakat", "mutabakat", mut2, f"mutabakat__gln-month-barcode__{tag}__r02.csv",
                    sep=";", decimal=",", period=per, available_at=rev_at, mode=mode, revision=2, supersedes=r01,
                    notes=f"Re-delivery: depot {REVISED_PARENT} complete")
            w.write("coverage_register", "coverage", cov_rows(ts[0], False),
                    f"coverage__source-month-scope__{tag}__r02.csv", period=per, available_at=rev_at, mode=mode,
                    revision=2, supersedes=c01)
        vendor = pd.concat([vendor_rows(t) for t in ts], ignore_index=True)
        vendor = _inject_problems(vendor, rng, meas, vendor_cols, n_bad_vendor, zero_glns)
        w.write("movement_vendor", "vendor", vendor, f"vendor__gln-month-barcode__{tag}__r01.csv", period=per,
                available_at=avail, mode=mode)

    # ---- sell-in (parent grain) and sub-depot stock snapshots ---------------------------
    out_parent = np.zeros((len(parents), T, B))
    for dep_arr, pd_arr, fr_arr, _ in flows:
        ok = pd.notna(dep_arr)
        for p in parents:
            sel = ok & (dep_arr == p)
            out_parent[pidx[p]] += (pd_arr[sel] + fr_arr[sel]).sum(axis=0)
    received = np.round(out_parent * rng.uniform(0.9, 1.15, out_parent.shape))
    stock = np.zeros_like(out_parent)
    level = out_parent[:, :3, :].mean(axis=1) * 1.5
    for t in range(T):
        level = np.clip(level + received[:, t] - out_parent[:, t] + rng.normal(0, 2, level.shape), 0, None)
        stock[:, t] = np.round(level)
    subs = uni.depots[uni.depots["level"] == "SUB"]
    sub_share = {p: rng.dirichlet(np.full((subs["parent_depot_code"] == p).sum(), 3.0)) for p in parents}
    for ms, tag, avail, mode in deliveries:
        ts = [m_index[m] for m in ms]
        si, st_rows = [], []
        for t in ts:
            for p in parents:
                for b in range(B):
                    if received[pidx[p], t, b] > 0:
                        si.append((month_s[t], p, barcodes[b], int(received[pidx[p], t, b]), 0,
                                   round(received[pidx[p], t, b] * uni.products.at[b, "price"] * 0.7, 2)))
                sub_codes = subs.loc[subs["parent_depot_code"] == p, "depot_code"].tolist()
                for j, sc in enumerate(sub_codes):
                    if rng.random() < 0.03:
                        continue  # missing snapshot -> parent aggregation is partial
                    snap = (months[t] + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
                    for b in range(B):
                        st_rows.append((snap, sc, barcodes[b], int(stock[pidx[p], t, b] * sub_share[p][j]), "SALEABLE"))
        w.write("sellin", "sellin", pd.DataFrame(si, columns=["Ay", "DepoKodu", "Barkod", "GirisAdet", "IadeAdet",
                                                              "TutarKDVDahil"]),
                f"sellin__month-depot-barcode__{tag}__r01.csv", period=(ms[0], ms[-1]), available_at=avail, mode=mode)
        w.write("depot_stock", "stock", pd.DataFrame(st_rows, columns=["StokTarihi", "AltDepoKodu", "Barkod",
                                                                      "EldekiAdet", "StokTipi"]),
                f"stock__subdepot-barcode__{tag}__r01.csv", period=(ms[0], ms[-1]), available_at=avail, mode=mode)

    # ---- CRM: visits and orders ------------------------------------------------------------
    vid = 0
    oid = 0
    for ms, tag, avail, mode in deliveries:
        v_rows, o_rows = [], []
        for t in (m_index[m] for m in ms):
            days = (months[t] + pd.offsets.MonthEnd(0)).day
            for i in np.nonzero(uni.visits[:, t] * uni.active[:, t])[0]:  # type: ignore[attr-defined]
                for _ in range(uni.visits[i, t]):
                    vid += 1
                    day = months[t] + timedelta(days=int(rng.integers(0, days)))
                    rec = day + timedelta(days=int(rng.integers(*cfg["crm"]["record_lag_days"])))
                    if rng.random() < 0.01:
                        rec = day + timedelta(days=40)   # late entry: must not leak into earlier cutoffs
                    outcome = "ORDER" if rng.random() < cfg["crm"]["order_per_visit_prob"] else \
                        str(rng.choice(["INFO", "NO_CONTACT"]))
                    v_rows.append((f"V{vid:09d}", glns[i], day.strftime("%Y-%m-%d"), rec.strftime("%Y-%m-%d"),
                                   str(rng.choice(["FIELD", "PHONE"], p=[0.8, 0.2])), outcome))
                    if outcome == "ORDER":
                        oid += 1
                        opts = np.nonzero(uni.listed[i] & uni.prod_active[t])[0]  # type: ignore[attr-defined]
                        for line, b in enumerate(rng.choice(opts, min(len(opts), int(rng.integers(1, 3))), replace=False)):
                            cancelled = rng.random() < 0.05
                            qty = int(rng.integers(2, 7))
                            o_rows.append((f"O{oid:08d}", str(line + 1), glns[i], barcodes[b], qty,
                                           qty if cancelled else 0, day.strftime("%Y-%m-%d"),
                                           (day + timedelta(days=int(rng.integers(1, 4)))).strftime("%Y-%m-%d"),
                                           "CANCELLED" if cancelled else "COMPLETED"))
        w.write("visits", "crm", pd.DataFrame(v_rows, columns=["ZiyaretID", "GLN", "ZiyaretTarihi", "KayitTarihi",
                                                               "Kanal", "Sonuc"]),
                f"visits__event__{tag}__r01.csv", period=(ms[0], ms[-1]), available_at=avail, mode=mode)
        w.write("orders", "crm", pd.DataFrame(o_rows, columns=["SiparisNo", "SatirNo", "GLN", "Barkod", "SiparisAdet",
                                                               "IptalAdet", "SiparisTarihi", "KayitTarihi", "Durum"]),
                f"orders__line__{tag}__r01.csv", period=(ms[0], ms[-1]), available_at=avail, mode=mode)

    # ---- commercial terms (one file per year, announcement date = known-at) --------------
    camp = uni.campaigns
    for year in sorted({m.year for m in months}):
        c = camp[camp["month"].dt.year == year]
        if c.empty:
            continue
        df = pd.DataFrame({"Ay": c["month"].dt.strftime("%Y-%m"), "Barkod": c["barcode"], "KampanyaID": c["campaign_id"],
                           "Kapsam": c["scope"], "MFTanimi": c["mf"], "IskontoOrani": c["discount"],
                           "FiyatBazi": c["price_basis"], "DuyuruTarihi": c["announced"].dt.strftime("%Y-%m-%d")})
        avail = pd.Timestamp(c["announced"].max()) if year > cfg["backfill_until_year"] else \
            pd.Timestamp(f"{cfg['backfill_until_year'] + 1}-01-20")
        w.write("commercial_terms", "commercial", df, f"commercial_terms__{year}__r01.csv",
                period=(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-01")), available_at=avail,
                mode="retrospective_lag" if year <= cfg["backfill_until_year"] else "batch")

    # ---- hidden truth (never read by the pipeline) ------------------------------------------
    truth_dir = settings.data_root / "synthetic_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame({
        "gln": np.repeat(glns, T), "month": np.tile(np.array(months, dtype="datetime64[ns]"), P),
        "realization": uni.realization.ravel(), "potential": uni.potential.ravel(),
        "true_units": uni.units.sum(axis=2).ravel(), "visits": uni.visits.ravel(),
        "active": uni.active.ravel()})  # type: ignore[attr-defined]
    long.to_parquet(truth_dir / "truth_pharmacy_month.parquet", index=False)
    uni.pharm[["gln", "city_code", "district_code", "kind", "open_idx", "close_idx"]].to_parquet(
        truth_dir / "truth_pharmacy.parquet", index=False)
    summary = {"synthetic_version": cfg["synthetic_version"], "seed": cfg["seed"], "scale": scale,
               "pharmacies": int(P), "districts": int(len(uni.geo)), "months": int(T), "barcodes": int(B),
               "files": w.files, "out_root": out_root, "truth_dir": truth_dir,
               "revised_month": month_s[t_rev] if t_rev is not None else None}
    write_json(truth_dir / "generation_summary.json", summary)
    log.info("synthetic universe: %d pharmacies, %d districts, %d months, %d files", P, len(uni.geo), T, len(w.files))
    return summary
