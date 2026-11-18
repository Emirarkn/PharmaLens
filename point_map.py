"""Pharmacy and clinic map: points are aggregated into H3 r7 cells, individual locations are never drawn.

Input CSV columns: lat, lng, type   (type: pharmacy | clinic)

    python point_map.py --input points.csv
    python point_map.py --sample            # try it with random sample points
"""
import argparse
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pydeck as pdk

OUTPUT = Path("point_map.html")
RESOLUTION = 7  # ~5 km2 cells
BLUE = [30, 90, 200]
RED = [210, 40, 50]


def sample(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = np.array([[41.01, 28.97], [40.99, 29.12], [41.08, 28.85], [40.90, 29.25]])  # one metro area, four districts
    spread = np.array([0.04, 0.05, 0.05, 0.06])
    k = rng.choice(len(centres), n, p=[0.4, 0.3, 0.2, 0.1])
    return pd.DataFrame({
        "lat": centres[k, 0] + rng.normal(0, 1, n) * spread[k], "lng": centres[k, 1] + rng.normal(0, 1, n) * spread[k] * 1.3,
        "type": rng.choice(["pharmacy", "clinic"], n, p=[0.98, 0.02]),
    })


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lng"]).copy()
    df["h3"] = [h3.latlng_to_cell(a, b, RESOLUTION) for a, b in zip(df.lat, df.lng)]
    return df.groupby(["h3", "type"]).size().rename("count").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="CSV with lat,lng,type columns")
    ap.add_argument("--sample", action="store_true", help="generate sample data")
    args = ap.parse_args()
    if args.sample:
        df = sample()
    elif args.input:
        df = pd.read_csv(args.input)
        missing = {"lat", "lng", "type"} - set(df.columns)
        if missing:
            raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    else:
        raise SystemExit("pass --input file.csv or --sample")

    cells = aggregate(df)
    pharm = cells[cells.type == "pharmacy"].copy()
    pharm["color"] = [BLUE + [int(min(230, 60 + 30 * n))] for n in pharm["count"]]
    pharm["label"] = pharm["count"].astype(str) + " pharmacies"
    clinic = cells[cells.type == "clinic"].copy()
    if len(clinic):
        clinic["lat"], clinic["lng"] = zip(*[h3.cell_to_latlng(c) for c in clinic.h3])
    else:
        clinic["lat"], clinic["lng"] = [], []
    clinic["label"] = clinic["count"].astype(str) + " clinics"

    layers = [
        pdk.Layer("H3HexagonLayer", data=pharm[["h3", "color", "label"]], get_hexagon="h3",
                  get_fill_color="color", extruded=False, pickable=True, stroked=False),
        pdk.Layer("ScatterplotLayer", data=clinic[["lat", "lng", "label"]], get_position=["lng", "lat"],
                  get_fill_color=RED + [230], get_radius=700, radius_min_pixels=4, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(df.lat.mean()), longitude=float(df.lng.mean()), zoom=9.5)
    pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip={"text": "{label}"}) \
        .to_html(str(OUTPUT), css_background_color="#f4f4f4", open_browser=False)
    print(f"{len(pharm)} pharmacy cells, {len(clinic)} clinic cells -> {OUTPUT}")


if __name__ == "__main__":
    main()
