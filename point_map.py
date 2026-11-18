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


def sample(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = np.array([[41.0, 29.0], [39.9, 32.8], [38.4, 27.1], [37.0, 35.3]])  # four large cities
    c = centres[rng.integers(0, len(centres), n)]
    return pd.DataFrame({
        "lat": c[:, 0] + rng.normal(0, 0.12, n), "lng": c[:, 1] + rng.normal(0, 0.15, n),
        "type": rng.choice(["pharmacy", "clinic"], n, p=[0.92, 0.08]),
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
                  get_fill_color=RED + [230], get_radius=1500, radius_min_pixels=5, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(df.lat.mean()), longitude=float(df.lng.mean()), zoom=6)
    pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip={"text": "{label}"}) \
        .to_html(str(OUTPUT), css_background_color="#f4f4f4", open_browser=False)
    print(f"{len(pharm)} pharmacy cells, {len(clinic)} clinic cells -> {OUTPUT}")


if __name__ == "__main__":
    main()
