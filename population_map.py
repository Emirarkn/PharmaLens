"""Population-density contour: Kontur Population (H3 r8) -> one HTML map.

    python population_map.py --download   # fetch the Kontur file (~34 MB), once
    python population_map.py              # writes population_map.html
"""
import argparse
import gzip
import shutil
import sqlite3
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import requests

URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_TR_20231101.gpkg.gz"
SOURCE = Path("data/kontur_population_TR_20231101.gpkg")
OUTPUT = Path("population_map.html")
ATTRIBUTION = "Population: Kontur Population 2023-11 (CC BY 4.0, modelled estimate)"

THRESHOLDS = [100, 1_000, 5_000, 10_000, 20_000, 40_000]  # people / km2
COLORS = [  # YlOrRd, 7 classes
    [255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60],
    [252, 78, 42], [227, 26, 28], [177, 0, 38],
]
CLASS_LABELS = ["< 100", "100-1,000", "1,000-5,000", "5,000-10,000", "10,000-20,000", "20,000-40,000", "> 40,000"]


def download() -> None:
    if SOURCE.exists():
        print(f"exists: {SOURCE}")
        return
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    gz = SOURCE.with_suffix(".gpkg.gz")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(gz, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    with gzip.open(gz, "rb") as src, open(SOURCE, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()
    print(f"downloaded: {SOURCE}")


def load() -> pd.DataFrame:
    con = sqlite3.connect(SOURCE)
    try:
        table = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()[0]
        df = pd.read_sql(f'SELECT h3, population FROM "{table}"', con)
    finally:
        con.close()
    if np.issubdtype(df.h3.dtype, np.integer):
        df["h3"] = df.h3.map(h3.int_to_str)
    return df[df.population > 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download the Kontur file")
    ap.add_argument("--min-density", type=float, default=THRESHOLDS[0],
                    help="cells below this density (people/km2) are not drawn; keeps the output small")
    args = ap.parse_args()
    if args.download:
        download()
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found; run: python population_map.py --download")

    cells = load()
    area = np.array([h3.cell_area(c, unit="km^2") for c in cells.h3])
    cells["density"] = cells.population.to_numpy() / area
    cells = cells[cells.density >= args.min_density].copy()
    cls = np.digitize(cells.density, THRESHOLDS)
    cells["color"] = [COLORS[i] + [255] for i in cls]
    cells["label"] = [f"{CLASS_LABELS[i]} people/km²" for i in cls]

    layer = pdk.Layer(
        "H3HexagonLayer", data=cells[["h3", "color", "label"]], get_hexagon="h3", get_fill_color="color",
        extruded=False, opacity=0.45, pickable=True, stroked=False,
    )
    deck = pdk.Deck(
        layers=[layer], initial_view_state=pdk.ViewState(latitude=39.0, longitude=35.0, zoom=5),
        map_style="light", tooltip={"text": "{label}"},
    )
    deck.to_html(str(OUTPUT), css_background_color="#f4f4f4", open_browser=False)
    print(f"{len(cells):,} cells -> {OUTPUT}\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
