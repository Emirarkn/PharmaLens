"""Nüfus yoğunluğu konturu: Kontur Population (H3 r8) -> tek HTML harita.

    python harita_nufus.py --indir        # Kontur dosyasını indirir (~34 MB), bir kez
    python harita_nufus.py                # nufus_haritasi.html üretir
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
DOSYA = Path("veri/kontur_population_TR_20231101.gpkg")
CIKTI = Path("nufus_haritasi.html")
ATIF = "Nüfus: Kontur Population 2023-11 (CC BY 4.0, modellenmiş tahmin)"

ESIKLER = [100, 1_000, 5_000, 10_000, 20_000, 40_000]  # kişi/km²
RENKLER = [  # YlOrRd, 7 sınıf
    [255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60],
    [252, 78, 42], [227, 26, 28], [177, 0, 38],
]
SINIF_ADI = ["< 100", "100–1.000", "1.000–5.000", "5.000–10.000", "10.000–20.000", "20.000–40.000", "> 40.000"]


def indir() -> None:
    if DOSYA.exists():
        print(f"var: {DOSYA}")
        return
    DOSYA.parent.mkdir(parents=True, exist_ok=True)
    gz = DOSYA.with_suffix(".gpkg.gz")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(gz, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    with gzip.open(gz, "rb") as src, open(DOSYA, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()
    print(f"indirildi: {DOSYA}")


def yukle() -> pd.DataFrame:
    con = sqlite3.connect(DOSYA)
    try:
        tablo = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()[0]
        df = pd.read_sql(f'SELECT h3, population AS nufus FROM "{tablo}"', con)
    finally:
        con.close()
    if np.issubdtype(df.h3.dtype, np.integer):
        df["h3"] = df.h3.map(h3.int_to_str)
    return df[df.nufus > 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", action="store_true", help="Kontur dosyasını indir")
    ap.add_argument("--min-yogunluk", type=float, default=ESIKLER[0],
                    help="Bu yoğunluğun (kişi/km²) altındaki hücreler çizilmez; çıktıyı küçük tutar")
    args = ap.parse_args()
    if args.indir:
        indir()
    if not DOSYA.exists():
        raise SystemExit(f"{DOSYA} yok; önce: python harita_nufus.py --indir")

    k = yukle()
    k["yogunluk"] = k.nufus.to_numpy() / np.array([h3.cell_area(c, unit="km^2") for c in k.h3])
    k = k[k.yogunluk >= args.min_yogunluk].copy()
    sinif = np.digitize(k.yogunluk, ESIKLER)
    k["renk"] = [RENKLER[i] + [255] for i in sinif]
    k["baslik"] = [f"{SINIF_ADI[i]} kişi/km²" for i in sinif]

    katman = pdk.Layer(
        "H3HexagonLayer", data=k[["h3", "renk", "baslik"]], get_hexagon="h3", get_fill_color="renk",
        extruded=False, opacity=0.45, pickable=True, stroked=False,
    )
    harita = pdk.Deck(
        layers=[katman], initial_view_state=pdk.ViewState(latitude=39.0, longitude=35.0, zoom=5),
        map_style="light", tooltip={"text": "{baslik}"},
    )
    harita.to_html(str(CIKTI), css_background_color="#f4f4f4", open_browser=False)
    print(f"{len(k):,} hücre -> {CIKTI}\n{ATIF}")


if __name__ == "__main__":
    main()
