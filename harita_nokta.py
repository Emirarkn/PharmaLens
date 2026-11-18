"""Eczane ve poliklinik haritası: noktalar H3 r7 hücrelerinde toplulaştırılır, tek tek konum çizilmez.

Girdi CSV: lat, lng, tip   (tip: eczane | poliklinik)

    python harita_nokta.py --girdi noktalar.csv
    python harita_nokta.py --ornek          # rastgele örnek noktalarla dener
"""
import argparse
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pydeck as pdk

CIKTI = Path("nokta_haritasi.html")
R = 7  # ~5 km² hücre
MAVI = [30, 90, 200]
KIRMIZI = [210, 40, 50]


def ornek(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    merkez = np.array([[41.0, 29.0], [39.9, 32.8], [38.4, 27.1], [37.0, 35.3]])  # örnek: 4 büyükşehir
    m = merkez[rng.integers(0, len(merkez), n)]
    return pd.DataFrame({
        "lat": m[:, 0] + rng.normal(0, 0.12, n), "lng": m[:, 1] + rng.normal(0, 0.15, n),
        "tip": rng.choice(["eczane", "poliklinik"], n, p=[0.92, 0.08]),
    })


def topla(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lng"]).copy()
    df["h3"] = [h3.latlng_to_cell(a, b, R) for a, b in zip(df.lat, df.lng)]
    return df.groupby(["h3", "tip"]).size().rename("adet").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--girdi", type=Path, help="lat,lng,tip sütunlu CSV")
    ap.add_argument("--ornek", action="store_true", help="örnek veri üret")
    args = ap.parse_args()
    if args.ornek:
        df = ornek()
    elif args.girdi:
        df = pd.read_csv(args.girdi)
        eksik = {"lat", "lng", "tip"} - set(df.columns)
        if eksik:
            raise SystemExit(f"CSV'de eksik sütun: {sorted(eksik)}")
    else:
        raise SystemExit("--girdi dosya.csv veya --ornek ver")

    g = topla(df)
    ecz = g[g.tip == "eczane"].copy()
    ecz["renk"] = [MAVI + [int(min(230, 60 + 30 * a))] for a in ecz.adet]
    ecz["baslik"] = ecz.adet.astype(str) + " eczane"
    pol = g[g.tip == "poliklinik"].copy()
    pol["lat"], pol["lng"] = zip(*[h3.cell_to_latlng(c) for c in pol.h3]) if len(pol) else ([], [])
    pol["baslik"] = pol.adet.astype(str) + " poliklinik"

    katmanlar = [
        pdk.Layer("H3HexagonLayer", data=ecz[["h3", "renk", "baslik"]], get_hexagon="h3", get_fill_color="renk",
                  extruded=False, pickable=True, stroked=False),
        pdk.Layer("ScatterplotLayer", data=pol[["lat", "lng", "baslik"]], get_position=["lng", "lat"],
                  get_fill_color=KIRMIZI + [230], get_radius=1500, radius_min_pixels=5, pickable=True),
    ]
    vs = pdk.ViewState(latitude=float(df.lat.mean()), longitude=float(df.lng.mean()), zoom=6)
    pdk.Deck(layers=katmanlar, initial_view_state=vs, map_style="light", tooltip={"text": "{baslik}"}) \
        .to_html(str(CIKTI), css_background_color="#f4f4f4", open_browser=False)
    print(f"{len(ecz)} eczane hücresi, {len(pol)} poliklinik hücresi -> {CIKTI}")


if __name__ == "__main__":
    main()
