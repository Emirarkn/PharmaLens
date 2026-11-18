# Eczane ve Nüfus Haritaları

Pilot illerde eczane/poliklinik dağılımı ile nüfus yoğunluğunu iki ayrı harita olarak gösteren küçük araç.
Çıktı dosyaları ve veriler repoda **yoktur**; yerelde üretilir.

| Harita | Betik | Ne gösterir |
|---|---|---|
| Nüfus yoğunluğu | `harita_nufus.py` | Kontur Population (H3 r8) kişi/km² kontur haritası, 7 sınıf |
| Eczane / poliklinik | `harita_nokta.py` | Noktalar H3 r7 hücrelerinde toplulaştırılır; tek tek konum, ad veya satış bilgisi yok |

```bash
pip install -r requirements.txt

python harita_nufus.py --indir && python harita_nufus.py   # nufus_haritasi.html
python harita_nokta.py --ornek                              # örnek veriyle nokta_haritasi.html
python harita_nokta.py --girdi noktalar.csv                 # kendi verin: lat,lng,tip (eczane|poliklinik)
```

Nüfus verisi: [Kontur Population](https://data.humdata.org/dataset/kontur-population-turkiye) 2023-11, CC BY 4.0 (modellenmiş tahmin).

Lisans: MIT
