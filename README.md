# PharmaLens

Small tooling for looking at pharmacy and clinic coverage against population density. Output files and data are **not** in this repo; they are generated locally.

## Maps in this repo

| Map | Script | What it shows |
|---|---|---|
| Population density | `population_map.py` | Contour of people/km² from Kontur Population (H3 r8), 7 classes |
| Pharmacies / clinics | `point_map.py` | Points aggregated into H3 r7 cells; no individual locations, names or sales data |

```bash
pip install -r requirements.txt

python population_map.py --download && python population_map.py   # population_map.html
python point_map.py --sample                                      # point_map.html from random sample points
python point_map.py --input points.csv                            # your own data: lat,lng,type (pharmacy|clinic)
```

## Territory explorer view (described, not included)

The full project also has a third, territory-explorer style map. It combines the layers above with a demand proxy and lets you see how pharmacies sit relative to where people and activity are.

- **Venue rating counts.** For businesses around each H3 cell, the number of user ratings (Google Places `userRatingCount`) is summed per cell and per 250 m radius around a pharmacy. It is used as a footfall proxy: many ratings means a busy area. It is measured only for a small set of pilot provinces because that API field is billed at the Enterprise tier, and a cell that hits the result cap is split into finer cells.
- **Pharmacy position vs. population density.** Each pharmacy gets the population of its own H3 cell and of the surrounding ring of cells, the density around it (people/km², computed from the true cell area), the count of commercial venues and transit stops within 250 m, and the distance to the nearest health facility. From these, pharmacies are labelled by setting, for example next to a clinic or in a dense residential area.
- **Display.** Density contour and rating-count cells are toggleable layers under the pharmacy points. Pharmacies with no measurement are shown greyed out instead of as zero.

This repo contains only the two simpler maps, since the explorer depends on paid API data and on private pharmacy records.

## Data and licence

Population: [Kontur Population](https://data.humdata.org/dataset/kontur-population-turkiye) 2023-11, CC BY 4.0 (modelled estimate). Code: MIT.
