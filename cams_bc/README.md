# cams_bc — CAMS-to-measurement correction for unseen locations

Sister of `era5_bc/` (wind) and of `NWP/ERA5/solar_bc/` (imputation at the measuring stations).
Question: can a pointwise model, trained where DWD measures GHI/DHI, sharpen the CAMS Radiation
Service at a location **without** measurement (a PV park)? That is spatial transfer, so the
validation is strict: test stations are never seen in training, the 3-fold cross-validation uses
disjoint station groups (`data/cams_bc/splits.csv`, stratified over a lat/lon grid).

| Module | Content |
|---|---|
| `config.py` | paths (`DATA_ROOT`), ring geometry, split constants |
| `features.py` | feature groups (`cams`, `geom`, `ring3`, `ring8`, `ring_agg`, `context`, `era5_solar`, `era5_met`, `static`) and the ladder of the predictor study |
| `targets.py` | target forms (`kc` with clear-sky² weights = loss in W/m², `kc_unweighted`, `ratio`, `residual`, `absolute`) |
| `data.py` | 30-min feature frames: `assemble()` is shared by the station dataset and the park application |
| `splits.py` | test set (20 %) and station folds |
| `models.py` | `XGBModel` (GPU hist, early stopping) and `MLPModel` (torch, two outputs, weighted MSE) |
| `metrics.py` | pooled/per-station/hourly metrics, skill vs raw CAMS |
| `hpo.py` | Optuna search spaces |

Scripts in `scripts/cams_bc/`: `build_dataset.py` → `predictor_study.py` → `hpo.py` → `final_eval.py`
→ `apply_parks.py`. Results in `results/cams_bc/`; models under `${DATA_ROOT}/synthetic/cams_bc/models/`.

Inputs are products of the NWP pipelines (read only): `solar_bc/processed_30min` (DWD 10-min
measurements → 30 min, ERA5 solar/met caches, CAMS at the station, QC flags), `cams_rad_cache/ring15min`
(CAMS at 3 and 8 km N/E/S/W around every station). Time labels are interval END, as there.

Environment: `~/Work/NWP/ERA5/.venv` (torch 2.7 + CUDA, optuna 4.9, xgboost 3.0.5 — the 3.4 wheel is
built for CUDA 13 and falls back to CPU with driver 570).
