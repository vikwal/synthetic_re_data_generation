# pv — modular synthetic PV chain

Replaces the monolithic `generate_pv.py` of 2025 (kept as `archiv/generate_pv_legacy.py`, not
tracked). Review and Trianel benchmark: `~/Work/FL_Contribution/reports/pv_generator_review.md`.

```
driver (pv.drivers)  ──►  solar position / clear sky (pv.timebase)
   station (imputed DWD 30 min) | CAMS (ADS CSV or WeatherDB) | ERA5 (WeatherDB cells)
   + ERA5 met (temp_air, wind_speed, pressure, albedo = fal)
        │
        ▼
DNI by closure, POA per sub-array (pv.transposition: Hay-Davies, IAM physical, PVGIS/DEM horizon)
        │   sub-arrays: fixed list, terrain-derived (pv.terrain + pv.arrays), Dirichlet, inter-row (gcr)
        ▼
cell temperature (Faiman | PVsyst) → DC (PVWatts | CEC single diode, pv.modules)
        │
        ▼
aging (pv.aging: LID + linear + late, era table) → soiling (pv.soiling: HSU) → DC losses
        │
        ▼
inverter (PVWatts | Sandia, pv.inverter; AC rating = kwp_dc / dc_ac → clipping) → AC losses, availability
```

Conventions: hourly UTC frames labelled by the interval **start**; solar position at the interval
centre; every driver converted in `pv.timebase.to_hourly_mean` with its own source convention
(DWD/CAMS/ERA5 label the interval end, the Trianel meters the start).

| module | content |
|---|---|
| `config.py` | park schema (`normalise`, `load_park`), legacy repair (`from_legacy`), station grid configs |
| `chain.py` | `run_park(cfg, driver, horizon, pm, rain)` → hourly frame (`OUT_COLS`) |
| `drivers.py` | `station_imputed`, `cams_from_csv`/`cams_station`, `era5_solar`, `era5_met`, `cams_pm`, `merge_driver` |
| `transposition.py` | `dni_from_closure`, `poa` (horizon shading, IAM) |
| `arrays.py` | `Array`, `effective_angles` (rows on slopes), `cluster_arrays`, `dirichlet_arrays`, `inter_row_factors`, `gcr_from_density` |
| `terrain.py` | GLO-30 slope/aspect in a footprint, `dem_horizon`, OSM/buffer polygons |
| `modules.py` | CEC libraries (bundled 2019 + 2026 CSV), `era_proxy_module`, `dc_pvwatts`, `dc_cec`, `low_light_curve` |
| `inverter.py` | `pick_inverter`, `ac_pvwatts`, `ac_sandia` |
| `aging.py` | `degradation_series` by era, `inverter_availability` |
| `soiling.py` | `hsu_ratio`, repaired `legacy_erf_loss` |
| `snow.py` | ERA5 snow fields at a point (`era5_snow_point`), Marion coverage (`coverage`, pvlib), depth gate, `loss_factor`; validated on Trianel winter 2023/24: linear depth rule 0–5 cm recommended |
| `angles.py` | `poa_bank`, `grid_fit`, `nnls_mixture`, `morning_evening_asymmetry` |
| `metrics.py` | benchmark metrics, plateau detection, residual by sun position, snow candidates |

Entry point of the park release: `generate_pv.py --config configs/runs/<release>.yaml` (stages in
`parks_pv/stages.py`). Benchmarks: `scripts/pv/benchmark_trianel.py`, `scripts/pv/terrain_trianel.py`,
`scripts/pv/module_spread.py` (read confidential Trianel inputs and write anonymised outputs only).
Tests: `tests/test_pv.py`.
