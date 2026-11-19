# 🌱 Synthetic Renewable Energy Data Generation

A workflow for retrieving and cleaning weather measurement and forecast data, and for
generating synthetic wind power time series from it: ERA5 reanalysis extrapolated to hub
height, a terrain-gated bias correction against local station measurements, an age-resolved
turbine degradation model, and optional wake losses.

---

## 📊 Workflow: Weather Data Processing

### 1. 📥 Download Raw Data

Raw data can be found on the DWD (German Weather Service) open data website:
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/

Relevant subdirectories:

- `air_temperature/`
- `precipitation/`
- `solar/`
- `wind_test/` *(contains wind speed standard deviation; `wind/` does not)*

Each directory contains:

- `historical/`: data until end of 2023
- `recent/`: last 17 months
- `now/`: current day

> ✅ **Only `historical/` and `recent/` are needed** <br>
> ✅ **You only need to update `recent/` regularly** <br>
> ✅ **You need to update `historical/` once a year**

You can control what gets downloaded using:

- `config['scraping']['get_historical']`
- `config['scraping']['get_recent']`

To clear old raw data:
→ Set `config['scraping']['overwrite'] = True`

**Run:**

```bash
python -m utils.scrape_stations
```

---

### 2. 📦 Unzip Raw Data

The downloaded files are zipped. To unzip:

```bash
python -m utils.unzip
```

---

### 3. 🗃️ Write Raw Data to the Database

This step:

- Combines weather variables per station into a list of DataFrames
- Stores them temporarily in `.pkl` files

> 💡 Make sure no old `.pkl` files are present before writing new data!

Control what gets written using:

- `config['write']['write_recent']`
- `config['write']['write_historical']`

**Run:**

```bash
python -m utils.raw_to_db
```

---

### 4. 🧹 Clean Weather Measurement Data

The raw data is written as-is. In this step:

- Stations with too much missing data are filtered out
- Use `config['write']['threshold']` to set the missing value limit

You can choose to clean:

- `config['write']['clean_pv'] = True` → for PV-relevant data
- `config['write']['clean_wind'] = True` → for wind-relevant data

**Run:**

```bash
python -m utils.clean_data
```

The result is a `raw` directory with one Parquet file per station, for PV and for wind.

---

### 5. ☁️ Extract Numerical Weather Predictions (NWP)

Forecast data is stored in two main tables:

- `SingleLevelFields`: relevant for PV
- `MultiLevelFields`: relevant for wind

You can control what gets queried using command-line flags:

- `--get_pv`: extract PV forecasts from `SingleLevelFields`
- `--get_wind`: extract wind forecasts from `MultiLevelFields`
- `--get_wind_vertical`: extract vertical wind profiles from `AnalysisFields`

**Run for PV, wind and vertical wind:**

```bash
python -m utils.get_nwp --get_pv --get_wind --get_wind_vertical
```

The result is a directory called `singlelevelfields` or `multilevelfields` for the requested NWP data.

---

## ⚡ Generate Synthetic Wind Power Time Series

`generate_wind.py` is the single entry point for wind power synthesis. Its behaviour is
fully controlled by a YAML config passed via `--config`; the chain logic itself lives in
the `round2/` package.

**Run with the default configuration** (`configs/round2/M4.yaml`):

```bash
python generate_wind.py
```

**Run a specific configuration:**

```bash
python generate_wind.py --config round2/S3.yaml
```

### Chain components

| Component | Config key | Options |
|---|---|---|
| Wind speed extrapolation to hub height | `shear` | `power_law` (dynamic exponent from ERA5 10 m/100 m) or `most` (Monin-Obukhov-stability-corrected log-linear law) |
| Bias correction of the ERA5 input | `correction` | `off`, `height_consistent` (terrain-gated quantile mapping) |
| Turbine aging / degradation | `aging_model` | `none`, `const` (linear), `weibull`, `weibull_step` |
| Wake losses | `wake.enabled` | `true`/`false`, model `noj` (PyWake, Jensen) with decay constant `wake.k` |
| Air density | `density` | `v1_mixed`, `static_1225`, `dynamic` |
| Sensitivity-analysis scalers | `wind_level_factor`, `power_curve_scale`, `z0_scale` | multiplicative overrides for global sensitivity analysis |
| Curtailment (after `power_park`) | `curtailment.enabled` (top-level block) | off by default (output unchanged); on: bat permits -> negative-price market response -> redispatch setpoints, `power_park` = curtailed, `power_park_avail` = available; see `curtailment/` and `parks/README_curt_release.md` |

`configs/round2/` ships a range of pre-built configurations covering these combinations
(single-component ablations, the stability-corrected alternative, sensitivity-analysis
variants); see `configs/round2/README.md` for an index. `configs/round2/SITE_v2.yaml` is
the configuration used to generate the published multi-site dataset (see Data Availability
in the paper for the dataset DOI).

`generate_wind_reninja.py` produces the Renewables.ninja comparison baseline used
throughout the validation.

---

## ☀️ Generate Synthetic PV Park Time Series

One entry point, one run config per release:

```bash
synthre/bin/python generate_pv.py --config configs/runs/parks_pv_v4.yaml                 # all stages of the config
synthre/bin/python generate_pv.py --config configs/runs/parks_pv_v4.yaml --stages generate,assemble --parks SEL1,SEL2
```

- **Locations** (data root, MaStR, CAMS models, ICON, DEM, ...) come from `settings.py`: environment variable >
  `configs/settings.yaml` (or `$SYNTHRE_SETTINGS`) > defaults. Template: `configs/settings.example.yaml`.
  `synthre/bin/python settings.py` prints the resolved values. Secrets (`WEATHER_DB_URL`, ADS key) stay in the environment.
- **Run config** (`configs/runs/*.yaml`, schema in `parks_pv/run_config.py`): release name, park selection, period,
  step, CAMS correction model, switches (`geometry.mode: estimate | register | manual`,
  `snow.model: off | depth_linear | depth_slide`, `aging.cohorts`, `horizon`).
- **Stages** (`parks_pv/stages.py`): `prepare` (OSM/PVGIS, CAMS centre + ring, corrected CAMS, ICON-D2) → `baseline`
  (register geometry, park parameters) → `geometry` → `sites` → `configs` → `generate` → `assemble` → `tables`.
- **Module angles** (`geometry.mode`): `estimate` measures tilt and azimuth from open state geodata (OSM outline,
  DOP20, DGM1, point cloud / DOM; `parks_pv/geometry_estimate.py`). Each park is estimated once and kept in the store
  `<geometry_dir>/<method>/<park_id>/` (`parks_pv/geometry_store.py`); later runs reuse it, parks without an accepted
  measurement fall back to the register geometry. `register` uses the MaStR classes only, `manual` takes the
  sub-arrays from a CSV (`park_id, tilt, azimuth, share`), e.g. for a fictitious park. The estimation also runs on
  its own: `synthre/bin/python scripts/pv/estimate_geometry.py --config configs/runs/parks_pv_v4.yaml [--parks ...] [--status]`.
- **MaStR**: `scripts/pv/mastr_ensure.py` checks for the parsed register, parses an unpacked export or downloads the
  current BNetzA full export (no automatic re-download). Missing ICON-D2 snow data only degrades the snow driver to ERA5
  (warning, manifest `icon_status`).
- **Park spec** (`parks` in the run config instead of `selection`): parks typed in by hand, fictitious or taken from
  the MaStR (`synthre/bin/python -m parks_pv.park_spec mastr|check ...`); format in [docs/park_spec.md](docs/park_spec.md).
- `configs/runs/parks_pv_v4.yaml` reproduces the frozen release `parks_pv_v4_30min` bit-identically; a release
  directory with a `FROZEN` file is never written (stages generate, assemble, tables).

## 🔧 Wind Turbine Power Curve Scraping

Information is scraped from:
https://www.wind-turbine-models.com

The scraping must follow this order:

1. `get_power_curve.py`
2. `get_power_curve_specs.py`

> ✅ No changes needed — scripts are ready to run.

This will generate **5 CSV files** for ~400 turbines:

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `turbine_power.csv`    | Power curves for all ~400 turbines                                          |
| `turbine_cp_data.csv`  | Cp values for a few turbines                                                |
| `turbine_ct_data.csv`  | Ct values for a few turbines                                                |
| `turbine_specs.csv`    | Rotor diameter and hub height (may contain non-numeric/missing entries)     |
| `turbine_names.csv`    | Basic metadata for all turbines                                             |

---

## ✅ Tests

```bash
pytest                    # fast unit tests
pytest -m slow            # + end-to-end generation smoke test (needs real ERA5 input on disk)
```

Covers the chain modules (`round2/aging.py`, `correction.py`, `stability.py`,
`evaluation.py`), the parks driver (`parks/`), the curtailment layers (`curtailment/`)
and an end-to-end run of `generate_wind.py`.
