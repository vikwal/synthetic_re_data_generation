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

`configs/round2/` ships a range of pre-built configurations covering these combinations
(single-component ablations, the stability-corrected alternative, sensitivity-analysis
variants); see `configs/round2/README.md` for an index. `configs/round2/SITE_v2.yaml` is
the configuration used to generate the published multi-site dataset (see Data Availability
in the paper for the dataset DOI).

`generate_wind_reninja.py` produces the Renewables.ninja comparison baseline used
throughout the validation.

---

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
`evaluation.py`) and an end-to-end run of `generate_wind.py`.
