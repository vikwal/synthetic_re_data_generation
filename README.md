# 🌱 Synthetic Renewable Energy Data Generation

This repository provides a full workflow for retrieving, processing, and managing weather measurement and forecast data, and for generating synthetic wind power time series from it.

It is the code base for *"A Measurement- and ERA5-Driven Simulation Framework for Synthetic Wind Power Time Series Accounting for Turbine Aging"* (Applied Energy, round-2 revision). The published dataset (200 German locations, 6 turbine configurations, hourly, July 2023-April 2026) is released separately under CC BY 4.0 on Mendeley Data; see the paper's Data Availability statement for the DOI.

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

## ⚡ Generate Synthetic Wind Power Time Series (round-2 / current)

`generate_wind_era5_v2.py` is the active pipeline that produced every result in the
paper. It extends the ERA5 shear method with the round-2 chain (implemented in the
`round2/` package):

| Component | Module | Paper section |
|---|---|---|
| Terrain-gated ERA5 bias correction (quantile mapping) | `round2.correction` | §3.3 (bias correction) |
| MOST-based stability-corrected extrapolation | `round2.stability` | §3.2 (extrapolation) |
| Weibull-based turbine aging | `round2.aging` | §3.4 (aging) |
| PyWake (Jensen) wake losses | `round2.wake` | validation framework, §4.2 |
| Global sensitivity analysis (Morris/Sobol via SALib) | `scripts/round2/wp6_morris.py`, `wp6_sobol.py` | §5.8 |

Each configuration in the paper's M0-M4 / S1-S4 ladder is a YAML file in
`configs/round2/` (e.g. `M4.yaml`, `SITE_v2.yaml` for the public dataset export).
`tests/test_v2_equals_v1.py` checks that the round-2 chain reduces to the round-1
generator bit-for-bit when its extensions are switched off.

**Run a single configuration:**

```bash
python generate_wind_era5_v2.py --config configs/round2/M4.yaml
```

**Run the full ladder and regenerate the paper's tables/figures:**

```bash
bash scripts/round2/run_full_sequence.sh   # or scripts/round2/run_ladder.py directly
python scripts/round2/make_summary.py      # -> results/quantitative/summary.md
python scripts/round2/regen_paper_figs.py  # -> figures used in results/figures/
```

`results/` in this repository holds exactly the quantitative results (CSVs/summary
tables) and figures reported in the paper — see `results/README.md`. The full
sensitivity/ablation sweep and intermediate diagnostics live in the (gitignored,
local-only) `figs/` and the server's working copy of `results/round2/`.

`era5_bc/` is a separate, self-contained experiment (LSTM/Transformer bias
correction of ERA5 station winds) discussed in the paper (§5.4) as a **negative
result**: it improves station-level RMSE but does not transport to farm-level
accuracy, so the deployed chain uses the terrain-gated quantile-mapping correction
instead. Kept for reproducibility of that finding, not part of the main chain.

### Superseded / round-1 scripts

`generate_wind.py`, `generate_wind_era5_fric.py`, `generate_wind_nwp.py` and
most legacy config directories (`_alphaI`, `_loglaw`, `_noage`, the plain
`real_wind_parks_era5_fric`) were round-1 (initial submission) variants and
are no longer used; they are kept locally under `archiv/` (not part of this
repository) for reference.

`generate_wind_era5.py` and `configs/real_wind_parks_era5/config_07374.yaml`
are the one round-1 file/config kept in the repository on purpose: they are
the v1 reference that `tests/test_v2_equals_v1.py` runs against to prove
`generate_wind_era5_v2.py` reduces to it bit-for-bit when the round-2
extensions are switched off. Do not delete them without updating that test.

`generate_wind_reninja.py` remains active — it produces the Renewables.ninja
comparison baseline used throughout the validation section.


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
pytest
```

Covers the round-2 modules (`round2/aging.py`, `correction.py`, `stability.py`,
`evaluation.py`) and a bit-exact regression check against the round-1 generator.
