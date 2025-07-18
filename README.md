# 🌱 Synthetic Renewable Energy Data Generation

This repository provides a full workflow for retrieving, processing, and managing weather measurement and forecast data, useful for synthetic generation of renewable energy datasets.

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

> ✅ **Only `historical/` and `recent/` are needed**
> ✅ **You only need to update `recent/` regularly**

You can control what gets downloaded using:

- `config['scraping']['get_historical']`
- `config['scraping']['get_recent']`

To clear old raw data:
→ Set `config['scraping']['overwrite'] = True`

**Run:**

```bash
python utils/scrape_stations.py
```

---

### 2. 📦 Unzip Raw Data

The downloaded files are zipped. To unzip:

```bash
python utils/unzip.py
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
python utils/raw_to_db.py
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
python utils/clean_data.py
```

The result of this script is a directory called raw, where are CSV-file stored, each for PV and for wind, with raw weather station data.

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
python utils/get_nwp.py --get_pv --get_wind --get_wind_vertical
```

The result is a directory called `singlelevelfields` or `multilevelfields` for the requested NWP data.

---

## Generate Synthetic Wind Power Time Series

```bash
python generate_wind.py
```

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