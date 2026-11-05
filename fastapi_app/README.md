# Synthetic Renewables — FastAPI app (v2)

A local scenario workspace for synthetic wind and solar generation. The Streamlit dashboard in `streamlit/` is untouched; `fastapi_app_v1_snapshot/` holds the earlier Streamlit-like port.

## Run

From the project root, with the pipeline's dependencies (pandas, scipy, metpy, pvlib, …) installed plus `pip install -r fastapi_app/requirements.txt`:

```
uvicorn fastapi_app.main:app --reload
```

Open http://127.0.0.1:8000. Scenarios and results persist in `fastapi_app/storage/` (override with `SYNTH_RE_STORAGE_DIR`). Python 3.9+.

## Features

- **Scenarios**: create, rename, duplicate, delete; edits autosave.
- **Setup**: Leaflet map (street/light/satellite, NWP grid layer), search, draggable turbines, click-to-place, wake (Jensen/Park) and per-turbine ageing.
- **Runs** execute in the background with a progress bar.
- **Results**: KPI cards; tabs Overview (loss waterfall ideal→ageing→wake→net, per-turbine losses, ageing timeline, per-turbine table), Resource (wind rose, speed histogram / irradiance), Performance (hour×month heatmap, duration curve, multi-year), Explorer (daily, quarterly, power curve), Data (paged table).
- **Compare**: 2–6 runs side by side.
- **Export**: CSV, Parquet, summary JSON, print-to-PDF.
- **Pre-trained model**: pre-train Linear regression, GRU, TCN or a compact TFT on one or more generated datasets (weather window → output as a share of capacity), then fine-tune on your measured power. GRU/TCN/TFT need PyTorch (`pip install torch`); linear regression works without it. Models are saved in `storage/models/` and can be downloaded. The weather the model learns from is an NWP forecast model (ICON-D2 or ECMWF IFS), read from the sample package locally or from the server's archive (see below); several models can be combined in one run.

## Layout

`mlprep.py`, `mlmodels.py`, `mlservice.py` pre-trained model pipeline · `realcompare.py` real vs synthetic · `api.py` routes · `service.py` run/describe/figures · `core.py` data + generation · `wake.py`, `layout.py`, `analytics.py` pure logic · `charts.py` Plotly dicts · `storage.py`, `jobs.py` persistence and background jobs · `static/js/` ES-module frontend.

## Tests

`pytest fastapi_app/tests`

Leftover stub files (`templates/index.html`, `templates/results.html`, `static/app.js`, `static/results.js`) are unused and can be deleted.

## Sample-data package (synthetic wind parks + forecast runs)

Unpack `beispieldaten_*` into `data/` (or set `SYNTH_RE_PACKAGE_DIR`). The package's wind parks are stored as generated
runs by `POST /api/ml/package/import` (the Pre-train form no longer has an import button). The Pre-train weather
models **ICON-D2** and **ECMWF IFS** then read the park's own forecast runs from the package and give one hourly series: for each hour the most recent forecast issued between
*lead time from* and *lead time to* hours earlier is used (12 to 36 h ≈ day-ahead), and the park's grid points are
averaged with the turbine weights from `turbine_points.csv`. ICON-D2 winds are interpolated from the model layers to
10 m and 100 m; ECMWF uses its single-level fields (no surface pressure). Code: `fastapi_app/parkdata.py`.
