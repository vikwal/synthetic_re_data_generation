# CAMS correction from PostgreSQL

The production entry point is `scripts/cams_bc/apply_parks.py` on every deployment.
It reads neighbour inputs and their station coordinates from PostgreSQL once per
requested period, then reuses them for every park. It does not load or fall back to
`neighbours_raw_30min.parquet`. The Parquet helpers remain for historical training
experiments and regression fixtures, not the production park correction.

Install the reproducible CPU environment with `python -m pip install -r requirements-cams.txt`.
The same versions are used on source and mirror deployments. PyTorch is optional
for historical MLP experiments; the deployed XGBoost model does not require it.

## Connections and files

Use `WEATHER_DB_URL` for inference and `WEATHER_DB_WRITE_URL` for maintenance, or
configure the libpq services `synthetic_weather` and `synthetic_weather_writer` in
`~/.pg_service.conf`. Keep connection values outside Git; use a password file with
mode 0600. Inference opens a read-only, repeatable-read transaction. The writer
needs SELECT on raw inputs, ownership of the two derived tables, and TEMP access.
Initial schema setup requires an administrator. No server name is embedded in code.

Model files, station registries, centre/ring CAMS files and outputs remain external
artifacts. Configure paths via the existing settings or CLI options. Model inference
defaults to CPU; `--device cuda:0` explicitly selects GPU. The PV release pipeline
uses the same entry point and `correction_device` setting (`PV_CORRECTION_DEVICE`). Early-stopping tree limits
are preserved on model reload.

## Database interface and setup

The four `cams_weather` views expose one interface:

| View | Key / contents |
| --- | --- |
| `met_points` | `cell_id`, `lat`, `lon` |
| `met` | `cell_id`, UTC `timestamp`, nine raw meteorological variables |
| `solar_points` | `cell_id`, `lat`, `lon` |
| `solar` | `cell_id`, `valid_time`, raw ERA5 solar variables |

Generate transactional setup SQL and apply it with an administrator connection:

```bash
python scripts/cams_bc/setup_database.py --layout spatial \
  --source-inputs --stations data/cams_bc/splits.csv > /tmp/cams_setup.sql
psql "$WEATHER_DB_WRITE_URL" -v ON_ERROR_STOP=1 -f /tmp/cams_setup.sql
```

`spatial` adapts the existing `era5_wind_grid` / geometry-keyed solar tables.
`point_id` adapts normalized `era5` / `grid_points` and solar tables. This is a
storage-schema choice, not a different inference method. With mirrored raw station
tables already present, omit `--source-inputs`; omit `--stations` to preserve an
installed model registry. `--reader-role NAME` grants access to the shared views
and neighbour tables. Re-running setup preserves existing observations and results.

With `--source-inputs`, raw observation and CAMS views project the existing
`stations` and `cams_radiation` tables without copying their bulk data. On a mirror,
tables with the same column contract supply those inputs. Do not replace physical
mirror tables with views. Registry coordinates must match the model's station
registry; additional stations require an explicit modelling decision.

## Neighbour inputs and quality

- `cams_neighbour_stations`: model station IDs and coordinates.
- `dwd_solar_observations_10min`: GHI/DHI in J/cm² per 10 minutes, UTC interval end.
- `cams_station_radiation_15min`: CAMS GHI, clear-sky GHI and TOA in Wh/m² per
  15 minutes, with original reliability values retained.
- `cams_neighbour_inputs_30min`: measured values in W/m², CAMS references,
  available-observation counts, day mask and derived BSRN validity flags.
- `cams_neighbour_dataset`: QC version, station count and published coverage.

Run the same refresh command on either deployment after raw inputs are updated:

```bash
python scripts/cams_bc/refresh_neighbours.py --start 2023-07-24
```

All station results and the completion marker publish in one transaction. A failed
refresh leaves the previous dataset intact. Readers keep a consistent snapshot.
The source DWD tables do not contain original QN codes: the stored validity flags
are derived using the established `solar_bc` rules, not presented as DWD flags.
Partial DWD half-hours retain the mean of available observations; CAMS half-hours
require both quarters. Measurements are not imputed. The four nearest stations,
missing-value handling and available-station weight normalization are unchanged.
Requests beyond published coverage yield missing neighbour features.

## Applying and checking

```bash
python scripts/cams_bc/apply_parks.py --points-json POINTS.json \
  --cams-dir CENTRE_CSVS --ring-dir RING_CSVS --models-dir MODELS/xgb_v5 \
  --out-dir OUTPUT --start 2026-08-01 --end 2026-09-01
```

Centre and ring CAMS files must already exist; this change does not add a download
step. The correction uses historical measurements and temporal context, including
future context, and is intended for retrospective synthetic time series.

`tests/test_cams_database.py` covers units, interval boundaries, partial inputs and
QC rejection. `scripts/cams_bc/check_database.py` exercises the shared weather and
neighbour database path and compares predictions using the same external feature
fixture and model files across deployments. No fixtures or model weights enter Git.
