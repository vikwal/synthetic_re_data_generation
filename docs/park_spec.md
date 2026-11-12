# Park spec: input of the PV generator

The interface between a front end (dashboard) and the PV generator `generate_pv.py`. A park spec describes one or more
PV parks, typed in by hand (also fictitious parks) or built from the Marktstammdatenregister (MaStR). The generator
turns every spec into the internal selection row (`parks_pv/park_spec.py:to_row`) and runs the same stages as for the
releases; a MaStR park given as spec gives bit-identical series to the release `parks_pv_v4_30min`.

## File

YAML or JSON, one park or a list under `parks`:

```yaml
parks:
  - park_id: demo_01                # letters, digits, _ and -; key of all caches and output files
    name: Demo park                 # optional label
    lat: 52.31                      # WGS84
    lon: 13.62
    ac_kw: 10000                    # AC rating (MaStR Nettonennleistung)
    dc_kwp: 12500                   # DC rating (MaStR Bruttoleistung)
    commissioning: 2021-06-01       # first commissioning (aging starts here)
    commissioning_full: 2023-04-01  # optional: park complete, the series starts here (default: last cohort / commissioning)
    cohorts:                        # optional construction phases (module era and aging per phase)
      - {commissioning: 2021-06-01, dc_kwp: 10000}
      - {commissioning: 2023-04-01, dc_kwp: 2500, area_ha: 2.0}
    area_ha: 12.0                   # optional fenced area -> row spacing (default: era density table)
    geometry:                       # optional, default mode = run config
      mode: manual                  # estimate | register | manual
      arrays:                       # manual: tilt against the horizontal, true azimuth (180 = south), share
        - {tilt: 15, azimuth: 90,  share: 0.5}
        - {tilt: 15, azimuth: 270, share: 0.5}
```

| key | required | meaning / default |
|---|---|---|
| `park_id`, `lat`, `lon`, `ac_kw`, `dc_kwp` | yes | DC/AC is clipped to 1.0-1.6 by the chain |
| `commissioning` | yes (not with `mastr.unit_ids`) | date of the first unit |
| `commissioning_full` | no | date the park reached its rating; default the latest cohort date |
| `cohorts` | no | `[{commissioning, dc_kwp, area_ha?}]`; default one cohort |
| `area_ha` | no | fenced area; default density by commissioning year |
| `geometry.mode` | no | `estimate`: angles measured from open state geodata (Germany, states with a fetcher), stored per park and reused; `register`: register classes only; `manual`: `geometry.arrays` |
| `orientation_class`, `tilt_class` | no | MaStR wording (`Süd`, `Ost-West`, ... / `< 20 Grad`, `20 - 40 Grad`, ...); default derived from `geometry.arrays`, else `Süd` / `20 - 40 Grad`. Used by mode `register` and as prior of `estimate` |
| `secondary_orientation`, `secondary_tilt_class`, `uniform_flag`, `power_limit` | no | further register fields (MaStR wording) |
| `bundesland`, `landkreis`, `gemeinde` | no | default from the nearest MaStR unit within 15 km (state of the geodata fetch); empty abroad |
| `mastr` | no | `{unit_ids: [...], lokation_ids: [...]}`: with unit ids the commissioning dates and age cohorts come from the units |

Fields of the federated-learning setup (`client`, `kind`, `cls`, `region`) are not part of the spec; they only label
the release tables.

## MaStR parks

```bash
synthre/bin/python -m parks_pv.park_spec mastr SEL917515222981 [SEL... of the same site] --out park.yaml [--geometry estimate]
```

writes the spec of a MaStR park from its Lokation number(s): capacities, coordinates, area, register classes, units
(`parks_pv.park_spec.from_mastr`). Several Lokationen of one construction site are merged; the largest is the default
`park_id`. Values may be edited afterwards (a "what if" variant of a real park).

## Check

```bash
synthre/bin/python -m parks_pv.park_spec check park.yaml
```

lists all errors (unknown keys, missing fields, ranges, dates, manual arrays) or prints the derived rows.

## Run

Run config with `parks` instead of `selection`:

```yaml
release: dashboard_demo
variant: _30min
parks: /path/to/park.yaml
period: {start: "2023-07-24", end: "2026-08-01"}
step: 30min
geometry: {mode: estimate, method: v4}     # default for parks without geometry.mode
snow: {model: depth_slide}
stages: [prepare, baseline, geometry, sites, configs, generate, assemble, tables]
```

```bash
synthre/bin/python generate_pv.py --config run.yaml
```

Output: `<data_root>/synthetic/solar/<release><variant>/synth_<park_id>.parquet` (columns in the release README),
`manifest_<park_id>.json`, `parks.csv`, `pv_arrays.csv`, `sites.csv`.

## Duration

| situation | what runs | duration |
|---|---|---|
| park whose inputs exist (all MaStR parks of the releases) | baseline ... tables | seconds per park |
| new location | stage `prepare`: OSM + PVGIS, CAMS from ADS (9 points, queue and daily quota), CAMS correction (GPU), ICON-D2 points (NAS) | hours, CAMS can take longer when the ADS queue is full |
| `geometry.mode: estimate` for a new park | state geodata tiles (download), window method | minutes per park, then stored |

Not tested yet: locations outside Germany (ERA5, ICON-D2 and the geodata cover Germany).
