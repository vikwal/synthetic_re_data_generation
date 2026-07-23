#!/usr/bin/env python3
"""Ask 33 driver: publishable v2 SITE dataset — 200 classified stations,
ERA5-only chain (M5-equivalent free-stream, no wakes).

Stages (subcommands, run in order; each is independently re-runnable):
  prep         write data/round2/site_commissioning.csv (v1 continuity, or
               --resample-ages for a fresh seeded fleet) + DB masterdata precheck
  generate     run generate_wind per station (subprocess, run_ladder
               pattern) into /mnt/nvme2/synthetic/wind/round2/SITE_v2/
  postprocess  drop internal columns, merge measured DWD weather (hourly,
               NaN gaps, measured_ prefix), write release synth_{id}.parquet
  tables       wind_missing_values_hourly.csv, site_branch_manifest.csv,
               nwp_coverage.csv into the release dir
  verify       consistency + physics checks; prints the HANDOFF summary

Chain config: configs/round2/SITE_v2.yaml. Spec: HANDOFF.md "Ask 33".
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

PYTHON = os.path.join(REPO, "synthre", "bin", "python")
EXPERIMENT = "SITE_v2"
SYNTH_BASE = "/mnt/nvme2/synthetic"
RUN_DIR = os.path.join(SYNTH_BASE, "wind", "round2", EXPERIMENT)
GEN_DIR = os.path.join(REPO, "configs", "round2", "_generated", "site_v2")
COMM_TABLE = os.path.join(REPO, "data", "round2", "site_commissioning.csv")
PREDICTED_CLASSES = os.path.join(REPO, "data", "round2", "predicted_classes.csv")
BRANCH_ASSIGNMENT = os.path.join(REPO, "data", "round2", "branch_assignment.csv")
STATION_CORR_TABLE = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")
TOPO = os.path.join(REPO, "data", "round2", "topo_features.csv")
# Site-only artifacts dir (referenced by SITE_v2.yaml artifacts_dir). Holds
# per-station correction JSONs keyed by station id, so the park/site filename
# collision in data/round2/correction/ (park_XXXXX shadows station XXXXX for 7
# ids) cannot leak a park's gate decision into a site. Sites anchor to their
# own station at 0 km per Ask 33.
SITE_ARTIFACTS = os.path.join(REPO, "data", "round2_site")
V1_WIND_PARAMETER = "/mnt/nvme2/synthetic/wind/wind_hourly_age_20251103/wind_parameter.csv"
MEASURED_DIR = "/mnt/lambda1/nvme1/synthetic/raw/wind"
ICON_DIR = "/mnt/nas/open-meteo/wind/icon_d2"
WIND_AGES = os.path.join(REPO, "data", "wind_ages.npy")
DEFAULT_RELEASE = os.path.join(SYNTH_BASE, "wind", "site_v2_20260714")

OUTPUT_START = "2023-07-24 00:00"
OUTPUT_END = "2026-04-30 23:00"
N_TURBINES = 6

# Human-readable published column names (Ask 35). The generator emits these
# directly when output_naming=='readable' (SITE_v2.yaml); postprocess also
# applies readable_rename_map() so already-generated native run CSVs are mapped
# idempotently without re-running the 200-station synthesis.
ERA5_BASE_COLS = ["u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
                  "wind_gust_10m", "friction_velocity", "temperature_2m",
                  "pressure", "dew_point_2m", "sensible_heat_flux",
                  "boundary_layer_height", "gravity_wave_dissipation"]
DERIVED_COLS = ["wind_speed_100m", "relative_humidity",
                "saturated_vapor_pressure", "density"]
TURBINE_COLS = [f"{base}_t{i}" for i in range(1, N_TURBINES + 1)
                for base in ("wind_speed", "surface_layer_flag", "density", "power")]
MEASURED_MAP = {"wind_speed": "measured_wind_speed",
                "wind_direction": "measured_wind_direction",
                "std_v_wind": "measured_wind_speed_std",
                "temperature_2m": "measured_temperature",
                "relative_humidity": "measured_relative_humidity",
                "pressure": "measured_pressure"}
RELEASE_COLS = ERA5_BASE_COLS + DERIVED_COLS + TURBINE_COLS + list(MEASURED_MAP.values())
DROP_COLS = ["qm_factor", "obukhov_L", "stability_class",
             "power_park_free", "power_park", "wake_factor"] + \
            [f"temperature_t{i}" for i in range(1, N_TURBINES + 1)] + \
            [f"w_most_t{i}" for i in range(1, N_TURBINES + 1)]


def site_ids() -> list:
    pc = pd.read_csv(PREDICTED_CLASSES, dtype={"location_id": str})
    st = pc[(pc["kind"] == "station") & pc["observed_class"].notna()]
    return sorted(st["location_id"])


def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range(OUTPUT_START, OUTPUT_END, freq="h", tz="UTC")


# ---------------------------------------------------------------- prep

def prep(resample_ages: bool = False) -> None:
    ids = site_ids()
    print(f"{len(ids)} classified stations")

    if resample_ages:
        ages = np.load(WIND_AGES)
        rng = np.random.default_rng(42)
        start = pd.Timestamp(OUTPUT_START, tz="UTC")
        rows = [{"park_id": sid,
                 "commissioning_date":
                     str((start - pd.Timedelta(days=float(a) * 365.25)).date())}
                for sid, a in zip(ids, rng.choice(ages, size=len(ids)))]
        table = pd.DataFrame(rows)
        source = f"fresh seeded draw from {WIND_AGES} (seed 42)"
    else:
        v1 = pd.read_csv(V1_WIND_PARAMETER, sep=";", dtype={"park_id": str})
        missing = sorted(set(ids) - set(v1["park_id"]))
        assert not missing, f"stations missing in v1 wind_parameter: {missing}"
        table = (v1[v1["park_id"].isin(ids)]
                 [["park_id", "commissioning_date"]]
                 .drop_duplicates("park_id").sort_values("park_id"))
        source = f"v1 continuity from {V1_WIND_PARAMETER}"

    assert len(table) == len(ids)
    late = table[pd.to_datetime(table["commissioning_date"]) >
                 pd.Timestamp(OUTPUT_START)]
    if len(late):
        print(f"NOTE: {len(late)} commissioning dates after output_start "
              f"(age clipped at 0): {late['park_id'].tolist()}")
    table.to_csv(COMM_TABLE, index=False)
    print(f"wrote {COMM_TABLE} ({len(table)} rows, {source})")

    # DB precheck: the generator's CSV fallback is broken, so masterdata
    # must be reachable for the whole run
    from utils import tools
    cfg = tools.load_config(os.path.join(REPO, "configs", "config_wind.yaml"))
    masterdata = tools.get_master_data(cfg["write"]["db_conf"])
    missing_md = sorted(set(ids) - set(masterdata["station_id"].astype(str)))
    assert not missing_md, f"stations missing in DB masterdata: {missing_md}"
    print(f"DB masterdata OK ({len(masterdata)} stations, all {len(ids)} covered)")

    # ERA5 inputs must be gap-free (knn_imputer in read_dfs must be a no-op)
    probe = pd.read_csv(f"/mnt/nvme2/synthetic/raw/wind_era5_v2/Station_{ids[0]}.csv")
    assert probe.isna().sum().sum() == 0, "ERA5 input has NaNs"
    print(f"ERA5 probe {ids[0]}: {len(probe)} rows, 0 NaNs")

    build_site_correction(ids)


def build_site_correction(ids: list) -> None:
    """Write authoritative per-station correction JSONs into SITE_ARTIFACTS,
    keyed by station id, with the branch from branch_assignment (station rows)
    and quantiles from station_correction_table. Avoids the park/site filename
    collision in data/round2/correction/."""
    from round2.correction import QUANTILES
    out = os.path.join(SITE_ARTIFACTS, "correction")
    os.makedirs(out, exist_ok=True)
    t = pd.read_parquet(STATION_CORR_TABLE)
    t["station_id"] = t["station_id"].astype(str)
    t = t.set_index("station_id")
    qe = [f"q_era5_{q:.3f}" for q in QUANTILES]
    qd = [f"q_dwd_{q:.3f}" for q in QUANTILES]
    ba = pd.read_csv(BRANCH_ASSIGNMENT, dtype={"location_id": str})
    ba = ba[ba["kind"] == "station"].set_index("location_id")
    n_collision = 0
    for sid in ids:
        branch = ba.loc[sid, "branch"]
        existing = os.path.join(REPO, "data", "round2", "correction", f"{sid}.json")
        if os.path.exists(existing):
            with open(existing) as f:
                if str(json.load(f).get("location_id", "")).startswith("park_"):
                    n_collision += 1
        payload = {"station_id": sid, "location_id": sid, "branch": branch,
                   "q_era5": t.loc[sid, qe].astype(float).tolist(),
                   "q_target_station": t.loc[sid, qd].astype(float).tolist(),
                   "q_target_model": None}
        with open(os.path.join(out, f"{sid}.json"), "w") as f:
            json.dump(payload, f)
    shutil.copy(TOPO, os.path.join(SITE_ARTIFACTS, "topo_features.csv"))
    print(f"site correction: {len(ids)} station JSONs -> {out} "
          f"({n_collision} corrected park/site collisions), topo_features.csv copied")


# ---------------------------------------------------------------- generate

def merged_config(station_id: str) -> str:
    with open(os.path.join(REPO, "configs", "config_wind.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(REPO, "configs", "round2", f"{EXPERIMENT}.yaml")) as f:
        r2 = yaml.safe_load(f)
    cfg["round2"] = r2["round2"]
    cfg["data"]["synth_dir"] = SYNTH_BASE
    os.makedirs(GEN_DIR, exist_ok=True)
    out = os.path.join(GEN_DIR, f"config_{station_id}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return f"round2/_generated/site_v2/config_{station_id}.yaml"


def generate(ids: list = None, force: bool = False) -> None:
    ids = ids or site_ids()
    os.makedirs(RUN_DIR, exist_ok=True)
    failed, skipped = [], 0
    t0 = time.time()
    for n, sid in enumerate(ids, 1):
        out_csv = os.path.join(RUN_DIR, f"synth_{sid}.csv")
        if os.path.exists(out_csv) and not force:
            skipped += 1
            continue
        rel = merged_config(sid)
        code = (f"import sys; sys.argv=['x']; "
                f"import generate_wind as m; m.main('{rel}')")
        res = subprocess.run([PYTHON, "-c", code], cwd=REPO,
                             capture_output=True, text=True, timeout=3600)
        if res.returncode != 0 or not os.path.exists(out_csv):
            print(f"FAILED {sid}:\n{res.stderr[-2000:]}", flush=True)
            failed.append(sid)
            continue
        rate = (time.time() - t0) / max(1, n - skipped)
        print(f"[{n}/{len(ids)}] {sid} done "
              f"({rate:.0f} s/station avg)", flush=True)
    print(f"generate: {len(ids) - skipped - len(failed)} generated, "
          f"{skipped} skipped (existing), {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


# ---------------------------------------------------------------- postprocess

def circular_mean_deg(series: pd.Series) -> pd.Series:
    """Hourly circular mean of wind direction in degrees, NaN-aware."""
    rad = np.deg2rad(series)
    s = np.sin(rad).resample("h", closed="left", label="left").mean()
    c = np.cos(rad).resample("h", closed="left", label="left").mean()
    mean = (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0
    mean[s.isna() | c.isna()] = np.nan
    return mean


def measured_hourly(station_id: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(MEASURED_DIR, f"Station_{station_id}.parquet"))
    df = df.drop(columns=["station_id"], errors="ignore")
    direction = df["wind_direction"].where(
        (df["wind_direction"] >= 0) & (df["wind_direction"] <= 360))
    plain = df.drop(columns=["wind_direction"]) \
              .resample("h", closed="left", label="left").mean()
    plain["wind_direction"] = circular_mean_deg(direction)
    return plain.rename(columns=MEASURED_MAP)


def postprocess(release_dir: str) -> None:
    ids = site_ids()
    os.makedirs(release_dir, exist_ok=True)
    expected_index = hourly_index()
    manifests = {}
    from generate_wind import readable_rename_map
    rename = readable_rename_map(N_TURBINES)
    for n, sid in enumerate(ids, 1):
        df = pd.read_csv(os.path.join(RUN_DIR, f"synth_{sid}.csv"),
                         sep=";", index_col=0, parse_dates=True)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        # Ask 35: map native run-CSV names to readable ones (idempotent — a no-op
        # if the run was generated with output_naming='readable').
        df = df.rename(columns=rename)
        meas = measured_hourly(sid).reindex(df.index)
        df = pd.concat([df, meas], axis=1)
        unexpected = [c for c in df.columns if c not in RELEASE_COLS]
        missing = [c for c in RELEASE_COLS if c not in df.columns]
        assert not unexpected and not missing, \
            f"{sid}: unexpected={unexpected} missing={missing}"
        df = df[RELEASE_COLS]
        assert df.index.equals(expected_index), f"{sid}: index mismatch"
        df.index.name = "timestamp"
        for i in range(1, N_TURBINES + 1):
            df[f"surface_layer_flag_t{i}"] = \
                df[f"surface_layer_flag_t{i}"].astype(str).eq("True")
        df.to_parquet(os.path.join(release_dir, f"synth_{sid}.parquet"))
        with open(os.path.join(RUN_DIR, f"manifest_{sid}.json")) as f:
            manifests[sid] = json.load(f)
        if n % 25 == 0:
            print(f"[{n}/{len(ids)}] postprocessed", flush=True)
    for name in ("wind_parameter.csv", "turbine_parameter.csv"):
        shutil.copy(os.path.join(RUN_DIR, name), os.path.join(release_dir, name))
    with open(os.path.join(release_dir, "manifests.json"), "w") as f:
        json.dump(manifests, f, indent=2, default=str)
    write_readme(release_dir)
    print(f"postprocess: {len(ids)} parquet files -> {release_dir}")


def write_readme(release_dir: str) -> None:
    readme = f"""# Synthetic wind power at DWD station sites — v2 (ERA5-driven)

One `synth_{{station_id}}.parquet` per site (200 sites), hourly UTC,
{OUTPUT_START} .. {OUTPUT_END}. All synthetic quantities are driven by ERA5
only (bias-corrected where the correction gate applies, see
`site_branch_manifest.csv`); measured DWD weather is merged for reference
with gaps left as NaN.

## Columns
- ERA5 base fields (grid-point extraction): u/v_wind_10m, u/v_wind_100m
  [m/s], wind_gust_10m [m/s], friction_velocity [m/s], temperature_2m [K],
  pressure [Pa], dew_point_2m [K], sensible_heat_flux [J/m2],
  boundary_layer_height [m], gravity_wave_dissipation [W/m2].
- Derived: wind_speed_100m [m/s] (after quantile-mapping correction where
  branch A), relative_humidity [-], saturated_vapor_pressure [Pa],
  density [kg/m3] (all from ERA5 dew point / temperature / pressure).
- Per turbine t1..t6 (see turbine_parameter.csv for types and hub heights):
  wind_speed_t{{i}} [m/s] (MOST profile anchored at 100 m),
  surface_layer_flag_t{{i}} [bool] (hub above 0.1 * boundary_layer_height,
  i.e. surface-layer criterion violated), density_t{{i}} [kg/m3] (hub
  height), power_t{{i}} [W] (incl. Weibull aging; commissioning dates in
  wind_parameter.csv).
- Measured DWD weather (10-min records aggregated to hourly means,
  closed/label left, NO imputation — gaps are NaN):
  measured_wind_speed [m/s], measured_wind_direction [deg, CIRCULAR mean],
  measured_wind_speed_std [m/s], measured_temperature [deg C],
  measured_relative_humidity [%], measured_pressure [hPa — note: the ERA5
  `pressure` column is in Pa].

## Companion tables
- wind_parameter.csv — site coordinates, altitude, commissioning date.
- turbine_parameter.csv — the six turbine types with hub heights and specs.
- wind_missing_values_hourly.csv — per-site, per-column missing shares [%]
  of the measured_* columns on the hourly grid.
- site_branch_manifest.csv — correction branch per site (A = quantile
  mapping anchored at the site's own station record; uncorrected = coastal
  guard).
- nwp_coverage.csv — ICON-D2 NWP point-file availability per site.
- manifests.json — full generation manifest per site (chain configuration,
  git commit, seeds).
"""
    with open(os.path.join(release_dir, "README.md"), "w") as f:
        f.write(readme)


# ---------------------------------------------------------------- tables

def tables(release_dir: str) -> None:
    ids = site_ids()

    # (a) measurement coverage on the hourly output grid
    rows = []
    for sid in ids:
        df = pd.read_parquet(os.path.join(release_dir, f"synth_{sid}.parquet"),
                             columns=list(MEASURED_MAP.values()))
        rows.append({"station_id": sid,
                     **(df.isna().mean() * 100).round(2).to_dict()})
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(release_dir, "wind_missing_values_hourly.csv"),
               index=False)
    print(f"wind_missing_values_hourly.csv: {len(cov)} stations, "
          f"median missing % per column:\n"
          f"{cov.drop(columns='station_id').median().round(2).to_dict()}")

    # (b) branch-assignment manifest, cross-checked against run manifests
    ba = pd.read_csv(BRANCH_ASSIGNMENT, dtype={"location_id": str})
    ba = ba[(ba["kind"] == "station") & ba["location_id"].isin(ids)]
    assert len(ba) == len(ids)
    label = {"A": "A (station-anchored QM)", "B": "B (LGBM model QM)",
             "C": "uncorrected"}
    man = ba[["location_id", "branch", "gate_reason", "distance_km"]].copy()
    man = man.rename(columns={"location_id": "station_id"})
    man["label"] = [
        "uncorrected (coastal guard)" if (b == "C" and "coastal" in str(r))
        else label[b] for b, r in zip(man["branch"], man["gate_reason"])]
    mismatches = []
    for sid, branch in zip(man["station_id"], man["branch"]):
        with open(os.path.join(RUN_DIR, f"manifest_{sid}.json")) as f:
            if json.load(f)["branch"] != branch:
                mismatches.append(sid)
    assert not mismatches, f"branch mismatch vs run manifests: {mismatches}"
    man.sort_values("station_id").to_csv(
        os.path.join(release_dir, "site_branch_manifest.csv"), index=False)
    print(f"site_branch_manifest.csv: {man['branch'].value_counts().to_dict()} "
          f"(0 mismatches vs run manifests)")

    # (c) NWP (ICON-D2) coverage
    nwp = []
    for sid in ids:
        path = os.path.join(ICON_DIR, f"icon_d2_{sid}.csv")
        if os.path.exists(path):
            with open(path, "rb") as f:
                f.seek(max(0, os.path.getsize(path) - 4096))
                last = f.read().decode(errors="ignore").strip().splitlines()[-1]
            nwp.append({"station_id": sid, "icon_d2": True,
                        "last_timestamp": last.split(",")[0]})
        else:
            nwp.append({"station_id": sid, "icon_d2": False,
                        "last_timestamp": None})
    nwp = pd.DataFrame(nwp)
    nwp.to_csv(os.path.join(release_dir, "nwp_coverage.csv"), index=False)
    print(f"nwp_coverage.csv: {int(nwp['icon_d2'].sum())}/{len(nwp)} sites "
          f"covered, last timestamps "
          f"{nwp['last_timestamp'].dropna().unique()[:3]} ...")


# ---------------------------------------------------------------- verify

def verify(release_dir: str) -> None:
    ids = site_ids()
    expected_index = hourly_index()
    problems = []

    # branch examples: one A, one coastal C
    ba = pd.read_csv(BRANCH_ASSIGNMENT, dtype={"location_id": str})
    ba = ba[(ba["kind"] == "station") & ba["location_id"].isin(ids)]
    example_a = ba[ba["branch"] == "A"]["location_id"].iloc[0]
    example_c = ba[ba["branch"] == "C"]["location_id"].iloc[0]

    comm = pd.read_csv(COMM_TABLE, dtype={"park_id": str}) \
             .set_index("park_id")["commissioning_date"]
    wp = pd.read_csv(os.path.join(release_dir, "wind_parameter.csv"),
                     sep=";", dtype={"park_id": str}) \
           .set_index("park_id")["commissioning_date"]

    blh_means = []
    for sid in ids:
        df = pd.read_parquet(os.path.join(release_dir, f"synth_{sid}.parquet"))
        if not df.index.equals(expected_index):
            problems.append(f"{sid}: index mismatch ({len(df)} rows)")
        if list(df.columns) != RELEASE_COLS:
            problems.append(f"{sid}: column mismatch")
        dens = df[[f"density_t{i}" for i in range(1, N_TURBINES + 1)]]
        # lower bound 0.7 admits mountain stations (05792 Zugspitze, 2904 m)
        if not ((dens.min().min() > 0.7) and (dens.max().max() < 1.45)):
            problems.append(f"{sid}: density out of [0.7, 1.45] "
                            f"({dens.min().min():.3f}..{dens.max().max():.3f})")
        if str(wp.get(sid)) != str(comm.get(sid)):
            problems.append(f"{sid}: commissioning {wp.get(sid)} != "
                            f"table {comm.get(sid)}")
        blh_means.append([df[f"surface_layer_flag_t{i}"].mean()
                          for i in range(1, N_TURBINES + 1)])

    # surface_layer_flag share must increase with hub height (57,138,149,95,
    # 119,78 -> order by hub height: t1(57) < t6(78) < t4(95) < t5(119)
    # < t2(138) < t3(149))
    order = [0, 5, 3, 4, 1, 2]
    med = np.median(np.array(blh_means), axis=0)
    if not all(med[order[i]] <= med[order[i + 1]] + 1e-9
               for i in range(len(order) - 1)):
        problems.append(f"blh_flag medians not monotone in hub height: {med}")

    # correction spot checks on raw run output (pre-merge). height_consistent
    # QM anchors at 10 m and scales 100 m by the same factor, so branch A must
    # move wind_speed_100m away from raw; branch C must leave it untouched.
    for sid, corrected in ((example_a, True), (example_c, False)):
        df = pd.read_csv(os.path.join(RUN_DIR, f"synth_{sid}.csv"),
                         sep=";", index_col=0, parse_dates=True)
        raw = pd.read_csv(f"/mnt/nvme2/synthetic/raw/wind_era5_v2/Station_{sid}.csv",
                          index_col=0, parse_dates=True)
        v_raw = np.sqrt(raw["u_wind_100m"] ** 2 + raw["v_wind_100m"] ** 2) \
                  .reindex(df.index)
        delta = (df["wind_speed_100m"] - v_raw).abs().mean()
        ratio = (df["wind_speed_100m"] / v_raw.replace(0, np.nan)).median()
        if corrected:
            print(f"branch A {sid}: correction applied, mean|Δ| vs raw "
                  f"{delta:.3f} m/s, median factor {ratio:.3f}")
            if delta < 0.01:
                problems.append(f"{sid} (branch A): output == raw ERA5, "
                                f"correction not applied")
        else:
            print(f"branch C {sid}: mean|Δ| vs raw {delta:.4f} m/s (expect ~0)")
            if delta > 0.02:  # rounding of wind_speed_100m only
                problems.append(f"{sid} (branch C): output != raw ERA5 "
                                f"(Δ={delta:.3f})")

    # synthetic columns identical pre/post merge (one sample). Map the native
    # run CSV to readable names first so the comparison is name-aligned.
    from generate_wind import readable_rename_map
    sid = ids[0]
    run = pd.read_csv(os.path.join(RUN_DIR, f"synth_{sid}.csv"),
                      sep=";", index_col=0, parse_dates=True) \
            .rename(columns=readable_rename_map(N_TURBINES))
    rel = pd.read_parquet(os.path.join(release_dir, f"synth_{sid}.parquet"))
    synth_cols = ERA5_BASE_COLS + DERIVED_COLS + \
        [c for c in TURBINE_COLS if "surface_layer_flag" not in c]
    if not np.allclose(run.loc[rel.index, synth_cols].values,
                       rel[synth_cols].values, equal_nan=True):
        problems.append(f"{sid}: synthetic columns changed by postprocess")

    # capacity factors vs rated power (from the power curves via turbine max)
    df = pd.read_parquet(os.path.join(release_dir, f"synth_{ids[0]}.parquet"))
    rated = {i: df[f"power_t{i}"].max() for i in range(1, 7)}
    cf = {f"t{i}": round(float(df[f"power_t{i}"].mean() / rated[i]), 3)
          for i in range(1, 7)}
    print(f"sample {ids[0]} capacity factor per turbine: {cf}")

    files = sorted(f for f in os.listdir(release_dir) if f.endswith(".parquet"))
    size_mb = sum(os.path.getsize(os.path.join(release_dir, f))
                  for f in files) / 1e6
    print(f"\nrelease: {len(files)} parquet files, {size_mb:.0f} MB total, "
          f"{len(expected_index)} rows each, {len(RELEASE_COLS)} data columns")
    print(f"sample columns: {RELEASE_COLS[:6]} ... {RELEASE_COLS[-6:]}")

    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nverify: ALL CHECKS PASSED")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--resample-ages", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    for name in ("postprocess", "tables", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--release-dir", default=DEFAULT_RELEASE)
    args = ap.parse_args()

    if args.stage == "prep":
        prep(resample_ages=args.resample_ages)
    elif args.stage == "generate":
        generate(ids=args.ids, force=args.force)
    elif args.stage == "postprocess":
        postprocess(args.release_dir)
    elif args.stage == "tables":
        tables(args.release_dir)
    elif args.stage == "verify":
        verify(args.release_dir)


if __name__ == "__main__":
    main()
