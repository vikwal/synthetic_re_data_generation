#!/usr/bin/env python
"""Synthetic twins of the six metered Trianel parks with the best chain we have, on a 30-min (or hourly) step.

Geometry (2026-10-22): arrays of the window method (geometry_trianel_windows.py ->
reports/figs_pv_generator/geodata_arrays_trianel_windows_v6.csv, all six metered parks incl. the bDOM park pv2 and the
east-west pair of pv7) -> register for parks missing in the table. The cascade of 2026-10-16 (laser surface model ->
image azimuth + register tilt -> register) is reproduced with --laser-csv (+ the image CSV). Chain = parks_pv_v1
production chain: corrected CAMS (cams_bc xgb_v5, 30-min product), PVGIS horizon, inter-row shading at the MaStR
GCR, PVWatts with era gamma and IAM, era aging, portfolio loss defaults (no meter calibration), ERA5 snow
depth_linear. Meters are written next to the twins on the same step (confidential directory outside git).

    synthre/bin/python scripts/pv/trianel_twins.py --step 30min [--pv pv1,pv7]
Inputs: trianel_benchmark/inputs (cams, cams_corrected, osm/pvgis_horizon.json, mastr, era5_snow_full),
        reports/figs_pv_generator/geodata_arrays_trianel_windows_v6.csv (legacy: geodata_arrays_trianel_surf_{laser,image}.csv)
Output: ${DATA_ROOT}/synthetic/solar/trianel_twins_<step>/{synth_<pv>.parquet, meter_<pv>.parquet, manifest_<pv>.json,
        parks.csv, sites_confidential.csv}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks import era5_db  # noqa: E402
from parks_pv import assemble as A, drivers as DRV, generate as GEN  # noqa: E402
from parks_pv.params import LOSSES  # noqa: E402
from pv import metrics, timebase  # noqa: E402
import trianel_common as tc  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
BENCH = f"{DATA_ROOT}/synthetic/solar/trianel_benchmark/inputs"
FIGS = os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator")
START, END = "2023-07-24", "2024-06-01"


def meters(parks: pd.DataFrame, step: str) -> pd.DataFrame:
    """Mean meter power (W) per park on the step, START labels; a step needs all its 15-min values."""
    md = pd.read_csv(os.path.join(tc.METER_DIR, "meterdata_solar_20240601.csv"), sep=";", index_col=0, parse_dates=True)
    md.index = pd.to_datetime(md.index, utc=True)
    need = int(pd.Timedelta(step) / pd.Timedelta("15min"))
    out = {}
    for pv, r in parks.iterrows():
        if r["malo"] not in md.columns:
            continue
        s = md[r["malo"]].astype(float).clip(lower=0.0) * 1000.0
        g = s.resample(pd.Timedelta(step), closed="left", label="left")
        out[pv] = g.mean().where(g.count() >= need)
    return pd.DataFrame(out)


def geometry(pv: str, p: pd.Series, laser: pd.DataFrame, image: pd.DataFrame, label: str = "geodata") -> tuple:
    """(arrays, source) by the cascade: measured arrays (window method; legacy: laser planes) -> image azimuth +
    register tilt (legacy only) -> register."""
    if pv in laser.park_id.values:
        g = laser[laser.park_id == pv]
        return [{"name": str(a.array_id), "tilt": float(a.tilt), "azimuth": float(a.azimuth), "share": float(a.share)} for a in g.itertuples()], f"{label}:{g.source.iloc[0]}"
    if pv in image.park_id.values:
        g = image[image.park_id == pv]
        az = float(np.average(g.azimuth, weights=g.share))
        return [{"name": "a0", "tilt": float(p["tilt_master"]), "azimuth": az, "share": 1.0}], f"image_azimuth+register_tilt:{g.source.iloc[0]}"
    return [{"name": "a0", "tilt": float(p["tilt_master"]), "azimuth": float(p["az_master"]), "share": 1.0}], "register"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="30min")
    ap.add_argument("--pv", default=None, help="comma-separated pv ids")
    ap.add_argument("--model-tag", default="xgb_v5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--register-ac", action="store_true", help="DC/AC from MaStR (else 1.10), never from the meter (D2)")
    ap.add_argument("--geodata-csv", default=os.path.join(FIGS, "geodata_arrays_trianel_windows_v6.csv"), help="arrays of the window method (default v6, 2026-10-22)")
    ap.add_argument("--laser-csv", default=None, help="LEGACY cascade of 2026-10-16: laser arrays CSV (e.g. geodata_arrays_trianel_surf_laser_v3.csv) + image CSV")
    a = ap.parse_args()
    suffix = "regac_" if a.register_ac else ""
    out_dir = a.out or f"{DATA_ROOT}/synthetic/solar/trianel_twins_{suffix}{a.step}"
    os.makedirs(out_dir, exist_ok=True)
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    met_h = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, os.path.join(BENCH, "mastr", "mastr_candidates.csv"), os.path.join(BENCH, "mastr", "mastr_xml_units.json"))
    par = tc.park_parameters(parks, facts, met_h, use_meter_ac=not a.register_ac)
    met = meters(parks, a.step)
    if a.laser_csv:
        laser, label = pd.read_csv(a.laser_csv), "laser"
        image = pd.read_csv(os.path.join(FIGS, "geodata_arrays_trianel_surf_image.csv"))
    else:
        laser, label = pd.read_csv(a.geodata_csv), "windows"
        image = pd.DataFrame(columns=["park_id", "azimuth", "share", "source"])
    conn = era5_db.connect()
    rows, sites = [], []
    pvs = [pv for pv in parks.index if pv in met and met[pv].notna().sum() > 1000 and (a.pv is None or pv in a.pv.split(","))]
    for pv in pvs:
        p = parks.loc[pv]
        arrays, geo_src = geometry(pv, p, laser, image, label)
        cfg = tc.base_config(pv, p, par.loc[pv], arrays=arrays, variant="PR")
        cfg["losses"] = dict(LOSSES)
        cfg["aging"]["enabled"] = True
        cfg["shading"]["horizon"] = True
        hz = tc.load_horizon(os.path.join(BENCH, "osm", "pvgis_horizon.json"), pv)
        drv, meta = DRV.build(pv, p["lat"], p["lon"], START, END, conn, a.model_tag, snow_cache=os.path.join(BENCH, "era5_snow_full", f"{pv}_era5_snow.parquet"),
                              step=a.step, corr_dir=os.path.join(BENCH, "cams_corrected"), cams_dir=os.path.join(BENCH, "cams"))
        res = GEN.run(cfg, drv, horizon=hz)
        rel = A.release_frame(drv, res)
        y = met[pv].reindex(rel.index)
        cap_w = cfg["kwp_dc"] * 1000.0
        day = rel["zenith"] < 90
        b = metrics.basic(y[day], rel.loc[day, "power_park"], cap_w)
        winter = day & (rel.index >= pd.Timestamp("2023-11-15", tz="UTC")) & (rel.index < pd.Timestamp("2024-03-15", tz="UTC"))
        bw = metrics.basic(y[winter], rel.loc[winter, "power_park"], cap_w)
        man = {"dataset": os.path.basename(out_dir), "park_id": pv, "step": str(pd.Timedelta(a.step)), "geometry_source": geo_src, "arrays": arrays,
               "config": {k: v for k, v in cfg.items() if k not in ("latitude", "longitude", "altitude")}, "params": par.loc[pv].to_dict(),
               "chain": "parks_pv_v1 production chain (corrected CAMS xgb_v5 at the step, PVGIS horizon, inter-row at MaStR GCR, PVWatts era gamma + IAM, "
                        "era aging, portfolio losses DC 5 % / AC 2 % / availability 0.98, ERA5 snow depth_linear); no meter calibration",
               "drivers": meta, "period": [str(rel.index[0]), str(rel.index[-1])], "n_steps": int(len(rel)),
               "meter_vs_twin": {"day": b, "winter": bw, "n_meter_steps": int(y.notna().sum())},
               "time_label": f"interval start, UTC, {pd.Timedelta(a.step)} means", "git": A.git_state(),
               "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        rel.to_parquet(os.path.join(out_dir, f"synth_{pv}.parquet"))
        pd.DataFrame({"power_w": y}).to_parquet(os.path.join(out_dir, f"meter_{pv}.parquet"))
        json.dump(man, open(os.path.join(out_dir, f"manifest_{pv}.json"), "w"), indent=1, default=str)
        rows.append({"park_id": pv, "kwp_dc": cfg["kwp_dc"], "dc_ac": cfg["dc_ac"], "geometry_source": geo_src,
                     "arrays": "; ".join(f"{x['tilt']:.1f}/{x['azimuth']:.0f}x{x['share']:.2f}" for x in arrays),
                     "r2_day": round(b["r2"], 4), "nrmse_day": round(b["nrmse_cap"], 4), "energy_ratio_day": round(b["energy_ratio"], 4),
                     "r2_winter": round(bw["r2"], 4), "n_meter_steps": int(y.notna().sum())})
        sites.append({"park_id": pv, "lat": p["lat"], "lon": p["lon"], "altitude": p["alt"]})
        print(pv, geo_src, f"R2 day {b['r2']:.3f} winter {bw['r2']:.3f} nRMSE {b['nrmse_cap'] * 100:.2f} % energy {b['energy_ratio']:.3f}", flush=True)
    conn.close()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "parks.csv"), index=False)
    pd.DataFrame(sites).to_csv(os.path.join(out_dir, "sites_confidential.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
