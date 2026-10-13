#!/usr/bin/env python
"""Synthetic PV reference arrays at the DWD stations (the station product: one 10-kW array per
(tilt, azimuth) pair), rebuilt on the modular pv package.

Changes against the legacy script (archiv/generate_pv_legacy.py, defects verified 2026-10-13):
  * aging is applied only when apply_ageing is set, with the PV degradation model (pv.aging) and
    the exact age per time step instead of 5 %/a on a linear ramp;
  * soiling reduces power (legacy returned power * loss);
  * inverter efficiency is drawn per station (legacy seeded every station with 42);
  * driver = imputed 30-min station series (GHI/DHI) + ERA5 met at the station, with the per-hour
    imputed share carried into the output; raw 10-min files + KNN imputation are gone;
  * output per station: DC power, cell temperature and POA per pair (DC so that parks can be
    assembled with their own DC/AC clipping), plus the AC of a 10-kW inverter for compatibility.

    python generate_pv.py --config configs/config_pv.yaml [--stations 00183,00232] [--out DIR]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from pv import chain, config as pvconfig, drivers, timebase
from parks import era5_db

REPO = os.path.dirname(os.path.abspath(__file__))


def station_coordinates(master_csv: str) -> pd.DataFrame:
    return pd.read_csv(master_csv, dtype={"station_id": str}).set_index("station_id")


def run_station(sid: str, row: pd.Series, cfg_pv: dict, start: str, end: str, conn, pairs: list, rng: np.random.Generator) -> tuple:
    idx = timebase.hourly_index(start, end)
    irr = drivers.station_imputed(sid, start, end)
    met, meta = drivers.era5_met(conn, row["latitude"], row["longitude"], start, end)
    drv = drivers.merge_driver(irr, met, idx)
    eta = float(rng.normal(cfg_pv.get("mean_eta_inv_nom", 0.96), cfg_pv.get("std_eta_inv_nom", 0.01)))
    out = pd.DataFrame(index=idx)
    for c in ("ghi", "dhi", "temp_air", "wind_speed", "albedo", "imputed_share", "missing_share"):
        if c in drv:
            out[c] = drv[c]
    first = True
    for tilt, az in pairs:
        cfg = pvconfig.station_grid(sid, row["latitude"], row["longitude"], float(row["station_height"]), float(tilt), float(az),
                                    kwp_dc=cfg_pv.get("dc_rating_watts", 10000) / 1000.0,
                                    dc_ac=cfg_pv.get("dc_ac_ratio", 1.0),
                                    module={"model": cfg_pv.get("module_model", "pvwatts"), "gamma_pdc": cfg_pv.get("gamma_pdc", -0.0035), "year": cfg_pv.get("year", 2015)},
                                    inverter={"model": "pvwatts", "eta_inv_nom": eta, "eta_inv_ref": cfg_pv.get("eta_inv_ref", 0.9637)},
                                    temperature={"model": "faiman", "u0": cfg_pv.get("faiman_u0", 25.0), "u1": cfg_pv.get("faiman_u1", 6.84)},
                                    transposition={"albedo": cfg_pv.get("albedo", "era5"), "iam": cfg_pv.get("iam", "physical")},
                                    aging={"enabled": bool(cfg_pv.get("apply_ageing", False)), "commissioning_date": cfg_pv.get("commissioning_date"), "era": cfg_pv.get("era")})
        r = chain.run_park(cfg, drv)
        if first:
            out["dni"], out["zenith"], out["azimuth_sun"] = r["dni"], r["zenith"], r["azimuth_sun"]
            first = False
        suf = f"tilt{int(tilt)}_az{int(az)}"
        out[f"poa_{suf}"] = r["poa_global"]
        out[f"tcell_{suf}"] = r["t_cell"]
        out[f"pdc_{suf}"] = r["p_dc_w"]
        out[f"10kw_{suf}"] = r["p_ac_w"]
    params = {"station_id": sid, "latitude": row["latitude"], "longitude": row["longitude"], "altitude": float(row["station_height"]),
              "eta_inv_nom": eta, "era5_met_cell": meta["era5_met_cell"], "imputed_share": float(irr["imputed_share"].mean()),
              "missing_share": float(irr["missing_share"].mean())}
    return out, params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "config_pv.yaml"))
    ap.add_argument("--stations", default=None, help="comma list; default: all imputed stations")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="output dir; default <synth_dir>/solar/stations_pv_v1")
    ap.add_argument("--master", default=os.path.expanduser("~/Work/stations_master.csv"))
    args = ap.parse_args()
    cfg = pvconfig.load_yaml(args.config)
    cfg_pv = cfg["pv_params"]
    start = args.start or cfg["data"].get("from_date", "2023-07-24")
    end = args.end or cfg["data"].get("to_date", "2026-08-01")
    out_dir = args.out or os.path.join(cfg["data"]["synth_dir"], "solar", "stations_pv_v1")
    os.makedirs(out_dir, exist_ok=True)
    master = station_coordinates(args.master)
    ids = args.stations.split(",") if args.stations else [s for s in drivers.imputed_stations() if s in master.index]
    pairs = cfg_pv["tilt_az_list"]
    rng = np.random.default_rng(int(cfg_pv.get("random_seed", 42)))
    conn = era5_db.connect()
    params = {}
    for i, sid in enumerate(ids):
        t0 = time.time()
        out, par = run_station(sid, master.loc[sid], cfg_pv, start, end, conn, pairs, rng)
        out.to_parquet(os.path.join(out_dir, f"synth_{sid}.parquet"))
        params[sid] = par
        print(f"[{i + 1}/{len(ids)}] {sid}: {len(out)} h, {len(pairs)} pairs, imputed {par['imputed_share']:.3f}, {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(params).T.to_csv(os.path.join(out_dir, "pv_parameter.csv"))
    json.dump({"config": cfg_pv, "start": start, "end": end, "pairs": pairs, "n_stations": len(ids)}, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
