#!/usr/bin/env python
"""Apply the CAMS correction to park points: CAMS centre + ring CSVs (ADS), ERA5 solar/met from the
WeatherDB, same feature assembly as the station dataset, fold-ensemble prediction, hourly output.

Point files are confidential (coordinates) and stay outside the repos; outputs carry only pv ids.

    python scripts/cams_bc/apply_parks.py --points-json <points.json> --cams-dir <dir> --ring-dir <dir> \
        --models-dir ${DATA_ROOT}/synthetic/cams_bc/models/xgb --out-dir <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import data, targets  # noqa: E402
from cams_bc.config import AZIMUTHS, RADII, STEP  # noqa: E402
from cams_bc.models import MLPModel  # noqa: E402
from pv import drivers  # noqa: E402
from parks import era5_db  # noqa: E402

AZ_FILE = {"N": "N", "O": "E", "S": "S", "W": "W"}
INTERVAL = ["era5_ghi", "era5_bhi", "era5_ghi_clear", "era5_toa", "era5_lw_cre", "era5_k_c", "era5_f_dir"]
MOMENT = ["era5_albedo", "era5_cloud_total", "era5_cloud_low", "era5_cloud_mid", "era5_cloud_high", "era5_water_vapour",
          "era5_temp_2m", "era5_pressure", "era5_dew_dep"]


def cams_30min(files: list) -> pd.DataFrame:
    """ADS CSVs (15-min sums) -> 30-min means with interval END labels, both quarters required."""
    df = pd.concat([drivers.read_cams_csv(f) for f in sorted(files)]).sort_index()
    df = df[~df.index.duplicated()]
    df.index = df.index + pd.Timedelta("15min")          # START -> END
    return data.to_30min(df)


def era5_solar_30min(conn, lat: float, lon: float, idx: pd.DatetimeIndex) -> pd.DataFrame:
    pts = drivers.era5_solar_points(conn)
    d = drivers.haversine_km(lat, lon, pts.lat.values, pts.lon.values)
    cell = int(pts.cell_id.iloc[int(np.argmin(d))])
    sql = (f"SELECT s.valid_time, s.ssrd, s.fdir, s.ssrdc, s.tisr, s.fal, s.tcc, s.lcc, s.mcc, s.hcc, s.tcwv, s.strd, s.strdc "
           f"FROM {drivers.ERA5_SOLAR} s JOIN {drivers.ERA5_SOLAR_POINTS} p ON p.geom = s.geom WHERE p.cell_id = %s ORDER BY s.valid_time;")
    with conn.cursor() as cur:
        cur.execute(sql, (cell,))
        rows = cur.fetchall()
    r = pd.DataFrame(rows, columns=["timestamp", "ssrd", "fdir", "ssrdc", "tisr", "fal", "tcc", "lcc", "mcc", "hcc", "tcwv", "strd", "strdc"])
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
    r = r.set_index("timestamp").astype(float).sort_index()
    h = pd.DataFrame(index=r.index)
    h["era5_ghi"], h["era5_bhi"], h["era5_ghi_clear"], h["era5_toa"] = r["ssrd"] / 3600, r["fdir"] / 3600, r["ssrdc"] / 3600, r["tisr"] / 3600
    h["era5_lw_cre"] = (r["strd"] - r["strdc"]) / 3600
    h["era5_k_c"] = np.where(h["era5_ghi_clear"] > 0, h["era5_ghi"] / h["era5_ghi_clear"], np.nan)
    h["era5_f_dir"] = np.clip(np.where(h["era5_ghi"] > 0, h["era5_bhi"] / h["era5_ghi"], np.nan), 0, 1)
    h["era5_albedo"], h["era5_cloud_total"], h["era5_cloud_low"], h["era5_cloud_mid"], h["era5_cloud_high"], h["era5_water_vapour"] = \
        r["fal"], r["tcc"], r["lcc"], r["mcc"], r["hcc"], r["tcwv"]
    return h


def era5_met_hourly(conn, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    pts = era5_db.load_grid_points(conn)
    ids, _ = era5_db.nearest_cells(pts, [lat], [lon])
    raw = era5_db.fetch_cell(conn, int(ids[0]), start, end)
    h = pd.DataFrame(index=raw.index)
    h["era5_temp_2m"], h["era5_pressure"] = raw["temp_2m"], raw["pressure"]
    h["era5_dew_dep"] = raw["temp_2m"] - raw["dew_point_2m"]
    return h


def onto_30min(h: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly ERA5 (END labels) onto the 30-min grid: interval means repeated (ceil to the hour),
    instantaneous values interpolated in time - as solar_bc.data.auf_raster."""
    parts = []
    iv = [c for c in h.columns if c in INTERVAL]
    if iv:
        t = h[iv].reindex(idx.ceil("h"))
        t.index = idx
        parts.append(t)
    mo = [c for c in h.columns if c in MOMENT]
    if mo:
        u = h[mo].reindex(h.index.union(idx)).interpolate(method="time", limit_area="inside").reindex(idx)
        parts.append(u)
    return pd.concat(parts, axis=1)


def load_models(models_dir: str, device: str) -> tuple:
    meta = json.load(open(os.path.join(models_dir, "meta.json")))
    models = []
    if meta["model"] == "xgb":
        from cams_bc.models import load_booster
        for k in range(3):
            # load_booster cuts the saved booster back to best_iteration (same trees as in final_eval)
            models.append({v: load_booster(os.path.join(models_dir, f"fold{k}_{v}.json")) for v in C.VARS})
    else:
        import torch
        for k in range(3):
            ck = torch.load(os.path.join(models_dir, f"fold{k}.pt"), weights_only=False)
            m = MLPModel(ck["params"], device=device)
            m.mu, m.sd, m.med = ck["mu"], ck["sd"], ck["med"]
            m.nan_cols = np.asarray(ck.get("nan_cols", []), dtype=int)   # missingness indicators fixed at training
            m.net = m._build(len(meta["columns"]) + len(m.nan_cols), 2).to(device)
            m.net.load_state_dict(ck["state"])
            models.append(m)
    return meta, models


def predict(meta: dict, models: list, feat: pd.DataFrame) -> dict:
    X = feat[meta["columns"]].to_numpy(np.float32)
    out = {v: [] for v in C.VARS}
    for m in models:
        if meta["model"] == "xgb":
            import xgboost as xgb
            dm = xgb.DMatrix(X, missing=np.nan)
            for v in C.VARS:
                out[v].append(targets.decode(meta["form"], m[v].predict(dm), feat, v))
        else:
            P = m.predict(X)
            for i, v in enumerate(C.VARS):
                out[v].append(targets.decode(meta["form"], P[:, i], feat, v))
    return {v: np.mean(out[v], axis=0) for v in C.VARS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-json", required=True)
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--ring-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--parks", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pts = json.load(open(args.points_json))
    conn = era5_db.connect()
    meta, models = load_models(args.models_dir, f"cuda:{args.gpu}")
    nb_table, nb_coords = data.neighbour_table(), data.coords_all()
    tag = os.path.basename(args.models_dir.rstrip("/"))
    s, e = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    idx = pd.date_range(s - pd.Timedelta("1D"), e + pd.Timedelta("1D"), freq=STEP, tz="UTC")
    for p in pts:
        pv = p["pv"]
        if args.parks and pv not in args.parks.split(","):
            continue
        ring_files = {(r, a): [os.path.join(args.ring_dir, f) for f in os.listdir(args.ring_dir) if f.startswith(f"CAMS_rad_{pv}_r{int(r)}_{AZ_FILE[a]}_")]
                      for r in RADII for a in AZIMUTHS}
        if any(len(v) == 0 for v in ring_files.values()):
            print(pv, "ring incomplete, skipped", flush=True)
            continue
        centre = cams_30min([os.path.join(args.cams_dir, f) for f in os.listdir(args.cams_dir) if f.startswith(f"CAMS_rad_{pv}_") and f.endswith(".csv")])
        d = centre.reindex(idx).add_prefix("cams_")
        ring = {k: cams_30min(v)[["ghi", "dhi", "ghi_clear"]] for k, v in ring_files.items()}
        h = era5_solar_30min(conn, p["lat"], p["lon"], idx).join(era5_met_hourly(conn, p["lat"], p["lon"], (s - pd.Timedelta("2D")).strftime("%Y-%m-%d %H:%M"), (e + pd.Timedelta("2D")).strftime("%Y-%m-%d %H:%M")))
        d = d.join(onto_30min(h, idx))
        nb = data.neighbour_features(p["lat"], p["lon"], idx, nb_table, nb_coords)   # all stations are available at a park
        feat = data.assemble(d, data.ring_features(ring, idx), p["lat"], p["lon"], p["alt"], nb)
        pr = predict(meta, models, feat)
        out = pd.DataFrame({"cams_ghi": feat["cams_ghi"], "cams_dhi": feat["cams_dhi"], "cams_ghi_clear": feat["cams_ghi_clear"],
                            "ghi_corr": pr["ghi"], "dhi_corr": pr["dhi"]}, index=idx)
        night = feat["cams_ghi_clear"] <= 10.0
        out.loc[night, "ghi_corr"] = out.loc[night, "cams_ghi"]
        out.loc[night, "dhi_corr"] = out.loc[night, "cams_dhi"]
        out["dhi_corr"] = np.minimum(out["dhi_corr"], out["ghi_corr"])
        out["ring_missing"] = feat["ring3N_kc"].isna()
        out.to_parquet(os.path.join(args.out_dir, f"{pv}_cams_corrected_30min_{tag}.parquet"))
        hourly = out.groupby((out.index - pd.Timedelta("1min")).floor("h")).agg(["mean", "count"])
        hh = hourly.xs("mean", axis=1, level=1).where(hourly.xs("count", axis=1, level=1)["ghi_corr"] == 2)
        hh = hh[(hh.index >= s) & (hh.index < e)]
        hh.to_parquet(os.path.join(args.out_dir, f"{pv}_cams_corrected_hourly_{tag}.parquet"))
        day = hh["cams_ghi_clear"] > 10
        print(pv, f"hours {len(hh)}, day ratio corr/cams ghi {hh.loc[day, 'ghi_corr'].sum() / hh.loc[day, 'cams_ghi'].sum():.3f} "
              f"dhi {hh.loc[day, 'dhi_corr'].sum() / hh.loc[day, 'cams_dhi'].sum():.3f}, nan {hh['ghi_corr'].isna().sum()}", flush=True)


if __name__ == "__main__":
    main()
