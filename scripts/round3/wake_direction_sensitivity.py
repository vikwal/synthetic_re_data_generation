#!/usr/bin/env python3
"""Round 3 (reviewer 2d): sensitivity of the hourly NOJ wake factor to the
inflow-direction assumption. The chain drives every turbine of a park with
one ERA5 100 m direction per hour (no veer, horizontally uniform). Here the
same PyWake setup (wp5_precompute_wakes.py, k = 0.075) is rerun with

  ref          ERA5 100 m direction (the chain's assumption)
  wd10         ERA5 10 m direction
  veer_hub     10 m -> 100 m veer linearly continued to the park's mean hub
  off_p10 ...  fixed offsets +10, -10, +20, -20 deg on the 100 m direction

and the mean wake loss per park and the hourly agreement with 'ref' are
reported, together with the ERA5 veer statistics (|wd100 - wd10|).

Run on l2: synthre/bin/python scripts/round3/wake_direction_sensitivity.py [--parks ...]
Output: results/round3/wake_direction_sensitivity.csv, wake_veer_stats.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "round2"))
from round2 import wake as r2_wake  # noqa: E402
import wp5_precompute_wakes as wp5  # noqa: E402

ERA5_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
OUT = os.path.join(REPO, "results", "round3")
K = 0.075


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def directions(era5_dir, station_id, hub_mean):
    df = pd.read_csv(os.path.join(era5_dir, f"Station_{station_id}.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    wd100 = r2_wake.wind_direction_met(df["u_wind_100m"].values, df["v_wind_100m"].values)
    wd10 = r2_wake.wind_direction_met(df["u_wind_10m"].values, df["v_wind_10m"].values)
    veer = ang_diff(wd100, wd10)                      # deg over 90 m
    wd_hub = (wd100 + veer * (hub_mean - 100.0) / 90.0) % 360.0
    v100 = np.hypot(df["u_wind_100m"], df["v_wind_100m"]).values
    return pd.DataFrame({"wd100": wd100, "wd10": wd10, "veer": veer,
                         "wd_hub": wd_hub, "v100": v100}, index=df.index)


def run_variant(wfm, x, y, type_idx, wts, unique_models, wind, wd):
    parts = []
    for _, chunk in wind.groupby(pd.Grouper(freq="MS")):
        if chunk.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = wfm(x, y, type=type_idx, wd=wd.reindex(chunk.index).values,
                      ws=chunk["ws"].values, time=True)
            p_waked = sim.Power.sum("wt").values
            p_free = np.zeros(len(chunk))
            for i, m in enumerate(unique_models):
                p_free += wts[m].power(chunk["ws"].values) * int((type_idx == i).sum())
        w = np.divide(p_waked, p_free, out=np.ones_like(p_free), where=p_free > 0)
        parts.append(pd.DataFrame({"w": np.clip(w, 1e-6, 1.0), "p_free": p_free}, index=chunk.index))
    return pd.concat(parts).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", nargs="*", default=None)
    args = ap.parse_args()
    from py_wake.literature.noj import Jensen_1983
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbines

    os.makedirs(OUT, exist_ok=True)
    lay = pd.read_csv(os.path.join(REPO, "data", "round2", "park_layouts.csv"),
                      dtype={"park_id": str}).dropna(subset=["park_id"])
    pc, ct = r2_wake.load_curves()
    rows, veer_rows = [], []
    for park_id, grp in lay.groupby("park_id"):
        if args.parks and park_id not in args.parks:
            continue
        if len(grp) < 2:
            continue  # single turbine: no wake
        station_id = park_id[:5]
        models = grp["model"].tolist()
        unique_models = sorted(set(models))
        wts = {}
        for m in unique_models:
            row = grp[grp["model"] == m].iloc[0]
            wts[m], _ = r2_wake.build_windturbine(m, float(row["hub_height"]),
                                                  float(row["rotor_diameter"]),
                                                  float(row["rated_kw"]), pc, ct)
        turbines = WindTurbines.from_WindTurbine_lst([wts[m] for m in unique_models])
        type_idx = np.array([unique_models.index(m) for m in models])
        x, y = grp["x_utm32"].values, grp["y_utm32"].values
        hub_mean = float(np.average(grp["hub_height"], weights=grp["rated_kw"]))
        wind = wp5.station_wind(ERA5_DIR, station_id, hub_mean)
        d = directions(ERA5_DIR, station_id, hub_mean).reindex(wind.index)
        pw = d.v100 >= 4.0  # power-relevant hours
        veer_rows.append({"park_id": park_id, "hub_mean": hub_mean,
                          "abs_veer_p50": float(d.veer.abs().median()),
                          "abs_veer_p90": float(d.veer.abs().quantile(0.9)),
                          "abs_veer_p50_v100ge4": float(d.veer[pw].abs().median()),
                          "abs_veer_p90_v100ge4": float(d.veer[pw].abs().quantile(0.9)),
                          "share_abs_veer_gt20_v100ge4": float((d.veer[pw].abs() > 20).mean()),
                          "hub_extrap_shift_p50": float(ang_diff(d.wd_hub, d.wd100)[pw].abs().median())})
        wfm = Jensen_1983(UniformSite(p_wd=[1.0], ti=0.1), turbines, k=K)
        variants = {"ref": d.wd100, "wd10": d.wd10, "veer_hub": d.wd_hub,
                    "off_p10": (d.wd100 + 10) % 360, "off_m10": (d.wd100 - 10) % 360,
                    "off_p20": (d.wd100 + 20) % 360, "off_m20": (d.wd100 - 20) % 360}
        ref = None
        for name, wd in variants.items():
            res = run_variant(wfm, x, y, type_idx, wts, unique_models, wind, wd)
            if name == "ref":
                ref = res
            loss = 1 - res.w
            e_loss = 1 - float((res.w * res.p_free).sum() / res.p_free.sum())
            dw = res.w - ref.w.reindex(res.index)
            rows.append({"park_id": park_id, "n_turbines": len(grp), "variant": name,
                         "mean_wake_loss_pct": float(loss.mean() * 100),
                         "energy_wake_loss_pct": float(e_loss * 100),
                         "delta_energy_loss_pp_vs_ref": float(e_loss * 100 - (
                             1 - float((ref.w * ref.p_free).sum() / ref.p_free.sum())) * 100),
                         "rms_hourly_dw": float(np.sqrt((dw ** 2).mean())),
                         "p90_abs_hourly_dw": float(dw.abs().quantile(0.9)),
                         "n_hours": int(len(res))})
            print(f"{park_id} {name:9s} energy wake loss {e_loss*100:6.2f} %  "
                  f"rms dw {rows[-1]['rms_hourly_dw']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "wake_direction_sensitivity.csv"), index=False)
    pd.DataFrame(veer_rows).to_csv(os.path.join(OUT, "wake_veer_stats.csv"), index=False)
    piv = out.pivot(index="park_id", columns="variant", values="energy_wake_loss_pct")
    print("\nenergy-weighted wake loss [%] per park and direction variant:")
    print(piv.round(2).to_string())
    print("\nfleet mean delta vs ref [pp]:")
    print(out.groupby("variant")["delta_energy_loss_pp_vs_ref"].agg(["mean", "min", "max"]).round(3).to_string())
    print("WAKE_SENS_DONE")


if __name__ == "__main__":
    main()
