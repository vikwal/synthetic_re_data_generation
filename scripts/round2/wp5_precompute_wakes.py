#!/usr/bin/env python3
"""WP5 — precompute hourly wake loss factors w(t) per park (NOJ, PyWake).

Per park: real turbine layout (park_layouts.csv), heterogeneous turbine types,
driving wind = ERA5 wind of the park's DWD station extrapolated to the
capacity-weighted mean hub height with the chain's power-law alpha; direction
from ERA5 100 m u/v. w(t) = P_waked / P_free clipped to (0, 1].

Output: /mnt/nvme2/synthetic/raw/round2/wakes/w_{park}_k{k:.3f}.parquet
        + wake_summary.csv + ct_assumptions.csv (data/round2/).

Usage: wp5_precompute_wakes.py [--era5-dir DIR] [--ks 0.05 0.075 0.10]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import wake as r2_wake  # noqa: E402

OUT_DIR = "/mnt/nvme2/synthetic/raw/round2/wakes"
DEFAULT_ERA5 = "/mnt/nas/synthetic/raw/wind_era5"


def station_wind(era5_dir: str, station_id: str, hub_height: float,
                 park_id: str = None, from_experiment: str = None):
    path = os.path.join(era5_dir, f"Station_{station_id}.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    wd = pd.Series(
        r2_wake.wind_direction_met(df["u_wind_100m"].values, df["v_wind_100m"].values),
        index=df.index)
    if from_experiment:
        # final-chain hub wind: mean of the per-turbine wind columns of the
        # experiment's synth output (corrected + MOST, guide 5.1)
        synth = pd.read_csv(os.path.join("/mnt/nvme2/synthetic/wind/round2",
                                         from_experiment, f"synth_{park_id}.csv"),
                            sep=";", index_col=0, parse_dates=True)
        wcols = [c for c in synth.columns if c.startswith("wind_speed_t")]
        ws = synth[wcols].mean(axis=1)
        return pd.DataFrame({"ws": ws, "wd": wd.reindex(ws.index)}).dropna()
    v10 = np.hypot(df["u_wind_10m"], df["v_wind_10m"])
    v100 = np.hypot(df["u_wind_100m"], df["v_wind_100m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(v100 / v10) / np.log(10.0)
    alpha = np.clip(np.nan_to_num(alpha), 0.0, 0.4)
    ws = v10 * (hub_height / 10.0) ** alpha
    return pd.DataFrame({"ws": ws, "wd": wd}, index=df.index).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-dir", default=DEFAULT_ERA5)
    ap.add_argument("--ks", nargs="+", type=float, default=[0.05, 0.075, 0.10])
    ap.add_argument("--from-experiment", default=None,
                    help="take hub wind from this ladder experiment's outputs")
    args = ap.parse_args()

    from py_wake.literature.noj import Jensen_1983
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbines

    os.makedirs(OUT_DIR, exist_ok=True)
    lay = pd.read_csv(os.path.join(REPO, "data", "round2", "park_layouts.csv"),
                      dtype={"park_id": str})
    lay = lay.dropna(subset=["park_id"])
    pc, ct = r2_wake.load_curves()

    ct_notes, summary = [], []
    for park_id, grp in lay.groupby("park_id"):
        station_id = park_id[:5]
        models = grp["model"].tolist()
        unique_models = sorted(set(models))
        wts, gen_flags = {}, {}
        for m in unique_models:
            row = grp[grp["model"] == m].iloc[0]
            wt, generic = r2_wake.build_windturbine(
                m, float(row["hub_height"]), float(row["rotor_diameter"]),
                float(row["rated_kw"]), pc, ct)
            wts[m] = wt
            gen_flags[m] = generic
            ct_notes.append({"park_id": park_id, "model": m,
                             "generic_ct": generic})
        turbines = WindTurbines.from_WindTurbine_lst([wts[m] for m in unique_models])
        type_idx = np.array([unique_models.index(m) for m in models])
        x, y = grp["x_utm32"].values, grp["y_utm32"].values

        hub_mean = float(np.average(grp["hub_height"], weights=grp["rated_kw"]))
        wind = station_wind(args.era5_dir, station_id, hub_mean,
                            park_id=park_id, from_experiment=args.from_experiment)

        for k in args.ks:
            wfm = Jensen_1983(UniformSite(p_wd=[1.0], ti=0.1), turbines, k=k)
            w_parts = []
            for _, chunk in wind.groupby(pd.Grouper(freq="MS")):
                if chunk.empty:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim = wfm(x, y, type=type_idx, wd=chunk["wd"].values,
                              ws=chunk["ws"].values, time=True)
                    p_waked = sim.Power.sum("wt").values
                    # free-stream park power = sum of per-type curves at
                    # the undisturbed wind speed
                    p_free = np.zeros(len(chunk))
                    for cnt_idx, m in enumerate(unique_models):
                        n_of_type = int((type_idx == cnt_idx).sum())
                        p_free += wts[m].power(chunk["ws"].values) * n_of_type
                w = np.divide(p_waked, p_free, out=np.ones_like(p_free),
                              where=p_free > 0)
                w_parts.append(pd.Series(np.clip(w, 1e-6, 1.0), index=chunk.index))
            w_series = pd.concat(w_parts).sort_index().rename("w")
            out = os.path.join(OUT_DIR, f"w_{park_id}_k{k:.3f}.parquet")
            w_series.to_frame().to_parquet(out)
            gen_mask = w_series.index[(w_series < 1.0)]
            summary.append({"park_id": park_id, "k": k,
                            "n_turbines": len(grp),
                            "mean_w": float(w_series.mean()),
                            "mean_wake_loss_pct": float((1 - w_series).mean() * 100),
                            "p10_w": float(w_series.quantile(0.10))})
            print(f"park {park_id} k={k}: mean wake loss "
                  f"{(1 - w_series.mean()) * 100:.2f} % over {len(w_series)} h")

    pd.DataFrame(ct_notes).drop_duplicates().to_csv(
        os.path.join(REPO, "data", "round2", "ct_assumptions.csv"), index=False)
    pd.DataFrame(summary).to_csv(
        os.path.join(REPO, "data", "round2", "wake_summary.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
