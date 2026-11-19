#!/usr/bin/env python3
"""Round 3: two small tables.

(a) reviewer 2a-2 — expected ERA5 error by terrain class: RMSE / bias of the
    raw ERA5 10 m wind against the DWD station per observed quality class
    (200 stations, data/round2/station_correction_table.parquet), and the
    deployed chain's farm-level |ER-1| and R2 by the parks' predicted class
    (results/quantitative/scores.csv, M5all_PL, clean).
(b) reviewer 3 / 2c-4 — hub-height air density actually used: per park
    (M5all_PL synth files, density_t* columns, capacity-unweighted mean over
    turbine groups) and per dataset turbine (site_v2 release, density_t1..t6
    over 200 sites): mean, p5, p50, p95, seasonal (DJF/JJA) means.

Run on l2: synthre/bin/python scripts/round3/class_error_and_density.py
Output: results/round3/era5_error_by_class.csv, parks_by_class.csv,
        density_parks.csv, density_dataset_turbines.csv
"""
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "round3")
SYNTH = "/mnt/nvme2/synthetic/wind/round2/M5all_PL"
SITES = "/mnt/nvme2/synthetic/wind/site_v2_20260714"
if not os.path.isdir(SITES):
    SITES = "/mnt/lambda1/nvme1/synthetic/wind/site_v2_20260714"


def by_class():
    t = pd.read_parquet(os.path.join(REPO, "data", "round2", "station_correction_table.parquet"))
    g = t.groupby("era5_class").agg(n=("station_id", "count"), rmse_mean=("rmse", "mean"),
                                    rmse_p10=("rmse", lambda x: x.quantile(.1)),
                                    rmse_p50=("rmse", "median"), rmse_p90=("rmse", lambda x: x.quantile(.9)),
                                    bias_mean=("bias", "mean"), bias_p10=("bias", lambda x: x.quantile(.1)),
                                    bias_p90=("bias", lambda x: x.quantile(.9)), corr_mean=("corr", "mean"))
    g.to_csv(os.path.join(OUT, "era5_error_by_class.csv"))
    print("ERA5 10 m wind error by observed quality class (200 stations):")
    print(g.round(3).to_string())

    b = pd.read_csv(os.path.join(REPO, "data", "round2", "branch_assignment.csv"), dtype=str)
    parks = b[b.kind == "park"].copy()
    parks["park_id"] = parks.location_id.str.replace("park_", "", regex=False)
    sc = pd.read_csv(os.path.join(REPO, "results", "quantitative", "scores.csv"), dtype={"park_id": str})
    m4 = sc[(sc.experiment == "M5all_PL") & (sc.variant == "clean")][["park_id", "r2", "energy_ratio"]]
    m4["abs_er_dev"] = (m4.energy_ratio - 1).abs()
    pk = parks.merge(m4, on="park_id", how="left")[["park_id", "predicted_class", "observed_class",
                                                    "branch", "distance_km", "r2", "abs_er_dev"]]
    pk.to_csv(os.path.join(OUT, "parks_by_class.csv"), index=False)
    print("\nparks (M4 clean) by predicted class:")
    print(pk.sort_values("predicted_class").round(3).to_string(index=False))
    print(pk.groupby("predicted_class")[["r2", "abs_er_dev"]].agg(["count", "median"]).round(3).to_string())


def density():
    rows = []
    for f in sorted(glob.glob(os.path.join(SYNTH, "synth_*.csv"))):
        park = os.path.basename(f)[6:-4]
        d = pd.read_csv(f, sep=";", index_col=0, parse_dates=True)
        dcols = [c for c in d.columns if c.startswith("density_t")]
        rho = d[dcols].mean(axis=1)
        rows.append({"park_id": park, "n_turbine_groups": len(dcols), "mean": rho.mean(),
                     "p05": rho.quantile(.05), "p50": rho.median(), "p95": rho.quantile(.95),
                     "min": rho.min(), "max": rho.max(),
                     "DJF": rho[rho.index.month.isin([12, 1, 2])].mean(),
                     "JJA": rho[rho.index.month.isin([6, 7, 8])].mean()})
    dp = pd.DataFrame(rows).set_index("park_id")
    dp.to_csv(os.path.join(OUT, "density_parks.csv"))
    print("\nhub-height air density per park [kg/m3] (M5all_PL):")
    print(dp.round(3).to_string())

    files = sorted(glob.glob(os.path.join(SITES, "synth_*.parquet")))
    if not files:
        print(f"no site parquet files under {SITES}")
        return
    acc = {f"t{i}": [] for i in range(1, 7)}
    for f in files:
        d = pd.read_parquet(f, columns=[f"density_t{i}" for i in range(1, 7)])
        for i in range(1, 7):
            acc[f"t{i}"].append(d[f"density_t{i}"].values)
    rows = []
    for t, parts in acc.items():
        a = np.concatenate(parts)
        rows.append({"turbine": t, "mean": a.mean(), "p05": np.percentile(a, 5), "p50": np.median(a),
                     "p95": np.percentile(a, 95), "min": a.min(), "max": a.max()})
    dt = pd.DataFrame(rows).set_index("turbine")
    dt.to_csv(os.path.join(OUT, "density_dataset_turbines.csv"))
    print(f"\nhub-height air density per dataset turbine over {len(files)} sites [kg/m3]:")
    print(dt.round(3).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    by_class()
    density()
    print("CLASS_DENSITY_DONE")
