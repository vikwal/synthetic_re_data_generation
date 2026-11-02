#!/usr/bin/env python3
"""Round 3 (reviewer 1c): distribution of the two-level ERA5 shear exponent
alpha = ln(v100/v10) / ln(10) over all 200 station sites and the 13 park
stations, July 2023 - April 2030 of the v2 station files. Reports the
quantiles, the shares of negative, > 0.4 (clip bound of the chain) and
> 0.6 values, and numerically fragile hours (v10 < 0.5 m/s).

Run on l2: synthre/bin/python scripts/round3/alpha_stats.py
Output: results/round3/alpha_stats_sites.csv (per site), alpha_stats_summary.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ERA5_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "round3")
PARK_STATIONS = ["00198", "00282", "00298", "01200", "01303", "02483",
                 "04745", "05347", "05426", "07374"]
CLIP = (0.0, 0.4)


def site_stats(path):
    df = pd.read_csv(path, usecols=["timestamp", "u_wind_10m", "v_wind_10m",
                                    "u_wind_100m", "v_wind_100m"])
    v10 = np.hypot(df.u_wind_10m, df.v_wind_10m)
    v100 = np.hypot(df.u_wind_100m, df.v_wind_100m)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(v100 / v10) / np.log(10.0)
    ok = np.isfinite(a)
    q = np.nanpercentile(a[ok], [1, 5, 25, 50, 75, 95, 99])
    return {
        "n_hours": int(len(a)), "n_nonfinite": int((~ok).sum()),
        "share_v10_lt_0p5": float((v10 < 0.5).mean()),
        "share_neg": float((a[ok] < CLIP[0]).mean()),
        "share_gt_0p4": float((a[ok] > CLIP[1]).mean()),
        "share_gt_0p6": float((a[ok] > 0.6).mean()),
        "share_clipped": float(((a[ok] < CLIP[0]) | (a[ok] > CLIP[1])).mean()),
        "p01": q[0], "p05": q[1], "p25": q[2], "p50": q[3], "p75": q[4], "p95": q[5], "p99": q[6],
        "mean": float(np.nanmean(a[ok])),
        # energy-weighted share: hours with v100 >= 4 m/s (power-relevant)
        "share_clipped_v100ge4": float(((a[ok] < CLIP[0]) | (a[ok] > CLIP[1]))[v100[ok] >= 4].mean()),
    }, a[ok]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, pooled, pooled_parks = [], [], []
    for f in sorted(glob.glob(os.path.join(ERA5_DIR, "Station_*.csv"))):
        sid = os.path.basename(f)[8:-4]
        st, a = site_stats(f)
        st["station_id"] = sid
        st["park_station"] = sid in PARK_STATIONS
        rows.append(st)
        pooled.append(a)
        if sid in PARK_STATIONS:
            pooled_parks.append(a)
    sites = pd.DataFrame(rows).set_index("station_id")
    sites.to_csv(os.path.join(OUT, "alpha_stats_sites.csv"))

    def summary(arrs, label):
        a = np.concatenate(arrs)
        q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
        return {"set": label, "n_hours": len(a), "share_neg": float((a < 0).mean()),
                "share_gt_0p4": float((a > 0.4).mean()), "share_gt_0p6": float((a > 0.6).mean()),
                "share_clipped": float(((a < 0) | (a > 0.4)).mean()),
                "p01": q[0], "p05": q[1], "p25": q[2], "p50": q[3], "p75": q[4], "p95": q[5], "p99": q[6],
                "mean": float(a.mean())}
    summ = pd.DataFrame([summary(pooled, "all_200_sites"), summary(pooled_parks, "13_park_stations")])
    summ.to_csv(os.path.join(OUT, "alpha_stats_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    print("\nper-site spread of the clipped share: min %.4f  median %.4f  max %.4f (site %s)" % (
        sites.share_clipped.min(), sites.share_clipped.median(), sites.share_clipped.max(),
        sites.share_clipped.idxmax()))
    print("park stations:\n", sites.loc[PARK_STATIONS, ["share_neg", "share_gt_0p4", "share_clipped",
                                                          "share_clipped_v100ge4", "p05", "p50", "p95"]].round(4).to_string())


if __name__ == "__main__":
    main()
