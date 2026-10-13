#!/usr/bin/env python
"""How good is the ERA5 snow driver near the parks? Daily ERA5 snow depth at the park point vs the
snow depth of the nearest DWD climate station (daily kl, SHK_TAG in cm). Anonymised output.

    python scripts/pv/snow_driver_check.py --snow-cache <dir with pv*_era5_snow.parquet> --dwd-csv <trianel_nearest_kl_daily.csv>
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snow-cache", required=True)
    ap.add_argument("--dwd-csv", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    args = ap.parse_args()
    dwd = pd.read_csv(args.dwd_csv, parse_dates=["date"])
    rows, series = [], {}
    for pv, g in dwd.groupby("pv"):
        f = os.path.join(args.snow_cache, f"{pv}_era5_snow.parquet")
        if not os.path.exists(f):
            continue
        e = pd.read_parquet(f)
        daily = pd.DataFrame({"era5_depth_cm": (e["snow_depth_m"] * 100).resample("1D").mean(), "era5_snowfall_mm": e["snowfall_mm_we"].resample("1D").sum(),
                              "era5_tmean": e["temp_air"].resample("1D").mean()})
        daily.index = daily.index.tz_localize(None)
        d = g.set_index("date")[["SHK_TAG", "TMK", "RSK"]].join(daily, how="inner")
        d = d[(d.index >= "2023-11-01") & (d.index <= "2024-03-31")]
        both = d["SHK_TAG"].notna() & d["era5_depth_cm"].notna()
        obs_snow, era_snow = d["SHK_TAG"] >= 1, d["era5_depth_cm"] >= 1
        rows.append({"pv": pv, "station_dist_km": float(g["dist_km"].iloc[0]), "days": int(both.sum()),
                     "dwd_snow_days": int((obs_snow & both).sum()), "era5_snow_days": int((era_snow & both).sum()),
                     "hit": int((obs_snow & era_snow & both).sum()), "miss": int((obs_snow & ~era_snow & both).sum()),
                     "false_alarm": int((~obs_snow & era_snow & both).sum()), "corr_depth": float(d.loc[both, ["SHK_TAG", "era5_depth_cm"]].corr().iloc[0, 1]),
                     "dwd_max_cm": float(d["SHK_TAG"].max()), "era5_max_cm": float(d["era5_depth_cm"].max()), "tmean_bias": float((d["era5_tmean"] - d["TMK"]).mean())})
        series[pv] = d
    res = pd.DataFrame(rows).set_index("pv")
    res.to_csv(os.path.join(args.out, "snow_driver_check.csv"))
    print(res.round(2).to_string())
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 1.6 * len(series)), sharex=True)
    for ax, (pv, d) in zip(np.atleast_1d(axes), series.items()):
        ax.fill_between(d.index, 0, d["SHK_TAG"], color="#52514e", alpha=0.5, step="mid", label="DWD station snow depth")
        ax.plot(d.index, d["era5_depth_cm"], color="#2a78d6", lw=1.2, label="ERA5 snow depth at park")
        ax.set_ylabel(f"{pv}\ncm")
    np.atleast_1d(axes)[0].legend(ncol=2, fontsize=7.5)
    fig.suptitle("Snow depth: nearest DWD climate station vs ERA5 at the park point, winter 2023/24", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_snow_driver_check.png"))


if __name__ == "__main__":
    main()
