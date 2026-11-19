#!/usr/bin/env python
"""Why the tilt needs seasonal spread. (a) cos(z - b) / cos z, the noon beam ratio of a south plane, over the noon
zenith z for b = 0, 10, 20, 30 deg, each scaled to 1 at z = 30 deg (the fit has a free scale), with the noon
zenith of the three calibration starts at 50 deg N. (b) Median absolute tilt error of the grid estimator on the
90 synthetic parks against the noon-zenith spread of its clear noons, POA bank (run 6) and chain bank (run 7),
all starts and windows pooled; the line marks TILT_ZENITH_SPREAD.

    synthre/bin/python scripts/pv/report_angle_identifiability.py
Output: reports/pv_angle_decomposition/figs/fig_identifiability.pdf and identifiability_bins.csv
"""
import os

import numpy as np
import pandas as pd

import report_angle_common as rc
from pv import plausibility as PL, timebase

BINS = [0, 5, 10, 15, 20, 30, 60]


def noon_zenith(date: str, lat: float = 50.0, lon: float = 10.0) -> float:
    idx = pd.date_range(f"{date} 09:00", f"{date} 14:00", freq="5min", tz="UTC")
    return float(timebase.solar_position(idx, lat, lon, 0.0)["apparent_zenith"].min())


def main():
    rc.style()
    plt = rc.plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(rc.COL, 1.85), gridspec_kw={"wspace": 0.42})
    z = np.linspace(20, 75, 200)
    zr = np.radians(z)
    for b, c in zip((0, 10, 20, 30), ("#B0B4BA", "#8FA6CE", "#4F6FB0", "#1F4596")):
        f = np.cos(zr - np.radians(b)) / np.cos(zr)
        f0 = np.cos(np.radians(30 - b)) / np.cos(np.radians(30))
        a1.plot(z, f / f0, color=c, lw=1.5, label=f"{b}°")
    starts = {"Jun": noon_zenith("2024-06-01"), "Mar": noon_zenith("2024-03-01"), "Dec": noon_zenith("2023-12-01")}
    for k, v in starts.items():
        a1.axvline(v, color="0.6", lw=0.8, ls=":")
        a1.text(v, 2.57, k, ha="center", va="bottom", fontsize=8, clip_on=False)
    a1.set_xlabel("noon zenith z (°)")
    a1.set_ylabel("cos(z−β)/cos z, scaled")
    a1.set_ylim(0.9, 2.55)
    a1.set_xlim(20, 75)
    a1.legend(title="tilt β", loc="upper left", fontsize=8, title_fontsize=8, handlelength=1.2, borderaxespad=0.1)
    rows = []
    for f, lab, c in (("plausibility_calibration.csv", "POA bank", rc.C_POA), ("plausibility_calibration_chain_none.csv", "chain bank", rc.C_CHAIN)):
        d = pd.read_csv(os.path.join(rc.FIGS_SYN, f)).dropna(subset=["err_tilt_grid", "noon_zenith_spread"])
        d["bin"] = pd.cut(d.noon_zenith_spread, BINS, right=False)
        g = d.groupby("bin", observed=True).agg(med=("err_tilt_grid", lambda s: float(np.median(np.abs(s)))),
                                                bias=("err_tilt_grid", "median"), n=("err_tilt_grid", "size"))
        mid = [0.5 * (iv.left + iv.right) for iv in g.index]
        a2.plot(mid, g.med, "o-", color=c, ms=3.5, lw=1.4, label=lab)
        for iv, r in g.iterrows():
            rows.append({"bank": lab, "spread_lo": iv.left, "spread_hi": iv.right, "median_abs_err": r.med, "median_err": r.bias, "n": int(r.n)})
    a2.axvline(PL.TILT_ZENITH_SPREAD, color="k", lw=0.9, ls="--")
    a2.set_xlabel("noon spread Δz (°)")
    a2.set_ylabel("median |tilt error| (°)")
    a2.set_ylim(0, None)
    a2.set_xlim(0, 45)
    a2.legend(loc="upper right", fontsize=8, handlelength=1.2, borderaxespad=0.1)
    for ax, tag in ((a1, "(a)"), (a2, "(b)")):
        ax.set_title(tag, loc="left", fontsize=9, pad=13)
    rc.save(fig, "fig_identifiability.pdf")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(rc.OUT, "identifiability_bins.csv"), index=False)
    print({k: round(v, 1) for k, v in starts.items()})
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
