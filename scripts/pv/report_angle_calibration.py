#!/usr/bin/env python
"""Calibration of the power-based estimator on the 90 synthetic parks over the window length: (a) median absolute
azimuth error on the 59 homogeneous parks, (b) median absolute tilt error on all parks, (c) share of the 15
register deviations found and the azimuth false-alarm rate. Spring and summer starts; winter starts are reported
in the text (unusable up to 30 days). Runs 6 (POA bank), 7 (chain bank) and 8 (chain bank + OSM row prior).

    synthre/bin/python scripts/pv/report_angle_calibration.py
Output: reports/pv_angle_decomposition/figs/fig_calibration.pdf and calibration_table.csv
"""
import os

import numpy as np
import pandas as pd

import report_angle_common as rc

RUNS = [("plausibility_calibration.csv", "POA bank", rc.C_POA),
        ("plausibility_calibration_chain_none.csv", "chain bank", rc.C_CHAIN),
        ("plausibility_calibration_chain_osm.csv", "chain bank + rows", rc.C_PRIOR)]
WINDOWS = [14, 30, 90, 365]


def scores(d: pd.DataFrame) -> pd.DataFrame:
    dev = d.deviation.fillna("").astype(str).str.len() > 0
    homog = ~dev & (d.register_orientation != "Ost-West")
    rows = []
    for (season, w), g in d.groupby(["season", "window_days"]):
        h = homog.loc[g.index]
        dv = dev.loc[g.index]
        az_dec = g.register_az_ok[~dv].dropna()
        found = g.register_az_ok[dv].dropna().astype(bool)
        tilt_found = g.register_tilt_ok[dv].dropna().astype(bool)
        rows.append({"season": season, "window": w,
                     "az_homog": float(np.median(np.abs(g.err_az_grid[h].dropna()))),
                     "az_all": float(np.median(np.abs(g.err_az_grid.dropna()))),
                     "tilt_all": float(np.median(np.abs(g.err_tilt_grid.dropna()))),
                     "tilt_bias": float(np.median(g.err_tilt_grid.dropna())),
                     "tilt_judged": float(g.tilt_identifiable.mean()),
                     "false_alarm_az": float((~az_dec.astype(bool)).mean()) if len(az_dec) else np.nan,
                     "found": float(((~found) | (~tilt_found.reindex(found.index).fillna(True))).mean()) if len(found) else np.nan,
                     "found_az": float((~found).mean()) if len(found) else np.nan,
                     "n_homog": int(h.sum()), "n_dev": int(dv.sum())})
    return pd.DataFrame(rows)


def main():
    rc.style()
    plt = rc.plt
    fig, axes = plt.subplots(1, 3, figsize=(rc.FULL, 1.9), gridspec_kw={"wspace": 0.32})
    tabs = []
    for f, lab, c in RUNS:
        s = scores(pd.read_csv(os.path.join(rc.FIGS_SYN, f)))
        s["run"] = lab
        tabs.append(s)
        for season, ls, mk in (("spring", "-", "o"), ("summer", "--", "s")):
            q = s[s.season == season].set_index("window").reindex(WINDOWS)
            x = np.arange(len(WINDOWS))
            kw = dict(color=c, ls=ls, marker=mk, ms=3.2, lw=1.3)
            axes[0].plot(x, q.az_homog, **kw, label=f"{lab}, {season}")
            axes[1].plot(x, q.tilt_all, **kw)
            axes[2].plot(x, 100 * q.found_az, **kw)
    for ax, yl in zip(axes, ("median |azimuth error| (°)", "median |tilt error| (°)", "deviations found (%)")):
        ax.set_xticks(range(len(WINDOWS)))
        ax.set_xticklabels([f"{w} d" for w in WINDOWS])
        ax.set_ylabel(yl)
        ax.set_xlabel("history")
    axes[0].set_ylim(0, 10)
    axes[1].set_ylim(0, 8)
    axes[2].set_ylim(50, 100)
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_title(tag, loc="left", fontsize=9, pad=3)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.38), handlelength=2.2, columnspacing=1.2)
    rc.save(fig, "fig_calibration.pdf")
    out = pd.concat(tabs)
    out.to_csv(os.path.join(rc.OUT, "calibration_table.csv"), index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
