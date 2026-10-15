#!/usr/bin/env python
"""pv7 on one clear summer day and one clear, snow-free winter day: the meter against the production chain with
three geometries, each scaled to the meter on all masked clear hours of the 312-day window (the shape is
compared, not the level): register 25/184 deg, the chain-bank estimate 10/180 deg, and the east-west pair 7.5 deg
at 77/257 deg (row axis from the DOM row band). Shows why flat south and east-west below 10 deg look alike.

    synthre/bin/python scripts/pv/report_angle_bankseries.py
Output: reports/pv_angle_decomposition/figs/fig_bankseries.pdf and bankseries_summary.csv
"""
import os

import numpy as np
import pandas as pd

import report_angle_common as rc
from pv import chain, plausibility as PL

GEOMS = [("register 25°/184°", [{"name": "a", "tilt": 25.0, "azimuth": 184.0, "share": 1.0}], rc.C_REG, "-"),
         ("estimate 10°/180°", [{"name": "a", "tilt": 10.0, "azimuth": 180.0, "share": 1.0}], rc.C_CHAIN, "-"),
         ("east-west 7.5°", [{"name": "e", "tilt": 7.5, "azimuth": 77.0, "share": 0.5},
                             {"name": "w", "tilt": 7.5, "azimuth": 257.0, "share": 0.5}], rc.C_EW, "--")]


def main():
    t = rc.trianel_park("pv7")
    p, y, corr, idx = t["p"], t["y"], t["corr"], t["idx"]
    series = {}
    for lab, arrays, _, _ in GEOMS:
        res = chain.run_park(t["cfg_for"](arrays), corr, horizon=t["hz"])
        series[lab] = res["p_ac_w"]
    bank = {(0.0, 0.0): series["estimate 10°/180°"]}
    mask, n_clear, n_sus = rc.estimation_mask(y, corr, p["lat"], p["lon"], p["alt"], bank, t["depth_cm"] > 0.5, t["ac_lim"])
    rows = []
    for lab, s in series.items():
        s_, r2 = PL._fit(y[mask].values, s[mask].values)
        series[lab] = s * s_
        tss = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        rows.append({"geometry": lab, "scale": s_, "r2_clear": 1 - r2 / tss, "n_hours": int(mask.sum())})
    per_day = mask.groupby(idx.date).sum()
    snow_free = (t["depth_cm"] < 0.1).groupby(idx.date).all()
    months = pd.Series(pd.to_datetime(per_day.index).month, index=per_day.index)
    summer = per_day[months.isin([7, 8]).values].idxmax()
    winter = per_day[(months.isin([12, 1, 2]).values) & snow_free.reindex(per_day.index).values].idxmax()
    rc.style()
    plt = rc.plt
    fig, axes = plt.subplots(1, 2, figsize=(rc.COL, 1.75), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, day, title in zip(axes, (summer, winter), ("clear summer day", "clear winter day")):
        sl = (idx.date == day)
        h = idx[sl].hour
        ax.plot(h, y[sl] / t["cap_w"], color=rc.C_METER, lw=0, marker="o", ms=3, label="meter")
        for lab, _, c, ls in GEOMS:
            ax.plot(h, series[lab][sl] / t["cap_w"], color=c, ls=ls, lw=1.4, label=lab)
        ax.set_title(title, fontsize=8.5, loc="left", pad=2)
        ax.set_xlabel("hour (UTC)")
        ax.set_xlim(3, 20)
        ax.set_xticks([6, 12, 18])
    axes[0].set_ylabel("power / capacity")
    axes[0].set_ylim(0, None)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.45), handlelength=1.6, columnspacing=1.0)
    rc.save(fig, "fig_bankseries.pdf")
    out = pd.DataFrame(rows)
    out["summer_day"], out["winter_day"], out["n_clear_before_suspect"], out["n_suspect"] = str(summer), str(winter), n_clear, n_sus
    out.to_csv(os.path.join(rc.OUT, "bankseries_summary.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
