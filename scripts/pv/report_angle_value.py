#!/usr/bin/env python
"""Value of the geometry source in the production chain at the six meters: Delta R2 against the register angles
(one plane) per park, day hours without curtailment suspects and winter (15 Nov - 15 Mar), 2023-07-24 to
2024-05-31 (312 days). Estimates are inserted unconditionally. Sources: chain bank 30 and 312 days, row prior +
chain bank 30 days, geodata without a meter (pv3, pv7 only).

    synthre/bin/python scripts/pv/report_angle_value.py
Output: reports/pv_angle_decomposition/figs/fig_value.pdf and value_table.csv (all sources, R2 / RMSE / nRMSE)
"""
import os

import numpy as np
import pandas as pd

import report_angle_common as rc

FILES = {"poa": "ablation_trianel_geometry.csv", "chain": "ablation_trianel_geometry_chain.csv",
         "prior": "ablation_trianel_geometry_chain_prior.csv"}
SOURCES = [("chain", "geometry_estimated_30d_chain", "chain bank, 30 d", rc.C_POA),
           ("chain", "geometry_estimated_312d_chain", "chain bank, 312 d", rc.C_CHAIN),
           ("prior", "geometry_estimated_30d_chain_prior", "rows + chain bank, 30 d", rc.C_PRIOR),
           ("poa", "geometry_geodata", "geodata, no meter", rc.C_GEO)]


def main():
    d = pd.concat([pd.read_csv(os.path.join(rc.FIGS_GEN, f)).assign(src=k) for k, f in FILES.items()])
    d = d[d.pv.isin(rc.PARKS) & d.subset.isin(["day_nosuspect", "winter", "cloudy"])]
    tab = d.drop_duplicates(["pv", "rung", "subset"])[["pv", "rung", "subset", "n", "r2", "rmse", "nrmse_cap", "energy_ratio"]]
    tab.to_csv(os.path.join(rc.OUT, "value_table.csv"), index=False)
    rc.style()
    plt = rc.plt
    fig, axes = plt.subplots(2, 1, figsize=(rc.COL, 3.1), sharex=True, sharey=True, gridspec_kw={"hspace": 0.18})
    x = np.arange(len(rc.PARKS))
    w = 0.2
    for ax, sub, title in zip(axes, ("day_nosuspect", "winter"), ("day hours", "winter")):
        q = tab[tab.subset == sub].pivot(index="rung", columns="pv", values="r2")[rc.PARKS]
        base = q.loc["geometry_master_only"]
        for k, (_, rung, lab, c) in enumerate(SOURCES):
            v = (q.loc[rung] - base).values
            ax.bar(x + (k - 1.5) * w, v, w, color=c, label=lab)
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_ylabel(f"ΔR$^2$, {title}")
        ax.set_ylim(-0.11, 0.3)
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        r7 = q.loc[[s[1] for s in SOURCES], "pv7"] - base["pv7"]
        print(sub, r7.round(3).to_dict())
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rc.PARKS)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8, handlelength=1.0, columnspacing=0.8)
    rc.save(fig, "fig_value.pdf")


if __name__ == "__main__":
    main()
