#!/usr/bin/env python
"""R2 surface of the grid fit (chain bank) over tilt x azimuth: a public synthetic park (summer start, 30 vs 365
days, raw CAMS as in the calibration) and pv7 (corrected CAMS, 30 vs 312 days from 2023-07-24). Marks the best
pair, the profile region R2 >= R2max - 0.005, the day-bootstrap picks (50 draws) and the truth or register.

    synthre/bin/python scripts/pv/report_angle_surface.py [--synthetic SEL981208943378]
Output: reports/pv_angle_decomposition/figs/fig_surface.pdf and surface_summary.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import report_angle_common as rc
from pv import angles, arrays as arr, plausibility as PL, timebase


def surface(power, bank, mask, n_boot=50, seed=0):
    g = angles.grid_fit(power, bank, mask)
    best = g.iloc[0]
    days = np.array(sorted(set(power.index[mask & power.notna()].date)))
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(n_boot):
        pick = set(rng.choice(days, len(days), replace=True))
        mb = mask & pd.Series(np.isin(power.index.date, list(pick)), index=power.index)
        gb = angles.grid_fit(power, bank, mb)
        picks.append((gb.iloc[0].tilt, gb.iloc[0].azimuth))
    return g, best, np.array(picks)


def synthetic_case(pid):
    from parks_pv import paths
    d = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{pid}.parquet"))
    cfg = json.load(open(os.path.join(paths.RELEASE_DIR, f"manifest_{pid}.json")))["config"]
    lat, lon, alt = cfg["latitude"], cfg["longitude"], cfg["altitude"]
    drv = pd.DataFrame({"ghi": d["ghi_cams_raw"], "dhi": np.minimum(d["dhi_cams_raw"], d["ghi_cams_raw"]),
                        "temp_air": d["temp_air"], "wind_speed": d["wind_speed"], "albedo": 0.2})
    exclude = (d["clipped"] > 0) | (d["snow_depth_cm"] > 0.5) | (d["imputed_share"] > 0)
    a_ = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    a_ = a_[a_.park_id == pid]
    t_true, az_true = arr.mean_angles(a_.tilt.values, a_.azimuth.values, a_.share.values)
    s = pd.Timestamp("2024-06-01", tz="UTC")
    out = []
    for w in (30, 365):
        sl = slice(s, s + pd.Timedelta(days=w))
        bank = PL.chain_bank(cfg, drv.loc[sl], PL.TILTS, PL.AZIMUTHS)
        mask, _, _ = rc.estimation_mask(d["power_park"].loc[sl], drv.loc[sl], lat, lon, alt, bank, exclude.loc[sl], None)
        zen = timebase.solar_position(drv.loc[sl].index, lat, lon, alt)["apparent_zenith"]
        spread = PL.noon_zenith_spread(mask & d["power_park"].loc[sl].notna(), zen)
        out.append((f"synthetic park, {w} d", d["power_park"].loc[sl], bank, mask, (float(t_true), float(az_true)), "truth", spread))
    return out


def pv7_case():
    t = rc.trianel_park("pv7")
    p = t["p"]
    bank_all = PL.chain_bank(t["cfg_for"](t["terr"]), t["corr"], PL.TILTS, PL.AZIMUTHS, horizon=t["hz"])
    out = []
    for w in (30, 312):
        sl = slice(t["idx"][0], t["idx"][0] + pd.Timedelta(days=w))
        bank = {k: v.loc[sl] for k, v in bank_all.items()}
        y = t["y"].loc[sl]
        mask, _, _ = rc.estimation_mask(y, t["corr"].loc[sl], p["lat"], p["lon"], p["alt"], bank, (t["depth_cm"] > 0.5).loc[sl], t["ac_lim"])
        zen = timebase.solar_position(t["corr"].loc[sl].index, p["lat"], p["lon"], p["alt"])["apparent_zenith"]
        spread = PL.noon_zenith_spread(mask & y.notna(), zen)
        out.append((f"pv7, {w} d", y, bank, mask, (float(p["tilt_master"]), float(p["az_master"])), "register", spread))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="SEL981208943378")
    a = ap.parse_args()
    cases = synthetic_case(a.synthetic) + pv7_case()
    rc.style()
    plt = rc.plt
    fig, axes = plt.subplots(1, 4, figsize=(rc.FULL, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    rows = []
    for ax, (title, y, bank, mask, ref, ref_lab, spread) in zip(axes, cases):
        g, best, picks = surface(y, bank, mask)
        g["d"] = g.r2 - best.r2
        piv = g.pivot(index="tilt", columns="azimuth", values="d")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap="Blues", vmin=-0.05, vmax=0, shading="nearest")
        reg = (piv.values >= -0.005).astype(float)
        ax.contour(piv.columns, piv.index, reg, levels=[0.5], colors=rc.C_METER, linewidths=1.0)
        jit = np.random.default_rng(1).normal(0, 1.2, picks.shape)
        ax.scatter(picks[:, 1] + 2 * jit[:, 1], picks[:, 0] + jit[:, 0], s=4, color=rc.C_EW, alpha=0.6, lw=0, zorder=3)
        ax.plot(best.azimuth, best.tilt, "o", ms=7, mfc="none", mec=rc.C_EW, mew=1.6, zorder=4)
        ax.plot(ref[1], ref[0], "x", ms=7, color="k", mew=1.6, zorder=4)
        ax.set_title(f"{title}\nnoon spread {spread:.0f}°", fontsize=8.5, loc="left", pad=2)
        ax.set_xticks([90, 180, 270])
        ax.set_xticklabels(["E", "S", "W"])
        ax.set_xlim(60, 300)
        ax.set_ylim(-2.5, 62.5)
        rows.append({"case": title, "best_tilt": best.tilt, "best_az": best.azimuth, "r2": best.r2,
                     "region_tilt": f"{g[g.d >= -0.005].tilt.min():.0f}-{g[g.d >= -0.005].tilt.max():.0f}",
                     "region_az": f"{g[g.d >= -0.005].azimuth.min():.0f}-{g[g.d >= -0.005].azimuth.max():.0f}",
                     "boot_tilt_5_95": f"{np.percentile(picks[:, 0], 5):.0f}-{np.percentile(picks[:, 0], 95):.0f}",
                     "n_hours": int(mask.sum()), "noon_spread": round(spread, 1), "ref": ref, "ref_kind": ref_lab})
    axes[0].set_ylabel("tilt (°)")
    for ax in axes:
        ax.set_xlabel("azimuth")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("R$^2$ − best", fontsize=8.5)
    rc.save(fig, "fig_surface.pdf")
    pd.DataFrame(rows).to_csv(os.path.join(rc.OUT, "surface_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
