#!/usr/bin/env python
"""Benchmark of the pv chain against the Trianel PV parks with meter data.

Three irradiance drivers (nearest imputed DWD station, CAMS at the park, ERA5 cell) x four chain
variants (L legacy-like PVWatts, P repaired PVWatts, C CEC+Sandia+aging+horizon, CF = C with
angles fitted from the meter data) per park. Outputs are anonymised (pv1..pv7): metric tables,
figures and an error decomposition in --out; hourly series go to --series-dir (outside any repo).

    python scripts/pv/benchmark_trianel.py --cams-dir <dir> --horizon-json <pvgis.json>
        --mastr-cand <csv> --mastr-xml <json> --out ~/Work/FL_Contribution/reports/figs_pv_generator
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pv import angle_diagnostics, angles, chain, metrics, timebase  # noqa: E402
from parks import era5_db  # noqa: E402
import trianel_common as tc  # noqa: E402

COL = {"meas": "#52514e", "station": "#2a78d6", "cams": "#eb6834", "era5": "#1baf7a", "fit": "#4a3aa7"}
DRIVERS = ["station", "cams", "era5"]
VARIANTS = ["L", "P", "C", "CR", "CF", "CRF"]
ALPHA = {"L": 0.25, "P": 0.4, "C": 0.55, "CR": 0.7, "CF": 0.85, "CRF": 1.0}
TILTS = list(range(5, 50, 5))
AZS = list(range(90, 275, 10))

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "font.size": 8.5,
                     "legend.frameon": False, "savefig.bbox": "tight", "lines.linewidth": 1.4})


def season(idx):
    return pd.Series(idx.month, index=idx).map(lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5) else "JJA" if m in (6, 7, 8) else "SON")


def scale_fit(y, p, mask):
    m = mask & y.notna() & p.notna() & (p > 0)
    x, t = p[m].values, y[m].values
    return float(np.dot(x, t) / np.dot(x, x)) if np.dot(x, x) > 0 else np.nan


def evaluate_park(pv, p, par, meter, drvs, horizon, args, log):
    cap_w = float(par["kwp_dc"]) * 1000.0
    y = meter.reindex(drvs["era5"].index)
    rows, series, extras = [], {}, {}
    # plateau / curtailment analysis of the meter data
    pl = metrics.plateaus(y, cap_w)
    plsum = metrics.plateau_summary(pl, cap_w)
    extras["plateaus"] = plsum
    off_mode_hours = pd.DatetimeIndex([])
    if not pl.empty and "mode_level_norm" in plsum:
        off = pl[(pl["level_norm"] - plsum["mode_level_norm"]).abs() > 0.03]
        off_mode_hours = pd.DatetimeIndex(np.concatenate([pd.date_range(a, b, freq="1h") for a, b in zip(off["start"], off["end"])])) if len(off) else off_mode_hours
    curt_mask = pd.Series(True, index=y.index)
    curt_mask.loc[curt_mask.index.isin(off_mode_hours)] = False

    # fitted angles on the CAMS driver (fallback station): grid + NNLS on clear hours
    fit_drv = drvs.get("cams", drvs["station"])
    sp = timebase.solar_position(fit_drv.index, p["lat"], p["lon"], p["alt"])
    bank = angles.poa_bank(fit_drv, p["lat"], p["lon"], p["alt"], TILTS, AZS, albedo=0.2, horizon=horizon)
    clear = angles.clear_hours(fit_drv["ghi"], p["lat"], p["lon"], p["alt"]) & curt_mask & y.notna()
    ynorm = y / cap_w
    fit_win = clear & (fit_drv.index < pd.Timestamp(args.fit_end, tz="UTC"))
    grid_all = angles.grid_fit(ynorm, bank, clear)
    grid_win = angles.grid_fit(ynorm, bank, fit_win)
    nn_all = angles.nnls_mixture(ynorm, bank, clear)
    nn_win = angles.nnls_mixture(ynorm, bank, fit_win)
    extras["angles"] = {"grid_all": grid_all.iloc[0].to_dict(), "grid_fitwin": grid_win.iloc[0].to_dict(),
                        "nnls_all": {k: v for k, v in nn_all.items() if k != "weights"},
                        "nnls_fitwin": {k: v for k, v in nn_win.items() if k != "weights"},
                        "n_clear_hours": int(clear.sum()), "n_clear_fitwin": int(fit_win.sum())}
    extras["grid_all"] = grid_all
    w = nn_win["weights"]
    w = w[w >= 0.02]
    fitted_arrays = [{"tilt": float(t), "azimuth": float(a), "share": float(v / w.sum())} for (t, a), v in w.items()]
    extras["fitted_arrays"] = fitted_arrays
    extras["nnls_weights"] = nn_win["weights"]

    zen = sp["apparent_zenith"]
    day = zen < 90.0
    tags = {"all": pd.Series(True, index=y.index), "day": day, "day_nocurt": day & curt_mask,
            "fit_window": day & (y.index < pd.Timestamp(args.fit_end, tz="UTC")), "eval_window": day & (y.index >= pd.Timestamp(args.fit_end, tz="UTC"))}
    # curtailment-suspect hours: clear sky (CAMS kc > 0.6), sun > 20 deg, measured < 50 % of the fitted chain
    for v in VARIANTS:
        arrays = fitted_arrays if v in ("CF", "CRF") else None
        cfg = tc.base_config(pv, p, par, arrays=arrays, variant={"CF": "C", "CRF": "CR"}.get(v, v))
        for d in DRIVERS:
            if d not in drvs:
                continue
            t0 = time.time()
            r = chain.run_park(cfg, drvs[d], horizon=horizon)
            ps = r["p_ac_w"]
            series[(v, d)] = r
            s = scale_fit(y, ps, day)
            for tag, m in tags.items():
                b = metrics.basic(y[m], ps[m], cap_w)
                bs = metrics.basic(y[m], ps[m] * s, cap_w)
                rows.append({"pv": pv, "variant": v, "driver": d, "subset": tag, **b, "r2_scaled": bs.get("r2"), "nrmse_scaled": bs.get("nrmse_cap"),
                             "scale": s, "energy_ratio_scaled": bs.get("energy_ratio")})
            if v == "CF" and d == ("cams" if "cams" in drvs else "era5"):
                cs_g = timebase.clearsky(y.index, p["lat"], p["lon"], p["alt"])["ghi"]
                kc = drvs[d]["ghi"] / cs_g.replace(0, np.nan)
                suspect = day & (kc > 0.6) & ((90.0 - zen) > 20.0) & (y < 0.5 * ps) & y.notna()
                extras["suspect_hours_by_month"] = {str(k)[:7]: int(v) for k, v in suspect.resample("MS").sum().items()}
                extras["suspect_hours_total"] = int(suspect.sum())
                tags["day_nosuspect"] = day & ~suspect
            if d == "station":
                imp = drvs[d]["imputed_share"] > 0
                for tag, m in {"imputed_hours": day & imp, "observed_hours": day & ~imp}.items():
                    b = metrics.basic(y[m], ps[m], cap_w)
                    rows.append({"pv": pv, "variant": v, "driver": d, "subset": tag, **b, "scale": s})
            log(f"  {pv} {v} {d}: R2 day {rows[-1 if d != 'station' else -3]['r2']:.3f}  {time.time() - t0:.1f}s")
    # clear-sky power (variant C, era5 met) for the clear-sky index of the measurement
    cs = timebase.clearsky(y.index, p["lat"], p["lon"], p["alt"])
    drv_cs = drvs["era5"].copy()
    drv_cs["ghi"], drv_cs["dhi"] = cs["ghi"].values, cs["dhi"].values
    series[("C", "clear")] = chain.run_park(tc.base_config(pv, p, par, variant="C"), drv_cs, horizon=horizon)
    return pd.DataFrame(rows), series, extras


# ------------------------------------------------------------------ figures

FIG_VARIANTS = ["L", "P", "C", "CF"]


def fig_overview(mt, out):
    sub = mt[(mt["subset"] == "day") & (mt["variant"].isin(FIG_VARIANTS))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col, ttl in zip(axes, ["r2", "nrmse_cap"], ["R² (Tagstunden)", "nRMSE / kWp (Tagstunden)"]):
        pvs = sorted(sub["pv"].unique(), key=lambda s: int(s[2:]))
        x = np.arange(len(pvs))
        wv = 0.8 / (len(FIG_VARIANTS) * len(DRIVERS))
        k = 0
        for v in FIG_VARIANTS:
            for d in DRIVERS:
                vals = [sub[(sub.pv == pv) & (sub.variant == v) & (sub.driver == d)][col].mean() for pv in pvs]
                ax.bar(x + k * wv - 0.4 + wv / 2, vals, wv, color=COL[d], alpha=ALPHA[v], edgecolor="white", linewidth=0.5,
                       label=f"{d} / {v}" if True else None)
                k += 1
        ax.set_xticks(x, pvs)
        ax.set_title(ttl, loc="left")
        if col == "r2":
            ax.set_ylim(0.5, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.suptitle("Trianel-PV-Benchmark: Antrieb (Farbe) × Kettenvariante (Sättigung L < P < C < CF)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_overview_r2_nrmse.png"))
    plt.close(fig)


def fig_diurnal(all_series, meters, caps, out):
    pvs = list(all_series)
    fig, axes = plt.subplots(len(pvs), 4, figsize=(12, 1.9 * len(pvs)), sharex=True, sharey=True, squeeze=False)
    for i, pv in enumerate(pvs):
        y = meters[pv] / caps[pv]
        ss = season(y.index)
        for j, se in enumerate(["DJF", "MAM", "JJA", "SON"]):
            ax = axes[i, j]
            m = ss == se
            ax.plot(y[m].groupby(y[m].index.hour).mean(), color=COL["meas"], lw=2.0, label="Messung")
            for d in DRIVERS:
                if ("C", d) in all_series[pv]:
                    ps = all_series[pv][("C", d)]["p_ac_w"] / caps[pv]
                    ax.plot(ps[m].groupby(ps[m].index.hour).mean(), color=COL[d], lw=1.2, label=d)
            if i == 0:
                ax.set_title(se, loc="left")
            if j == 0:
                ax.set_ylabel(f"{pv}\nP / kWp")
    axes[-1, 0].set_xlabel("Stunde UTC")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Mittlerer Tagesgang je Jahreszeit: Messung vs. Variante C mit drei Antrieben", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_diurnal_seasons.png"))
    plt.close(fig)


def fig_monthly(all_series, meters, out):
    pvs = list(all_series)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.8), sharey=True)
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        y = meters[pv]
        for d in DRIVERS:
            if ("C", d) in all_series[pv]:
                me = metrics.monthly_energy(y, all_series[pv][("C", d)]["p_ac_w"])
                ax.plot(me.index, me["ratio"], color=COL[d], marker="o", ms=3, label=d)
        ax.axhline(1.0, color="#52514e", lw=0.8)
        ax.set_title(pv, loc="left")
        ax.set_ylim(0.5, 1.6)
        ax.tick_params(axis="x", rotation=60, labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("Energie synth / gemessen")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Monatlicher Energiequotient, Variante C", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_monthly_energy_ratio.png"))
    plt.close(fig)


def fig_kc_hist(all_series, meters, out):
    pvs = list(all_series)
    bins = np.linspace(0, 1.3, 27)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.6), sharey=True)
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        pc = all_series[pv][("C", "clear")]["p_ac_w"]
        zen = all_series[pv][("C", "clear")]["zenith"]
        kc_m = metrics.clearsky_index(meters[pv], pc, zen)
        ax.hist(kc_m.dropna(), bins=bins, density=True, histtype="step", color=COL["meas"], lw=2.0, label="Messung")
        for d in DRIVERS:
            if ("C", d) in all_series[pv]:
                kc = metrics.clearsky_index(all_series[pv][("C", d)]["p_ac_w"], pc, zen)
                ax.hist(kc.dropna(), bins=bins, density=True, histtype="step", color=COL[d], lw=1.2, label=d)
        ax.set_title(pv, loc="left")
        ax.set_xlabel("Leistungs-Clear-Sky-Index")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Verteilung des Clear-Sky-Index der Leistung (Sonnenhöhe > 10°)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_kc_histogram.png"))
    plt.close(fig)


def fig_clipping(all_series, meters, caps, out):
    pvs = list(all_series)
    bins = np.linspace(0.5, 1.05, 45)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.6), sharey=True)
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        y = meters[pv] / caps[pv]
        ax.hist(y[y > 0.5].dropna(), bins=bins, histtype="step", color=COL["meas"], lw=2.0, label="Messung")
        for v, d, ls in [("L", "cams", ":"), ("C", "cams", "-")]:
            if (v, d) in all_series[pv]:
                ps = all_series[pv][(v, d)]["p_ac_w"] / caps[pv]
                ax.hist(ps[ps > 0.5], bins=bins, histtype="step", color=COL[d], lw=1.2, ls=ls, label=f"{d} / {v}")
        ax.set_title(pv, loc="left")
        ax.set_xlabel("P / kWp")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Obere Verteilung der normierten Leistung: Clipping-/Abregelungsplateaus", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_clipping_histogram.png"))
    plt.close(fig)


def fig_ramps(all_series, meters, caps, out):
    pvs = list(all_series)
    bins = np.linspace(-0.5, 0.5, 51)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.6), sharey=True)
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        y = meters[pv] / caps[pv]
        ax.hist(metrics.ramps(y), bins=bins, density=True, histtype="step", color=COL["meas"], lw=2.0, label="Messung")
        for d in DRIVERS:
            if ("C", d) in all_series[pv]:
                ax.hist(metrics.ramps(all_series[pv][("C", d)]["p_ac_w"] / caps[pv]), bins=bins, density=True, histtype="step", color=COL[d], lw=1.2, label=d)
        ax.set_yscale("log")
        ax.set_title(pv, loc="left")
        ax.set_xlabel("ΔP / kWp je Stunde")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Rampenverteilung (stündliche Änderung der normierten Leistung)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_ramps.png"))
    plt.close(fig)


def fig_asymmetry(all_series, meters, caps, parks, out):
    pvs = list(all_series)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.8), sharey=True)
    rows = []
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        r = all_series[pv][("C", "cams")] if ("C", "cams") in all_series[pv] else all_series[pv][("C", "era5")]
        sp = pd.DataFrame({"apparent_zenith": r["zenith"], "azimuth": r["azimuth_sun"]})
        y = meters[pv] / caps[pv]
        am = angle_diagnostics.morning_evening_asymmetry(y, sp)
        ax.plot(am["elev_lo"] + 5, am["west_minus_east"], color=COL["meas"], marker="o", lw=2.0, label="Messung")
        for v, c in [("C", COL["cams"]), ("CF", COL["fit"])]:
            key = (v, "cams") if (v, "cams") in all_series[pv] else (v, "era5")
            a2 = angle_diagnostics.morning_evening_asymmetry(all_series[pv][key]["p_ac_w"] / caps[pv], sp)
            ax.plot(a2["elev_lo"] + 5, a2["west_minus_east"], color=c, marker="s", ms=3, label=f"{v} (Stammdaten)" if v == "C" else "CF (gefittet)")
            for _, q in a2.iterrows():
                rows.append({"pv": pv, "series": v, "elev_lo": q["elev_lo"], "west_minus_east": q["west_minus_east"]})
        for _, q in am.iterrows():
            rows.append({"pv": pv, "series": "meas", "elev_lo": q["elev_lo"], "west_minus_east": q["west_minus_east"]})
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(pv, loc="left")
        ax.set_xlabel("Sonnenhöhe (°)")
    np.atleast_1d(axes)[0].set_ylabel("West − Ost (P / kWp)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Morgen/Abend-Asymmetrie bei gleicher Sonnenhöhe (Azimutfehler-Indikator)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_asymmetry.png"))
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_residual_sun(all_series, meters, caps, out):
    pvs = list(all_series)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.3 * len(pvs), 2.8))
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        key = ("C", "cams") if ("C", "cams") in all_series[pv] else ("C", "era5")
        r = all_series[pv][key]
        tab = metrics.residual_by_sun(meters[pv], r["p_ac_w"], r["zenith"], r["azimuth_sun"], caps[pv])
        im = ax.imshow(tab.values, origin="lower", cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(tab.columns)), [f"{c.left:.0f}" for c in tab.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(tab.index)), [f"{c.left:.0f}" for c in tab.index], fontsize=6)
        ax.set_title(pv, loc="left")
        ax.set_xlabel("Sonnenazimut (°)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Sonnenhöhe (°)")
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.02, label="(synth − mess) / kWp")
    fig.suptitle("Residuum nach Sonnenstand, Variante C / CAMS (Neigungs-, Azimut- und Horizontfehler)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_residual_sun.png"))
    plt.close(fig)


def fig_anglefit(extras, parks, out):
    pvs = list(extras)
    fig, axes = plt.subplots(1, len(pvs), figsize=(2.3 * len(pvs), 2.8))
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        g = extras[pv]["grid_all"].pivot(index="tilt", columns="azimuth", values="r2")
        im = ax.imshow(g.values, origin="lower", cmap="Blues", vmin=max(0, g.values.max() - 0.15), vmax=g.values.max(), aspect="auto",
                       extent=[g.columns.min() - 5, g.columns.max() + 5, g.index.min() - 2.5, g.index.max() + 2.5])
        p = parks.loc[pv]
        ax.plot(p["az_master"], p["tilt_master"], marker="x", color="#e34948", ms=8, mew=2, label="Stammdaten")
        b = extras[pv]["angles"]["grid_all"]
        ax.plot(b["azimuth"], b["tilt"], marker="o", mfc="none", color="#0b0b0b", ms=8, mew=1.5, label="bester Einzelwinkel")
        n = extras[pv]["angles"]["nnls_all"]
        ax.plot(n["azimuth_eff"], n["tilt_eff"], marker="^", color=COL["fit"], ms=7, label="NNLS-Mischung")
        ax.set_title(pv, loc="left")
        ax.set_xlabel("Azimut (°)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Neigung (°)")
    np.atleast_1d(axes)[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Winkelschätzung aus Messdaten (R² je Winkelpaar an klaren Stunden, CAMS-Antrieb)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "fig_angle_fit.png"))
    plt.close(fig)


def fig_example_weeks(all_series, meters, caps, pv, out):
    r = all_series[pv]
    weeks = [("2023-08-07", "2023-08-14"), ("2024-01-15", "2024-01-22"), ("2024-04-22", "2024-04-29")]
    fig, axes = plt.subplots(len(weeks), 1, figsize=(11, 2.3 * len(weeks)))
    for ax, (a, b) in zip(axes, weeks):
        y = (meters[pv] / caps[pv])[a:b]
        ax.plot(y.index, y, color=COL["meas"], lw=2.0, label="Messung")
        for d in DRIVERS:
            if ("C", d) in r:
                ps = (r[("C", d)]["p_ac_w"] / caps[pv])[a:b]
                ax.plot(ps.index, ps, color=COL[d], lw=1.0, label=d)
        ax.set_ylabel("P / kWp")
        ax.set_title(f"{pv}, {a} bis {b}", loc="left")
    axes[0].legend(ncol=4, fontsize=7)
    fig.savefig(os.path.join(out, f"fig_example_weeks_{pv}.png"))
    plt.close(fig)


def fig_station_distance(mt, info, out):
    sub = mt[(mt.subset == "day") & (mt.variant == "C")]
    fig, ax = plt.subplots(figsize=(5, 3))
    for d in DRIVERS:
        s = sub[sub.driver == d].set_index("pv")
        x = [info[pv]["station_dist_km"] for pv in s.index]
        ax.scatter(x, s["r2"], color=COL[d], label=d, s=28)
        for pv, xi, yi in zip(s.index, x, s["r2"]):
            if d == "station":
                ax.annotate(pv, (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("Entfernung Park → nächste imputierte DWD-Station (km)")
    ax.set_ylabel("R² Tagstunden, Variante C")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, "fig_station_distance.png"))
    plt.close(fig)


def fig_imputed(mt, out):
    sub = mt[(mt.variant == "C") & (mt.driver == "station") & (mt.subset.isin(["imputed_hours", "observed_hours"]))]
    piv = sub.pivot(index="pv", columns="subset", values="nrmse_cap")
    n = sub.pivot(index="pv", columns="subset", values="n")
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv["observed_hours"], 0.4, color=COL["station"], label="beobachtete Stunden")
    ax.bar(x + 0.2, piv["imputed_hours"], 0.4, color="#9085e9", label="imputierte Stunden")
    for i, pv in enumerate(piv.index):
        ax.annotate(f"n={int(n.loc[pv, 'imputed_hours'])}", (i + 0.2, piv.loc[pv, "imputed_hours"]), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("nRMSE / kWp (Tagstunden)")
    ax.legend(fontsize=7)
    ax.set_title("Stationsantrieb: Fehler an imputierten vs. beobachteten Stunden", loc="left")
    fig.savefig(os.path.join(out, "fig_imputed_hours.png"))
    plt.close(fig)
    return piv.join(n, rsuffix="_n")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--fit-end", default="2024-02-01", help="angles fitted on hours before this date (CF variant)")
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--horizon-json", required=True)
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--series-dir", default=os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic", "solar", "trianel_benchmark"))
    ap.add_argument("--parks", default=None, help="comma list of pv ids (default: all with meter data)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.series_dir, exist_ok=True)
    logf = open(os.path.join(args.out, "benchmark.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== benchmark {time.strftime('%Y-%m-%d %H:%M')} window {args.start}..{args.end}")
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    anon = parks.drop(columns=["lat", "lon", "malo"]).copy()
    anon["arrays"] = anon["arrays"].map(lambda a: "; ".join(f"{x['tilt']:.0f}°/{x['azimuth']:.0f}°×{x['share']:.2f}" for x in a))
    anon = anon.join(facts).join(par.drop(columns=["module_lib"], errors="ignore"))
    anon["meter_first"] = [meter[pv].first_valid_index() if pv in meter else None for pv in anon.index]
    anon["meter_last"] = [meter[pv].last_valid_index() if pv in meter else None for pv in anon.index]
    anon["meter_max_kw"] = [meter[pv].max() / 1000 if pv in meter else np.nan for pv in anon.index]
    conn = era5_db.connect()
    pvs = args.parks.split(",") if args.parks else [pv for pv in parks.index if pv in meter and meter[pv].notna().sum() > 1000]
    all_mt, all_series, all_extras, info, caps = [], {}, {}, {}, {}
    for pv in pvs:
        p = parks.loc[pv]
        log(f"-- {pv}: kWp {par.loc[pv, 'kwp_dc']:.0f} ({par.loc[pv, 'kwp_src']}), dc_ac {par.loc[pv, 'dc_ac']:.2f} ({par.loc[pv, 'dc_ac_src']}), year {par.loc[pv, 'year']} ({par.loc[pv, 'year_src']}), module {par.loc[pv, 'module_lib']}")
        drvs, inf = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, pv)
        inf["horizon_max_deg"] = float(tc.load_horizon(args.horizon_json, pv).max())
        info[pv] = inf
        log(f"   drivers: {inf}")
        hz = tc.load_horizon(args.horizon_json, pv)
        y = meter[pv].reindex(timebase.hourly_index(args.start, args.end))
        mt, series, extras = evaluate_park(pv, p, par.loc[pv], y, drvs, hz, args, log)
        all_mt.append(mt)
        all_series[pv], all_extras[pv] = series, extras
        caps[pv] = float(par.loc[pv, "kwp_dc"]) * 1000.0
        # persist hourly series (confidential-free: anonymised id, no coordinates) outside the repos
        pack = pd.DataFrame({"p_meas_w": y})
        for (v, d), r in series.items():
            pack[f"p_ac_{v}_{d}_w"] = r["p_ac_w"]
            if v == "C":
                for c in ("ghi", "dhi", "dni", "poa_global", "t_cell", "imputed_share"):
                    pack[f"{c}_{d}"] = r[c]
        pack.to_parquet(os.path.join(args.series_dir, f"{pv}_hourly.parquet"))
        json.dump({k: (v if not isinstance(v, pd.DataFrame) else None) for k, v in extras.items() if k not in ("grid_all", "nnls_weights")},
                  open(os.path.join(args.out, f"extras_{pv}.json"), "w"), default=float, indent=1)
        extras["grid_all"].to_csv(os.path.join(args.out, f"anglefit_grid_{pv}.csv"), index=False)
    mt = pd.concat(all_mt, ignore_index=True)
    mt.to_csv(os.path.join(args.out, "metrics_long.csv"), index=False)
    pd.DataFrame(info).T.to_csv(os.path.join(args.out, "driver_info.csv"))
    anon.to_csv(os.path.join(args.out, "parks_anon.csv"))

    # summary tables
    day = mt[mt.subset == "day"]
    piv = day.pivot_table(index=["pv", "variant"], columns="driver", values=["r2", "nrmse_cap", "energy_ratio", "r2_scaled"])
    piv.to_csv(os.path.join(args.out, "summary_day.csv"))
    nocurt = mt[mt.subset == "day_nocurt"].pivot_table(index=["pv", "variant"], columns="driver", values=["r2", "nrmse_cap"])
    nocurt.to_csv(os.path.join(args.out, "summary_day_nocurt.csv"))
    nosus = mt[mt.subset == "day_nosuspect"].pivot_table(index=["pv", "variant"], columns="driver", values=["r2", "nrmse_cap", "energy_ratio"])
    nosus.to_csv(os.path.join(args.out, "summary_day_nosuspect.csv"))
    sus = pd.DataFrame({pv: e.get("suspect_hours_by_month", {}) for pv, e in all_extras.items()})
    sus = sus.sort_index()
    sus.to_csv(os.path.join(args.out, "suspect_hours_by_month.csv"))
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for pv in sus.columns:
        ax.plot(sus.index, sus[pv], marker="o", ms=3, label=pv)
    ax.set_ylabel("Stunden je Monat")
    ax.set_title("Abregelungsverdacht: klarer Himmel, Sonne > 20°, Messung < 50 % der gefitteten Kette", loc="left", fontsize=8)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(os.path.join(args.out, "fig_suspect_hours.png"))
    plt.close(fig)
    split = mt[mt.subset.isin(["fit_window", "eval_window"]) & mt.variant.isin(["C", "CF"])].pivot_table(index=["pv", "driver"], columns=["variant", "subset"], values="r2")
    split.to_csv(os.path.join(args.out, "summary_cf_split.csv"))

    # error decomposition (variant C/CF, day): driver spread, geometry gain, level/loss gain
    rows = []
    for pv in pvs:
        d = day[day.pv == pv]
        c = d[d.variant == "C"].set_index("driver")
        cf = d[d.variant == "CF"].set_index("driver")
        cr = d[d.variant == "CR"].set_index("driver")
        best = c["nrmse_cap"].idxmin()
        rows.append({"pv": pv, "best_driver": best, "rows_gain": c.loc[best, "nrmse_cap"] - cr.loc[best, "nrmse_cap"],
                     "gcr": par.loc[pv, "gcr"], "suspect_hours": all_extras[pv].get("suspect_hours_total"), "nrmse_L_best": d[(d.variant == "L")].set_index("driver").loc[best, "nrmse_cap"],
                     "nrmse_P_best": d[(d.variant == "P")].set_index("driver").loc[best, "nrmse_cap"],
                     "nrmse_C_best": c.loc[best, "nrmse_cap"], "nrmse_C_worst": c["nrmse_cap"].max(), "worst_driver": c["nrmse_cap"].idxmax(),
                     "driver_spread": c["nrmse_cap"].max() - c["nrmse_cap"].min(),
                     "geometry_gain": c.loc[best, "nrmse_cap"] - cf.loc[best, "nrmse_cap"],
                     "level_gain": cf.loc[best, "nrmse_cap"] - cf.loc[best, "nrmse_scaled"],
                     "floor_nrmse": cf.loc[best, "nrmse_scaled"], "scale_CF_best": cf.loc[best, "scale"],
                     "r2_C_best": c.loc[best, "r2"], "r2_CF_best": cf.loc[best, "r2"], "r2_CF_scaled_best": cf.loc[best, "r2_scaled"],
                     **{f"plateau_{k}": v for k, v in all_extras[pv]["plateaus"].items()}})
    dec = pd.DataFrame(rows).set_index("pv")
    dec.to_csv(os.path.join(args.out, "error_decomposition.csv"))
    ang = pd.DataFrame({pv: {"tilt_master": parks.loc[pv, "tilt_master"], "az_master": parks.loc[pv, "az_master"],
                             "tilt_grid": e["angles"]["grid_all"]["tilt"], "az_grid": e["angles"]["grid_all"]["azimuth"], "r2_grid": e["angles"]["grid_all"]["r2"],
                             "tilt_nnls": e["angles"]["nnls_all"]["tilt_eff"], "az_nnls": e["angles"]["nnls_all"]["azimuth_eff"], "r2_nnls": e["angles"]["nnls_all"]["r2"],
                             "n_comp_nnls": e["angles"]["nnls_all"]["n_components"], "tilt_nnls_fitwin": e["angles"]["nnls_fitwin"]["tilt_eff"],
                             "az_nnls_fitwin": e["angles"]["nnls_fitwin"]["azimuth_eff"], "n_clear": e["angles"]["n_clear_hours"],
                             "mastr_orientation": facts.loc[pv, "orientation_class"], "mastr_tilt_class": facts.loc[pv, "tilt_class"]}
                        for pv, e in all_extras.items()}).T
    ang.to_csv(os.path.join(args.out, "angles_estimated.csv"))
    snow = {}
    for pv in pvs:
        key = ("C", "cams") if ("C", "cams") in all_series[pv] else ("C", "era5")
        r = all_series[pv][key]
        sc = metrics.snow_candidates(meter[pv].reindex(r.index), r["p_ac_w"], r["temp_air"], caps[pv])
        snow[pv] = {"snow_candidate_days": int(sc.sum()), "days_evaluated": int(len(sc)), "dates": ";".join(str(d.date()) for d in sc[sc].index)}
    pd.DataFrame(snow).T.to_csv(os.path.join(args.out, "snow_candidates.csv"))

    meters = {pv: meter[pv].reindex(all_series[pv][("C", "era5")].index) for pv in pvs}
    fig_overview(mt, args.out)
    fig_diurnal(all_series, meters, caps, args.out)
    fig_monthly(all_series, meters, args.out)
    fig_kc_hist(all_series, meters, args.out)
    fig_clipping(all_series, meters, caps, args.out)
    fig_ramps(all_series, meters, caps, args.out)
    asym = fig_asymmetry(all_series, meters, caps, parks, args.out)
    asym.to_csv(os.path.join(args.out, "asymmetry.csv"), index=False)
    fig_residual_sun(all_series, meters, caps, args.out)
    fig_anglefit(all_extras, parks, args.out)
    for pv in pvs[:2] + [pvs[-1]]:
        fig_example_weeks(all_series, meters, caps, pv, args.out)
    fig_station_distance(mt, info, args.out)
    fig_imputed(mt, args.out).to_csv(os.path.join(args.out, "imputed_hours.csv"))
    log("done")


if __name__ == "__main__":
    main()
