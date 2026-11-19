#!/usr/bin/env python3
"""HANDOFF asks 9-11 — regenerate the manuscript figures.

Same export names in figs/round2/summary/ so the paper side can pull-replace:
  dumbbells_headline.png, kriging_example_week.png, bias_vs_age.png,
  morris_mu_star.png, sobol_indices.png, acf_M5.png, bars_r2.png,
  ladder_spaghetti.png, effect_matrix.png
plus (ask 11) power_curves/pc_<Pseudonym>.png and error_distribution.png.

Rules (ask 9): paper pseudonyms only (no station IDs / real names), print
fonts, no debug titles (captions live in LaTeX), human-readable SA parameter
names, no overlapping labels.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import aging, evaluation, meterdata  # noqa: E402
from round2.paper_style import (PSEUDONYM, PSEUDONYM_SHORT, SA_NAME,  # noqa: E402
                                apply_print_style, parks_by_age, rung)

SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")
SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"
MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
M0_DIR = "/mnt/nas/renewables.ninja/Wind/data/real_parks"
WINDOW = ("2023-06-01", "2024-06-01")

HEADLINE = [
    ("Extrapolation (MOST)", "M5all", "M5all_PL"),
    ("Aging (Weibull)", "M5all", "M5all_noage"),
    ("Downscaling (QM)", "M5all", "M5all_noQM"),
    ("Measurement pathway", "K5", "M5all"),
]
# ask 19 clean ladder: M1-M3 without aging, fleet Weibull enters at M4
LADDER_M = ["M1noage", "M2noage", "M3noage", "M4all", "M5all"]
LADDER_K = ["K1", "K3", "K5"]


def scores(variant="clean"):
    s = pd.read_csv(os.path.join(SUM_DIR, "scores.csv"), dtype={"park_id": str})
    s = s[s["variant"] == variant].copy()
    s["abs_er_dev"] = (s["energy_ratio"] - 1.0).abs()
    return s.set_index(["experiment", "park_id"])


def paired(sc, a, b, metric):
    try:
        return pd.concat([sc.loc[a][metric].rename("a"),
                          sc.loc[b][metric].rename("b")], axis=1).dropna()
    except KeyError:
        return None


# ---------------------------------------------------------------- ask 9 figs
DUMBBELL_PANELS = [
    # Extrapolation panel removed: that contrast is now Table 5.2/tab:pl_height
    # (per-farm, hub-height ordered) rather than a fleet dumbbell. Remaining
    # three panels use M6 (M5all_PL) as the base chain throughout, so all three
    # isolate their component on the same, best-performing configuration.
    # "M5all_PL_noWake" is M6 with the precomputed wake factor divided back
    # out (exact reconstruction, no rerun; wake factors are stored per
    # park/k, independent of the shear method).
    ("Aging (Weibull)", "M5all_PL", "M5all_PL_noage"),
    ("Bias correction (QM)", "M5all_PL", "M5all_PL_noQM"),
    ("Wakes (NOJ)", "M5all_PL", "M5all_PL_noWake"),
]


def fig_dumbbells(sc):
    """Ask 13b (revised): 1x3 layout, all three panels on the M6 base
    chain; green = component improves the park, red = degrades; grey dot =
    without-component baseline. QM panel encodes gate branches by marker
    shape (A circle, B square; C = baseline dot only)."""
    branches = pd.read_csv(os.path.join(REPO, "data", "round2",
                                        "branch_assignment.csv"),
                           dtype={"location_id": str})
    br = branches[branches["kind"] == "park"].set_index(
        branches.loc[branches["kind"] == "park", "location_id"]
        .str.replace("park_", "", regex=False))["branch"]
    order = [p for p, _, _ in parks_by_age()]
    # figsize calibrated against the measured printed width (~6.8in at
    # full \\linewidth in a figure*): fontsizes below give ~13-15pt effective
    # on the page rather than the ~6pt that a naive figsize=(30,11) produced.
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    for ax, (label, a, b) in zip(axes, DUMBBELL_PANELS):
        is_qm = "QM" in label or "Downscaling" in label
        pr = paired(sc, a, b, "abs_er_dev")
        for yi, pid in enumerate(order):
            if pr is None or pid not in pr.index:
                continue
            va, vb = pr.loc[pid, "a"], pr.loc[pid, "b"]
            branch = br.get(pid, "C")
            if is_qm and branch == "C":
                ax.scatter([vb], [yi], color="lightgray", s=90, zorder=3)
                continue
            color = "tab:green" if va < vb else ("tab:red" if va > vb
                                                 else "tab:gray")
            marker = {"A": "o", "B": "s"}.get(branch, "o") if is_qm else "o"
            ax.plot([vb, va], [yi, yi], color=color, lw=2.6, alpha=0.8)
            ax.scatter([vb], [yi], color="lightgray", s=90, zorder=3)
            ax.scatter([va], [yi], color=color, s=115, zorder=4, marker=marker)
        # final gate: no park uses branch B -> no marker legend needed
        ax.set_title(f"{label}\n({rung(b)} $\\rightarrow$ {rung(a)})", fontsize=18)
        ax.grid(alpha=0.3, axis="x")
        ax.set_xlabel("|ER $-$ 1|", fontsize=18)
        ax.set_xlim(0, 0.5)
        ax.set_xticks(np.arange(0, 0.51, 0.1))
        ax.tick_params(axis="x", labelsize=18)
    axes[0].set_yticks(range(len(order)), [PSEUDONYM_SHORT[p] for p in order],
                       fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dumbbells_headline.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_ladder_spaghetti(sc):
    metrics = [("r2", "R²"), ("abs_er_dev", "|ER − 1|"), ("wasserstein", "W₁")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (metric, name) in zip(axes, metrics):
        for rungs, color, lbl in ((LADDER_M, "tab:blue", "ERA5 pathway"),
                                  (LADDER_K, "tab:orange", "Measurement pathway")):
            xs = (np.arange(len(rungs)) if rungs is LADDER_M
                  else np.arange(len(LADDER_M), len(LADDER_M) + len(rungs)))
            ys_all = []
            for pid in PSEUDONYM:
                ys = [sc.loc[(r, pid), metric] if (r, pid) in sc.index
                      else np.nan for r in rungs]
                ax.plot(xs, ys, color=color, alpha=0.18, lw=0.8)
                ys_all.append(ys)
            ax.plot(xs, np.nanmedian(np.array(ys_all, float), axis=0),
                    color=color, lw=2.6, marker="o", label=lbl)
        ax.set_xticks(range(len(LADDER_M) + len(LADDER_K)),
                      [rung(r) for r in LADDER_M + LADDER_K], rotation=45,
                      fontsize=16)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ladder_spaghetti.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_effect_matrix(sc):
    rows = HEADLINE + [("MOST incremental", "M3noage", "M2noage"),
                       ("QM incremental", "M2noage", "M1noage")]
    order = [p for p, _, _ in parks_by_age()]
    mat = np.full((len(rows), len(order)), np.nan)
    for ri, (label, a, b) in enumerate(rows):
        pr = paired(sc, a, b, "abs_er_dev")
        if pr is None:
            continue
        for ci, pid in enumerate(order):
            if pid in pr.index:
                mat[ri, ci] = pr.loc[pid, "a"] - pr.loc[pid, "b"]
    fig, ax = plt.subplots(figsize=(20, 6.8))
    vmax = np.nanmax(np.abs(mat)) or 0.1
    im = ax.pcolormesh(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(order)) + 0.5,
                  [PSEUDONYM[p] for p in order], rotation=40, ha="right",
                  fontsize=15)
    ax.set_yticks(np.arange(len(rows)) + 0.5,
                  [f"{lbl} ({rung(b)}$\\rightarrow${rung(a)})"
                   for lbl, a, b in rows], fontsize=16)
    fig.colorbar(im, label=r"$\Delta$ |ER $-$ 1| (blue = improvement)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "effect_matrix.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_kriging_week():
    pid, window = "04745", ("2024-02-05", "2024-02-12")
    fig, ax = plt.subplots(figsize=(14, 5.4))
    meas = meterdata.load_park_power(pid, window) / 1e6
    ax.plot(meas.index, meas.values, "k-", lw=1.8, label="measured")
    for exp, color in (("M5all", "tab:blue"), ("K5", "tab:orange")):
        path = os.path.join(SYNTH_BASE, exp, f"synth_{pid}.csv")
        s = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
            .loc[window[0]:window[1], "power_park"] / 1e6
        ax.plot(s.index, s.values, color=color, lw=1.3, alpha=0.9,
                label=rung(exp))
    ax.set_ylabel("Park power [MW]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kriging_example_week.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_bias_vs_age():
    eff = pd.read_csv(os.path.join(REPO, "results", "round2",
                                   "wp4_fleet_efficiency.csv"),
                      dtype={"park_id": str})
    comp = pd.read_csv(os.path.join(REPO, "results", "round2",
                                    "wp4_curve_comparison.csv"))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(eff["age"], eff["efficiency"], zorder=3, s=42,
               label="parks (efficiency)")
    # manual label offsets against overlaps (points cluster at 6-10 yr)
    OFF = {"Hesse South": (-8, 8, "right"), "Saxony-Anhalt": (8, 8, "left"),
           "Thuringia": (8, -4, "left"), "Bavaria": (-8, 8, "right"),
           "Palatinate": (8, -16, "left"),
           "Mecklenburg-Western Pomerania": (8, 4, "left"),
           "Hesse North": (8, 6, "left"),
           "North Rhine-Westphalia North": (-10, 6, "right"),
           "North Rhine-Westphalia West": (-10, -18, "right"),
           "Lower Saxony": (-10, 16, "right"),
           "Schleswig-Holstein East": (-10, 2, "right"),
           "Schleswig-Holstein West": (-10, 2, "right")}
    for _, r in eff.iterrows():
        name = PSEUDONYM.get(r["park_id"], "")
        dx, dy, ha = OFF.get(name, (8, 5, "left"))
        ax.annotate(name, (r["age"], r["efficiency"]), fontsize=14,
                    textcoords="offset points", xytext=(dx, dy), ha=ha)
    a_grid = np.linspace(0, eff["age"].max() + 2, 100)
    # ask 22: EEG-step variant left the manuscript — two schedule curves only
    for name, key, fn, style in (("const ADR", "const_adr", aging.DF_const, "--"),
                                 ("Weibull", "weibull", aging.DF_weibull, "-")):
        c = comp.loc[comp["model"] == key, "level_offset"]
        c = float(c.iloc[0]) if len(c) else 0.0
        ax.plot(a_grid, np.exp(np.log(fn(a_grid)) + c), style, label=name)
    ax.set_xlabel("Park age [yr]")
    ax.set_ylabel(r"Efficiency $\sum P_{meas}\,/\,\sum P_{synth}$ (no aging)")
    ax.legend(loc="upper right", fontsize=16)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bias_vs_age.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_morris_sobol():
    """Ask 20a / print-size rework: Morris as a standalone single-column
    figure. Single-column figures shrink ~3x in print, so fonts are sized for
    that (like the placebo fig): the top-5 parameters (the ones quoted in
    Sec 5.8) are annotated directly, the remaining four are one cluster note,
    no legend."""
    mr = pd.read_csv(os.path.join(REPO, "results", "round2",
                                  "morris_ranking.csv"))
    mr = mr.sort_values("mu_star", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.scatter(mr["mu_star"], mr["sigma"], s=340, color="tab:blue",
               edgecolors="k", linewidths=0.9, zorder=3)
    # per-parameter label placement: (dx pt, dy pt, ha, va)
    offs = {
        "wind_level_factor": (0, -24, "center", "top"),
        "correction_mode": (0, -24, "center", "top"),
        "aging_kappa": (12, 12, "left", "bottom"),
        "power_curve_scale": (14, 0, "left", "center"),
        "aging_lambda": (0, -24, "center", "top"),
    }
    for _, r in mr.iterrows():
        if r["parameter"] not in offs:
            continue
        dx, dy, ha, va = offs[r["parameter"]]
        ax.annotate(SA_NAME.get(r["parameter"], r["parameter"]),
                    (r["mu_star"], r["sigma"]), fontsize=24,
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va)
    rest = mr[~mr["parameter"].isin(offs)]
    ax.annotate(f"{len(rest)} remaining parameters",
                (rest["mu_star"].max(), rest["sigma"].min()), fontsize=22,
                color="dimgray", style="italic",
                textcoords="offset points", xytext=(16, -8),
                ha="left", va="center")
    ax.set_xlim(-0.002, mr["mu_star"].max() * 1.28)
    ax.set_ylim(bottom=-0.001)
    # caption carries the full axis meaning; the long form overflows the top
    ax.set_xlabel(r"$\mu^*$ (mean |elementary effect|)", fontsize=28)
    ax.set_ylabel(r"$\sigma$ (interactions)", fontsize=28)
    ax.tick_params(labelsize=24)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "morris_mu_star.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)

    sb = pd.read_csv(os.path.join(REPO, "results", "round2",
                                  "sobol_indices.csv"))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(sb))
    ax.bar(x - 0.2, sb["S1"], 0.4, yerr=sb["S1_conf"], label="$S_1$ (first order)")
    ax.bar(x + 0.2, sb["ST"], 0.4, yerr=sb["ST_conf"], label="$S_T$ (total)")
    ax.set_xticks(x, [SA_NAME.get(p, p) for p in sb["parameter"]], rotation=15,
                  ha="right", fontsize=16)
    ax.set_ylabel("Sobol index")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sobol_indices.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_acf_m5():
    # switched from M5all (S3, alt. chain) to M5all_PL (M4, deployed) for
    # the renumbering; ACF is not expected to depend on the extrapolation
    # law choice (paper text, Sec. 5.3), so this reuses the same data logic.
    from statsmodels.tsa.stattools import acf as sm_acf
    fig, ax = plt.subplots(figsize=(10, 6.5))
    acc_m, acc_s = [], []
    for pid in PSEUDONYM:
        path = os.path.join(SYNTH_BASE, "M5all_PL", f"synth_{pid}.csv")
        synth = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
            .loc[WINDOW[0]:WINDOW[1], "power_park"]
        meas = meterdata.load_park_power(pid, WINDOW)
        both = pd.concat([meas.rename("m"), synth.rename("s")], axis=1).dropna()
        if len(both) < 1000:
            continue
        acc_m.append(sm_acf(both["m"], nlags=48))
        acc_s.append(sm_acf(both["s"], nlags=48))
    lags = np.arange(49)
    # ask 18b: IQR band (25th-75th percentile across the 13 parks) instead of
    # per-park spaghetti; IQR chosen over min-max so a single worst park does
    # not dominate the band
    am, as_ = np.array(acc_m), np.array(acc_s)
    ax.fill_between(lags, np.percentile(am, 25, axis=0),
                    np.percentile(am, 75, axis=0), color="gray", alpha=0.35,
                    label="measured (IQR)")
    ax.fill_between(lags, np.percentile(as_, 25, axis=0),
                    np.percentile(as_, 75, axis=0), color="tab:red",
                    alpha=0.25, label="synthetic M4 (IQR)")
    ax.plot(lags, am.mean(axis=0), "k-", lw=3.2, label="measured (mean)")
    ax.plot(lags, as_.mean(axis=0), color="tab:red", lw=3.2,
            label="synthetic M4 (mean)")
    # print-size fonts: single-column figure shrinks ~3x, match the placebo fig
    ax.set_xlabel("Lag [h]", fontsize=28)
    ax.set_ylabel("ACF of park power", fontsize=28)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=22)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "acf_M5.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- ask 10 bars
def fig_bars_round1_style(sc):
    # ask 32: K5 (measurement pathway) removed from the manuscript
    rungs = ["M0", "M1noage", "M2noage", "M3noage", "M4all", "M5all", "M5all_PL"]
    parks = parks_by_age()
    fig, axes = plt.subplots(5, 3, figsize=(19, 23), sharey=True)
    for ax, (pid, name, age) in zip(axes.flat, parks):
        vals = [sc.loc[(r, pid), "r2"] if (r, pid) in sc.index else np.nan
                for r in rungs]
        xs = np.arange(len(rungs))
        ax.bar(xs, [0 if (np.isnan(v) or v < 0) else v for v in vals],
               color="seagreen", width=0.65)
        for x, v in zip(xs, vals):
            if np.isnan(v):
                ax.text(x, 0.04, "n/a", ha="center", fontsize=14,
                        color="dimgray")
            elif v < 0:
                ax.text(x, 0.04, "<0", ha="center", fontsize=14,
                        color="firebrick")
        ax.axhline(1.0, color="red", ls="--", lw=1.1)
        ax.set_xticks(xs, [rung(r) for r in rungs], fontsize=15)
        ax.set_title(f"{name} (Age: {age:.0f} years)", fontsize=17)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes.flat[len(parks):]:
        ax.axis("off")
    for row in range(5):
        axes[row, 0].set_ylabel("R²")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(os.path.join(FIG_DIR, "bars_r2.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- ask 11 figs
def _park_ideal_curve(pid):
    import yaml
    from round2.chain import _power_curves
    with open(os.path.join(REPO, "configs", "real_wind_parks_era5",
                           f"config_{pid}.yaml")) as f:
        p = yaml.safe_load(f)["params"]
    specs = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_specs.csv"),
                        sep=";").drop_duplicates(subset="Turbine").set_index("Turbine")
    cut_outs = [float(specs.loc[t, "Abschaltgeschwindigkeit"]) for t in p["turbines"]]
    curves = _power_curves(p["turbines"], cut_outs)
    grid = np.arange(0, 30.01, 0.05)
    total = np.zeros_like(grid)
    for t in p["turbines"]:
        ticks, watts = curves[t]
        total += np.interp(grid, ticks, watts, right=0.0)
    return grid, total / 1e6


def _load_rn(pid):
    """Renewables.ninja per-park power (MW) + hub wind (m/s), or None."""
    p = os.path.join(M0_DIR, f"{pid}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.loc[WINDOW[0]:WINDOW[1]]
    pcols = [c for c in df.columns if c.startswith("power_t")]
    wcols = [c for c in df.columns if c.startswith("wind_speed_t")]
    return pd.DataFrame({"synth": df[pcols].sum(axis=1) / 1e3,
                         "v": df[wcols].mean(axis=1)})


def _mask_for(exp, pid, index):
    p = os.path.join(MASK_BASE, exp, f"mask_{pid}.parquet")
    if not os.path.exists(p):
        return pd.Series(True, index=index)
    keep = pd.read_parquet(p)["keep"]
    return keep.reindex(index).fillna(False).astype(bool)


def _pc_panel(ax, meas_v, meas_p, keep, synth_v, synth_p, grid, ideal,
              rated, xmax):
    """One power-curve panel (ask 15: stronger markers)."""
    ax.scatter(meas_v[keep], meas_p[keep], s=14, alpha=0.5,
               color="tab:blue", label="measured", edgecolors="none")
    ax.scatter(meas_v[~keep], meas_p[~keep], s=22, alpha=0.6,
               color="tab:red", marker="x", lw=1.2,
               label="measured (outlier)")
    ax.scatter(synth_v, synth_p, s=12, alpha=0.35, color="tab:orange",
               label="synthetic", edgecolors="none")
    ax.plot(grid, ideal, "k-", lw=2.0, label="aggregated power curve")
    ax.set_xlabel("Extrapolated wind speed at hub [m/s]")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02 * rated, rated * 1.15)
    ax.grid(alpha=0.3)


def fig_power_curves_and_errors(exp="M5all", panel_label="Framework (M5)",
                                subdir="power_curves",
                                err_name="error_distribution.png"):
    out_dir = os.path.join(FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    err_m5, err_ninja = [], []
    for pid, name in PSEUDONYM.items():
        path = os.path.join(SYNTH_BASE, exp, f"synth_{pid}.csv")
        df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
            .loc[WINDOW[0]:WINDOW[1]]
        wcols = [c for c in df.columns if c.startswith("wind_speed_t")]
        v_hub = df[wcols].mean(axis=1)
        synth = df["power_park"] / 1e6
        meas = meterdata.load_park_power(pid, WINDOW) / 1e6
        rated = meterdata.rated_power_w(pid) / 1e6
        both = pd.concat([meas.rename("meas"), synth.rename("synth"),
                          v_hub.rename("v")], axis=1).dropna()
        keep_m5 = _mask_for(exp, pid, both.index)
        grid, ideal = _park_ideal_curve(pid)
        rn = _load_rn(pid)
        xmax = min(30, max(both["v"].max(),
                           rn["v"].max() if rn is not None else 0) * 1.08)

        ncols = 2 if rn is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 6.5),
                                 sharey=True)
        axes = [axes] if ncols == 1 else list(axes)
        _pc_panel(axes[0], both["v"], both["meas"], keep_m5,
                  both["v"][keep_m5], both["synth"][keep_m5],
                  grid, ideal, rated, xmax)
        axes[0].set_ylabel("Park power [MW]")
        axes[0].set_title(panel_label)
        axes[0].legend(markerscale=2.2, framealpha=0.9, fontsize=15,
                       loc="upper left")
        if rn is not None:
            rboth = pd.concat([meas.rename("meas"), rn["synth"].rename("synth"),
                               rn["v"].rename("v")], axis=1).dropna()
            keep_rn = _mask_for("M0", pid, rboth.index)
            _pc_panel(axes[1], rboth["v"], rboth["meas"], keep_rn,
                      rboth["v"][keep_rn], rboth["synth"][keep_rn],
                      grid, ideal, rated, xmax)
            axes[1].set_title("Renewables.ninja")
        else:
            axes[0].annotate("not in RN catalog", xy=(0.97, 0.05),
                             xycoords="axes fraction", ha="right",
                             fontsize=15, color="dimgray")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pc_{name.replace(' ', '_')}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

        err_m5.append(((both.loc[keep_m5, "synth"] - both.loc[keep_m5, "meas"])
                       / rated).values)
        if rn is not None:
            nj_p = rn["synth"].reindex(both.index)
            ok = keep_m5 & nj_p.notna()
            err_ninja.append(((nj_p[ok] - both.loc[ok, "meas"]) / rated).values)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.6, 0.6, 81)
    ax.hist(np.concatenate(err_m5), bins=bins, density=True, alpha=0.55,
            color="tab:blue", label=rung(exp))
    ax.hist(np.concatenate(err_ninja), bins=bins, density=True, alpha=0.55,
            color="tab:orange", label="Renewables.ninja (9 parks)")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Normalized error  (P$_{synth}$ $-$ P$_{meas}$) / P$_{rated}$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, err_name), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def main():
    apply_print_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    sc = scores("clean")
    fig_dumbbells(sc)
    fig_ladder_spaghetti(sc)
    fig_effect_matrix(sc)
    fig_kriging_week()
    fig_bias_vs_age()
    fig_morris_sobol()
    fig_acf_m5()
    fig_bars_round1_style(sc)
    # renumbering: M4 (M5all_PL) is now the deployed configuration, replacing
    # the M5all/"M5" default used before the M6-flagship revision.
    fig_power_curves_and_errors(exp="M5all_PL", panel_label="Framework (M4)")
    fig_power_curves_and_errors(exp="K5", panel_label="Framework (K5)",
                                subdir="power_curves_K5",
                                err_name="error_distribution_K5.png")
    print("paper figures regenerated ->", FIG_DIR)


if __name__ == "__main__":
    main()
