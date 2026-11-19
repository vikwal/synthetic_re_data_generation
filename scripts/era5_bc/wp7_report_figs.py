#!/usr/bin/env python3
"""WP7c — report figures for the ERA5 bias-correction evaluation.

Per (eval split, window):
1. boxplots of per-station metrics (MAE/MBE/PCC/RMSE/R2/Skill) per model
2. observed vs predicted median wind speed scatter per model
3. two example weeks (median-MAE coastal + inland station)
4. skill (1 - RMSE_model/RMSE_ERA5) vs distance to coast

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp7_report_figs.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc.config import load_config  # noqa: E402
from era5_bc.data import load_processed  # noqa: E402

MODELS = ["uc_era5", "gboost", "lstm", "transformer"]
COLORS = {"uc_era5": "#888888", "gboost": "#4477aa",
          "lstm": "#ee6677", "transformer": "#228833"}


def fig_boxplots(ps, tag, figs_dir):
    metrics = ["mae", "mbe", "pcc", "rmse", "r2", "skill"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4))
    labels = [m for m in MODELS if m in ps.model.unique()]
    for ax, met in zip(axes, metrics):
        data = [ps.query("model == @m")[met].dropna() for m in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        showfliers=False)
        for patch, m in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)
        if met in ("mbe", "skill"):
            ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(met.upper())
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Per-station hourly metrics — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, f"boxplots_{tag}.png"), dpi=150)
    plt.close(fig)


def fig_median_scatter(ps, tag, figs_dir):
    models = [m for m in MODELS if m in ps.model.unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4),
                             sharex=True, sharey=True)
    for ax, m in zip(axes, models):
        sub = ps.query("model == @m")
        ax.scatter(sub["median_obs"], sub["median_pred"], s=18, alpha=0.7,
                   color=COLORS[m])
        lim = (0, max(ps["median_obs"].max(), ps["median_pred"].max()) * 1.05)
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set(title=m, xlabel="observed median [m/s]", xlim=lim, ylim=lim)
    axes[0].set_ylabel("predicted median [m/s]")
    fig.suptitle(f"Median wind speed — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, f"median_scatter_{tag}.png"), dpi=150)
    plt.close(fig)


def fig_example_weeks(cfg, ps, split, window, start, tag, figs_dir):
    """One coastal + one inland station (median MAE rank of the LSTM)."""
    res_dir = cfg["paths"]["results_dir"]
    sf = {}
    for m in ("lstm", "transformer"):
        p = os.path.join(res_dir, f"pred_{m}_{split}_{window}.parquet")
        if os.path.exists(p):
            sf[m] = (pd.read_parquet(p)
                     .set_index(["station_id", "timestamp"])["sf_pred"])
    gb = pd.read_csv(os.path.join(res_dir, f"gboost_predictions_{split}.csv"),
                     dtype={"station_id": str}).set_index("station_id")["sf_pred"]

    ref = ps.query("model == 'lstm'") if "lstm" in ps.model.unique() else ps
    picks = []
    for areas in (("coastal",), ("low_srl", "high_srl", "hilly")):
        sub = ref[ref["area"].isin(areas)].sort_values("mae")
        if len(sub):
            picks.append(sub.iloc[len(sub) // 2]["station_id"])
    if not picks:
        return

    week = pd.Timedelta(days=7)
    t0 = start + pd.Timedelta(days=60)
    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 3.5 * len(picks)),
                             squeeze=False)
    for ax, sid in zip(axes[:, 0], picks):
        frame = load_processed(cfg, sid).loc[t0:t0 + week]
        ax.plot(frame.index, frame["ws_obs"], "k-", lw=1.2, label="obs")
        ax.plot(frame.index, frame["ws10"], color=COLORS["uc_era5"], lw=1,
                label="UC-ERA5")
        ax.plot(frame.index, frame["ws10"] * gb.loc[sid],
                color=COLORS["gboost"], lw=1, label="TI-GBOOST")
        for m in sf:
            s = sf[m].loc[sid].reindex(frame.index)
            ax.plot(frame.index, frame["ws10"] * s, color=COLORS[m], lw=1,
                    label=f"TR-{m.upper()}")
        ax.set_title(f"station {sid} "
                     f"({ps.set_index('station_id')['area'].get(sid, '?')})")
        ax.set_ylabel("ws 10m [m/s]")
        ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, f"example_week_{tag}.png"), dpi=150)
    plt.close(fig)


def fig_skill_dist_coast(ps, tag, figs_dir, static_raw):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ("gboost", "lstm", "transformer"):
        sub = ps.query("model == @m").set_index("station_id")
        if len(sub) == 0:
            continue
        ax.scatter(static_raw.loc[sub.index, "dist_coast"],
                   sub["skill"] * 100, s=16, alpha=0.7, color=COLORS[m],
                   label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="distance to coast [km]",
           ylabel="skill vs UC-ERA5 [%]  (1 - RMSE_m/RMSE_ERA5)",
           title=f"BC skill vs coast distance — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, f"skill_dist_coast_{tag}.png"), dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    res_dir, figs_dir = cfg["paths"]["results_dir"], cfg["paths"]["figs_dir"]
    os.makedirs(figs_dir, exist_ok=True)
    static_raw = S.load_static_table(cfg)
    per = cfg["periods"]
    starts = {"spatial": per["train_start"], "temporal": per["eval_start"]}

    for split in ("val", "test"):
        for window, start in starts.items():
            tag = f"{split}_{window}"
            path = os.path.join(res_dir, f"per_station_{tag}.csv")
            if not os.path.exists(path):
                print(f"skip {tag}: {path} missing")
                continue
            ps = pd.read_csv(path, dtype={"station_id": str})
            fig_boxplots(ps, tag, figs_dir)
            fig_median_scatter(ps, tag, figs_dir)
            fig_example_weeks(cfg, ps, split, window, start, tag, figs_dir)
            fig_skill_dist_coast(ps, tag, figs_dir, static_raw)
            print(f"{tag}: 4 figures -> {figs_dir}")


if __name__ == "__main__":
    main()
