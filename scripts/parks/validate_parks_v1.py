#!/usr/bin/env python3
"""parks_v1 validation: capacity factors, comparison with the 200-site
dataset, wake efficiencies, group aging, and the one overlap with the round-2
meter-data parks (MaStR Lokation SEL923305996505 = park 05426).

Writes figures + validation_summary.json to FL_Contribution/reports/figs_park_synthesis_v1/.
Usage: validate_parks_v1.py
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from parks import library, paths  # noqa: E402
from round2 import aging, evaluation, meterdata  # noqa: E402

OUT = os.path.join(paths.FL_DIR, "reports", "figs_park_synthesis_v1")
SITE_DIR = "/mnt/nvme2/synthetic/wind/site_v2_20260714"
SITE_WINDOW = ("2023-07-24", "2026-04-30 23:00")
TRIANEL = {"SEL923305996505": "05426"}
METER_WINDOW = ("2023-07-24", "2024-06-01")

# reference palette (light): surface, ink, categorical slots 1-3 (validated all-pairs)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
COHORTS = [("<=2004", S1), ("2005-14", S2), ("2015+", S3)]


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
        "lines.linewidth": 2.0})


def cohort(y: float) -> str:
    return "<=2004" if y <= 2004 else "2005-14" if y <= 2014 else "2015+"


def load():
    R = paths.RELEASE_DIR
    parks = pd.read_csv(os.path.join(R, "parks.csv"))
    groups = pd.read_csv(os.path.join(R, "wind_groups.csv"))
    w = groups.assign(w=groups.n_turbines * groups.rated_kw)
    wavg = lambda col: w.groupby("park_id").apply(lambda d: np.average(d[col], weights=d.w), include_groups=False)  # noqa: E731
    parks = parks.join(wavg("hub_height_m").rename("hub_w"), on="park_id")
    parks = parks.join(wavg("commissioning_year").rename("year_w"), on="park_id")
    a = w.assign(A=w.n_turbines * np.pi * (w.rotor_diameter_m / 2) ** 2).groupby("park_id")[["A", "w"]].sum()
    parks = parks.join((a.w * 1000 / a.A).rename("specific_power_w_m2"), on="park_id")
    v100 = {lk: pd.read_parquet(os.path.join(R, f"synth_{lk}.parquet"), columns=["wind_speed_100m"])
            .wind_speed_100m.mean() for lk in parks.park_id}
    parks["v100_mean"] = parks.park_id.map(v100)
    return parks, groups


def group_cf(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lk, gg in groups.groupby("park_id"):
        r = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{lk}.parquet")).loc[SITE_WINDOW[0]:SITE_WINDOW[1]]
        for x in gg.itertuples():
            rows.append({"park_id": lk, "group_id": x.group_id, "type": x.turbine_type,
                         "hub": x.hub_height_m, "year": x.commissioning_year,
                         "cf": r[f"power_{x.group_id}"].mean() / (x.n_turbines * x.rated_kw * 1000)})
    return pd.DataFrame(rows)


def site_cf() -> pd.DataFrame:
    types = ["Enercon E-70 E4 2.300", "Enercon E-82 E2 2.000", "Enercon E-115 2.500",
             "Vestas V90", "Vestas V112-3.45", "Vestas V80-1.8"]
    cmax = library.curve_max_kw(types)
    rows = []
    for f in sorted(glob.glob(os.path.join(SITE_DIR, "synth_*.parquet"))):
        r = pd.read_parquet(f, columns=[f"power_t{i}" for i in range(1, 7)])
        for i, t in enumerate(types, start=1):
            rows.append({"site": os.path.basename(f)[6:11], "type": t, "cf": r[f"power_t{i}"].mean() / (cmax[t] * 1000)})
    return pd.DataFrame(rows)


def fig_cf_latitude(parks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, col in COHORTS:
        d = parks[parks.year_w.map(cohort) == name]
        ax.scatter(d.latitude, d.cf * 100, s=40, color=col, edgecolor=SURFACE, linewidth=1.5,
                   label=f"{name} (n={len(d)})", zorder=3)
    ax.set_xlabel("latitude [deg N]")
    ax.set_ylabel("capacity factor (waked) [%]")
    ax.set_title("Park capacity factor vs latitude, by capacity-weighted commissioning cohort",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left", title="cohort", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cf_vs_latitude.png"), dpi=160)
    plt.close(fig)


def fig_cf_distribution(gcf: pd.DataFrame, scf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d, col, lab in ((gcf.cf, S1, f"parks_v1 turbine groups (n={len(gcf)})"),
                        (scf.cf, S2, f"200-site dataset, 6 turbines per site (n={len(scf)})")):
        x = np.sort(d.values) * 100
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, label=lab)
    ax.set_xlabel("capacity factor [%] (2023-07-24 to 2026-04-30)")
    ax.set_ylabel("cumulative share")
    ax.set_title("Capacity-factor distribution: parks_v1 groups vs site_v2 turbines", loc="left", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cf_distribution_vs_sites.png"), dpi=160)
    plt.close(fig)


def fig_wake(parks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(85, 97, color=GRID, alpha=0.6, zorder=0, label="typical range 85-97 %")
    ax.scatter(parks.n_turbines, parks.wake_efficiency * 100, s=40, color=S1, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="park")
    ax.set_xscale("log")
    ax.set_xlabel("turbines in park (log)")
    ax.set_ylabel("wake efficiency [%] (energy-weighted)")
    ax.set_title("NOJ wake efficiency (k = 0.075) vs park size", loc="left", fontsize=10)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wake_efficiency.png"), dpi=160)
    plt.close(fig)


def fig_aging(groups: pd.DataFrame) -> None:
    age = (pd.Timestamp("2025-02-15") - pd.to_datetime(groups.commissioning_date)).dt.days / 365.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    a = np.linspace(0, 33, 200)
    ax.plot(a, aging.DF_weibull(a) * 100, color=INK2, linewidth=1.5, label="Weibull DF (lambda 54.5, kappa 2)")
    ax.scatter(age, groups.aging_factor_mean * 100, s=30, color=S1, edgecolor=SURFACE, linewidth=1.2,
               zorder=3, label=f"turbine group, period mean (n={len(groups)})")
    ax.set_xlabel("group age at mid-period (2025-02) [years]")
    ax.set_ylabel("retained load factor [%]")
    ax.set_title("Aging per turbine group", loc="left", fontsize=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aging_groups.png"), dpi=160)
    plt.close(fig)


def trianel_check() -> dict:
    out = {}
    for lk, pid in TRIANEL.items():
        rel = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{lk}.parquet"))
        meas = meterdata.load_park_power(pid, METER_WINDOW)
        synth = rel.power_park.loc[METER_WINDOW[0]:METER_WINDOW[1]]
        m = evaluation.evaluate(meas, synth, p_rated=meterdata.rated_power_w(pid))
        m = {k: v for k, v in m.items() if not k.startswith("acf")}
        out[lk] = {"round2_park": pid, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in m.items()}}
        both = pd.concat([meas.rename("measured"), synth.rename("synthetic")], axis=1).dropna()
        daily = both.resample("D").mean() / 1e3
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(daily.index, daily.measured, color=S2, linewidth=1.4, label="measured (meter data)")
        ax.plot(daily.index, daily.synthetic, color=S1, linewidth=1.4, label="parks_v1 synthetic")
        ax.set_ylabel("daily mean power [kW]")
        ax.set_title(f"{lk} (= round-2 park {pid}, 1x V112-3.3, 140 m): daily means", loc="left", fontsize=10)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"meter_check_{lk}.png"), dpi=160)
        plt.close(fig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    style()
    parks, groups = load()
    gcf, scf = group_cf(groups), site_cf()
    fig_cf_latitude(parks)
    fig_cf_distribution(gcf, scf)
    fig_wake(parks)
    fig_aging(groups)
    X = np.c_[np.ones(len(parks)), parks.latitude, np.log(parks.hub_w), parks.specific_power_w_m2 / 100]
    beta, *_ = np.linalg.lstsq(X, parks.cf, rcond=None)
    resid = parks.cf - X @ beta
    multi = parks[parks.n_turbines > 1]
    summary = {
        "cf": parks.cf.describe().round(4).to_dict(),
        "cf_capacity_weighted": float(np.average(parks.cf, weights=parks.capacity_kw)),
        "cf_bins": {"<10%": int((parks.cf < .10).sum()), "10-15%": int(((parks.cf >= .10) & (parks.cf < .15)).sum()),
                    "15-35%": int(((parks.cf >= .15) & (parks.cf <= .35)).sum()), ">35%": int((parks.cf > .35).sum())},
        "corr_v100_latitude": float(parks[["v100_mean", "latitude"]].corr().iloc[0, 1]),
        "ols_cf_lat_lnhub_spec": {"coef": dict(zip(["const", "lat", "ln_hub", "spec_per_100W"], beta.round(4))),
                                  "r2": float(1 - (resid ** 2).sum() / ((parks.cf - parks.cf.mean()) ** 2).sum())},
        "group_cf": gcf.cf.describe().round(4).to_dict(), "site_cf": scf.cf.describe().round(4).to_dict(),
        "wake_eff_multi": multi.wake_efficiency.describe().round(4).to_dict(),
        "wake_outside_85_97_multi": multi[(multi.wake_efficiency < .85) | (multi.wake_efficiency > .97)]
        [["park_id", "n_turbines", "wake_efficiency"]].round(4).to_dict("records"),
        "aging_factor_mean": groups.aging_factor_mean.describe().round(4).to_dict(),
        "trianel": trianel_check(),
    }
    band = pd.cut(parks.latitude, [47, 50, 51.5, 53, 56], labels=["<50", "50-51.5", "51.5-53", ">53"])
    summary["by_latitude_band"] = parks.groupby(band, observed=True).agg(
        n=("cf", "size"), cf=("cf", "median"), v100=("v100_mean", "median"), hub=("hub_w", "median"),
        spec=("specific_power_w_m2", "median"), year=("year_w", "median")).round(3).reset_index().to_dict("records")
    parks.to_csv(os.path.join(OUT, "parks_validation.csv"), index=False)
    gcf.to_csv(os.path.join(OUT, "group_cf.csv"), index=False)
    with open(os.path.join(OUT, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
