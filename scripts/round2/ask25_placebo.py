#!/usr/bin/env python3
"""HANDOFF ask 25 — aging placebo (P1) + age-permutation test (P2).

Confound under test: the aging on/off twin contrast could be a generic level
correction (any ~right-sized derating helps a high-biased chain), not
age-resolution. Two placebos, evaluated at M5 (primary: MOST + wakes) and M2
(robustness: power-law, no wakes), clean variant, canonical final-gate state:

  P1 — energy-matched uniform derating: DF_i(t) -> fleet scalar DF* =
       mean over the 13 parks of their time-mean DF_i (sensitivity: the
       measured-energy-weighted mean, 'P1w').
  P2 — age permutation: permute the 13 commissioning dates across parks,
       n_perm sampled draws (identity excluded); test statistic = fleet
       median improvement of the aged-vs-no-aged twin, one-sided.

Everything (true / P1 / no-aging reference / permutation draws) runs on the
WP6 fast chain (round2/chain.py mechanics, aging-injectable) so the
chain-vs-full-pipeline error cancels in the paired differences; the
fast-chain-vs-ladder deltas for the true assignment are exported once as a
calibration check. Masks: the existing per-model families (M5all / M2all) —
aging does not change the wind, so hours are identical by construction.

Outputs -> results/round2/ask25/:
  placebo_scores.csv           variant x park x {r2, energy_ratio, abs_er_dev,
                               wasserstein, bias_n} at M5 + M2
  wilcoxon_true_vs_p1.csv      true-vs-P1 rows, wilcoxon_key_comparisons format
  perm_null.csv                per-draw fleet-median statistics + true rows
  perm_null_hist.png           null histograms, true value marked (2x2 panels)
  placebo_dumbbells.png        per-park |ER-1| dumbbells true vs P1 vs no-aging
  calibration_vs_ladder.csv    fast chain vs ladder scores (true + noage twins)

Usage: ask25_placebo.py [--n-perm 1000] [--seed 25]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from round2 import aging as r2_aging  # noqa: E402
from round2 import correction as r2_correction  # noqa: E402
from round2 import stability as r2_stability  # noqa: E402
from round2 import evaluation, meterdata, parkinfo  # noqa: E402
from round2.chain import DEFAULT_THETA, ParkCache  # noqa: E402
from round2.paper_style import PSEUDONYM, apply_print_style, parks_by_age  # noqa: E402

MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
OUT_DIR = os.path.join(REPO, "results", "round2", "ask25")
LAM, KAP = 54.5, 2.0  # fleet Weibull, ladder canon

# model -> chain theta + mask family + ladder twin experiments
MODELS = {
    "M5": {"theta": {"correction": "height_consistent", "shear": "most",
                     "wake_enabled": True, "wake_k": 0.075},
           "mask_exp": "M5all", "ladder_aged": "M5all",
           "ladder_noage": "M5all_noage"},
    "M2": {"theta": {"correction": "height_consistent", "shear": "power_law",
                     "wake_enabled": False},
           "mask_exp": "M2all", "ladder_aged": "M2all",
           "ladder_noage": "M2noage"},
}


class AgingChain:
    """Aging-injectable replica of ParkCache.energy_ratio (round2/chain.py).

    Everything upstream of the degradation shift (correction, shear, density,
    curves, wakes) is frozen at construction; power(df_vec) applies only the
    power-curve wind-axis shift (1/DF)^(1/3) with release above the shifted
    rated wind speed — bit-identical mechanics to the validated fast chain.
    """

    def __init__(self, pc: ParkCache, theta: dict):
        th = {**DEFAULT_THETA, **theta}
        v10, v100 = pc.v10.copy(), pc.v100.copy()
        if th["correction"] != "off" and pc.q_target is not None:
            c = r2_correction.correction_factor(v10, pc.q_era5, pc.q_target)
            v10 = v10 * c
            if th["correction"] == "height_consistent":
                v100 = v100 * c
        self.n = len(pc.index)
        self.turbs = []
        for t, h in zip(pc.turbines, pc.hub_heights):
            if th["shear"] == "most" and pc.L is not None:
                v_hub = r2_stability.most_wind_profile(
                    v100, h, pc.z0 * th["z0_scale"], pc.L)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.log(v100 / v10) / np.log(10.0)
                alpha = np.clip(np.nan_to_num(alpha), 0.0, 0.4)
                v_hub = v10 * (h / 10.0) ** alpha
            v_hub = np.round(v_hub, 2)
            s = pc.specs[t]
            ticks, watts = pc.curves[t]
            watts = watts * th["power_curve_scale"]
            self.turbs.append({
                "v_hub": v_hub, "ticks": ticks, "watts": watts,
                "rated_power": watts.max(), "rated_ws": s["rated_ws"],
                "dead": (v_hub < s["cut_in"]) | (v_hub > s["cut_out"]),
                "rho": 1.225 if th["density"] == "static_1225" else pc.rho_hub[h],
            })
        self.wake = None
        if th["wake_enabled"] and pc.wake:
            k = float(np.clip(th["wake_k"], min(pc.wake), max(pc.wake)))
            assert k in pc.wake, f"wake k={k} not on the precomputed grid"
            self.wake = pc.wake[k]

    def power(self, df_vec) -> np.ndarray:
        df_vec = np.asarray(df_vec, dtype=float)
        p_park = np.zeros(self.n)
        for tb in self.turbs:
            v_hub = tb["v_hub"]
            dr = np.broadcast_to(df_vec, v_hub.shape).copy()
            rated_ws_vec = tb["rated_ws"] * (1.0 / dr) ** (1.0 / 3.0)
            dr[v_hub >= rated_ws_vec] = 1.0
            v_eff = v_hub * dr ** (1.0 / 3.0)
            p = np.interp(v_eff, tb["ticks"], tb["watts"]) * (tb["rho"] / 1.225)
            p = np.where(tb["dead"], 0.0, p)
            p = np.minimum(p, tb["rated_power"])
            p_park += p
        if self.wake is not None:
            p_park = p_park * self.wake
        return p_park


def ages_from_date(index: pd.DatetimeIndex, date: str) -> np.ndarray:
    return np.clip(np.asarray(
        (index - pd.to_datetime(date, utc=True)).days, dtype=float) / 365.25,
        0.0, None)


def clean_arrays(pc: ParkCache, mask_exp: str):
    """(bool ok over pc.index, measured[ok]) for the clean variant."""
    mask = pd.read_parquet(os.path.join(
        MASK_BASE, mask_exp, f"mask_{pc.park_id}.parquet"))["keep"]
    keep = mask.reindex(pc.index).fillna(False).values.astype(bool)
    ok = keep & ~np.isnan(pc.meas)
    return ok, pc.meas[ok]


def metrics(synth_ok: np.ndarray, meas_ok: np.ndarray, rated: float) -> dict:
    """Mirror of evaluation._metrics on pre-aligned arrays + bias_n."""
    ss_tot = float(((meas_ok - meas_ok.mean()) ** 2).sum())
    return {
        "r2": 1.0 - float(((meas_ok - synth_ok) ** 2).sum()) / ss_tot,
        "energy_ratio": float(synth_ok.sum() / meas_ok.sum()),
        "wasserstein": float(stats.wasserstein_distance(
            meas_ok / rated, synth_ok / rated)),
        "bias_n": float((synth_ok - meas_ok).mean() / rated),
        "n_hours": int(len(meas_ok)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    parks = sorted(meterdata.load_mapping()["park_id"])
    print(f"building park caches ({len(parks)} parks) ...", flush=True)
    pcs = {pid: ParkCache(pid) for pid in parks}
    rated = {pid: meterdata.rated_power_w(pid) for pid in parks}
    dates = {pid: parkinfo.commissioning_date(pid) for pid in parks}

    # DF vectors for every (park, commissioning date) pair: the permutation
    # test only ever reindexes into this 13x13 table
    df_tab = {pid: {d: r2_aging.DF_weibull(ages_from_date(pcs[pid].index, d),
                                           LAM, KAP)
                    for d in dates.values()} for pid in parks}
    df_true = {pid: df_tab[pid][dates[pid]] for pid in parks}

    # P1: energy-matched uniform derating (fleet mean of time-mean DF_i);
    # P1w sensitivity: measured-energy-weighted fleet mean
    tmean = np.array([df_true[pid].mean() for pid in parks])
    e_meas = np.array([np.nansum(pcs[pid].meas) for pid in parks])
    df_star = float(tmean.mean())
    df_star_w = float((tmean * e_meas).sum() / e_meas.sum())
    print(f"P1 uniform DF* = {df_star:.4f} | energy-weighted = {df_star_w:.4f}")

    chains, oks, meas_ok = {}, {}, {}
    for m, spec in MODELS.items():
        for pid in parks:
            chains[(m, pid)] = AgingChain(pcs[pid], spec["theta"])
            oks[(m, pid)], meas_ok[(m, pid)] = clean_arrays(
                pcs[pid], spec["mask_exp"])

    def score(m, pid, df_vec):
        ok = oks[(m, pid)]
        p = chains[(m, pid)].power(df_vec)[ok]
        return metrics(p, meas_ok[(m, pid)], rated[pid])

    def er_r2(m, pid, df_vec):
        ok = oks[(m, pid)]
        p = chains[(m, pid)].power(df_vec)[ok]
        mm = meas_ok[(m, pid)]
        r2 = 1.0 - float(((mm - p) ** 2).sum()) / float(((mm - mm.mean()) ** 2).sum())
        return float(p.sum() / mm.sum()), r2

    # ---- deliverable 1: placebo scores (full metrics) --------------------
    variants = {"noage": lambda pid: 1.0,
                "true": lambda pid: df_true[pid],
                "P1_uniform": lambda pid: df_star,
                "P1_uniform_eweighted": lambda pid: df_star_w}
    rows = []
    for m in MODELS:
        for var, df_of in variants.items():
            for pid in parks:
                rows.append({"model": m, "variant": var, "park_id": pid,
                             **score(m, pid, df_of(pid))})
    sc = pd.DataFrame(rows)
    sc["abs_er_dev"] = (sc["energy_ratio"] - 1.0).abs()
    sc.to_csv(os.path.join(OUT_DIR, "placebo_scores.csv"), index=False)

    # ---- calibration: fast chain (true / noage) vs ladder scores ---------
    ladder = pd.read_csv(os.path.join(REPO, "results", "round2", "summary",
                                      "scores.csv"), dtype={"park_id": str})
    ladder = ladder[ladder["variant"] == "clean"].set_index(
        ["experiment", "park_id"])
    cal = []
    for m, spec in MODELS.items():
        for var, exp in (("true", spec["ladder_aged"]),
                         ("noage", spec["ladder_noage"])):
            ch = sc[(sc.model == m) & (sc.variant == var)].set_index("park_id")
            for pid in parks:
                lr = ladder.loc[(exp, pid)]
                cal.append({"model": m, "variant": var, "experiment": exp,
                            "park_id": pid,
                            "er_chain": ch.loc[pid, "energy_ratio"],
                            "er_ladder": float(lr["energy_ratio"]),
                            "d_er": ch.loc[pid, "energy_ratio"] - float(lr["energy_ratio"]),
                            "r2_chain": ch.loc[pid, "r2"],
                            "r2_ladder": float(lr["r2"]),
                            "d_r2": ch.loc[pid, "r2"] - float(lr["r2"])})
    cal = pd.DataFrame(cal)
    cal.to_csv(os.path.join(OUT_DIR, "calibration_vs_ladder.csv"), index=False)
    print("calibration fast chain vs ladder (median |delta| per exp):")
    print(cal.groupby(["model", "variant"])[["d_er", "d_r2"]]
          .apply(lambda g: g.abs().median()).round(4).to_string())

    # ---- deliverable 3: Wilcoxon true vs P1 ------------------------------
    wrows = []
    for m in MODELS:
        t = sc[(sc.model == m) & (sc.variant == "true")].set_index("park_id")
        p1 = sc[(sc.model == m) & (sc.variant == "P1_uniform")].set_index("park_id")
        for metric, a, b in (("abs_er_dev", p1["abs_er_dev"], t["abs_er_dev"]),
                             ("r2", t["r2"], p1["r2"])):
            # orientation: positive median_diff = true assignment better
            res = evaluation.compare_pathways(a, b)
            wrows.append({
                "comparison": f"aging true vs P1-uniform ({m})",
                "a": "true_ages" if metric == "r2" else "P1_uniform",
                "b": "P1_uniform" if metric == "r2" else "true_ages",
                "metric": metric, "variant": "clean",
                "n_parks": res["n_parks"],
                "median_diff_a_minus_b": res["median_diff"],
                "wilcoxon_p_exact": res.get("wilcoxon_p", np.nan),
                "rank_biserial": res.get("rank_biserial", np.nan),
                "boot_ci_lo": res.get("boot_ci_lo", np.nan),
                "boot_ci_hi": res.get("boot_ci_hi", np.nan),
                "sign_test_p": res.get("sign_test_p", np.nan)})
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(os.path.join(OUT_DIR, "wilcoxon_true_vs_p1.csv"), index=False)
    print(wdf.round(4).to_string(index=False))

    # ---- deliverable 2: age-permutation null -----------------------------
    noage = {(m, pid): er_r2(m, pid, 1.0) for m in MODELS for pid in parks}

    def fleet_stats(m, date_of):
        """Fleet-median improvement of aged-vs-no-aged under an assignment."""
        d_er, d_r2 = [], []
        for pid in parks:
            er, r2 = er_r2(m, pid, df_tab[pid][date_of(pid)])
            er0, r20 = noage[(m, pid)]
            d_er.append(abs(er0 - 1.0) - abs(er - 1.0))
            d_r2.append(r2 - r20)
        return float(np.median(d_er)), float(np.median(d_r2))

    true_stat = {m: fleet_stats(m, lambda pid: dates[pid]) for m in MODELS}

    rng = np.random.default_rng(args.seed)
    date_list = [dates[pid] for pid in parks]
    prows = []
    for i in range(args.n_perm):
        perm = rng.permutation(len(parks))
        while np.array_equal(perm, np.arange(len(parks))):
            perm = rng.permutation(len(parks))
        assign = {pid: date_list[perm[j]] for j, pid in enumerate(parks)}
        for m in MODELS:
            s_er, s_r2 = fleet_stats(m, lambda pid: assign[pid])
            prows.append({"draw": i, "model": m,
                          "d_abs_er_dev_median": s_er, "d_r2_median": s_r2})
        if (i + 1) % 100 == 0:
            print(f"  perm {i + 1}/{args.n_perm}", flush=True)
    null = pd.DataFrame(prows)
    for m in MODELS:
        null.loc[len(null)] = {"draw": "true", "model": m,
                               "d_abs_er_dev_median": true_stat[m][0],
                               "d_r2_median": true_stat[m][1]}
    null.to_csv(os.path.join(OUT_DIR, "perm_null.csv"), index=False)

    pperm = {}
    for m in MODELS:
        nm = null[(null.model == m) & (null.draw != "true")]
        for col, k in (("d_abs_er_dev_median", "abs_er_dev"), ("d_r2_median", "r2")):
            pperm[(m, k)] = float(
                (1 + (nm[col] >= true_stat[m][0 if k == "abs_er_dev" else 1]).sum())
                / (len(nm) + 1))
        print(f"{m}: true d|ER-1| = {true_stat[m][0]:+.4f} "
              f"(p_perm {pperm[(m, 'abs_er_dev')]:.4f}) | "
              f"true dR2 = {true_stat[m][1]:+.4f} "
              f"(p_perm {pperm[(m, 'r2')]:.4f})")

    # ---- figures ----------------------------------------------------------
    apply_print_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for i, m in enumerate(MODELS):
        nm = null[(null.model == m) & (null.draw != "true")]
        for j, (col, lbl, key) in enumerate((
                ("d_abs_er_dev_median",
                 r"fleet median $\Delta|\mathrm{ER}-1|$ (no-aging $-$ aged)",
                 "abs_er_dev"),
                ("d_r2_median",
                 r"fleet median $\Delta R^2$ (aged $-$ no-aging)", "r2"))):
            ax = axes[i, j]
            ax.hist(nm[col].astype(float), bins=40, color="#9ecae1",
                    edgecolor="white")
            tv = true_stat[m][0 if key == "abs_er_dev" else 1]
            ax.axvline(tv, color="#d62728", lw=3)
            ax.text(0.97, 0.95,
                    f"{m}\ntrue = {tv:+.3f}\n$p_{{perm}}$ = "
                    f"{pperm[(m, key)]:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=17,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax.set_xlabel(lbl)
            if j == 0:
                ax.set_ylabel("permutation draws")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "perm_null_hist.png"), dpi=180)
    plt.close(fig)

    order = [(pid, name) for pid, name, _ in parks_by_age()]
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = sc[sc.model == m].set_index(["variant", "park_id"])["abs_er_dev"]
        ys = np.arange(len(order))
        for y, (pid, _) in zip(ys, order):
            ax.plot([sub.loc[("noage", pid)], sub.loc[("true", pid)]],
                    [y, y], color="lightgray", lw=2, zorder=1)
        ax.scatter([sub.loc[("noage", pid)] for pid, _ in order], ys,
                   s=140, color="#bbbbbb", label="no aging", zorder=2)
        ax.scatter([sub.loc[("P1_uniform", pid)] for pid, _ in order], ys,
                   s=140, color="#1f77b4", marker="s",
                   label="P1 uniform derating", zorder=3)
        ax.scatter([sub.loc[("true", pid)] for pid, _ in order], ys,
                   s=160, color="#d62728", marker="D", label="true ages",
                   zorder=4)
        ax.set_yticks(ys, [name for _, name in order])
        ax.set_xlabel(r"$|\mathrm{ER}-1|$ (clean)")
        ax.set_title(m)
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "placebo_dumbbells.png"), dpi=180)
    plt.close(fig)

    # ---- pre-registered read-out summary ----------------------------------
    med = sc.groupby(["model", "variant"])[["abs_er_dev", "r2"]].median()
    print("\nfleet medians (clean):")
    print(med.round(4).to_string())
    for m in MODELS:
        d_er = float(med.loc[(m, "P1_uniform"), "abs_er_dev"]
                     - med.loc[(m, "true"), "abs_er_dev"])
        d_r2 = float(med.loc[(m, "true"), "r2"]
                     - med.loc[(m, "P1_uniform"), "r2"])
        print(f"{m}: true vs P1 median advantage: |ER-1| {d_er:+.4f}, "
              f"R2 {d_r2:+.4f}")
    print(f"\ndone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
