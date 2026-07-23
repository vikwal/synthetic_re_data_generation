#!/usr/bin/env python3
"""Ask 31 — diagnostics + contrasts for the validity-gated profile.

Reads the M3_vg synth outputs (which store per-turbine w_most/blh_flag and
the met columns) plus scores.csv, and produces the four required diagnostics:
1. hourly attribution: |v_hub(MOST) − v_hub(PL)| stratified by blh-flag class
2. switch-discontinuity: |Δv_hub| at rule-switch transitions vs non-switch
3. ACF of synthetic power (M5_vg vs M5all) vs measured, clean window
4. sanity: PL-hour share vs blh_flag share
plus the paired contrasts (M3_vg vs M2/M3; M5_vg vs M5all/M5all_PL) and a
per-park R² dumbbell figure.

Outputs -> results/round2/ask31/
Run: synthre/bin/python scripts/round2/ask31_diagnostics.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata, stability  # noqa: E402
from round2.paper_style import PSEUDONYM, apply_print_style  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(REPO, "results", "round2", "ask31")
SYNTH = "/mnt/nvme2/synthetic/wind/round2"
WINDOW = ("2023-07-24", "2024-06-01")
PARKS = None  # filled in main from scores


def hub_cols(df):
    """(first-turbine hub col, w_most col, blh_flag col) of the synth frame."""
    hubs = [c for c in df.columns if c.startswith("wind_speed_t")]
    return hubs[0], "w_most_t1", "blh_flag_t1"


def load_synth(exp, pid):
    df = pd.read_csv(os.path.join(SYNTH, exp, f"synth_{pid}.csv"), sep=";",
                     index_col=0, parse_dates=True)
    return df.loc[WINDOW[0]:WINDOW[1]]


def diag_attribution_and_switches(parks):
    """Diagnostics 1, 2, 4 from the M3_vg outputs (per first turbine)."""
    rows_att, rows_sw = [], []
    for pid in parks:
        df = load_synth("M3_vg", pid)
        hub_col, w_col, flag_col = hub_cols(df)
        # reconstruct both profile variants at the same turbine
        m2 = load_synth("M2", pid)      # pure dyn-alpha PL, same corrections
        m3 = load_synth("M3", pid)      # pure MOST, same corrections
        v_pl = m2[hub_col].reindex(df.index)
        v_most = m3[hub_col].reindex(df.index)
        dv = (v_most - v_pl).abs()
        flagged = df[flag_col].astype(bool)

        rows_att.append({
            "park_id": pid, "pseudonym": PSEUDONYM.get(pid, pid),
            "share_flagged": float(flagged.mean()),
            "share_pl_hours": float((df[w_col] < 0.5).mean()),
            "dv_mean_flagged": float(dv[flagged].mean()),
            "dv_mean_unflagged": float(dv[~flagged].mean()),
            "dv_p95_flagged": float(dv[flagged].quantile(0.95)),
            "dv_p95_unflagged": float(dv[~flagged].quantile(0.95)),
            "dv_concentration": float(dv[flagged].mean()
                                      / max(dv[~flagged].mean(), 1e-9)),
        })

        # switch discontinuity on the GATED hub series
        v_hub = df[hub_col]
        gate = (df[w_col] >= 0.5).astype(int)
        step = v_hub.diff().abs()
        switch = gate.diff().abs() == 1
        rows_sw.append({
            "park_id": pid, "pseudonym": PSEUDONYM.get(pid, pid),
            "n_switches": int(switch.sum()),
            "step_mean_switch": float(step[switch].mean()),
            "step_p95_switch": float(step[switch].quantile(0.95)),
            "step_mean_nonswitch": float(step[~switch].mean()),
            "step_p95_nonswitch": float(step[~switch].quantile(0.95)),
        })
    return pd.DataFrame(rows_att), pd.DataFrame(rows_sw)


def diag_acf(parks):
    """Diagnostic 3: ACF(48) of park power, M5_vg / M5all vs measured."""
    rows = []
    for pid in parks:
        meas = meterdata.load_park_power(pid, WINDOW).dropna()
        if len(meas) < 500:
            continue
        a_meas = sm_acf(meas, nlags=48, missing="drop")
        r = {"park_id": pid, "pseudonym": PSEUDONYM.get(pid, pid)}
        for exp in ("M5all", "M5_vg", "M5_vg_blend"):
            df = load_synth(exp, pid)
            p = df["power_park"] if "power_park" in df.columns else \
                df.filter(regex=r"^power_t\d+$").sum(axis=1)
            p = p.reindex(meas.index).dropna()
            a = sm_acf(p, nlags=48, missing="drop")
            r[f"acf_maxdev_{exp}"] = float(np.max(np.abs(a - a_meas[:len(a)])))
            r[f"acf_lag24_{exp}"] = float(a[24])
        r["acf_lag24_meas"] = float(a_meas[24])
        rows.append(r)
    return pd.DataFrame(rows)


def contrasts_and_dumbbell(parks):
    sc = pd.read_csv(os.path.join(REPO, "results", "round2", "summary",
                                  "scores.csv"), dtype={"park_id": str})
    sc = sc[sc.variant == "clean"]
    piv_r2 = sc.pivot_table(index="park_id", columns="experiment", values="r2")
    piv_er = sc.pivot_table(index="park_id", columns="experiment",
                            values="energy_ratio")
    rows = []
    for label, a, b in [("M3_vg vs M2", "M3_vg", "M2"),
                        ("M3_vg vs M3", "M3_vg", "M3"),
                        ("M5_vg vs M5all", "M5_vg", "M5all"),
                        ("M5_vg vs M5all_PL", "M5_vg", "M5all_PL"),
                        ("M5_vg_blend vs M5all", "M5_vg_blend", "M5all")]:
        if a not in piv_r2.columns or b not in piv_r2.columns:
            continue
        for metric, piv, better in (("r2", piv_r2, "higher"),
                                    ("abs_er_dev", (piv_er - 1).abs(), "lower")):
            st = evaluation.compare_pathways(piv[a], piv[b])
            rows.append({"contrast": label, "metric": metric,
                         "median_a": float(piv[a].median()),
                         "median_b": float(piv[b].median()),
                         "median_diff": st["median_diff"],
                         "wilcoxon_p": st.get("wilcoxon_p"),
                         "rank_biserial": st.get("rank_biserial"),
                         "n_a_better": int(((piv[a] > piv[b]) if better == "higher"
                                            else (piv[a] < piv[b])).sum())})
    contrasts = pd.DataFrame(rows)

    # dumbbell: per-park clean R2 for M2 / M3 / M3_vg
    apply_print_style()
    order = piv_r2["M3_vg"].loc[parks].sort_values().index
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    y = np.arange(len(order))
    for yi, pid in zip(y, order):
        vals = [piv_r2.loc[pid, e] for e in ("M2", "M3", "M3_vg")]
        ax.plot([min(vals), max(vals)], [yi, yi], color="0.8", lw=2, zorder=1)
    for exp, color, marker in (("M2", "tab:gray", "o"), ("M3", "tab:blue", "s"),
                               ("M3_vg", "tab:red", "D")):
        ax.scatter(piv_r2.loc[order, exp], y, s=110, color=color, marker=marker,
                   label=exp, zorder=3)
    ax.set_yticks(y, [PSEUDONYM.get(p, p) for p in order], fontsize=15)
    ax.set_xlabel("R² (clean)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_vg_dumbbell_r2.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)
    return contrasts


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv(os.path.join(REPO, "results", "round2", "summary",
                                  "scores.csv"), dtype={"park_id": str})
    parks = sorted(sc[sc.experiment == "M3_vg"].park_id.unique())
    print(f"{len(parks)} parks")

    att, sw = diag_attribution_and_switches(parks)
    att.round(4).to_csv(os.path.join(OUT, "diag1_attribution.csv"), index=False)
    sw.round(4).to_csv(os.path.join(OUT, "diag2_switch_jumps.csv"), index=False)
    print("\n=== Diag 1: |v_MOST - v_PL| flagged vs unflagged ===")
    print(att.round(3).to_string(index=False))
    print("\n=== Diag 2: switch discontinuity ===")
    print(sw.round(3).to_string(index=False))

    acf_df = diag_acf(parks)
    acf_df.round(4).to_csv(os.path.join(OUT, "diag3_acf.csv"), index=False)
    print("\n=== Diag 3: ACF max deviation vs measured ===")
    print(acf_df.round(3).to_string(index=False))

    contrasts = contrasts_and_dumbbell(parks)
    contrasts.round(4).to_csv(os.path.join(OUT, "contrasts_vg.csv"), index=False)
    print("\n=== Contrasts (clean, paired) ===")
    print(contrasts.round(4).to_string(index=False))

    # ask-31 scores subset
    keep = ["M2", "M3", "M3_vg", "M5all", "M5all_PL", "M5_vg", "M5_vg_blend"]
    sc[sc.experiment.isin(keep)].to_csv(
        os.path.join(OUT, "scores_vg.csv"), index=False)
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
