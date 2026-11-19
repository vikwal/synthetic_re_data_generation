#!/usr/bin/env python3
"""Round 3 (reviewer 4d): what the operational-data cleaning removes.

For the deployed chain (M5all_PL) and every park, the kept/removed mask of
score_ladder.py is split into
  stage1_zero      measured 0 while hub wind >= smallest cut-in (outage/curtailment)
  under_at_wind    DBSCAN-removed, measured < 50 % of synthetic while the
                   synthetic park power exceeds 30 % of rated (derating, partial
                   availability, curtailment)
  over_vs_model    DBSCAN-removed, measured > 150 % of synthetic (model input
                   error rather than an operational state)
  other            remaining DBSCAN-removed hours
plus the share of removed hours falling into the documented night-curtailment
window (22-06 CET) for parks that have night curtailment in the master data.
Also tabulates the ladder medians per filter variant (raw / clean /
clean_exkl_curt) from results/quantitative/scores.csv (reviewer 4d-3).

Run on l2: synthre/bin/python scripts/round3/filter_classification.py [--exp M5all_PL]
Output: results/round3/filter_classes_<exp>.csv, filter_variant_medians.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import meterdata  # noqa: E402

MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"
OUT = os.path.join(REPO, "results", "round3")
WINDOW = ("2023-06-01", "2024-06-01")
NIGHT_PARKS = {"50082792290", "50446515501", "50646350997"}  # masterdata: Nachtabregelung True


def min_cut_in(exp, park_id):
    p = os.path.join(SYNTH_BASE, exp, "turbine_parameter.csv")
    if not os.path.exists(p):
        return 3.0
    t = pd.read_csv(p, sep=None, engine="python", dtype={"park_id": str})
    col = "cut_in" if "cut_in" in t.columns else None
    sub = t[t.park_id.astype(str) == park_id] if "park_id" in t.columns else t
    return float(sub[col].min()) if col and not sub.empty else 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="M5all_PL")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mapping = meterdata.load_mapping()
    rows = []
    for mask_file in sorted(os.listdir(os.path.join(MASK_BASE, args.exp))):
        park_id = mask_file[len("mask_"):-len(".parquet")]
        keep = pd.read_parquet(os.path.join(MASK_BASE, args.exp, mask_file))["keep"]
        synth = pd.read_csv(os.path.join(SYNTH_BASE, args.exp, f"synth_{park_id}.csv"),
                            sep=";", index_col=0, parse_dates=True)
        wcols = [c for c in synth.columns if c.startswith("wind_speed_t")]
        v_hub = synth[wcols].mean(axis=1)
        p_synth = synth["power_park"] if "power_park" in synth else synth[[c for c in synth.columns if c.startswith("power_t")]].sum(axis=1)
        meas = meterdata.load_park_power(park_id, WINDOW)
        rated = meterdata.rated_power_w(park_id)
        df = pd.concat([meas.rename("meas"), p_synth.rename("synth"), v_hub.rename("v"),
                        keep.rename("keep")], axis=1).dropna(subset=["meas", "synth", "v"])
        df["keep"] = df["keep"].fillna(True).astype(bool)
        rem = df[~df.keep]
        ci = min_cut_in(args.exp, park_id)
        stage1 = (rem.meas == 0) & (rem.v >= ci)
        dbs = ~stage1
        under = dbs & (rem.meas < 0.5 * rem.synth) & (rem.synth > 0.3 * rated)
        over = dbs & (rem.meas > 1.5 * rem.synth)
        other = dbs & ~under & ~over
        local = rem.index.tz_convert("Europe/Berlin")
        night = (local.hour >= 22) | (local.hour < 6)
        malo = mapping.loc[mapping.park_id == park_id, "malo_id"].iloc[0]
        rows.append({
            "park_id": park_id, "n_hours": len(df), "removed_share": float((~df.keep).mean()),
            "stage1_zero_share": float(stage1.sum() / len(df)),
            "under_at_wind_share": float(under.sum() / len(df)),
            "over_vs_model_share": float(over.sum() / len(df)),
            "other_share": float(other.sum() / len(df)),
            "removed_night_share": float(night.mean()) if len(rem) else np.nan,
            "night_curtailment_documented": str(malo) in NIGHT_PARKS,
            "energy_share_removed_meas": float(rem.meas.sum() / df.meas.sum()),
            "energy_share_removed_synth": float(rem.synth.sum() / df.synth.sum()),
        })
        print(f"{park_id}: removed {rows[-1]['removed_share']*100:5.1f} %  "
              f"zero {rows[-1]['stage1_zero_share']*100:4.1f}  under {rows[-1]['under_at_wind_share']*100:4.1f}  "
              f"over {rows[-1]['over_vs_model_share']*100:4.1f}  other {rows[-1]['other_share']*100:4.1f}  "
              f"night {rows[-1]['removed_night_share']*100:4.1f} %", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f"filter_classes_{args.exp}.csv"), index=False)
    print("\nfleet mean shares [% of hours]:")
    print((out[["removed_share", "stage1_zero_share", "under_at_wind_share", "over_vs_model_share",
                "other_share"]].mean() * 100).round(2).to_string())

    # 4d-3: ladder medians per filter variant
    sc = pd.read_csv(os.path.join(REPO, "results", "quantitative", "scores.csv"))
    exps = ["M0", "M1noage", "M2noage", "M2all", "M5all_PL", "M5all"]
    med = (sc[sc.experiment.isin(exps)]
           .groupby(["experiment", "variant"])[["r2", "energy_ratio", "rmse_n", "wasserstein"]]
           .median().reset_index())
    med["abs_er_dev"] = (sc[sc.experiment.isin(exps)].assign(d=lambda x: (x.energy_ratio - 1).abs())
                         .groupby(["experiment", "variant"])["d"].median().values)
    med.to_csv(os.path.join(OUT, "filter_variant_medians.csv"), index=False)
    print("\nladder medians per filter variant:")
    print(med.pivot(index="experiment", columns="variant", values=["r2", "abs_er_dev"]).round(3).to_string())
    print("FILTER_DONE")


if __name__ == "__main__":
    main()
