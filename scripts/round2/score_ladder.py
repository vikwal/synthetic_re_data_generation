#!/usr/bin/env python3
"""Unified re-scorer: all ladder experiments x 13 parks x 3 filter variants.

Variants: raw (all overlapping hours), clean (DBSCAN + cut-in filter,
model-dependent mask), clean_exkl_curt (clean plus price<=0 hours removed).
No regeneration — reads the existing synth outputs.

Outputs: results/round2/summary/scores.csv (tidy),
         results/round2/summary/outlier_shares.csv,
         masks -> /mnt/nvme2/synthetic/raw/round2/masks/{exp}/mask_{park}.parquet

Usage: score_ladder.py [--experiments M1 M5all ...] [--check-vs-ladder]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata, outliers  # noqa: E402

SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"
MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
OUT_DIR = os.path.join(REPO, "results", "round2", "summary")
WINDOW = ("2023-06-01", "2024-06-01")


def min_cut_in(park_id: str) -> float:
    with open(os.path.join(REPO, "configs", "real_wind_parks_era5",
                           f"config_{park_id}.yaml")) as f:
        turbines = yaml.safe_load(f)["params"]["turbines"]
    specs = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_specs.csv"),
                        sep=";").drop_duplicates(subset="Turbine").set_index("Turbine")
    return float(min(float(specs.loc[t, "Einschaltgeschwindigkeit"])
                     for t in set(turbines)))


def score_one(exp: str, park_id: str, prices: pd.Series):
    path = os.path.join(SYNTH_BASE, exp, f"synth_{park_id}.csv")
    df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
        .loc[WINDOW[0]:WINDOW[1]]
    synth = df["power_park"]
    wcols = [c for c in df.columns if c.startswith("wind_speed_t")]
    v_hub = df[wcols].mean(axis=1)
    meas = meterdata.load_park_power(park_id, WINDOW)
    rated = meterdata.rated_power_w(park_id)

    keep = outliers.compute_mask(park_id, meas, v_hub,
                                 min_cut_in=min_cut_in(park_id), synth=synth)
    os.makedirs(os.path.join(MASK_BASE, exp), exist_ok=True)
    keep.rename("keep").to_frame().to_parquet(
        os.path.join(MASK_BASE, exp, f"mask_{park_id}.parquet"))
    outlier_share = float(1.0 - keep.mean())

    rows = []
    variants = {
        "raw": (meas, synth, None),
        "clean": (meas[keep.reindex(meas.index).fillna(False)], synth, None),
        "clean_exkl_curt": (meas[keep.reindex(meas.index).fillna(False)], synth, prices),
    }
    for variant, (m, s, pr) in variants.items():
        res = evaluation.evaluate(m, s, p_rated=rated, prices=pr)
        res.pop("acf_meas", None), res.pop("acf_synth", None)
        if variant == "clean_exkl_curt":
            # keep only the price-screened numbers from the companion run
            res = {k.replace("_excl_curt", ""): v for k, v in res.items()
                   if (k.endswith("_excl_curt") and not k.startswith("acf"))
                   or k in ("n_hours", "window", "curtailed_share")}
        rows.append({"experiment": exp, "park_id": park_id, "variant": variant,
                     "outlier_share": outlier_share, **res})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--check-vs-ladder", action="store_true")
    args = ap.parse_args()

    exps = args.experiments or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(SYNTH_BASE, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith("regtest"))
    prices = pd.read_csv(os.path.join(REPO, "data", "round2", "prices_delu.csv"),
                         index_col=0, parse_dates=True)["price_eur_mwh"]
    parks = sorted(meterdata.load_mapping()["park_id"])

    all_rows, problems = [], []
    for exp in exps:
        for pid in parks:
            try:
                all_rows.extend(score_one(exp, pid, prices))
            except FileNotFoundError:
                problems.append((exp, pid, "missing synth output"))
            except Exception as e:
                problems.append((exp, pid, str(e)[:120]))
        done = [r for r in all_rows if r["experiment"] == exp and r["variant"] == "clean"]
        if done:
            med = float(np.median([r["r2"] for r in done]))
            print(f"{exp}: {len(done)} parks | median clean R2 = {med:.3f}", flush=True)

    scores = pd.DataFrame(all_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "scores.csv")
    if args.experiments and os.path.exists(out_csv):
        # partial run: merge into the existing table instead of overwriting
        old = pd.read_csv(out_csv, dtype={"park_id": str})
        old = old[~old["experiment"].isin(set(scores["experiment"]))]
        scores = pd.concat([old, scores], ignore_index=True)
    scores.to_csv(out_csv, index=False)
    shares = scores[scores["variant"] == "clean"][
        ["experiment", "park_id", "outlier_share"]]
    shares.pivot(index="park_id", columns="experiment", values="outlier_share") \
        .round(4).to_csv(os.path.join(OUT_DIR, "outlier_shares.csv"))
    print(f"scores: {len(scores)} rows -> {OUT_DIR}/scores.csv")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" ", p)

    if args.check_vs_ladder:
        ladder = pd.read_csv(os.path.join(REPO, "results", "round2",
                                          "ladder_metrics.csv"),
                             dtype={"park_id": str})
        raw = scores[scores["variant"] == "raw"].set_index(["experiment", "park_id"])
        n_ok = n_bad = 0
        for _, lr in ladder.iterrows():
            key = (lr["experiment"], lr["park_id"])
            if key not in raw.index:
                continue
            if abs(raw.loc[key, "r2"] - lr["r2"]) < 1e-6:
                n_ok += 1
            else:
                n_bad += 1
                print(f"MISMATCH {key}: scorer {raw.loc[key, 'r2']:.6f} "
                      f"vs ladder {lr['r2']:.6f}")
        print(f"consistency raw vs ladder_metrics: {n_ok} ok, {n_bad} mismatch")


if __name__ == "__main__":
    main()
