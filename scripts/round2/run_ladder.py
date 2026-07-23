#!/usr/bin/env python3
"""M-ladder driver: run one experiment (M1..M5) over the 13 parks and
evaluate with WP1.

Per park: merge the park YAML (configs/real_wind_parks_era5/config_XXXXX.yaml)
with the experiment's round2 block (configs/round2/<EXP>.yaml), write the
merged config to configs/round2/_generated/, run generate_wind.main()
in a subprocess, then evaluate synthetic park power vs meter data.

Usage: run_ladder.py M2 [--parks 07374 04745] [--skip-generate]
Results: results/round2/ladder_metrics.csv (append/update)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata  # noqa: E402

PYTHON = os.path.join(REPO, "synthre", "bin", "python")
PARK_CONFIG_DIR = os.path.join(REPO, "configs", "real_wind_parks_era5")
GEN_DIR = os.path.join(REPO, "configs", "round2", "_generated")
SYNTH_BASE = "/mnt/nvme2/synthetic"
RESULTS = os.path.join(REPO, "results", "round2", "ladder_metrics.csv")
VALIDATION_WINDOW = ("2023-06-01", "2024-06-01")


def merged_config(park_id: str, experiment: str) -> str:
    with open(os.path.join(PARK_CONFIG_DIR, f"config_{park_id}.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(REPO, "configs", "round2", f"{experiment}.yaml")) as f:
        r2 = yaml.safe_load(f)
    cfg["round2"] = r2["round2"]
    cfg["data"]["synth_dir"] = SYNTH_BASE
    os.makedirs(GEN_DIR, exist_ok=True)
    out = os.path.join(GEN_DIR, f"config_{park_id}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return f"round2/_generated/config_{park_id}.yaml"


def run_park(park_id: str, experiment: str) -> bool:
    rel = merged_config(park_id, experiment)
    code = (f"import sys; sys.argv=['x']; "
            f"import generate_wind as m; m.main('{rel}')")
    res = subprocess.run([PYTHON, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        print(f"FAILED {experiment}/{park_id}:\n{res.stderr[-2000:]}")
        return False
    return True


def evaluate_park(park_id: str, experiment: str, prices: pd.Series) -> dict:
    synth_path = os.path.join(SYNTH_BASE, "wind", "round2", experiment,
                              f"synth_{park_id}.csv")
    df = pd.read_csv(synth_path, sep=";", index_col=0, parse_dates=True)
    synth = df["power_park"]
    meas = meterdata.load_park_power(park_id, VALIDATION_WINDOW)
    synth = synth.loc[VALIDATION_WINDOW[0]:VALIDATION_WINDOW[1]]
    rated = meterdata.rated_power_w(park_id)
    m = evaluation.evaluate(meas, synth, p_rated=rated, prices=prices)
    m.pop("acf_meas", None), m.pop("acf_synth", None)
    return {"experiment": experiment, "park_id": park_id, "rated_w": rated, **m}


def sanity_gates(row: dict) -> list:
    issues = []
    if not (0.5 <= row.get("energy_ratio", np.nan) <= 2.0):
        issues.append(f"energy_ratio={row.get('energy_ratio'):.3f} outside [0.5,2]")
    if row.get("n_hours", 0) < 4000:
        issues.append(f"only {row.get('n_hours')} overlapping hours")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--parks", nargs="*", default=None)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    parks = args.parks or sorted(
        os.path.basename(p)[len("config_"):-len(".yaml")]
        for p in glob.glob(os.path.join(PARK_CONFIG_DIR, "config_*.yaml")))
    prices = pd.read_csv(os.path.join(REPO, "data", "round2", "prices_delu.csv"),
                         index_col=0, parse_dates=True)["price_eur_mwh"]

    rows, problems = [], []
    for pid in parks:
        if not args.skip_generate:
            ok = run_park(pid, args.experiment)
            if not ok:
                problems.append((pid, "generation failed"))
                continue
        try:
            row = evaluate_park(pid, args.experiment, prices)
        except Exception as e:
            print(f"EVAL FAILED {pid}: {e}", flush=True)
            problems.append((pid, f"evaluation failed: {e}"))
            continue
        for issue in sanity_gates(row):
            problems.append((pid, issue))
        rows.append(row)
        print(f"{args.experiment} {pid}: R2={row['r2']:.3f} "
              f"ER={row['energy_ratio']:.3f} W1={row['wasserstein']:.4f} "
              f"({row['n_hours']} h)")

    new = pd.DataFrame(rows)
    if new.empty:
        print("NO RESULTS — all parks failed:", problems)
        sys.exit(1)
    try:
        old = pd.read_csv(RESULTS, dtype={"park_id": str})
        # replace only the rows actually re-evaluated (experiment x parks) —
        # a partial --parks run must not drop the other parks' rows
        done = set(new["park_id"])
        old = old[~((old["experiment"] == args.experiment)
                    & (old["park_id"].isin(done)))]
        new = pd.concat([old, new], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        pass
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new.to_csv(RESULTS, index=False)

    sub = new[new["experiment"] == args.experiment]
    print(f"\n== {args.experiment}: {len(sub)} parks ==")
    print(sub[["park_id", "r2", "energy_ratio", "wasserstein", "rmse_n"]]
          .round(4).to_string(index=False))
    print("medians:", sub[["r2", "energy_ratio", "wasserstein"]].median().round(4).to_dict())
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" ", p)
    report = os.path.join(REPO, "results", "round2", "RUN_REPORT.md")
    with open(report, "a") as f:
        f.write(f"\n## ladder {args.experiment} ({pd.Timestamp.now()})\n")
        f.write(sub[["park_id", "r2", "energy_ratio", "wasserstein"]]
                .round(4).to_markdown(index=False) + "\n")
        for p in problems:
            f.write(f"- PROBLEM: {p}\n")


if __name__ == "__main__":
    main()
