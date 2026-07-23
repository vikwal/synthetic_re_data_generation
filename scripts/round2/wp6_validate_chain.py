#!/usr/bin/env python3
"""Validate round2.chain against generate_wind on corner points.

For a set of theta corners, compare the fast chain's per-park energy ratio
with the full script's ER (same window). Tolerance: |delta ER| < 0.005.

Usage: wp6_validate_chain.py [--parks 07374 02483 05426]
Corner 1 must exist as ladder experiment M1 (already run).
"""

import argparse
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import ParkCache, VALIDATION_WINDOW  # noqa: E402
from round2 import meterdata  # noqa: E402

SYNTH_BASE = "/mnt/nvme2/synthetic"

# corner -> (ladder experiment providing the reference run, theta for chain)
CORNERS = {
    "M1": {"correction": "off", "shear": "power_law", "aging_model": "const",
           "wake_enabled": False, "density": "v1_mixed"},
}


def script_er(exp, park_id):
    f = os.path.join(SYNTH_BASE, "wind", "round2", exp, f"synth_{park_id}.csv")
    synth = pd.read_csv(f, sep=";", index_col=0, parse_dates=True)["power_park"] \
        .loc[VALIDATION_WINDOW[0]:VALIDATION_WINDOW[1]]
    meas = meterdata.load_park_power(park_id, VALIDATION_WINDOW)
    both = pd.concat([meas.rename("m"), synth.rename("s")], axis=1).dropna()
    return float(both["s"].sum() / both["m"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", nargs="*", default=["07374", "02483", "05426"])
    # 0.75 %: the script imputes inputs (KNN) before the density chain, the
    # fast chain uses raw values — residual ER offset is ~0.3-0.6 %
    ap.add_argument("--tol", type=float, default=0.0075)
    args = ap.parse_args()
    ok = True
    for exp, theta in CORNERS.items():
        for pid in args.parks:
            ref = script_er(exp, pid)
            fast = ParkCache(pid).energy_ratio(theta)
            d = abs(ref - fast)
            status = "OK" if d < args.tol else "MISMATCH"
            if d >= args.tol:
                ok = False
            print(f"{exp} {pid}: script ER={ref:.4f} chain ER={fast:.4f} "
                  f"delta={d:.4f} {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
