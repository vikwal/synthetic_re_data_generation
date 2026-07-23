#!/usr/bin/env python3
"""WP7.1 — delta-method uncertainty paragraph (mandatory minimum).

sigma_v per ERA5 class from the WP2-A RMSE table; propagates to power via the
first-order formula. Writes results/round2/uncertainty.md.
"""

import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import uncertainty  # noqa: E402

TABLE = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")
OUT = os.path.join(REPO, "results", "round2", "uncertainty.md")


def main():
    table = pd.read_parquet(TABLE)
    lines = ["# WP7.1 — First-order (delta-method) uncertainty band", ""]
    lines.append("| ERA5 class | stations | mean RMSE [m/s] | sigma_v/v | "
                 "sigma_P/P | 95 % band [%] | amplification |")
    lines.append("|---|---|---|---|---|---|---|")
    q50_col = [c for c in table.columns if c.startswith("q_dwd_0.5")][0]
    for cls, grp in table.groupby("era5_class"):
        rmse = float(grp["rmse"].mean())
        mean_wind = float(grp[q50_col].mean())
        sv = rmse / mean_wind
        s = uncertainty.delta_band_summary(sv)
        lines.append(f"| {cls} | {len(grp)} | {rmse:.2f} | {sv:.3f} | "
                     f"{s['sigma_P_rel']:.3f} | +-{s['band_95_pct']:.0f} | "
                     f"x{s['amplification']:.2f} |")
    lines += [
        "",
        "sigma_v/v uses each class's mean observed median 10 m wind; at hub "
        "height the RELATIVE error is smaller (shear raises the level more "
        "than the absolute error), so these bands are conservative upper "
        "bounds at 10 m.",
        "",
        "Below rated power, the first-order propagation "
        "(sigma_P/P)^2 = (3 sigma_v/v)^2 + (sigma_rho/rho)^2 + (sigma_DF/DF)^2 "
        "+ (sigma_curve/P)^2 shows the wind-speed error is amplified by a "
        "factor of ~3 through the power-curve cube and dominates every other "
        "input by an order of magnitude. The approximation breaks near "
        "cut-in and rated speed and is used as a cross-check, not as a "
        "calibrated band (Monte-Carlo propagation with coverage calibration "
        "is deferred to follow-up work).",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
