#!/usr/bin/env python3
"""Unit tests for era5_bc.metrics.

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/test_metrics.py
"""

import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import metrics as M  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 20_000) * 5.0

    # identity
    assert M.mae(x, x) == 0 and M.rmse(x, x) == 0 and M.mbe(x, x) == 0
    assert abs(M.pcc(x, x) - 1) < 1e-12 and abs(M.r2(x, x) - 1) < 1e-12
    assert abs(M.pss(x, x, 0.5, 40.0) - 100.0) < 1e-9, "PSS(x,x) must be 100"

    # constant offset: MBE sign convention pred - obs
    assert abs(M.mbe(x, x + 1.0) - 1.0) < 1e-9
    assert abs(M.mae(x, x + 1.0) - 1.0) < 1e-9

    # PSS of disjoint distributions ~ 0
    assert M.pss(x, x + 30.0, 0.5, 40.0) < 1.0

    # PSS invariant to sample order, bounded [0, 100]
    y = rng.weibull(2.2, 20_000) * 5.5
    p1 = M.pss(x, y, 0.5, 40.0)
    p2 = M.pss(x, rng.permutation(y), 0.5, 40.0)
    assert abs(p1 - p2) < 1e-9 and 0 <= p1 <= 100

    # pct improvement: model halves the MAE -> 50 % (perf = 0)
    assert abs(M.pct_improvement(0.5, 1.0, 0.0) - 50.0) < 1e-9
    # PCC 0.8 -> 0.9 with perf 1.0 -> 50 %
    assert abs(M.pct_improvement(0.9, 0.8, 1.0) - 50.0) < 1e-9

    # station_metrics smoke
    ev = {"pss_tail_pcts": [10, 90], "pss_bin_width": 0.5, "pss_max_ws": 40.0,
          "quantiles": [0.1, 0.5, 0.9]}
    res = M.station_metrics(x, y, ev)
    assert set(res) >= {"mae", "mbe", "rmse", "pcc", "pss_all", "pss_lwt",
                        "pss_upt", "median_obs", "median_pred", "q50_obs"}
    assert res["pss_lwt"] <= 100 and res["pss_upt"] <= 100

    print("metrics tests PASSED")


if __name__ == "__main__":
    main()
