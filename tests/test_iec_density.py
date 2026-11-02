"""Numeric check of the IEC density normalisation in generate_wind.py:
  rho = 1.225  -> identical to the round-2 multiplicative form
  rho = 1.10, cubic region -> both forms scale by rho/rho_0
  rho = 1.10, above rated  -> IEC keeps rated power, multiplicative loses 10 %
Run from the repo root on l2 with synthre/bin/python.
"""
import sys
import numpy as np
import pandas as pd

sys.argv = ["x"]
import generate_wind as gw

ws = np.arange(0, 26, 1.0)
rated = 2000e3
curve = pd.Series(np.where(ws < 3, 0, np.where(ws < 12, rated * ((ws - 3) / 9) ** 3, rated)), index=ws)
D = 90.0
A = np.pi * (D / 2) ** 2


def power(v, rho, mode):
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    data = pd.DataFrame({"wind_speed_hub": [v], "density_hub": [rho]}, index=idx)
    cp = gw.get_cp_from_power_curve_hardcoded(data, curve, "wind_speed_hub", "density_hub", D,
                                              density_mode=mode)
    params = {"noise": 0.0}
    p = gw.generate_wind_power(data, params, rated, cp, D, 3.0, 25.0, 12.0,
                               density_mode=mode)
    return float(np.asarray(p)[0])


ok = True
for v in (6.0, 9.0, 11.5, 14.0):
    a, b = power(v, 1.225, "v1_mixed"), power(v, 1.225, "iec")
    same = abs(a - b) < 1e-6 * rated
    ok &= same
    print(f"v={v:5.1f} rho=1.225  v1_mixed={a/1e3:8.1f} kW  iec={b/1e3:8.1f} kW  equal={same}")
for v in (6.0, 9.0, 11.5, 14.0):
    a, b = power(v, 1.10, "v1_mixed"), power(v, 1.10, "iec")
    expect_iec = float(np.interp(v * (1.10 / 1.225) ** (1 / 3), curve.index, curve.values))
    ok &= abs(b - expect_iec) < 1e-6 * rated
    print(f"v={v:5.1f} rho=1.100  v1_mixed={a/1e3:8.1f} kW  iec={b/1e3:8.1f} kW  "
          f"(expected iec {expect_iec/1e3:8.1f}, ratio mult/iec {a/max(b,1):.3f})")
print("IEC_TEST", "PASS" if ok else "FAIL")
