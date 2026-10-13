"""Run the PV chain for one park on its driver frame and add the snow loss (E8, depth_linear: coverage =
min(snow depth / 5 cm, 1), string loss model of pv.snow) as a multiplicative factor on the AC power."""
import numpy as np
import pandas as pd

from pv import chain, snow

SNOW_FULL_COVER_CM = 5.0
OUT_EXTRA = ["snow_depth_cm", "snow_coverage", "snow_factor", "power_park_free", "power_park"]


def run(cfg: dict, drv: pd.DataFrame, horizon: pd.Series = None, snow_on: bool = True) -> pd.DataFrame:
    res = chain.run_park(cfg, drv, horizon=horizon)
    res["snow_depth_cm"] = drv["snow_depth_cm"].reindex(res.index).fillna(0.0)
    cov = (res["snow_depth_cm"] / SNOW_FULL_COVER_CM).clip(0.0, 1.0) if snow_on else pd.Series(0.0, index=res.index)
    res["snow_coverage"] = cov
    res["snow_factor"] = snow.loss_factor(cov).reindex(res.index).fillna(1.0) if snow_on else 1.0
    res["power_park_free"] = res["p_ac_w"]
    res["power_park"] = res["p_ac_w"] * res["snow_factor"]
    return res
