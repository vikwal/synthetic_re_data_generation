"""Shared setup for the angle-decomposition report figures (scripts/pv/report_angle_*.py).

Figures are written at their final width for an IEEE two-column page (3.5 in per column, 7.16 in across)
and included at scale 1, so the font sizes below are the sizes in the PDF. Trianel parks appear only as
pv1..pv7; no coordinates, names or meter ids leave this module.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = os.path.expanduser("~/Work/FL_Contribution")
FIGS_GEN = os.path.join(PROJECT, "reports", "figs_pv_generator")
FIGS_SYN = os.path.join(PROJECT, "reports", "figs_pv_park_synthesis_v1")
OUT = os.path.join(PROJECT, "reports", "pv_angle_decomposition", "figs")
BENCH = os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic", "solar", "trianel_benchmark", "inputs")
START, END = "2023-07-24", "2024-06-01"           # meter window, 312 days
PARKS = ["pv1", "pv2", "pv3", "pv5", "pv6", "pv7"]

COL, FULL = 3.5, 7.16                              # inches
C_REG = "#8A8A8A"       # register / master data
C_POA = "#8FA6CE"       # POA bank
C_CHAIN = "#1F4596"     # chain bank
C_PRIOR = "#6A4C93"     # row prior + chain bank / geodata
C_GEO = "#2E8B57"       # geodata without a meter
C_METER = "#222222"     # meter
C_EW = "#B3121A"        # east-west pair


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42, "font.family": "sans-serif",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def trianel_park(pv: str) -> dict:
    """Everything the ablation's geometry rungs use for one metered park: corrected CAMS driver, meter,
    ERA5 snow depth, horizon, terrain arrays around the master angles and a config factory."""
    import ablation_trianel as ab
    import trianel_common as tc
    from parks import era5_db
    from pv import snow, timebase
    style()
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, os.path.join(BENCH, "mastr", "mastr_candidates.csv"),
                           os.path.join(BENCH, "mastr", "mastr_xml_units.json"))
    par = tc.park_parameters(parks, facts, meter)
    osm = json.load(open(os.path.join(BENCH, "osm", "trianel_pv_osm.json")))
    p = parks.loc[pv]
    idx = timebase.hourly_index(START, END)
    drvs, _ = tc.build_drivers(p, START, END, era5_db.connect(), os.path.join(BENCH, "cams"), pv)
    raw = drvs["cams"]
    c = pd.read_parquet(os.path.join(BENCH, "cams_corrected", f"{pv}_cams_corrected_hourly_xgb_v5.parquet")).reindex(idx)
    corr = raw.copy()
    corr["ghi"] = c["ghi_corr"].fillna(raw["ghi"]).clip(lower=0)
    corr["dhi"] = np.minimum(c["dhi_corr"].fillna(raw["dhi"]).clip(lower=0), corr["ghi"])
    corr["bhi"] = corr["ghi"] - corr["dhi"]
    hz = tc.load_horizon(os.path.join(BENCH, "osm", "pvgis_horizon.json"), pv)
    master, terr, _ = ab.geometry(pv, p, facts, par, osm)
    sn = pd.read_parquet(os.path.join(BENCH, "era5_snow_full", f"{pv}_era5_snow.parquet"))
    sn.index = sn.index - pd.Timedelta("1h")
    depth_cm = (sn["snow_depth_m"] * 100.0).reindex(idx).ffill(limit=6).fillna(0.0)
    snow_factor = snow.loss_factor((depth_cm / ab.SNOW_FULL_COVER_CM).clip(0.0, 1.0))
    from parks_pv.params import LOSSES

    def cfg_for(arrays):
        cfg = tc.base_config(pv, p, par.loc[pv], arrays=arrays, variant="PR")
        cfg["shading"]["horizon"] = True
        cfg["losses"] = dict(LOSSES)
        cfg["aging"]["enabled"] = True
        return cfg

    ac = par.loc[pv, "ac_cap_meter_kw"]
    style()                                        # ablation_trianel sets its own rcParams on import
    return {"pv": pv, "p": p, "idx": idx, "y": meter[pv].reindex(idx), "corr": corr, "raw": raw, "hz": hz,
            "terr": terr, "master": master, "depth_cm": depth_cm, "snow_factor": snow_factor, "cfg_for": cfg_for,
            "cap_w": float(par.loc[pv, "kwp_dc"]) * 1000.0, "ac_lim": float(ac) * 1000.0 if np.isfinite(ac) else None}


def estimation_mask(power: pd.Series, drv: pd.DataFrame, lat, lon, alt, bank: dict, exclude: pd.Series, ac_lim):
    """The masks of plausibility.assess in the same order: clear hours minus exclude and clipping, then the
    suspect refit against the bank. Returns (mask, n_clear_before_suspect, n_suspect)."""
    from pv import plausibility as PL
    excl = PL.clipping_mask(power, ac_lim) | exclude.reindex(power.index).fillna(False).astype(bool)
    mask = PL.clear_mask(drv, lat, lon, alt, excl)
    sus = PL.suspect_mask(power, bank, mask)
    return mask & ~sus, int(mask.sum()), int(sus.sum())
