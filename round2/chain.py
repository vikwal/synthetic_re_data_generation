"""WP6/WP7 — vectorized fast chain over cached per-park arrays.

run_chain(theta) reproduces the v2 power chain in pure NumPy for the
sensitivity analysis (guide 6.1: scalar output = mean over parks of
|energy_ratio - 1|). Validated against generate_wind on corner
points by scripts/round2/wp6_validate_chain.py (tolerance < 0.5 % ER).

theta keys (defaults = M-ladder baseline):
  wind_level_factor  float             multiplies v10 and v100
  correction         'off'|'wind10_only'|'height_consistent'
  shear              'power_law'|'most'
  z0_scale           float
  aging_model        'const'|'weibull'|'weibull_step'|'off' ('none' alias, DF=1)
  aging_lambda, aging_kappa  floats
  power_curve_scale  float
  density            'v1_mixed'|'static_1225'|'dynamic'
  wake_k             float in [0.05, 0.10] (interpolated on the 3-point grid)
  wake_enabled       bool
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from round2 import aging as r2_aging
from round2 import correction as r2_correction
from round2 import stability as r2_stability
from round2 import meterdata

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ERA5_DIR_DEFAULT = "/mnt/nas/synthetic/raw/wind_era5"
ERA5_DIR_V2 = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
WAKE_DIR = "/mnt/nvme2/synthetic/raw/round2/wakes"
WAKE_KS = [0.05, 0.075, 0.10]
# effective window: chain outputs start 2023-07-24 (round-1 convention),
# meter data ends 2024-05 -> align the fast chain to the same hours the
# ladder evaluation actually uses
VALIDATION_WINDOW = ("2023-07-24", "2024-06-01")

DEFAULT_THETA = {
    "wind_level_factor": 1.0, "correction": "height_consistent",
    "shear": "power_law", "z0_scale": 1.0,
    "aging_model": "weibull", "aging_lambda": 54.5, "aging_kappa": 2.0,
    "power_curve_scale": 1.0, "density": "v1_mixed",
    "wake_k": 0.075, "wake_enabled": True,
}


def _load_park_config(park_id):
    path = os.path.join(REPO, "configs", "real_wind_parks_era5",
                        f"config_{park_id}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _power_curves(turbines, cut_outs):
    """Interpolated curves exactly like generate_wind.py:interpolate()."""
    pc = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_power.csv"),
                     sep=";", decimal=",", index_col=0)
    pc = pc.loc[:, ~pc.columns.duplicated()]
    curves = {}
    for t, co in zip(turbines, cut_outs):
        ticks = np.arange(0, co * 100, 1) / 100.0
        curve = (pc[t] * 1000.0)
        interp = pd.Series(index=ticks, dtype=float).to_frame() \
            .merge(curve.to_frame(), how="left", left_index=True, right_index=True)
        interp.drop(columns=[0], inplace=True)
        interp.iloc[-1] = curve.max()
        interp = interp.interpolate(method="polynomial", order=3)
        interp = interp.clip(upper=curve.max(), lower=0).fillna(0)
        curves[t] = (ticks, interp.iloc[:, -1].values)
    return curves


class ParkCache:
    def __init__(self, park_id: str, era5_dir: str = None):
        self.park_id = park_id
        cfg = _load_park_config(park_id)
        p = cfg["params"]
        self.turbines = p["turbines"]
        self.hub_heights = p["hub_heights"]
        station_id = park_id[:5]

        era5_dir = era5_dir or (
            ERA5_DIR_V2 if os.path.exists(
                os.path.join(ERA5_DIR_V2, f"Station_{station_id}.csv"))
            else ERA5_DIR_DEFAULT)
        df = pd.read_csv(os.path.join(era5_dir, f"Station_{station_id}.csv"))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.loc[VALIDATION_WINDOW[0]:VALIDATION_WINDOW[1]]
        self.index = df.index
        self.v10 = np.hypot(df["u_wind_10m"], df["v_wind_10m"]).values
        self.v100 = np.hypot(df["u_wind_100m"], df["v_wind_100m"]).values

        # density chain at 2 m (see generate_wind.py:get_rho + huang p_s)
        t2 = df["temp_2m"].values
        td = df["dew_point_2m"].values
        press = df["pressure"].values
        e = _huang(td - 273.15)
        e_s = _huang(t2 - 273.15)
        relhum = e / e_s
        p_w = relhum * e_s
        p_g = press - p_w
        self.rho2m = p_g / (287.05 * t2) + p_w / (461.5 * t2)
        self.t2 = t2
        # hub density per turbine height (barometric, as in v1)
        self.rho_hub = {}
        for h in sorted(set(self.hub_heights)):
            self.rho_hub[h] = self.rho2m * (
                1 - (0.00649 * (h - 2)) / t2) ** ((0.028949 * 9.81) / (0.00649 * 8.31451) - 1)

        self.ustar = df["friction_wind"].values if "friction_wind" in df else None
        self.sshf = df["sshf"].values if "sshf" in df else None
        self.L = (r2_stability.obukhov_length(self.ustar, self.rho2m, t2, self.sshf)
                  if self.sshf is not None else None)

        # correction tables
        self.q_era5, self.q_target = None, None
        for name in (park_id, station_id):
            path = os.path.join(REPO, "data", "round2", "correction", f"{name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    tab = json.load(f)
                key = {"A": "q_target_station", "B": "q_target_model"}.get(tab["branch"])
                if key and tab.get(key):
                    self.q_era5 = np.array(tab["q_era5"])
                    self.q_target = np.array(tab[key])
                break

        # z0
        topo = pd.read_csv(os.path.join(REPO, "data", "round2", "topo_features.csv"),
                           dtype={"location_id": str})
        row = topo[topo["location_id"] == station_id]
        self.z0 = float(row["z0"].iloc[0]) if len(row) and pd.notna(row["z0"].iloc[0]) else 0.1

        # turbine specs + curves
        specs = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_specs.csv"), sep=";")
        self.specs = {}
        for t in set(self.turbines):
            s = specs[specs["Turbine"] == t].iloc[0]
            self.specs[t] = {"cut_in": float(s["Einschaltgeschwindigkeit"]),
                             "cut_out": float(s["Abschaltgeschwindigkeit"]),
                             "rated_ws": float(s["Nennwindgeschwindigkeit"])}
        self.curves = _power_curves(self.turbines,
                                    [self.specs[t]["cut_out"] for t in self.turbines])

        # aging: base start age from comm_dates / park config; parks with
        # apply_ageing=False in their round-1 config are never aged (the
        # v2 script keeps that behaviour, so the fast chain must too)
        # Ask 12: authoritative table (configs) is the single age source
        from round2 import parkinfo
        try:
            self.commissioning_date = parkinfo.commissioning_date(park_id)
        except KeyError:
            comm = pd.read_csv(os.path.join(REPO, "data", "comm_dates.csv"), sep=";",
                               dtype={"park_id": str})
            cd = comm[comm["park_id"] == park_id]["commissioning_date"].values
            self.commissioning_date = cd[0] if len(cd) else p.get("commissioning_date")
        self.apply_ageing = bool(p.get("apply_ageing", True))
        self.ages = np.clip(
            (self.index - pd.to_datetime(self.commissioning_date, utc=True)).days / 365.25,
            0, None) if (self.commissioning_date is not None and self.apply_ageing) else None

        # wake factors on the 3-point k grid
        self.wake = {}
        for k in WAKE_KS:
            path = os.path.join(WAKE_DIR, f"w_{park_id}_k{k:.3f}.parquet")
            if os.path.exists(path):
                w = pd.read_parquet(path)["w"].reindex(self.index).ffill().fillna(1.0)
                self.wake[k] = w.values

        # measured energy over the same window
        meas = meterdata.load_park_power(park_id, VALIDATION_WINDOW)
        both = pd.Series(True, index=self.index).to_frame("m").join(meas, how="left")
        self.meas = both.iloc[:, 1].values  # W, NaN where missing

    def force_all_aging(self):
        """Ask 29 (b): SA baseline = paper-M5 (server M5all) — age ALL parks
        from the authoritative parkinfo dates, ignoring the round-1
        apply_ageing flags. Call once on the Chain-cached parks before an SA
        sweep; the ladder itself achieves the same via apply_ageing_override."""
        from round2 import parkinfo
        if self.ages is None:
            cd = parkinfo.commissioning_date(self.park_id)
            self.ages = np.clip(
                (self.index - pd.to_datetime(cd, utc=True)).days / 365.25,
                0, None)

    def energy_ratio(self, theta: dict) -> float:
        th = {**DEFAULT_THETA, **theta}
        v10, v100 = self.v10.copy(), self.v100.copy()
        if th["correction"] != "off" and self.q_target is not None:
            c = r2_correction.correction_factor(v10, self.q_era5, self.q_target)
            v10 = v10 * c
            if th["correction"] == "height_consistent":
                v100 = v100 * c
        wlf = th["wind_level_factor"]
        v10, v100 = v10 * wlf, v100 * wlf

        # degradation factor
        if self.ages is not None:
            if th["aging_model"] == "weibull":
                df_vec = r2_aging.DF_weibull(self.ages, th["aging_lambda"], th["aging_kappa"])
            elif th["aging_model"] == "weibull_step":
                df_vec = r2_aging.DF_weibull_step(self.ages, th["aging_lambda"], th["aging_kappa"])
            elif th["aging_model"] in ("off", "none"):
                # ask 29: aging disabled, DF = 1 (the M5all_noage state)
                df_vec = np.ones(len(self.index))
            else:
                df_vec = np.linspace(r2_aging.DF_const(self.ages[0]),
                                     r2_aging.DF_const(self.ages[-1]), len(self.ages))
        else:
            df_vec = np.ones(len(self.index))

        p_park = np.zeros(len(self.index))
        for t, h in zip(self.turbines, self.hub_heights):
            if th["shear"] == "most" and self.L is not None:
                v_hub = r2_stability.most_wind_profile(
                    v100, h, self.z0 * th["z0_scale"], self.L)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.log(v100 / v10) / np.log(10.0)
                alpha = np.clip(np.nan_to_num(alpha), 0.0, 0.4)
                v_hub = v10 * (h / 10.0) ** alpha
            v_hub = np.round(v_hub, 2)
            s = self.specs[t]
            ticks, watts = self.curves[t]
            watts = watts * th["power_curve_scale"]
            rated_power = watts.max()
            # degradation as curve shift, released above the shifted rated ws
            dr = df_vec.copy()
            rated_ws_vec = s["rated_ws"] * (1.0 / dr) ** (1.0 / 3.0)
            dr[v_hub >= rated_ws_vec] = 1.0
            v_eff = v_hub * dr ** (1.0 / 3.0)
            p_curve = np.interp(v_eff, ticks, watts)
            if th["density"] == "static_1225":
                rho = 1.225
            else:
                rho = self.rho_hub[h]
            p = p_curve * (rho / 1.225)
            p = np.where((v_hub < s["cut_in"]) | (v_hub > s["cut_out"]), 0.0, p)
            p = np.minimum(p, rated_power)
            p_park += p

        if th["wake_enabled"] and self.wake:
            k = float(np.clip(th["wake_k"], WAKE_KS[0], WAKE_KS[-1]))
            ks = np.array(sorted(self.wake))
            ws = np.stack([self.wake[kk] for kk in sorted(self.wake)])
            # linear interpolation across the k grid, per hour
            w = np.array([np.interp(k, ks, ws[:, i]) for i in range(ws.shape[1])]) \
                if len(ks) > 1 else ws[0]
            p_park = p_park * w

        ok = ~np.isnan(self.meas)
        if ok.sum() < 100 or np.nansum(self.meas[ok]) <= 0:
            return np.nan
        return float(p_park[ok].sum() / self.meas[ok].sum())


class Chain:
    def __init__(self, park_ids=None):
        if park_ids is None:
            park_ids = sorted(
                os.path.basename(p)[len("config_"):-len(".yaml")]
                for p in glob.glob(os.path.join(
                    REPO, "configs", "real_wind_parks_era5", "config_*.yaml")))
        self.parks = [ParkCache(pid) for pid in park_ids]

    def run_chain(self, theta: dict) -> float:
        """Guide 6.1 scalar: mean over parks of |ER - 1|."""
        ers = np.array([p.energy_ratio(theta) for p in self.parks])
        return float(np.nanmean(np.abs(ers - 1.0)))

    def energy_ratios(self, theta: dict) -> dict:
        return {p.park_id: p.energy_ratio(theta) for p in self.parks}


def _huang(temp_c):
    return np.where(
        temp_c > 0,
        np.exp(34.494 - (4924.99 / (temp_c + 237.1))) / (temp_c + 105) ** 1.57,
        np.exp(43.494 - (6545.8 / (temp_c + 278))) / (temp_c + 868) ** 2)
