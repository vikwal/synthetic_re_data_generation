"""Synthetic wind power time series generator.

Builds hourly synthetic wind power output from ERA5 reanalysis (extrapolated
to hub height), a terrain-gated bias correction of the ERA5 input, an
age-resolved turbine degradation model, and optional wake losses. Behaviour
is fully controlled by the YAML config passed via --config; see
configs/round2/M4.yaml for the default (deployed) configuration and
configs/round2/ for validated alternatives (e.g. S1.yaml: stability-corrected
extrapolation instead of the dynamic power law; *_noage / *_noQM / *_noWake:
single-component ablations).

Chain components (see the round2/ package):
  correction     gated quantile-mapping bias correction, height-consistent on v10+v100
  shear          power_law (dynamic exponent from v10/v100) | most (stability-corrected)
  aging_model    const | weibull | weibull_step
  density        v1_mixed | static_1225 | dynamic
  wake           wake loss factor w(t) applied to the park sum
  scalers        wind_level_factor, power_curve_scale, z0_scale (sensitivity analysis)
"""

import os
import json
import math
import shutil
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import metpy.calc as mpcalc
from metpy.units import units
from collections import defaultdict

from utils import clean_data, tools
from round2 import aging as r2_aging
from round2 import correction as r2_correction
from round2 import stability as r2_stability

R2_DEFAULTS = {
    'experiment_id': 'M1',
    'correction': 'off',              # off | wind10_only | height_consistent | tr_bc | tr_bc10
    'correction_branch': 'auto',      # auto | A | B | C
    # tr_bc: hourly era5_bc scaling factors (always height-consistent)
    'bc_factors_parquet': 'results/era5_bc/park_sf_transformer.parquet',
    'shear': 'power_law',             # power_law | most
    'aging_model': 'const',           # none | const | weibull | weibull_step
    'aging': {'lambda': 54.5, 'kappa': 2.0, 'step_delta': 0.02},
    'wake': {'enabled': False, 'model': 'noj', 'k': 0.075},
    'apply_ageing_override': None,    # None = Park-Config; True/False = für ALLE Parks erzwingen
    'commissioning_table': None,      # CSV mit autoritativen Park-Inbetriebnahmedaten (Ask 12)
    'density': 'v1_mixed',            # v1_mixed | static_1225 | dynamic
    'wind_level_factor': 1.0,
    'power_curve_scale': 1.0,
    'z0_scale': 1.0,
    'z0_default': 0.1,
    'output_start': '2023-07-24',
    'era5_dir_v2': None,
    'artifacts_dir': 'data/round2',
    'wake_dir': '/mnt/nvme2/synthetic/raw/round2/wakes',
    'output_naming': 'native',        # native | readable (Ask 35: human-readable
                                      # column names for the published dataset;
                                      # 'native' keeps the park-ladder names)
}

# Ask 35: code-native -> human-readable output column names. Applied only when
# round2.output_naming == 'readable' (site dataset); the park ladder stays on
# native names, so round2 artefacts and score_ladder (reads power_park) are
# unaffected. Per-turbine blh_flag handled by readable_rename_map().
READABLE_RENAME_BASE = {
    'friction_wind': 'friction_velocity',
    'sshf': 'sensible_heat_flux',
    'gwd': 'gravity_wave_dissipation',
    'blh': 'boundary_layer_height',
    'temp_2m': 'temperature_2m',
    'relhum': 'relative_humidity',
    'sat_vap_pressure': 'saturated_vapor_pressure',
}


def readable_rename_map(n_turbines: int) -> dict:
    m = dict(READABLE_RENAME_BASE)
    for i in range(1, n_turbines + 1):
        m[f'blh_flag_t{i}'] = f'surface_layer_flag_t{i}'
    return m


def get_round2_params(config: dict) -> dict:
    r2 = dict(R2_DEFAULTS)
    user = config.get('round2', {}) or {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(r2.get(k), dict):
            r2[k] = {**r2[k], **v}
        else:
            r2[k] = v
    return r2


def load_correction_tables(r2: dict, station_id: str, park_id: str = None):
    """Per-location QM tables written by scripts/round2/wp2d_gate_table.py.
    Park-specific table (park_id.json) wins over the driving station's one.
    Returns (q_era5, q_target, branch) or (None, None, 'C') when absent."""
    path = None
    if park_id:
        cand = os.path.join(r2['artifacts_dir'], 'correction', f'{park_id}.json')
        if os.path.exists(cand):
            path = cand
    if path is None:
        path = os.path.join(r2['artifacts_dir'], 'correction', f'{station_id}.json')
    if not os.path.exists(path):
        return None, None, 'C'
    with open(path) as f:
        tab = json.load(f)
    branch = tab['branch'] if r2['correction_branch'] == 'auto' else r2['correction_branch']
    key = {'A': 'q_target_station', 'B': 'q_target_model'}.get(branch)
    if key is None or tab.get(key) is None:
        return None, None, 'C'
    return np.array(tab['q_era5']), np.array(tab[key]), branch


_BC_SF_CACHE = {}


def load_bc_factors(r2: dict, park_id: str) -> pd.Series:
    """Hourly era5_bc scaling factors for one park (Hook A'').
    Parquet from scripts/era5_bc/wp9_downscale_parks.py; park ids there carry
    a 'park_' prefix. Parquet is read once per process."""
    path = r2['bc_factors_parquet']
    if path not in _BC_SF_CACHE:
        sf = pd.read_parquet(path)
        sf['timestamp'] = pd.to_datetime(sf['timestamp'], utc=True)
        _BC_SF_CACHE[path] = sf.set_index(['park_id', 'timestamp'])['sf_pred']
    sf = _BC_SF_CACHE[path]
    key = f'park_{park_id}'
    assert key in sf.index.get_level_values(0), f'{key} missing in {path}'
    return sf.loc[key]


def load_z0(r2: dict, station_id: str) -> float:
    path = os.path.join(r2['artifacts_dir'], 'topo_features.csv')
    if os.path.exists(path):
        topo = pd.read_csv(path, dtype={'location_id': str})
        row = topo.loc[topo['location_id'] == station_id]
        if len(row) and pd.notna(row['z0'].iloc[0]):
            return float(row['z0'].iloc[0]) * r2['z0_scale']
    return r2['z0_default'] * r2['z0_scale']


def load_wake_factor(r2: dict, park_id: str, index: pd.DatetimeIndex) -> pd.Series:
    """Precomputed hourly wake loss factor w(t) in (0,1] (wp5_precompute_wakes)."""
    k = r2['wake']['k']
    path = os.path.join(r2['wake_dir'], f'w_{park_id}_k{k:.3f}.parquet')
    w = pd.read_parquet(path)['w']
    w = w.reindex(index).ffill().fillna(1.0).clip(1e-6, 1.0)
    return w

def get_park_params(station_id: str,
                        masterdata: pd.DataFrame,
                        params: dict,
                        commissioning_date: str) -> dict:
    specific_params = params.copy()
    # drop turbine information
    if 'rated' in specific_params: del specific_params['rated']
    del specific_params['turbines']
    del specific_params['hub_heights']
    del specific_params['h1']
    del specific_params['p_s_model']
    del specific_params['v2_method']
    del specific_params['karman']
    del specific_params['temp_gradient']
    del specific_params['noise']
    del specific_params['annual_degradation']
    del specific_params['apply_ageing']
    del specific_params['hourly_resolution']
    del specific_params['random_seed']
    del specific_params['nwp_heights']
    del specific_params['wind_speed_col_list']
    # from stations masterdata
    longitude = masterdata.loc[masterdata.station_id == station_id]['longitude'].iloc[0]
    latitude = masterdata.loc[masterdata.station_id == station_id]['latitude'].iloc[0]
    altitude = masterdata.loc[masterdata.station_id == station_id]['station_height'].iloc[0]
    specific_params['longitude'] = longitude
    specific_params['latitude'] = latitude
    specific_params['altitude'] = altitude
    specific_params['commissioning_date'] = commissioning_date
    return specific_params

def get_alpha(data: pd.DataFrame,
              features: dict,
              params: dict,
              hub_height: float,
              h1: float = 10,) -> pd.Series:
    alpha = pd.Series(np.nan, index=data.index)
    nwp_heights = params['nwp_heights']
    heights = params['wind_speed_col_list']
    #v1 = data[features['wind_speed']['name']]
    wind_speed_col_list = [features[f"wind_speed_{h}"]["name"] for h in heights]
    #for i in range(len(heights)-1):
    #    if hub_height <= nwp_heights[i+1] and hub_height >= nwp_heights[i]:
    v1 = data[wind_speed_col_list[0]]
    v2 = data[wind_speed_col_list[1]]
    h1 = nwp_heights[0]
    h2 = nwp_heights[1]
    # break
    mask = (v1 > 0) & (v2 > 0)
    alpha = np.log(v2[mask] / v1[mask]) / np.log(h2 / h1)
    alpha.fillna(0, inplace=True)
    alpha[alpha < 0] = 0
    alpha[alpha > 0.4] = 0.4
    return alpha


def calculate_wind_speed_from_components(u: pd.Series, v: pd.Series) -> pd.Series:
    """
    Calculate wind speed magnitude from u and v components.

    Parameters:
    -----------
    u : pd.Series
        Wind speed u-component (m/s)
    v : pd.Series
        Wind speed v-component (m/s)

    Returns:
    --------
    wind_speed : pd.Series
        Wind speed magnitude (m/s)
    """
    return np.sqrt(u**2 + v**2)


def get_windspeed_at_height(data: pd.DataFrame,
                            h2: float,
                            suffix: str = ''
                            ):
    """
    Calculate the wind speed at height h2 using wind speeds at 10m and 100m.

    This function computes alpha from the two available wind speed measurements
    (10m and 100m) and then extrapolates to the desired height h2.

    Parameters:
    -----------
    data : pd.DataFrame
        Must contain columns 'wind_speed_10m' and 'wind_speed_100m'
    h2 : float
        Target height in meters
    suffix : str
        Suffix for alpha column name (optional)

    Returns:
    --------
    v2 : pd.Series
        The calculated wind speed at height h2.
    """
    # Hardcoded column names and heights
    v_10m = data['wind_speed_10m']
    v_100m = data['wind_speed_100m']
    h1 = 10  # meters
    h2_ref = 100  # meters

    # Calculate alpha from the two wind speeds
    if 'alpha' not in data.columns:
        mask = (v_10m > 0) & (v_100m > 0)
        alpha = pd.Series(np.nan, index=data.index)
        alpha[mask] = np.log(v_100m[mask] / v_10m[mask]) / np.log(h2_ref / h1)
        alpha.fillna(0, inplace=True)
        alpha[alpha < 0] = 0
        alpha[alpha > 0.4] = 0.4

    # Store alpha if suffix is provided
    if suffix:
        data[f'alpha{suffix}'] = alpha

    # Calculate wind speed at desired height
    if h2 < 0:
        h2 = h1
    v2 = v_10m * (h2 / h1) ** alpha

    return v2


def get_temperature_at_height(data: pd.DataFrame,
                              params: dict,
                              h2: float) -> pd.Series:
    """Temperature extrapolation using hardcoded column name 'temp_2m'.
    temp_2m is already in Kelvin from ERA5 data."""
    t1 = data['temp_2m'].copy()  # Already in Kelvin
    h1 = 2 # in meters
    temp_gradient = params['temp_gradient']
    delta_h = h2 - h1
    t2 = t1 - temp_gradient * delta_h
    return t2  # Return in Kelvin


def get_pressure_at_height(data: pd.DataFrame,
                           params: dict,
                           h1: float,
                           h2: float) -> pd.Series:
    """Pressure extrapolation using hardcoded column names.
    pressure and temp_2m are already in Pascal and Kelvin from ERA5 data."""
    R = 8.31451
    M_air = 0.028949 # dry air
    g = 9.81
    p1 = data['pressure'].copy()  # Already in Pascal
    t1 = data['temp_2m'].copy()  # Already in Kelvin
    temp_gradient = params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    p2 = p1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) )
    return p2

def get_density_at_height(data: pd.DataFrame,
                          params: dict,
                          h2: float) -> pd.Series:
    """Density extrapolation using hardcoded column name 'density'.
    temp_2m is already in Kelvin from ERA5 data."""
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water
    g = 9.81
    rho1 = data['density'].copy()
    t1 = data['temp_2m'].copy()  # Already in Kelvin
    h1 = 2 # because of temperature measured at 2 m
    temp_gradient = params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    rho2 = rho1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) - 1)
    return rho2

def merge_curve(data: pd.DataFrame,
                curve: pd.Series,
                features: dict,
                suffix: str = '_hub') -> pd.Series:
    wind_speed_hub_col = f'{features["wind_speed_hub"]['name']}{suffix}'
    wind = data[wind_speed_hub_col].to_frame()
    values = pd.merge(wind, curve,left_on=wind_speed_hub_col, right_on='wind_speed', how="left")
    values.index = wind.index
    return values[curve.name]

def merge_curve_hardcoded(data: pd.DataFrame,
                curve: pd.Series,
                wind_speed_hub_col: str) -> pd.Series:
    """Merge power curve with wind speed data using hardcoded column name."""
    wind = data[wind_speed_hub_col].to_frame()
    values = pd.merge(wind, curve, left_on=wind_speed_hub_col, right_on='wind_speed', how="left")
    values.index = wind.index
    return values[curve.name]

def interpolate(power_curve: pd.Series, cut_out: float):
    ticks = np.arange(0, cut_out*100, 1) / 100.0
    rated_power = power_curve.max()
    interpol = pd.Series(index=ticks, dtype=float)
    interpol.index.name = 'wind_speed'
    interpol = interpol.to_frame().merge(power_curve.to_frame(), how='left', left_index=True, right_index=True)
    interpol.drop(columns=[0], inplace=True)
    interpol.iloc[-1] = rated_power
    interpol = interpol.interpolate(method='polynomial', order=3)
    interpol = interpol.clip(upper=rated_power, lower=0)
    interpol.fillna(0, inplace=True)
    return pd.Series(interpol.iloc[:,-1], index=interpol.index)


def get_cp_from_power_curve(data: pd.DataFrame,
                            power_curve: pd.Series,
                            features: dict,
                            rotor_diameter: float,
                            degradation_vector: np.ndarray = None,
                            suffix: str = '_hub') -> pd.Series:
    wind_speed_hub_col = f'{features["wind_speed_hub"]['name']}{suffix}'
    rho_hub_col = f'{features['density_hub']['name']}{suffix}'
    wind_speed = data[wind_speed_hub_col]
    rho = data[rho_hub_col]
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    if degradation_vector is None:
        degradation_vector = np.ones(len(data))
    degradation_vector = pd.Series(degradation_vector, index=wind_speed.index)
    cp_values = []
    for t in wind_speed.index:
        v = wind_speed.loc[t]
        if v == 0:
            cp_values.append(0)
            continue
        dr = degradation_vector.loc[t]
        rho_t = 1.225 # standard air density for stable cp values # rho.loc[t]
        degraded_ws = power_curve.index * (1/dr)**(1/3)
        degraded_power_curve = pd.Series(power_curve.values, index=degraded_ws).sort_index()
        power = np.interp(v, degraded_power_curve.index, degraded_power_curve.values)
        cp = power / (0.5 * rho_t * rotor_area * v**3)
        cp_values.append(cp)
    Cp = pd.Series(cp_values, index=wind_speed.index, name="Cp").clip(lower=0, upper=1)
    return Cp

def get_cp_from_power_curve_hardcoded(data: pd.DataFrame,
                            power_curve: pd.Series,
                            wind_speed_hub_col: str,
                            density_hub_col: str,
                            rotor_diameter: float,
                            degradation_vector: np.ndarray = None,
                            density_mode: str = 'v1_mixed') -> pd.Series:
    """Calculate Cp from power curve using hardcoded column names.

    Hook D: density_mode 'dynamic' uses the extrapolated hub density here;
    'v1_mixed' (round-1 behaviour) and 'static_1225' keep the 1.225 constant.
    """
    wind_speed = data[wind_speed_hub_col]
    rho = data[density_hub_col]
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    if degradation_vector is None:
        degradation_vector = np.ones(len(data))
    degradation_vector = pd.Series(degradation_vector, index=wind_speed.index)
    cp_values = []
    for t in wind_speed.index:
        v = wind_speed.loc[t]
        if v == 0:
            cp_values.append(0)
            continue
        dr = degradation_vector.loc[t]
        if density_mode == 'dynamic':
            rho_t = rho.loc[t]
        else:
            rho_t = 1.225 # standard air density for stable cp values # rho.loc[t]
        degraded_ws = power_curve.index * (1/dr)**(1/3)
        degraded_power_curve = pd.Series(power_curve.values, index=degraded_ws).sort_index()
        power = np.interp(v, degraded_power_curve.index, degraded_power_curve.values)
        cp = power / (0.5 * rho_t * rotor_area * v**3)
        cp_values.append(cp)
    Cp = pd.Series(cp_values, index=wind_speed.index, name="Cp").clip(lower=0, upper=1)
    return Cp

def get_saturated_vapor_pressure(temperature: pd.Series,
                                 model: str = 'improved_magnus') -> pd.Series:
    def huang(temp):
        return np.where(
            temp > 0,
            np.exp(34.494 - (4924.99 / (temp + 237.1))) / (temp + 105) ** 1.57,
            np.exp(43.494 - (6545.8 / (temp + 278))) / (temp + 868) ** 2
        )
    def improved_magnus(temp):
        return np.where(
            temp > 0,
            610.94 * np.exp((17.625 * temp) / (temp + 243.04)),
            611.21 * np.exp((22.587 * temp) / (temp + 273.86))
        )
    model_functions = {
        'huang': huang,
        'improved_magnus': improved_magnus,
    }
    if model not in model_functions:
        raise ValueError(f"Unknown model: {model}")
    return model_functions[model](temperature)

def get_rho(data: pd.DataFrame) -> pd.Series:
    """Calculate air density using hardcoded column names.
    pressure and temp_2m are already in Pascal and Kelvin from ERA5 data."""
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    air_pressure = data['pressure'].copy()  # Already in Pascal
    temperature_kelvin = data['temp_2m'].copy()  # Already in Kelvin
    relhum = data['relhum'].copy()
    p_s = data['sat_vap_pressure'].copy()
    # check if relative humidity is in the range between 0 and 1
    if relhum.max() > 1:
        relhum /= 100
    p_w = relhum * p_s
    p_g = air_pressure - p_w
    rho_g = p_g / (R_dry * temperature_kelvin)
    rho_w = p_w / (R_w * temperature_kelvin)
    rho = rho_g + rho_w
    return rho


def read_dfs(path: str,
             features: list,
             features_dict: dict,
             drop_threshold: float = 1,
             v2_method: str = 'alphaI',
             masterdata: pd.DataFrame = None,
             hourly_resolution: bool = True,
             specific_id: str = None,
             nwp_data_path: str = None,
             wind_speed_heights: list = None) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in tqdm(os.listdir(path), desc='Reading DataFrames'):
        station_id = file.split('.csv')[0].split('_')[1]
        skip_station = False
        if specific_id and (station_id != specific_id):
            continue
        try:
            data = pd.read_csv(os.path.join(path, file), delimiter= ",")
        except Exception as e:
            logging.warning(f"Error reading file {file}: {e}")
            return [], []
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        data = tools.knn_imputer(data=data, n_neighbors=5)
        dfs.append(data)
        station_ids.append(station_id)
        if specific_id:
            break
    return dfs, station_ids


def get_turbines(turbine_path: str,
                 cp_path: str,
                 specs_path: str,
                 params: dict):
    turbines = params['turbines']
    # read power curves
    power_curves = pd.read_csv(turbine_path, sep=";", decimal=",")
    power_curves.set_index('wind_speed', inplace=True)
    power_curves = power_curves[turbines] * 1000 # kW -> W
    if power_curves.columns.duplicated().any():
        power_curves = power_curves.loc[:, ~power_curves.columns.duplicated()]
    # read cp curves
    cp_curves = pd.read_csv(cp_path, sep=";", decimal=".")
    cp_curves.set_index('wind_speed', inplace=True)
    #cp_curves = cp_curves[turbines]
    # read turbine specs
    turbine_specs = pd.read_csv(specs_path, sep=';')
    specs = {}
    for height, turbine in enumerate(turbines):
        specs[turbine] = {
            'diameter': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Rotordurchmesser'].values[0]),
            #'height': params['hub_heights'][height],
            'cut_in': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Einschaltgeschwindigkeit'].values[0]),
            'cut_out': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Abschaltgeschwindigkeit'].values[0]),
            'rated': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Nennwindgeschwindigkeit'].values[0])
        }
    return power_curves, cp_curves, specs

def get_ageing_degradation(
        time_vector: pd.DatetimeIndex,
        mean_age_years: float = 15.0,
        std_dev_age_years: float = 5.0,
        annual_load_factor_loss_rate: float = 0.0063,
        real_ages: np.ndarray = None,
        commissioning_date: str = None,
        random_seed: int = 42) -> (np.ndarray, str):
    np.random.seed(random_seed)
    if real_ages is None:
        start_age = np.random.normal(loc=mean_age_years, scale=std_dev_age_years)
    else:
        start_age = float(np.random.choice(real_ages, size=1, replace=False)[0])
    start_age = max(0.0, start_age)
    if commissioning_date is not None:
        start_age = (time_vector[0] - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
    else:
        commissioning_date = str((time_vector[0] - pd.Timedelta(days=start_age*365.25)).date())
    end_age = (time_vector[-1] - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
    base_efficiency = 1.0 - annual_load_factor_loss_rate
    efficiency_factor_start = base_efficiency ** start_age
    efficiency_factor_end = base_efficiency ** end_age
    efficiency_vector = np.linspace(efficiency_factor_start, efficiency_factor_end, num=len(time_vector))
    return efficiency_vector, commissioning_date

def apply_rotor_inertia(power: np.ndarray,
                        wind_speed: pd.Series,
                        cut_in: float,
                        alpha: float = 1.0,
                        tau: float = 2.0,
                        min_tau: float = 0.2) -> np.ndarray:
    adjusted_power = power.copy()
    tail_power = 0.0
    if wind_speed.index is not None and len(wind_speed.index) > 1:
        dt_hours = (wind_speed.index[1] - wind_speed.index[0]) / np.timedelta64(1, 'h')
    else:
        dt_hours = 1.0
    decay_factor = np.exp(-dt_hours / tau)
    for i in range(1, len(power)):
        if power[i] < power[i-1]:
            delta_v = cut_in - wind_speed.iloc[i]
            tau_eff = max(min_tau, tau * (1 + alpha * delta_v))
            decay_factor = np.exp(-dt_hours / tau_eff)
            tail_power *= decay_factor
            adjusted_power[i] = tail_power
        else:
            tail_power = adjusted_power[i]
    return adjusted_power

def apply_noise(wind_power: pd.Series,
                rated_power: float,
                noise: float = 0.05,
                random_seed: int = 42) -> np.ndarray:
    np.random.seed(random_seed)
    noise_factor = np.random.normal(loc=1, scale=noise, size=len(wind_power))
    noisy_power = wind_power * noise_factor
    noisy_power = np.maximum(0, noisy_power)
    noisy_power = np.minimum(rated_power, noisy_power)
    return noisy_power

def get_features(data: pd.DataFrame,
                 params: dict,
                 hub_height: float,
                 suffix: str = '_hub',
                 r2: dict = None,
                 station_ctx: dict = None) -> pd.DataFrame:
    """
    Calculate all features needed for wind power generation.
    Uses hardcoded ERA5 column names and computes:
    - Wind speeds at 10m and 100m from u/v components
    - Relative humidity from dew point and temperature
    - Air density
    - Extrapolated values at hub height
    """
    # Hardcoded column names
    wind_speed_hub_col = f"wind_speed{suffix}"
    temperature_hub_col = f"temp_2m{suffix}"
    sat_vap_ps_hub_col = f"sat_vap_pressure{suffix}"
    density_hub_col = f"density{suffix}"

    # Step 1: Calculate wind speeds from u and v components
    data['wind_speed_10m'] = calculate_wind_speed_from_components(
        data['u_wind_10m'], data['v_wind_10m']
    )
    data['wind_speed_100m'] = calculate_wind_speed_from_components(
        data['u_wind_100m'], data['v_wind_100m']
    )

    # Hook A'' (era5_bc): hourly TR-BC scaling factor instead of the gated QM.
    # tr_bc = height-consistent; tr_bc10 = 10 m level only (profile re-anchor,
    # correction fades with height). Same insertion point as Hook A.
    if r2 is not None and r2['correction'] in ('tr_bc', 'tr_bc10') \
            and station_ctx and station_ctx.get('bc_sf') is not None:
        data = r2_correction.apply_bc_factor(
            data, station_ctx['bc_sf'],
            mode='wind10_only' if r2['correction'] == 'tr_bc10'
            else 'height_consistent')

    # Hook A (WP2): gated quantile-mapping correction, height-consistent by
    # default. Idempotent because steps above recompute v10/v100 from u/v.
    elif r2 is not None and r2['correction'] != 'off' and station_ctx \
            and station_ctx.get('q_target') is not None:
        data = r2_correction.apply_gated_correction(
            data, station_ctx['q_era5'], station_ctx['q_target'],
            mode=r2['correction'])

    # Hook F (WP6): wind-level scaler on both levels
    if r2 is not None and r2.get('wind_level_factor', 1.0) != 1.0:
        data['wind_speed_10m'] = data['wind_speed_10m'] * r2['wind_level_factor']
        data['wind_speed_100m'] = data['wind_speed_100m'] * r2['wind_level_factor']

    # Step 2: Calculate actual vapor pressure from dew point using Huang formula
    # dew_point_2m is in Kelvin from ERA5, convert to Celsius for Huang formula
    e = get_saturated_vapor_pressure(
        temperature=data['dew_point_2m'] - 273.15,
        model=params['p_s_model']
    )

    # Step 3: Calculate saturation vapor pressure from temperature using Huang formula
    # temp_2m is in Kelvin from ERA5, convert to Celsius for Huang formula
    e_s = get_saturated_vapor_pressure(
        temperature=data['temp_2m'] - 273.15,
        model=params['p_s_model']
    )

    # Step 4: Calculate relative humidity
    data['relhum'] = e / e_s

    # Step 5: Store saturation vapor pressure
    data['sat_vap_pressure'] = e_s

    # Step 6: Calculate air density at 2m height
    data['density'] = get_rho(data=data)

    # Step 7: Extrapolate wind speed to hub height
    # Hook B (WP3): MOST stability-corrected profile as alternative shear.
    # Ask 31 (exploratory): 'validity_gated' switches hourly to the dynamic-
    # alpha power law where the surface-layer criterion is violated
    # (hub > 0.1*blh, Stull); 'validity_blend' soft-blends both with a
    # logistic weight on the validity margin (K_VG_BLEND fixed below).
    if r2 is not None and r2['shear'] in ('most', 'validity_gated',
                                          'validity_blend'):
        z0 = station_ctx['z0'] if station_ctx else r2['z0_default']
        if 'obukhov_L' not in data.columns:
            data['obukhov_L'] = r2_stability.obukhov_length(
                u_star=data['friction_wind'].values,
                rho=data['density'].values,
                temp_k=data['temp_2m'].values,
                sshf=data['sshf'].values)
            data['stability_class'] = r2_stability.stability_class(
                data['obukhov_L'].values)
        v_most = r2_stability.most_wind_profile(
            data['wind_speed_100m'].values, hub_height, z0,
            data['obukhov_L'].values)
        if r2['shear'] == 'most':
            data[wind_speed_hub_col] = np.round(v_most, 2)
        else:
            v_pl = get_windspeed_at_height(
                data=data, h2=hub_height, suffix=suffix).values
            blh = data['blh'].values
            if r2['shear'] == 'validity_gated':
                invalid = r2_stability.blh_flag(hub_height, blh)
                w_most = np.where(invalid, 0.0, 1.0)
            else:  # validity_blend
                K_VG_BLEND = 3.0  # w=0.5 at hub == 0.1*blh; ~0.95 at 2.7x margin
                with np.errstate(divide='ignore'):
                    margin = np.log(np.clip(0.1 * blh, 1e-6, None) / hub_height)
                w_most = 1.0 / (1.0 + np.exp(-K_VG_BLEND * margin))
            data[wind_speed_hub_col] = np.round(
                w_most * v_most + (1.0 - w_most) * v_pl, 2)
            data[f'w_most{suffix}'] = w_most
        if 'blh' in data.columns:
            data[f'blh_flag{suffix}'] = r2_stability.blh_flag(
                hub_height, data['blh'].values)
    else:
        data[wind_speed_hub_col] = round(
            get_windspeed_at_height(data=data, h2=hub_height, suffix=suffix), 2
        )

    # Step 8: Extrapolate temperature to hub height
    data[temperature_hub_col] = get_temperature_at_height(
        data=data, params=params, h2=hub_height
    )

    # Step 9: Calculate saturation vapor pressure at hub height
    temperature_hub = data[temperature_hub_col]
    data[sat_vap_ps_hub_col] = get_saturated_vapor_pressure(
        temperature=temperature_hub,
        model=params['p_s_model']
    )

    # Step 10: Extrapolate density to hub height
    data[density_hub_col] = get_density_at_height(
        data=data, params=params, h2=hub_height
    )

    return data

def generate_wind_power(data: pd.DataFrame,
                        params: dict,
                        rated_power: float,
                        Cp: pd.Series,
                        rotor_diameter: float,
                        cut_in: float,
                        cut_out: float,
                        rated_speed: float,
                        degradation_vector: np.ndarray = None,
                        suffix: str = '_hub',
                        density_mode: str = 'v1_mixed') -> pd.Series:
    """Generate wind power using hardcoded column names.

    Hook D: 'v1_mixed' (round 1) and 'dynamic' use the extrapolated hub
    density in the power equation; 'static_1225' forces 1.225 everywhere.
    """
    wind_speed_hub_col = f"wind_speed{suffix}"
    density_hub_col = f"density{suffix}"
    if density_mode == 'static_1225':
        rho = pd.Series(1.225, index=data.index)
    else:
        rho = data[density_hub_col]
    wind_speed_hub = data[wind_speed_hub_col]
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    if degradation_vector is None:
        degradation_vector = np.ones(len(data))
    else:
        degradation_vector = degradation_vector.copy()
        rated_speed_vector = rated_speed * (1/degradation_vector)**(1/3)
        degradation_vector[wind_speed_hub >= rated_speed_vector] = 1.0
    wind_power = np.where(
        wind_speed_hub < cut_in, 0,
        np.where(
            wind_speed_hub <= cut_out,
            0.5 * rho * rotor_area * Cp * wind_speed_hub ** 3,
            #np.minimum(rated_power, 0.5 * rho * rotor_area * Cp * wind_speed_hub ** 3 * degradation_vector),
            0
        )
    )
    #power = apply_rotor_inertia(power=wind_power,
    #                            wind_speed=wind_speed_hub,
    #                            cut_in=cut_in)
    wind_power = np.minimum(wind_power, rated_power)
    power = apply_noise(wind_power=wind_power,
                        rated_power=rated_power,
                        noise=params['noise'])
    return power

def gen_full_dataframe(power_curves: pd.DataFrame,
                       turbine: str,
                       params: dict,
                       df: pd.DataFrame,
                       hub_height: float,
                       specs: dict,
                       rated_power = None,
                       degradation_vector: np.ndarray = None,
                       suffix_for_turbine_cols: str = '',
                       r2: dict = None,
                       station_ctx: dict = None) -> pd.DataFrame:
    """Generate full dataframe with wind power using hardcoded column names."""
    rotor_diameter = specs[turbine]['diameter']
    #hub_height = specs[turbine]['height']
    cut_in = specs[turbine]['cut_in']
    cut_out = specs[turbine]['cut_out']
    rated_speed = specs[turbine]['rated']
    density_mode = r2['density'] if r2 else 'v1_mixed'

    # Calculate all features (wind speeds, humidity, density, etc.)
    df = get_features(data=df,
            params=params,
            hub_height=hub_height,
            suffix=suffix_for_turbine_cols,
            r2=r2,
            station_ctx=station_ctx)

    power_curve = interpolate(power_curve=power_curves[turbine], # filter
                              cut_out=cut_out)
    # Hook F (WP6): manufacturer-tolerance scaler on the power curve
    if r2 is not None and r2.get('power_curve_scale', 1.0) != 1.0:
        power_curve = power_curve * r2['power_curve_scale']
    if rated_power == None:
        rated_power = power_curve.max()

    # Calculate Cp using hardcoded column names
    wind_speed_hub_col = f"wind_speed{suffix_for_turbine_cols}"
    density_hub_col = f"density{suffix_for_turbine_cols}"
    Cp = get_cp_from_power_curve_hardcoded(data=df,
                                power_curve=power_curve,
                                wind_speed_hub_col=wind_speed_hub_col,
                                density_hub_col=density_hub_col,
                                rotor_diameter=rotor_diameter,
                                degradation_vector=degradation_vector,
                                density_mode=density_mode)
    df[f'Cp{suffix_for_turbine_cols}'] = Cp

    # Merge power curve
    power_curve_merged = merge_curve_hardcoded(data=df,
                            curve=power_curve,
                            wind_speed_hub_col=wind_speed_hub_col)

    # Generate wind power
    power = generate_wind_power(data=df,
                            params=params,
                            rated_power=rated_power,
                            Cp=Cp,
                            rotor_diameter=rotor_diameter,
                            cut_in=cut_in,
                            cut_out=cut_out,
                            rated_speed=rated_speed,
                            degradation_vector=degradation_vector,
                            suffix=suffix_for_turbine_cols,
                            density_mode=density_mode)
    df[f"power{suffix_for_turbine_cols}"] = power
    #drop_columns = ['wind_speed_hub', 'temperature_hub', 'density_hub', 'sat_vap_pressure_hub']
    #df.drop(drop_columns, axis=1, inplace=True)
    return df


DEFAULT_CONFIG = 'round2/M4.yaml'  # framework default: dyn. power law, gated
                                    # correction, Weibull aging, wakes on (deployed)


def main(config_file: str = None) -> None:

    parser = argparse.ArgumentParser(description="Synthetic Wind Power Time Series Simulation")
    parser.add_argument('-p', '--park_id', type=str, default='', help='Select park_id (default: None)')
    parser.add_argument('-c', '--config', type=str, default='',
                        help=f'Config path under configs/ (default: {DEFAULT_CONFIG})')
    args = parser.parse_args()

    if not config_file:
        config_file = args.config or DEFAULT_CONFIG

    # supports nested paths like 'round2/M4.yaml' or 'round2/M2/config_07374.yaml'
    base = os.path.basename(config_file).split('.')[0]
    args.park_id = base[7:] if base.startswith('config_') else ''

    if args.park_id == '':
        config_suffix = '_wind'
        park_id = None
        dwd_station_id = None
    else:
        park_id = args.park_id
        config_suffix = '_wind'#f'_{park_id}'
        dwd_station_id = str(park_id[:5])

    if not config_file:
        config = tools.load_config(f"configs/config{config_suffix}.yaml")
    else:
        config = tools.load_config(f"configs/{config_file}")
    db_config = config['write']['db_conf']
    features = config['features']
    params = config['params']
    params['wind_speed_col_list'] = [10,80,120,180]
    r2 = get_round2_params(config)
    if r2['apply_ageing_override'] is not None:
        params['apply_ageing'] = bool(r2['apply_ageing_override'])

    raw_dir = os.path.join(config['data']['synth_dir'], 'raw', 'wind')
    ageing_flag = 'noage'
    if params['apply_ageing'] and r2['aging_model'] != 'none':
        ageing_flag = 'age'
    if params['hourly_resolution']:
        resolution = 'hourly'
    else:
        resolution = '10min'
    synth_dir = os.path.join(config['data']['synth_dir'],
                             'wind', 'round2',
                             str(r2['experiment_id']))
    # delete dir
    #shutil.rmtree(synth_dir)

    os.makedirs(synth_dir, exist_ok=True)
    w_vert_dir = config['data']['w_vert_dir']
    era5_dir = r2['era5_dir_v2'] or config['data']['era5_dir']
    if not (os.path.isdir(era5_dir) and os.listdir(era5_dir)):
        logging.warning(f"era5_dir_v2 {era5_dir} missing/empty, "
                        f"falling back to {config['data']['era5_dir']}")
        era5_dir = config['data']['era5_dir']
    if r2['shear'] == 'most':
        probe = os.path.join(era5_dir, sorted(os.listdir(era5_dir))[0])
        cols = pd.read_csv(probe, nrows=0).columns
        assert 'sshf' in cols, ("shear='most' needs sshf/blh columns — run "
                                "scripts/round2/wp0_extract_points.py first")
    turbine_dir = config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)
    wind_ages_path = config['data']['wind_ages']
    wind_ages = np.load(wind_ages_path)
    params['random_seed'] = params.get('random_seed', 42)

    commissioning_date = params.get('commissioning_date', None)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        #datefmt=datefmt,
        handlers=[
            #logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
    )

    _, wind_features = clean_data.relevant_features(features=features)

    try:
        masterdata = tools.get_master_data(db_config)
    except Exception as e:
        logging.warning(f"DB masterdata unavailable ({e}), "
                        f"falling back to data/master_data.csv")
        masterdata = pd.read_csv('data/master_data.csv', dtype={'station_id': str})

    # load here comm_dates
    comm_dates = pd.read_csv('data/comm_dates.csv', sep=";", decimal=".", dtype={'park_id': str})
    #comm_dates.set_index('park_id', inplace=True)

    frames, station_ids = read_dfs(path=era5_dir,
                                   features=wind_features,
                                   masterdata=masterdata,
                                   features_dict=features,
                                   drop_threshold=1,
                                   v2_method=params['v2_method'],
                                   hourly_resolution=params['hourly_resolution'],
                                   specific_id=dwd_station_id,
                                   nwp_data_path=config['data']['nwp_data_path'],
                                   wind_speed_heights=params['wind_speed_col_list'])

    if frames == []:
        logging.warning(f"No data found for station {dwd_station_id}")
        return

    power_curves, _, specs = get_turbines(
        turbine_path=turbine_path,
        cp_path=cp_path,
        specs_path=specs_path,
        params=params
    )
    power_master = {}
    logging.info(f'Starting wind power generation. Ageing: {params["apply_ageing"]}, Hourly resolution: {params['hourly_resolution']}')
    turbine_master = defaultdict(dict)
    for station_id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        #tqdm.write(f'Processing station {str(station_id)}')
        if not args.park_id:
            park_id = station_id
        logging.debug(f'Processing station {str(station_id)}')
        df = frame.copy()
        # commissioning date priority (Ask 12): authoritative round2 table
        # (if configured) > comm_dates.csv > park config. Without the table
        # key the v1 behaviour is preserved (v1==v2 regression test).
        commissioning_date = None
        if r2['commissioning_table'] and os.path.exists(r2['commissioning_table']):
            auth = pd.read_csv(r2['commissioning_table'], dtype={'park_id': str})
            hit = auth[auth['park_id'] == park_id]['commissioning_date'].values
            if len(hit):
                commissioning_date = hit[0]
        if commissioning_date is None:
            cd_hits = comm_dates[comm_dates['park_id'] == park_id]['commissioning_date'].values
            if len(cd_hits):
                commissioning_date = cd_hits[0]
            else:
                commissioning_date = params.get('commissioning_date', None)
                logging.info(f"park {park_id} not in comm_dates.csv, using config "
                             f"commissioning_date={commissioning_date}")
        # Hook C (WP4): switchable degradation schedule; 'const' == round 1
        degradation_vector, commissioning_date = r2_aging.get_degradation_vector(
            time_vector=df.index,
            model=r2['aging_model'] if r2['aging_model'] != 'none' else 'const',
            real_ages=wind_ages,
            commissioning_date=commissioning_date,
            random_seed=params['random_seed'],
            lam=r2['aging']['lambda'],
            kappa=r2['aging']['kappa'],
            step_delta=r2['aging']['step_delta'])
        if not params['apply_ageing'] or r2['aging_model'] == 'none':
            degradation_vector = None
            commissioning_date = '-'
        # Hook A context: correction tables + z0 for this location
        q_era5, q_target, branch = load_correction_tables(r2, station_id, park_id)
        station_ctx = {'q_era5': q_era5, 'q_target': q_target,
                       'branch': branch, 'z0': load_z0(r2, station_id)}
        if r2['correction'] in ('tr_bc', 'tr_bc10'):
            station_ctx['bc_sf'] = load_bc_factors(r2, park_id)
            station_ctx['branch'] = 'BC'
        specific_params = get_park_params(station_id=station_id,
                                          masterdata=masterdata,
                                          params=params,
                                          commissioning_date=commissioning_date)
        power_master[park_id] = specific_params
        for turbine_id, turbine in enumerate(params['turbines'], start=1):
            hub_height = params['hub_heights'][turbine_id-1]
            rated_power = None
            if 'rated' in params:
                rated_power = params['rated'][turbine_id-1]*1000 # * 1000: kW -> W
            df = gen_full_dataframe(
                    power_curves=power_curves,
                    turbine=turbine,
                    params=params,
                    df=df,
                    hub_height=hub_height,
                    rated_power=rated_power, # only needed when to curtail rated power
                    specs=specs,
                    degradation_vector=degradation_vector,
                    suffix_for_turbine_cols=f'_t{turbine_id}',
                    r2=r2,
                    station_ctx=station_ctx
            )
            turbine_master[f't{turbine_id}']['diameter'] = specs[turbine]['diameter']
            turbine_master[f't{turbine_id}']['hub_height'] = hub_height
            turbine_master[f't{turbine_id}']['cut_in'] = specs[turbine]['cut_in']
            turbine_master[f't{turbine_id}']['cut_out'] = specs[turbine]['cut_out']
            turbine_master[f't{turbine_id}']['rated'] = specs[turbine]['rated']
            turbine_master[f't{turbine_id}']['turbine_name'] = turbine
            turbine_master[f't{turbine_id}']['park_id'] = park_id
        params['random_seed'] += 1
        # Hook E (WP5): park sum and precomputed wake loss factor
        turbine_power_cols = [f'power_t{i}' for i in range(1, len(params['turbines']) + 1)]
        df['power_park_free'] = df[turbine_power_cols].sum(axis=1)
        if r2['wake']['enabled']:
            w = load_wake_factor(r2, park_id, df.index)
            df['wake_factor'] = w
            df['power_park'] = df['power_park_free'] * w
        else:
            df['power_park'] = df['power_park_free']
        cols_to_drop = [col for col in df.columns if 'alpha' in col or 'sat_vap_pressure_t' in col or 'temp_2m_t' in col or 'Cp' in col or 'wind_speed_10m' in col]
        df.drop(columns=cols_to_drop, inplace=True)
        df = df[r2['output_start']:]
        # Ask 35: emit human-readable column names for the published dataset
        # (opt-in; the park ladder keeps native names via output_naming='native')
        if r2.get('output_naming') == 'readable':
            df = df.rename(columns=readable_rename_map(len(params['turbines'])))
        df.to_csv(os.path.join(synth_dir, f'synth_{park_id}.csv'), sep=";", decimal=".")
        # experiment manifest for reproducibility
        try:
            git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True).stdout.strip()
        except Exception:
            git_commit = 'unknown'
        manifest = {'experiment_id': r2['experiment_id'], 'park_id': park_id,
                    'round2': {k: v for k, v in r2.items()},
                    'branch': station_ctx['branch'], 'z0': station_ctx['z0'],
                    'random_seed': params['random_seed'] - 1,
                    'git_commit': git_commit,
                    'era5_dir': era5_dir}
        with open(os.path.join(synth_dir, f'manifest_{park_id}.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        commissioning_date = None
        break
    # Save the park parameters
    wind_parameter_path = os.path.join(synth_dir, 'wind_parameter.csv')
    write_header = not os.path.exists(wind_parameter_path) #and not park_id
    df_power_master = pd.DataFrame.from_dict(power_master, orient='index')
    df_power_master.index.name = 'park_id'
    # check if the entries are already in the file if it exists
    df_power_master.to_csv(wind_parameter_path, sep=";", decimal=".", mode='a', header=write_header)
    # drop duplicates
    df_power_master = pd.read_csv(wind_parameter_path, sep=";", decimal=".", dtype={'park_id': str})
    df_power_master.drop_duplicates(inplace=True)
    df_power_master.to_csv(wind_parameter_path, sep=";", decimal=".", index=False)
    # save the turbine parameters
    turbine_master_path = os.path.join(synth_dir, 'turbine_parameter.csv')
    write_header = not os.path.exists(turbine_master_path) #and not park_id
    df_turbine_master = pd.DataFrame.from_dict(turbine_master, orient='index')
    df_turbine_master.index.name = 'turbine'
    df_turbine_master.to_csv(turbine_master_path, sep=";", decimal=".", mode='a', header=write_header)
    # drop duplicates
    df_turbine_master = pd.read_csv(turbine_master_path, sep=";", decimal=".", dtype={'park_id': str})
    df_turbine_master.drop_duplicates(inplace=True)
    df_turbine_master.to_csv(turbine_master_path, sep=";", decimal=".", index=False)

if __name__ == "__main__":
    #config_files = os.listdir('configs/')
    # for config_name in config_files:
    #     if config_name != 'config_07374.yaml': # only for testing
    #         continue
    #     print(f'Config: {config_name}')
    #     main(config_name)
    main('config_wind.yaml')

