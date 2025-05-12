import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import clean_data
import utils

def get_windspeed_at_height(data: pd.DataFrame,
                            params: dict,
                            adj_params: dict
                            ):
    """
    Calculate the wind speed (v2) using different methods.

    Parameters:
    -----------
    data : pd.DataFrame
    params: dict
        Dictionary with parameters names
        params['v_wind']['param']: Wind Speed
        params['d_wind']['param']: Wind Direction
        params['w_wind']['param']: Vertical Wind Speed
        params['sigma_wind_lon']['param']: Standard Deviation longitudinal wind speed
    adj_params : dict
        Dictionary with neccesary parameters
        adj_params['h1']: Height of measurements in m
        adj_params['h2']: Hub height in m
        adj_params['karman']: von Karmans constant
        adj_params['method']: Height of measurement in m
            The method used to calculate v2. Options are:
            - 'alphaI': Uses linear relationship between turbulence intensity and alpha (Ishizaki, 1983)
            - 'seven_power': A default method using the 1/7 power law.
            If no method is provided, the default 'seven_power' is used.
    Returns:
    --------
    v2 : float
        The calculated wind speed at height h2.
    """
    v1 = data[params['v_wind']['param']]
    method = adj_params['v2_method']
    h1 = adj_params['h1']
    h2 = adj_params['hub_height']
    if method == 'alphaI':
        direction = data[params['d_wind']['param']]
        sigma_u = data[params['sigma_wind_lon']['param']]
        w = data[params['v_wind_vert']['param']]
        k = adj_params['karman']
        theta = (270 - direction).apply(math.radians)
        u = v1 * theta.apply(math.cos)
        v = v1 * theta.apply(math.sin)
        I = sigma_u / v1 # its okay to divide by v1 according to IEC 61400-1
        u_star = ( (w*u)**2 + (w*v)**2 ) ** 0.25
        Au = sigma_u / u_star
        b = 1/(k*Au)
        alpha = b * I
    elif method == 'seven_power':
        alpha = 1/7
    else:
        alpha = 1/7 # 'seven_power'
    v2 = v1 * (h2 / h1) ** alpha
    return v2


def get_temperature_at_height(data: pd.DataFrame,
                              params: dict,
                              adj_params: dict) -> pd.Series:
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h1 = 2 # in meters
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    delta_h = h2 - h1
    t2 = t1 - temp_gradient * delta_h
    t2 = t2 - 273.15
    return t2


def get_pressure_at_height(data: pd.DataFrame,
                           params: dict,
                           adj_params: dict,
                           h1: float) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    g = 9.81
    p1 = data[params['pressure']['param']]
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    p2 = p1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) )
    return p2

def get_density_at_height(data: pd.DataFrame,
                          params: dict,
                          adj_params: dict) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water
    g = 9.81
    rho1 = data[params['density']['param']]
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h1 = 2 # because of temperature measured at 2 m
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    rho2 = rho1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) - 1)
    return rho2

def get_power_curve(data: pd.DataFrame,
                    power_curve: pd.Series,
                    params: dict) -> pd.Series:
    wind = data[[params["v_wind_hub"]["param"]]]
    power = pd.merge(wind, power_curve,left_on=params["v_wind_hub"]["param"], right_on='wind_speed', how="left")
    power.index = wind.index
    #C_p = (power["power"]/1000) / (0.5 * rho * rotor_area * wind["v_wind_hub"]**3)
    return power[power_curve.name]

def get_Cp(data: pd.DataFrame,
           power_curve: pd.Series,
           cp_curve: pd.Series,
           params: dict,
           adj_params: dict,):
    turbine = cp_curve.name
    if cp_curve.isna().all():
        Cp = get_cp_from_power_curve(data=data,
                                    power_curve=power_curve,
                                    params=params,
                                    adj_params=adj_params)
        return Cp
    wind = data[[params["v_wind_hub"]["param"]]]
    ticks = np.arange(0, 2501, 1) / 100.0
    wind_speed_index = pd.DataFrame(ticks, columns=['wind_speed'])
    cp_curve = pd.merge(wind_speed_index, cp_curve, how='left', right_index=True, left_on='wind_speed')
    if adj_params['interpol_method'] == 'linear':
        cp_curve = cp_curve.interpolate(method='linear', axis=0)
    elif adj_params['interpol_method'] == 'polynomial':
        cp_curve = cp_curve.interpolate(method='polynomial', order=adj_params['polynom_grad'], axis=0)
    cp_curve.fillna(0, inplace=True)
    Cp = pd.merge(wind, cp_curve,left_on=params["v_wind_hub"]["param"], right_on='wind_speed', how="left")[turbine]
    Cp.index = wind.index
    return Cp

def get_cp_from_power_curve(data: pd.DataFrame,
                            power_curve: pd.Series,
                            params: dict,
                            adj_params: dict) -> pd.Series:
    wind = data[[params["v_wind_hub"]["param"]]]
    power = pd.merge(wind, power_curve.reset_index(),left_on=params["v_wind_hub"]["param"], right_on='wind_speed', how="left")
    power.index = wind.index
    rotor_diameter = adj_params['rotor_diameter']
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    Cp = power.iloc[:,-1] / (0.5 * rotor_area * wind.iloc[:,0]**3)
    return Cp

def get_saturated_vapor_pressure(temperature: pd.Series,
                                 model: str = 'improved magnus') -> pd.Series:
    if model == 'huang':
        p_s = np.where(
            temperature > 0,
            np.exp(34.494 - (4924.99 / (temperature + 237.1))) / (temperature + 105) ** 1.57,
            np.exp(43.494 - (6545.8 / (temperature + 278))) / (temperature + 868) ** 2
        )
    elif model == 'improved_magnus':
        p_s = np.where(
            temperature > 0,
            610.94 * np.exp((17.625 * temperature) / (temperature + 243.04)),
            611.21 * np.exp((22.587 * temperature) / (temperature + 273.86))
        )
    return p_s

def get_rho(data: pd.DataFrame,
            params: dict) -> pd.Series:
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    air_pressure = data[params['pressure']['param']] * 100 # * 100 -> hPa to Pa
    temperature = data[params['temperature']['param']]
    relhum = data[params['relhum']['param']]
    p_s = data[params['sat_vap_pressure']['param']]
    # check if relative humidity is in the range between 0 and 1
    if relhum.max() > 1:
        relhum /= 100
    temperature_kelvin = temperature + 273.15
    p_w = relhum * p_s
    p_g = air_pressure - p_w
    rho_g = p_g / (R_dry * temperature_kelvin)
    rho_w = p_w / (R_w * temperature_kelvin)
    rho = rho_g + rho_w
    return rho


def read_dfs(dir: str,
             w_vert_dir: str,
             features: list) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in os.listdir(dir):
        station_id = file.split('.csv')[0].split('_')[1]
        w_vert_file = f'w_vert_{station_id}.csv'
        data = pd.read_csv(os.path.join(dir, file), delimiter= ",")
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        data = data.resample('1H', closed='left', label='left', origin='start').mean()
        data = data[features]
        # get the vertical wind_speed
        w_vert = pd.read_csv(os.path.join(w_vert_dir, w_vert_file), delimiter=",")
        w_vert['timestamp'] = pd.to_datetime(w_vert['timestamp'], utc=True)
        w_vert.set_index('timestamp', inplace=True)
        w_vert = w_vert.resample('1H', closed='left', label='left', origin='start').mean()
        data = pd.merge(data, w_vert, left_index=True, right_index=True, how='inner')
        # knn impute the data
        data = knn_imputer(data=data, n_neighbors=5)
        dfs.append(data)
        station_ids.append(station_id)
    return dfs, station_ids

def knn_imputer(data: pd.DataFrame,
               n_neighbors: int = 5):
    # To help KNNImputer estimating the temporal saisonalities we add encoded temporal features.
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df_scaled)), columns=data.columns, index=data.index)
    df.drop(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'], axis=1, inplace=True)
    return df

def get_turbines(turbine_path: str,
                 cp_path: str,
                 specs_path: str,
                 adj_params: dict):
    turbines = adj_params['turbines']
    # read power curves
    power_curves = pd.read_csv(turbine_path, sep=";", decimal=".")
    power_curves.set_index('wind_speed', inplace=True)
    power_curves = power_curves[turbines] * 1000
    # read cp curves
    cp_curves = pd.read_csv(cp_path, sep=";", decimal=".")
    cp_curves.set_index('wind_speed', inplace=True)
    cp_curves = cp_curves[turbines]
    # read turbine specs
    turbine_specs = pd.read_csv(specs_path)
    diameter_height = {}
    for height, turbine in enumerate(turbines):
        diameter_height[turbine] = {
            'diameter': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Rotordurchmesser'].values[0]),
            'height': adj_params['hub_heights'][height],
        }
    return power_curves, cp_curves, diameter_height

def get_cut_in_cut_out_speeds(power_curve: pd.Series):
    cut_in = power_curve[power_curve > 0].index.min()
    cut_out_candidates = power_curve[power_curve > 0].index
    cut_out = cut_out_candidates.max() if not cut_out_candidates.empty else None
    return cut_in, cut_out

def apply_noise(wind_power: pd.Series,
                rated_power: float,
                noise: float = 0.05) -> pd.DataFrame:
    noise_factor = np.random.normal(loc=1, scale=noise, size=len(wind_power))
    noisy_power = wind_power * noise_factor
    noisy_power = np.maximum(0, noisy_power)
    noisy_power = np.minimum(rated_power, noisy_power)
    return noisy_power

def get_features(data: pd.DataFrame,
                 params: dict,
                 adj_params: dict) -> pd.DataFrame:
    data[params['v_wind_hub']['param']] = round(get_windspeed_at_height(data=data,
                                                                  params=params,
                                                                  adj_params=adj_params), 2)
    data[params['temperature_hub']['param']] = get_temperature_at_height(data=data,
                                                                         params=params,
                                                                         adj_params=adj_params)
    temperature = data[params['temperature']['param']]
    data[params['sat_vap_pressure']['param']] = get_saturated_vapor_pressure(temperature=temperature,
                                                                             model=adj_params['p_s_model'])
    temperature_hub = data[params['temperature_hub']['param']]
    data[params['sat_vap_pressure_hub']['param']] = get_saturated_vapor_pressure(temperature=temperature_hub,
                                                                                 model=adj_params['p_s_model'])
    data[params['density']['param']] = get_rho(data=data,
                                               params=params)
    data[params['density_hub']['param']] = get_density_at_height(data=data,
                                                                 params=params,
                                                                 adj_params=adj_params)
    return data

def generate_wind_power(data: pd.DataFrame,
                        params: dict,
                        adj_params: dict,
                        power_curve: pd.DataFrame,
                        Cp: pd.Series) -> pd.Series:
    rated_power = power_curve.max()  # Watts
    rotor_diameter = adj_params['rotor_diameter']
    cut_in, cut_out = adj_params['cut_in'], adj_params['cut_out']
    rho = data[params['density_hub']['param']]
    wind_speed_hub = data[params['v_wind_hub']['param']]
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    wind_power = np.where(
        wind_speed_hub < cut_in, 0,
        np.where(
            wind_speed_hub <= cut_out,
            np.minimum(rated_power, 0.5 * rho * rotor_area * Cp * wind_speed_hub ** 3),
            0
        )
    )
    # apply noise
    power = apply_noise(wind_power=wind_power,
                        rated_power=rated_power,
                        noise=adj_params['noise'])
    return power

def gen_full_dataframe(power_curves: pd.DataFrame,
                       turbine: str,
                       adj_params: dict,
                       params: dict,
                       df: pd.DataFrame,
                       cp_curves: pd.DataFrame,
                       diameter_height: dict) -> pd.DataFrame:
    cut_in, cut_out = get_cut_in_cut_out_speeds(power_curve=power_curves[turbine])
    adj_params['cut_in'], adj_params['cut_out'] = cut_in, cut_out
    adj_params['rotor_diameter'] = diameter_height[turbine]['diameter']
    adj_params['hub_height'] = diameter_height[turbine]['height']
    df = get_features(data=df,
            params=params,
            adj_params=adj_params)
    Cp = get_Cp(data=df,
                power_curve=power_curves[turbine],
                cp_curve=cp_curves[turbine],
                params=params,
                adj_params=adj_params)
    power_curve = get_power_curve(data=df,
                                power_curve=power_curves[turbine],
                                params=params)
    power = generate_wind_power(data=df,
                            params=params,
                            adj_params=adj_params,
                            power_curve=power_curve,
                            Cp=Cp)
    df[turbine] = power
    #drop_columns = [params['v_wind_hub']['param'], 'temperature_hub', 'density_hub', 'sat_vap_pressure_hub']
    #df.drop(drop_columns, axis=1, inplace=True)
    return df

def main() -> None:
    config = utils.load_config("config.yaml")

    dir = config['data']['wind_dir']
    synth_dir = config['data']['synth_dir']
    w_vert_dir = config['data']['w_vert_dir']
    synth_dir = config['data']['synth_dir']
    turbine_dir = config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)

    os.makedirs(synth_dir, exist_ok=True)

    params = config['synth']
    adj_params = config['adjustable_wind_params']

    _, wind_features = clean_data.relevant_features(params=params)

    frames, station_ids = read_dfs(dir=dir, w_vert_dir=w_vert_dir, features=wind_features)
    power_curves, cp_curves, diameter_height = get_turbines(
        turbine_path=turbine_path,
        cp_path=cp_path,
        specs_path=specs_path,
        adj_params=adj_params
    )
    turbine_list = list(power_curves.columns)
    for id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        df = frame.copy()
        for turbine in turbine_list:
            df = gen_full_dataframe(
                    power_curves=power_curves,
                    turbine=turbine,
                    adj_params=adj_params,
                    params=params,
                    df=df,
                    cp_curves=cp_curves,
                    diameter_height=diameter_height
            )
        df.to_csv(os.path.join(synth_dir, f'synth_{id}.csv'), sep=";", decimal=".")


if __name__ == "__main__":
    main()

