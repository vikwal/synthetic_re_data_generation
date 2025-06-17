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
                            features: dict,
                            params: dict
                            ):
    """
    Calculate the wind speed (v2) using different methods.

    Parameters:
    -----------
    data : pd.DataFrame
    features: dict
        Dictionary with parameters names
        features['wind_speed']['name']: Wind Speed
        features['d_wind']['name']: Wind Direction
        features['w_wind']['name']: Vertical Wind Speed
        features['sigma_wind_lon']['name']: Standard Deviation longitudinal wind speed
    params : dict
        Dictionary with neccesary parameters
        params['h1']: Height of measurements in m
        params['h2']: Hub height in m
        params['karman']: von Karmans constant
        params['method']: Height of measurement in m
            The method used to calculate v2. Options are:
            - 'alphaI': Uses linear relationship between turbulence intensity and alpha (Ishizaki, 1983)
            - 'seven_power': A default method using the 1/7 power law.
            If no method is provided, the default 'seven_power' is used.
    Returns:
    --------
    v2 : float
        The calculated wind speed at height h2.
    """
    v1 = data[features['wind_speed']['name']]
    method = params['v2_method']
    h1 = params['h1']
    h2 = params['hub_height']
    if method == 'alphaI':
        direction = data[features['d_wind']['name']]
        sigma_u = data[features['sigma_wind_lon']['name']]
        w = data[features['wind_speed_vertical']['name']]
        k = params['karman']
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
                              features: dict,
                              params: dict) -> pd.Series:
    t1 = data[features['temperature']['name']]
    t1 = t1 + 273.15
    h1 = 2 # in meters
    h2 = params['hub_height']
    temp_gradient = params['temp_gradient']
    delta_h = h2 - h1
    t2 = t1 - temp_gradient * delta_h
    t2 = t2 - 273.15
    return t2


def get_pressure_at_height(data: pd.DataFrame,
                           features: dict,
                           params: dict,
                           h1: float) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    g = 9.81
    p1 = data[features['pressure']['name']]
    t1 = data[features['temperature']['name']]
    t1 = t1 + 273.15
    h2 = params['hub_height']
    temp_gradient = params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    p2 = p1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) )
    return p2

def get_density_at_height(data: pd.DataFrame,
                          features: dict,
                          params: dict) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water
    g = 9.81
    rho1 = data[features['density']['name']]
    t1 = data[features['temperature']['name']]
    t1 = t1 + 273.15
    h1 = 2 # because of temperature measured at 2 m
    h2 = params['hub_height']
    temp_gradient = params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    rho2 = rho1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) - 1)
    return rho2

def merge_curve(data: pd.DataFrame,
                curve: pd.Series,
                features: dict) -> pd.Series:
    wind = data[[features["wind_speed_hub"]['name']]]
    values = pd.merge(wind, curve,left_on=features["wind_speed_hub"]['name'], right_on='wind_speed', how="left")
    values.index = wind.index
    return values[curve.name]

def interpolate(power_curve: pd.DataFrame, cut_out: float):
    ticks = np.arange(0, cut_out*100, 1) / 100.0
    rated_power = power_curve.max()
    interpol = pd.Series(index=ticks, dtype=float)
    interpol.index.name = 'wind_speed'
    interpol = interpol.to_frame().merge(power_curve.to_frame(), how='left', left_index=True, right_index=True)
    interpol.drop(columns=[0], inplace=True)
    interpol.iloc[-1] = rated_power
    interpol = interpol.interpolate(method='polynomial', order=3)
    interpol = interpol.clip(upper=rated_power)
    interpol.fillna(0, inplace=True)
    return pd.Series(interpol.iloc[:,-1], index=interpol.index)


def get_cp_from_power_curve(data: pd.DataFrame,
                            power_curve: pd.Series,
                            features: dict,
                            params: dict,
                            degradation_vector: np.ndarray = None) -> pd.Series:
    wind_speed = data[[features["wind_speed_hub"]['name']]]
    rho = data[features['density_hub']['name']]
    power = pd.merge(wind_speed, power_curve.reset_index(),left_on=features["wind_speed_hub"]['name'], right_on='wind_speed', how="left")
    power.index = wind_speed.index
    rotor_diameter = params['rotor_diameter']
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    if degradation_vector is None:
        degradation_vector = np.ones(len(data))
    degradation_vector = pd.Series(degradation_vector, index=wind_speed.index)
    wind_speed_after_degradation = wind_speed * (1/degradation_vector)**(1/3)
    Cp = power.iloc[:,-1] / (0.5 * rho * rotor_area * wind_speed_after_degradation**3)
    #Cp = Cp * degradation_vector
    Cp.clip(lower=0, upper=1, inplace=True)
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
            features: dict) -> pd.Series:
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    air_pressure = data[features['pressure']['name']] * 100 # * 100 -> hPa to Pa
    temperature = data[features['temperature']['name']]
    relhum = data[features['relhum']['name']]
    p_s = data[features['sat_vap_pressure']['name']]
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
                 params: dict):
    turbines = params['turbines']
    # read power curves
    power_curves = pd.read_csv(turbine_path, sep=";", decimal=".")
    power_curves.set_index('wind_speed', inplace=True)
    power_curves = power_curves[turbines] * 1000
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
            'height': params['hub_heights'][height],
            'cut_in': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Einschaltgeschwindigkeit'].values[0]),
            'cut_out': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Abschaltgeschwindigkeit'].values[0]),
            'rated': float(turbine_specs[turbine_specs['Turbine'] == turbine]['Nennwindgeschwindigkeit'].values[0])
        }
    return power_curves, cp_curves, specs

def get_ageing_degradation(
        time_vector: pd.DatetimeIndex,
        mean_age_years: float = 15.0,
        std_dev_age_years: float = 5.0,
        annual_load_factor_loss_rate: float = 0.0063):
    start_age = np.random.normal(loc=mean_age_years, scale=std_dev_age_years)
    start_age = max(0.0, start_age)
    end_age = start_age + 1
    commissioning_date = str((time_vector[0] - pd.Timedelta(days=start_age*365.25)).date())
    base_efficiency = 1.0 - annual_load_factor_loss_rate
    efficiency_factor_start = base_efficiency ** start_age
    efficiency_factor_end = base_efficiency ** end_age
    efficiency_vector = np.linspace(efficiency_factor_start, efficiency_factor_end, num=len(time_vector))
    return efficiency_vector, commissioning_date

def apply_noise(wind_power: pd.Series,
                rated_power: float,
                noise: float = 0.05) -> pd.DataFrame:
    noise_factor = np.random.normal(loc=1, scale=noise, size=len(wind_power))
    noisy_power = wind_power * noise_factor
    noisy_power = np.maximum(0, noisy_power)
    noisy_power = np.minimum(rated_power, noisy_power)
    return noisy_power

def get_features(data: pd.DataFrame,
                 features: dict,
                 params: dict) -> pd.DataFrame:
    data[features['wind_speed_hub']['name']] = round(get_windspeed_at_height(data=data,
                                                                  features=features,
                                                                  params=params), 2)
    data[features['temperature_hub']['name']] = get_temperature_at_height(data=data,
                                                                         features=features,
                                                                         params=params)
    temperature = data[features['temperature']['name']]
    data[features['sat_vap_pressure']['name']] = get_saturated_vapor_pressure(temperature=temperature,
                                                                             model=params['p_s_model'])
    temperature_hub = data[features['temperature_hub']['name']]
    data[features['sat_vap_pressure_hub']['name']] = get_saturated_vapor_pressure(temperature=temperature_hub,
                                                                                 model=params['p_s_model'])
    data[features['density']['name']] = get_rho(data=data,
                                               features=features)
    data[features['density_hub']['name']] = get_density_at_height(data=data,
                                                                 features=features,
                                                                 params=params)
    return data

def generate_wind_power(data: pd.DataFrame,
                        features: dict,
                        params: dict,
                        power_curve: pd.DataFrame,
                        Cp: pd.Series,
                        degradation_vector: np.ndarray = None) -> pd.Series:
    rated_power = power_curve.max()  # Watts
    rotor_diameter = params['rotor_diameter']
    cut_in, cut_out, rated_speed = params['cut_in'], params['cut_out'], params['rated']
    rho = data[features['density_hub']['name']]
    wind_speed_hub = data[features['wind_speed_hub']['name']]
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
    # apply noise
    power = apply_noise(wind_power=wind_power,
                        rated_power=rated_power,
                        noise=params['noise'])
    return power

def gen_full_dataframe(power_curves: pd.DataFrame,
                       turbine: str,
                       params: dict,
                       features: dict,
                       df: pd.DataFrame,
                       cp_curves: pd.DataFrame,
                       specs: dict,
                       degradation_vector: np.ndarray = None) -> pd.DataFrame:
    params['rotor_diameter'] = specs[turbine]['diameter']
    params['hub_height'] = specs[turbine]['height']
    params['cut_in'] = specs[turbine]['cut_in']
    params['cut_out'] = specs[turbine]['cut_out']
    df = get_features(data=df,
            features=features,
            params=params)
    power_curve = interpolate(power_curve=power_curves[turbine],
                              cut_out=params['cut_out'])
    Cp = get_cp_from_power_curve(data=df,
                                power_curve=power_curve,
                                features=features,
                                params=params)
    power_curve = merge_curve(data=df,
                            curve=power_curve,
                            features=features)
    degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index)
    #params['commissioning_date'] = commissioning_date
    power = generate_wind_power(data=df,
                            features=features,
                            params=params,
                            power_curve=power_curve,
                            Cp=Cp,
                            degradation_vector=degradation_vector)
    df[turbine] = power
    #drop_columns = [features['wind_speed_hub']['name'], 'temperature_hub', 'density_hub', 'sat_vap_pressure_hub']
    #df.drop(drop_columns, axis=1, inplace=True)
    return df


def main() -> None:
    config = utils.load_config("config.yaml")

    raw_dir = os.path.join(config['data']['raw_dir'], 'wind')
    synth_dir = os.path.join(config['data']['synth_dir'], 'wind')
    w_vert_dir = config['data']['w_vert_dir']
    turbine_dir = config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)

    os.makedirs(synth_dir, exist_ok=True)

    features = config['features']
    params = config['wind_params']

    _, wind_features = clean_data.relevant_features(features=features)

    frames, station_ids = read_dfs(dir=raw_dir, w_vert_dir=w_vert_dir, features=wind_features)
    power_curves, cp_curves, specs = get_turbines(
        turbine_path=turbine_path,
        cp_path=cp_path,
        specs_path=specs_path,
        params=params
    )
    turbine_list = list(power_curves.columns)
    for id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        df = frame.copy()
        degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index)
        for turbine in turbine_list:
            df = gen_full_dataframe(
                    power_curves=power_curves,
                    turbine=turbine,
                    params=params,
                    features=features,
                    df=df,
                    cp_curves=cp_curves,
                    specs=specs,
                    degradation_vector=degradation_vector
            )
        df.to_csv(os.path.join(synth_dir, f'synth_{id}.csv'), sep=";", decimal=".")
    # Save the turbine specs
    #specs_df = pd.DataFrame.from_dict(specs, orient='index')

if __name__ == "__main__":
    main()

