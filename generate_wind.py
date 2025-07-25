import os
import math
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from utils import tools, clean_data

def get_park_params(station_id: str,
                        masterdata: pd.DataFrame,
                        params: dict,
                        commissioning_date: str) -> dict:
    specific_params = params.copy()
    # drop turbine information
    del specific_params['turbines']
    del specific_params['hub_heights']
    del specific_params['h1']
    del specific_params['p_s_model']
    del specific_params['v2_method']
    del specific_params['karman']
    del specific_params['temp_gradient']
    del specific_params['interpol_method']
    del specific_params['polynom_grad']
    del specific_params['noise']
    del specific_params['mean_age_years']
    del specific_params['std_dev_age_years']
    del specific_params['annual_degradation']
    del specific_params['apply_ageing']
    del specific_params['hourly_resolution']
    # from stations masterdata
    longitude = masterdata.loc[masterdata.station_id == station_id]['longitude'].iloc[0]
    latitude = masterdata.loc[masterdata.station_id == station_id]['latitude'].iloc[0]
    altitude = masterdata.loc[masterdata.station_id == station_id]['station_height'].iloc[0]
    specific_params['longitude'] = longitude
    specific_params['latitude'] = latitude
    specific_params['altitude'] = altitude
    specific_params['commissioning_date'] = commissioning_date
    return specific_params

def get_windspeed_at_height(data: pd.DataFrame,
                            features: dict,
                            params: dict,
                            h2: float
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
                              params: dict,
                              h2: float) -> pd.Series:
    t1 = data[features['temperature']['name']].copy()
    t1 = t1 + 273.15
    h1 = 2 # in meters
    temp_gradient = params['temp_gradient']
    delta_h = h2 - h1
    t2 = t1 - temp_gradient * delta_h
    t2 = t2 - 273.15
    return t2


def get_pressure_at_height(data: pd.DataFrame,
                           features: dict,
                           params: dict,
                           h1: float,
                           h2: float) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    g = 9.81
    p1 = data[features['pressure']['name']].copy()
    t1 = data[features['temperature']['name']].copy()
    t1 = t1 + 273.15
    temp_gradient = params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    p2 = p1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) )
    return p2

def get_density_at_height(data: pd.DataFrame,
                          features: dict,
                          params: dict,
                          h2: float) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water
    g = 9.81
    rho1 = data[features['density']['name']].copy()
    t1 = data[features['temperature']['name']].copy()
    t1 = t1 + 273.15
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
        dr = degradation_vector.loc[t]
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

def get_rho(data: pd.DataFrame,
            features: dict) -> pd.Series:
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    air_pressure = data[features['pressure']['name']].copy()
    air_pressure *= 100 # * 100 -> hPa to Pa
    temperature = data[features['temperature']['name']].copy()
    relhum = data[features['relhum']['name']].copy()
    p_s = data[features['sat_vap_pressure']['name']].copy()
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


def read_dfs(path: str,
             w_vert_dir: str,
             features: list,
             hourly_resolution: bool = True,
             specific_id: str = None) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in os.listdir(path):
        station_id = file.split('.csv')[0].split('_')[1]
        if specific_id and (station_id != specific_id):
            continue
        w_vert_file = f'w_vert_{station_id}.csv'
        data = pd.read_csv(os.path.join(path, file), delimiter= ",")
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        if hourly_resolution:
            data = data.resample('h', closed='left', label='left', origin='start').mean()
        data = data[features]
        # get the vertical wind_speed
        w_vert = pd.read_csv(os.path.join(w_vert_dir, w_vert_file), delimiter=",")
        w_vert['timestamp'] = pd.to_datetime(w_vert['timestamp'], utc=True)
        w_vert.set_index('timestamp', inplace=True)
        w_vert = w_vert.resample('h', closed='left', label='left', origin='start').mean()
        #data = pd.merge(data, w_vert, left_index=True, right_index=True, how='inner')
        data = pd.merge_asof(
            data,
            w_vert,
            left_index=True,
            right_index=True,
            direction='backward',         # wähle den letzten bekannten Wert
        )
        # knn impute the data
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
    power_curves = power_curves[turbines] * 1000 # MW -> kW
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
        annual_load_factor_loss_rate: float = 0.0063,
        real_ages: np.ndarray = None,
        commissioning_date: str = None):
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
                noise: float = 0.05) -> pd.DataFrame:
    noise_factor = np.random.normal(loc=1, scale=noise, size=len(wind_power))
    noisy_power = wind_power * noise_factor
    noisy_power = np.maximum(0, noisy_power)
    noisy_power = np.minimum(rated_power, noisy_power)
    return noisy_power

def get_features(data: pd.DataFrame,
                 features: dict,
                 params: dict,
                 hub_height: float,
                 suffix: str = '_hub') -> pd.DataFrame:
    wind_speed_hub_col = f"{features['wind_speed_hub']['name']}{suffix}"
    temperature_col = features['temperature']['name']
    temperature_hub_col = f"{features['temperature_hub']['name']}{suffix}"
    sat_vap_ps_col = features['sat_vap_pressure']['name']
    sat_vap_ps_hub_col = f"{features['sat_vap_pressure']['name']}{suffix}"
    density_col = features['density']['name']
    density_hub_col = f"{features['density']['name']}{suffix}"

    data[wind_speed_hub_col] = round(get_windspeed_at_height(data=data,
                                                                  features=features,
                                                                  params=params,
                                                                  h2=hub_height), 2)
    data[temperature_hub_col] = get_temperature_at_height(data=data,
                                                          features=features,
                                                          params=params,
                                                          h2=hub_height)
    temperature = data[temperature_col]
    data[sat_vap_ps_col] = get_saturated_vapor_pressure(temperature=temperature,
                                                        model=params['p_s_model'])
    temperature_hub = data[temperature_hub_col]
    data[sat_vap_ps_hub_col] = get_saturated_vapor_pressure(temperature=temperature_hub,
                                                            model=params['p_s_model'])
    data[density_col] = get_rho(data=data,
                                               features=features)
    data[density_hub_col] = get_density_at_height(data=data,
                                                                 features=features,
                                                                 params=params,
                                                                 h2=hub_height)
    return data

def generate_wind_power(data: pd.DataFrame,
                        features: dict,
                        params: dict,
                        rated_power: float,
                        Cp: pd.Series,
                        rotor_diameter: float,
                        cut_in: float,
                        cut_out: float,
                        rated_speed: float,
                        degradation_vector: np.ndarray = None,
                        suffix: str = '_hub') -> pd.Series:
    wind_speed_hub_col = f"{features['wind_speed_hub']['name']}{suffix}"
    density_hub_col = f"{features['density']['name']}{suffix}"
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
    power = apply_noise(wind_power=wind_power,
                        rated_power=rated_power,
                        noise=params['noise'])
    return power

def gen_full_dataframe(power_curves: pd.DataFrame,
                       turbine: str,
                       params: dict,
                       features: dict,
                       df: pd.DataFrame,
                       specs: dict,
                       degradation_vector: np.ndarray = None,
                       suffix_for_turbine_cols: str = '') -> pd.DataFrame:
    rotor_diameter = specs[turbine]['diameter']
    hub_height = specs[turbine]['height']
    cut_in = specs[turbine]['cut_in']
    cut_out = specs[turbine]['cut_out']
    rated_speed = specs[turbine]['rated']
    df = get_features(data=df,
            features=features,
            params=params,
            hub_height=hub_height,
            suffix=suffix_for_turbine_cols)
    power_curve = interpolate(power_curve=power_curves[turbine], # filter
                              cut_out=cut_out)
    Cp = get_cp_from_power_curve(data=df,
                                power_curve=power_curve,
                                features=features,
                                rotor_diameter=rotor_diameter,
                                degradation_vector=degradation_vector,
                                suffix=suffix_for_turbine_cols)
    power_curve = merge_curve(data=df,
                            curve=power_curve,
                            features=features,
                            suffix=suffix_for_turbine_cols)
    #degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index)
    #params['commissioning_date'] = commissioning_date
    power = generate_wind_power(data=df,
                            features=features,
                            params=params,
                            rated_power=power_curve.max(),
                            Cp=Cp,
                            rotor_diameter=rotor_diameter,
                            cut_in=cut_in,
                            cut_out=cut_out,
                            rated_speed=rated_speed,
                            degradation_vector=degradation_vector,
                            suffix=suffix_for_turbine_cols)
    df[f"power{suffix_for_turbine_cols}"] = power
    #drop_columns = [features['wind_speed_hub']['name'], 'temperature_hub', 'density_hub', 'sat_vap_pressure_hub']
    #df.drop(drop_columns, axis=1, inplace=True)
    return df


def main() -> None:
    config = tools.load_config("config.yaml")
    db_config = config['write']['db_conf']

    raw_dir = os.path.join(config['data']['raw_dir'], 'wind')
    synth_dir = os.path.join(config['data']['synth_dir'], 'wind', 'real_parks_noage')
    w_vert_dir = config['data']['w_vert_dir']
    turbine_dir = config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)
    wind_ages_path = config['data']['wind_ages']
    wind_ages = np.load(wind_ages_path)

    specific_id = '07374'
    commissioning_date = None#'2003-06-01'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        #datefmt=datefmt,
        handlers=[
            #logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
    )

    os.makedirs(synth_dir, exist_ok=True)

    features = config['features']
    params = config['wind_params']

    _, wind_features = clean_data.relevant_features(features=features)

    frames, station_ids = read_dfs(path=raw_dir,
                                   w_vert_dir=w_vert_dir,
                                   features=wind_features,
                                   hourly_resolution=params['hourly_resolution'],
                                   specific_id=specific_id)

    masterdata = tools.get_master_data(db_config)
    power_curves, _, specs = get_turbines(
        turbine_path=turbine_path,
        cp_path=cp_path,
        specs_path=specs_path,
        params=params
    )
    power_master = {}
    logging.info(f'Starting wind power generation. Ageing: {params["apply_ageing"]}, Hourly resolution: {params['hourly_resolution']}')
    for station_id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        logging.debug(f'Processing station {str(station_id)}')
        df = frame.copy()
        degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index,
                                                                        real_ages=wind_ages, commissioning_date=commissioning_date)
        if not params['apply_ageing']:
            degradation_vector = None
            commissioning_date = '-'
        specific_params = get_park_params(station_id=station_id,
                                          masterdata=masterdata,
                                          params=params,
                                          commissioning_date=commissioning_date)
        power_master[station_id] = specific_params
        for turbine_id, turbine in enumerate(params['turbines'], start=1):
            df = gen_full_dataframe(
                    power_curves=power_curves,
                    turbine=turbine,
                    params=params,
                    features=features,
                    df=df,
                    specs=specs,
                    degradation_vector=degradation_vector,
                    suffix_for_turbine_cols=f'_t{turbine_id}'
            )
            if 'turbine_id' not in specs[turbine].keys():
                specs[turbine]['turbine_id'] = f't{turbine_id}'
        df.to_csv(os.path.join(synth_dir, f'synth_{station_id}.csv'), sep=";", decimal=".")
        commissioning_date = None
    # Save the park parameters
    df_power_master = pd.DataFrame.from_dict(power_master, orient='index')
    df_power_master.index.name = 'park_id'
    df_power_master.to_csv(os.path.join(synth_dir, 'wind_parameter.csv'), sep=";", decimal=".")
    # save the turbine parameters
    df_turbine_master = pd.DataFrame.from_dict(specs, orient='index')
    df_turbine_master.index.name = 'turbine'
    df_turbine_master.to_csv(os.path.join(synth_dir, 'turbine_parameter.csv'), sep=";", decimal=".")

if __name__ == "__main__":
    main()

