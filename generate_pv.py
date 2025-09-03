# Generating Synthetic PV Power Time Series

import os
import numpy as np
import pandas as pd
import pvlib
import shutil
import logging
from tqdm import tqdm
from typing import List
from scipy.special import erf

from utils import tools, clean_data

def read_dfs(path: str,
             cams_dir: str,
             features: list,
             hourly_resolution: bool = True) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in os.listdir(path):
        station_id = file.split('.csv')[0].split('_')[1]
        data = pd.read_csv(os.path.join(path, file), delimiter= ",")
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        if hourly_resolution:
            aggregation_rules = {
            col: 'sum' if col in ['precipitation', 'precipitation_rate'] else 'mean'
            for col in data.columns
            }
            data = data.resample('h', closed='left', label='left', origin='start').agg(aggregation_rules)
        data = data[features]
        cams = pd.read_csv(os.path.join(cams_dir, f'cams_{station_id}.csv'), sep=',', decimal='.')
        cams['timestamp'] = pd.to_datetime(cams['timestamp'], utc=True)
        cams.set_index('timestamp', inplace=True)
        cams = cams.resample('h', closed='left', label='left', origin='start').mean()
        #data = pd.merge(data, w_vert, left_index=True, right_index=True, how='inner')
        data = pd.merge_asof(
            data,
            cams,
            left_index=True,
            right_index=True,
            direction='backward', # wähle den letzten bekannten Wert
        )
        # knn impute the data
        data = tools.knn_imputer(data=data, n_neighbors=5)
        dfs.append(data)
        station_ids.append(station_id)
    return dfs, station_ids

def get_location_and_elevation(station_id: str,
                               db_config: dict):
    masterdata = tools.get_master_data(db_config)
    latitude = masterdata.loc[masterdata.station_id == station_id]['latitude'].iloc[0]
    longitude = masterdata.loc[masterdata.station_id == station_id]['longitude'].iloc[0]
    elevation = masterdata.loc[masterdata.station_id == station_id]['station_height'].iloc[0]
    return latitude, longitude, elevation

def get_specific_params(station_id: str,
                        masterdata: pd.DataFrame,
                        params: dict,
                        commissioning_date: str) -> dict:
    specific_params = params.copy()
    if 'eta_inv_nom' not in params:
        eta_inv_nom = np.random.normal(loc=specific_params['mean_eta_inv_nom'], scale=0.01)
        specific_params['eta_inv_nom'] = eta_inv_nom
    longitude = masterdata.loc[masterdata.station_id == station_id]['longitude'].iloc[0]
    latitude = masterdata.loc[masterdata.station_id == station_id]['latitude'].iloc[0]
    altitude = masterdata.loc[masterdata.station_id == station_id]['station_height'].iloc[0]
    specific_params['longitude'] = longitude
    specific_params['latitude'] = latitude
    specific_params['altitude'] = altitude
    specific_params['commissioning_date'] = commissioning_date
    return specific_params

def get_ageing_degradation(
        time_vector: pd.DatetimeIndex,
        mean_age_years: float = 10.0,
        std_dev_age_years: float = 3.33,
        annual_load_factor_loss_rate: float = 0.05,
        real_ages: np.ndarray = None,
        commissioning_date: str = None,
        random_seed: int = 42):
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

def apply_soiling(power: pd.Series, # in Watts
                  pm2_5: pd.Series, # in kg/m^3
                  pm10: pd.Series, # in kg/m^3
                  prec: pd.Series, # in mm,
                  prec_dau: pd.Series, # in mins
                  tilt: float, # in degrees
                  v_depos: float = 0.009, # in m/s
                  cleaning_thresh_day: float = 5.0, # in mm/day
                  cleaning_factor_day: float = 0.1, # in [0,1]
                  random_seed: int = 42): # in [0,1]
    pm2_5 *= 1e3 # convert to g/m^3
    pm10 *= 1e3 # convert to g/m^3
    dt = (power.index[1] - power.index[0]).seconds
    deposition_flux = pm10 * v_depos            # g/m²/s
    deposition = deposition_flux * dt * np.cos(np.radians(tilt))        # g/m² per prec_dau

    freq_per_hour = 3600 / dt
    cleaning_thresh = cleaning_thresh_day / 24 / freq_per_hour
    cleaning_factor = cleaning_factor_day / 24 / freq_per_hour
    cleaning_events = (prec >= cleaning_thresh).astype(float)

    np.random.seed(random_seed)
    w = np.zeros(len(power))  # kumuliertes Gewicht pro Fläche (g/m²)
    w[0] = max(0, np.random.normal(loc=4.0, scale=1.0))

    for i in range(1, len(power)):
        if cleaning_events.iloc[i] == 0:
            w[i] = w[i-1] + deposition.iloc[i]
        else:
            w[i] = w[i-1] * (1 - cleaning_factor * (prec_dau.iloc[i] / (dt / 60)))

    soiling_loss = 0.3434 * erf(0.17 * w ** 0.8473)
    return power * soiling_loss

def get_features(data: pd.DataFrame,
                 features: dict,
                 params: dict
                ):
    # calculate pressure
    #pressure = pvlib.atmosphere.alt2pres(altitude)
    #pressure = data[features['pressure']['name']]
    temperature = data[features['temperature']['name']]
    wind_speed = data[features['wind_speed']['name']]

    latitude = params['latitude']
    longitude = params['longitude']
    altitude = params['altitude']
    surface_tilt = params['surface_tilt']
    surface_azimuth = params['surface_azimuth']
    albedo = params['albedo']

    faiman_u0 = params['faiman_u0']
    faiman_u1 = params['faiman_u1']

    # get solar position
    solpos = pvlib.solarposition.get_solarposition(
        time=data.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=temperature,
        #pressure=pressure,
    )
    solar_zenith = solpos['zenith']
    solar_azimuth = solpos['azimuth']

    # GHI and DHI in W/m^2 --> J / cm^2 = J / 0,0001 m^2 = 10000 J / m^2 --> Dividing by 600 seconds (DWD is giving GHI as sum of 10 minutes))
    dhi_col = features['dhi']['name']
    ghi_col = features['ghi']['name']
    dhi = data[dhi_col] * 1e4 / 600
    ghi = data[ghi_col] * 1e4 / 600
    data[dhi_col] = dhi
    data[ghi_col] = ghi
    # get albedo column
    #albedo = data[params['albedo']['name']]

    # set extremely low values to zero
    data.loc[data[dhi_col]< 0.01, dhi_col] = 0
    data.loc[data[ghi_col] < 0.01, ghi_col] = 0

    # get dni from ghi, dni and zenith
    dni = pvlib.irradiance.dni(ghi=ghi,
                               dhi=dhi,
                               zenith=solar_zenith)
    dni.fillna(0, inplace=True)
    data[features['dni']['name']] = dni

    # get total irradiance
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(data.index),
        albedo=albedo,
        model='haydavies',
    )
    total_irradiance.fillna(0, inplace=True)
    total_irradiance[total_irradiance < 0.01] = 0

    ghi_poa = total_irradiance['poa_global']
    dhi_poa = total_irradiance['poa_diffuse']
    dni_poa = total_irradiance['poa_direct']
    #sky_dhi = total_irradiance['poa_sky_diffuse']
    #ground_dhi = total_irradiance['poa_ground_diffuse']

    data[features['dhi_poa']['name']] = dhi_poa
    data[features['ghi_poa']['name']] = ghi_poa
    data[features['dni_poa']['name']] = dni_poa

    cell_temperature = pvlib.temperature.faiman(total_irradiance['poa_global'],
                                                temperature,
                                                wind_speed,
                                                u0=faiman_u0,
                                                u1=faiman_u1)
    data[features['cell_temperature']['name']] = cell_temperature
    return data


def generate_pv_power(data: pd.DataFrame,
                      features: dict,
                      params: dict,
                      degradation_vector: np.ndarray = None
                      ) -> pd.Series:
    np.random.seed(params['random_seed'])
    # get data
    ghi_poa = data[features['ghi_poa']['name']]
    cell_temperature = data[features['cell_temperature']['name']]
    pm2_5 = data[features['pm2_5']['name']]
    pm10 = data[features['pm10']['name']]
    prec = data[features['prec']['name']]
    prec_dau = data[features['prec_dau']['name']]

    # get static params
    dc_rating = params['dc_rating_watts']
    ac_rating = dc_rating / params['dc_ac_ratio']
    gamma_pdc = params['gamma_pdc']
    eta_inv_nom = params['eta_inv_nom']
    eta_inv_ref = params['eta_inv_ref']
    tilt = params['surface_tilt']
    v_depos = params['v_depos']
    cleaning_tresh_day = params['cleaning_thresh_day']
    cleaning_factor_day = params['cleaning_factor_day']
    random_seed = params['random_seed']

    power_dc = pvlib.pvsystem.pvwatts_dc(ghi_poa,
                                         cell_temperature,
                                         dc_rating,
                                         gamma_pdc=gamma_pdc,
                                         temp_ref=25.0)
    # aging degradation
    power_dc *= degradation_vector if degradation_vector is not None else 1.0

    # applying soiling losses
    power_dc = apply_soiling(power=power_dc,
                             pm2_5=pm2_5,
                             pm10=pm10,
                             prec=prec,
                             prec_dau=prec_dau,
                             tilt=tilt,
                             v_depos=v_depos,
                             cleaning_thresh_day=cleaning_tresh_day,
                             cleaning_factor_day=cleaning_factor_day,
                             random_seed=random_seed
                             ) if params['apply_soiling'] else power_dc

    power_ac = pvlib.inverter.pvwatts(power_dc,
                                      ac_rating,
                                      eta_inv_nom=eta_inv_nom,
                                      eta_inv_ref=eta_inv_ref)
    # drop columns
    data.drop(columns=[features['dhi_poa']['name'],
                       features['ghi_poa']['name'],
                       features['dni_poa']['name'],
                       features['cell_temperature']['name']], inplace=True)
    return power_ac


def gen_full_dataframe(params: dict,
                       features: dict,
                       df: pd.DataFrame,
                       degradation_vector: np.ndarray = None) -> pd.DataFrame:
    tilt, az = params['surface_tilt'], params['surface_azimuth']
    df = get_features(data=df,
                      features=features,
                      params=params)
    power = generate_pv_power(data=df,
                              features=features,
                              params=params,
                              degradation_vector=degradation_vector)
    df[f'10kw_tilt{tilt}_az{az}'] = power
    return df

def main() -> None:
    config = tools.load_config("configs/config_pv.yaml")
    db_config = config['write']['db_conf']

    raw_dir = os.path.join(config['data']['raw_dir'], 'solar')
    cams_dir = config['data']['cams_dir']
    pv_ages_path = config['data']['pv_ages']
    pv_ages = np.load(pv_ages_path)

    features = config['features']
    params = config['pv_params']
    tilt_az_list = params['tilt_az_list']

    aging_flag = '_noage'
    if params['apply_ageing']:
        aging_flag = 'age'
    if params['hourly_resolution']:
        resolution = 'hourly'
    else:
        resolution = '10min'
    synth_dir = os.path.join(config['data']['synth_dir'], 'solar', f'solar_{resolution}_{aging_flag}')
    if os.path.exists(synth_dir):
        shutil.rmtree(synth_dir)
    os.makedirs(synth_dir, exist_ok=True)

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
    pv_features, _ = clean_data.relevant_features(features=features)

    frames, station_ids = read_dfs(path=raw_dir,
                                   cams_dir=cams_dir,
                                   features=pv_features,
                                   hourly_resolution=params['hourly_resolution'])
    masterdata = tools.get_master_data(db_config)
    logging.info(f'Starting solar power generation. Ageing: {params["apply_ageing"]}, Hourly resolution: {params['hourly_resolution']}')
    power_master = {}
    for station_id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        df = frame.copy()
        degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index,
                                                                        real_ages=pv_ages,
                                                                        random_seed=params['random_seed'])
        specific_params = get_specific_params(station_id=station_id,
                                              masterdata=masterdata,
                                              params=params,
                                              commissioning_date=commissioning_date)
        for tilt, az in tilt_az_list:
            specific_params['surface_tilt'] = tilt
            specific_params['surface_azimuth'] = az
            df = gen_full_dataframe(
                    params=specific_params,
                    features=features,
                    df=df,
                    degradation_vector=degradation_vector
            )
        params['random_seed'] += 1
        del specific_params['surface_tilt']
        del specific_params['surface_azimuth']
        del specific_params['dc_rating_watts']
        del specific_params['random_seed']
        del specific_params['mean_eta_inv_nom']
        del specific_params['apply_ageing']
        del specific_params['apply_soiling']
        del specific_params['hourly_resolution']
        del specific_params['tilt_az_list']
        df.to_csv(os.path.join(synth_dir, f'synth_{station_id}.csv'), sep=";", decimal=".")
        power_master[station_id] = specific_params
    # Save the park parameters
    pv_parameter_path = os.path.join(synth_dir, 'pv_parameter.csv')
    write_header = not os.path.exists(pv_parameter_path) #and not park_id
    df_power_master = pd.DataFrame.from_dict(power_master, orient='index')
    df_power_master.index.name = 'park_id'
    # check if the entries are already in the file if it exists
    df_power_master.to_csv(pv_parameter_path, sep=";", decimal=".", mode='a', header=write_header)
    # drop duplicates
    df_power_master = pd.read_csv(pv_parameter_path, sep=";", decimal=".", dtype={'park_id': str})
    df_power_master.drop_duplicates(inplace=True)
    df_power_master.to_csv(pv_parameter_path, sep=";", decimal=".", index=False)

if __name__ == '__main__':
    main()