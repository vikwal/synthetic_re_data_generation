# Generating Synthetic PV Power Time Series

import os
import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm
from typing import List

from utils import tools, clean_data

def read_dfs(path: str,
             features: list) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in os.listdir(path):
        station_id = file.split('.csv')[0].split('_')[1]
        data = pd.read_csv(os.path.join(path, file), delimiter= ",")
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        data = data.resample('1H', closed='left', label='left', origin='start').mean()
        data = data[features]
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
    del specific_params['gamma_pdc']
    del specific_params['eta_inv_ref']
    longitude = masterdata.loc[masterdata.station_id == station_id]['longitude'].iloc[0]
    latitude = masterdata.loc[masterdata.station_id == station_id]['latitude'].iloc[0]
    altitude = masterdata.loc[masterdata.station_id == station_id]['station_height'].iloc[0]
    specific_params['longitude'] = longitude
    specific_params['latitude'] = latitude
    specific_params['altitude'] = altitude
    specific_params['commissioning_date'] = commissioning_date
    return specific_params

def get_features(data: pd.DataFrame,
                 features: dict,
                 params: dict
                ):
    # calculate pressure
    #pressure = pvlib.atmosphere.alt2pres(elevation)
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
    cell_temperature = pvlib.temperature.faiman(total_irradiance['poa_global'],
                                                temperature,
                                                wind_speed,
                                                u0=faiman_u0,
                                                u1=faiman_u1)
    return total_irradiance, cell_temperature


def generate_pv_power(total_irradiance: pd.Series,
                      cell_temperature: pd.Series,
                      params: dict,
                      degradation_vector: np.ndarray = None
                      ) -> pd.Series:

    dc_rating = params['dc_rating_watts']
    ac_rating = dc_rating / params['dc_ac_ratio']
    gamma_pdc = params['gamma_pdc']
    eta_inv_nom = params['eta_inv_nom']
    eta_inv_ref = params['eta_inv_ref']

    power_dc = pvlib.pvsystem.pvwatts_dc(total_irradiance,
                                         cell_temperature,
                                         dc_rating,
                                         gamma_pdc=gamma_pdc,
                                         temp_ref=25.0)
    power_ac = pvlib.inverter.pvwatts(power_dc,
                                      ac_rating,
                                      eta_inv_nom=eta_inv_nom,
                                      eta_inv_ref=eta_inv_ref)
    return power_ac

def get_ageing_degradation(
        time_vector: pd.DatetimeIndex,
        mean_age_years: float = 10,
        std_dev_age_years: float = 3.33,
        annual_load_factor_loss_rate: float = 0.08,
        real_ages: np.ndarray = None):
    if real_ages is None:
        start_age = np.random.normal(loc=mean_age_years, scale=std_dev_age_years)
    else:
        start_age = float(np.random.choice(real_ages, size=1, replace=False)[0])
    start_age = max(0.0, start_age)
    end_age = start_age + 1
    commissioning_date = str((time_vector[0] - pd.Timedelta(days=start_age*365.25)).date())
    base_efficiency = 1.0 - annual_load_factor_loss_rate
    efficiency_factor_start = base_efficiency ** start_age
    efficiency_factor_end = base_efficiency ** end_age
    efficiency_vector = np.linspace(efficiency_factor_start, efficiency_factor_end, num=len(time_vector))
    return efficiency_vector, commissioning_date

def gen_full_dataframe(params: dict,
                       features: dict,
                       df: pd.DataFrame,
                       degradation_vector: np.ndarray = None) -> pd.DataFrame:
    total_irradiance, cell_temperature = get_features(data=df,
                                                      features=features,
                                                      params=params)
    total = total_irradiance['poa_global']
    #direct = total_irradiance['poa_direct']
    #diffuse = total_irradiance['poa_diffuse']
    #sky_dhi = total_irradiance['poa_sky_diffuse']
    #ground_dhi = total_irradiance['poa_ground_diffuse']
    power = generate_pv_power(total_irradiance=total,
                              cell_temperature=cell_temperature,
                              params=params,
                              degradation_vector=degradation_vector)
    df['power'] = power
    return df

def main() -> None:
    config = tools.load_config("config.yaml")
    db_config = config['write']['db_conf']

    raw_dir = os.path.join(config['data']['raw_dir'], 'solar')
    synth_dir = os.path.join(config['data']['synth_dir'], 'solar')
    pv_ages_path = config['data']['wind_ages']
    pv_ages = np.load(pv_ages_path)

    os.makedirs(synth_dir, exist_ok=True)

    features = config['features']
    params = config['pv_params']

    pv_features, _ = clean_data.relevant_features(features=features)

    frames, station_ids = read_dfs(path=raw_dir, features=pv_features)
    masterdata = tools.get_master_data(db_config)

    power_master = {}
    for station_id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        df = frame.copy()
        degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index, real_ages=pv_ages)
        specific_params = get_specific_params(station_id=station_id,
                                              masterdata=masterdata,
                                              params=params,
                                              commissioning_date=commissioning_date)
        power_master[station_id] = specific_params
        df = gen_full_dataframe(
                params=specific_params,
                features=features,
                df=df,
                degradation_vector=degradation_vector
        )
        df.to_csv(os.path.join(synth_dir, f'synth_{station_id}.csv'), sep=";", decimal=".")
    # Save the technical parameters
    df_power_master = pd.DataFrame.from_dict(power_master, orient='index')
    df_power_master.index.name = 'park_id'
    df_power_master.to_csv(os.path.join(synth_dir, 'pv_parameter.csv'), sep=";", decimal=".")

if __name__ == '__main__':
    main()