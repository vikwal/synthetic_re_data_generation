# Generating Synthetic PV Power Time Series

import os
import yaml
import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import utils, clean_data

def read_dfs(dir: str,
             features: list) -> List[pd.DataFrame]:
    dfs = []
    station_ids = []
    for file in os.listdir(dir):
        station_id = file.split('.csv')[0].split('_')[1]
        data = pd.read_csv(os.path.join(dir, file), delimiter= ",")
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data.set_index('timestamp', inplace=True)
        data = data.resample('1H', closed='left', label='left', origin='start').mean()
        data = data[features]
        # knn impute the data
        data = utils.knn_imputer(data=data, n_neighbors=5)
        dfs.append(data)
        station_ids.append(station_id)
    return dfs, station_ids

def get_features(data: pd.DataFrame,
                 features: dict,
                 params: dict
                ):
    # calculate pressure
    #pressure = pvlib.atmosphere.alt2pres(elevation)
    dhi = data[features['dhi']['name']]
    ghi = data[features['ghi']['name']]
    #pressure = data[features['pressure']['name']]
    temperature = data[features['temperature']['name']]
    wind_speed = data[features['wind_speed']['name']]
    latitude = data[features['latitude']['name']]
    longitude = data[features['longitude']['name']]
    elevation = data[features['elevation']['name']]

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
        altitude=elevation,
        temperature=temperature,
        #pressure=pressure,
    )
    solar_zenith = solpos['zenith']
    solar_azimuth = solpos['azimuth']

    # GHI and DHI in W/m^2 --> J / cm^2 = J / 0,0001 m^2 = 10000 J / m^2 --> Dividing by 600 seconds (DWD is giving GHI as sum of 10 minutes))
    dhi = data[features['dhi']['name']] * 1e4 / 600
    ghi = data[features['ghi']['name']] * 1e4 / 600

    # get dni from ghi, dni and zenith
    dni = pvlib.irradiance.dni(ghi=ghi,
                               dhi=dhi,
                               zenith=solar_zenith)

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
    cell_temperature = pvlib.temperature.faiman(total_irradiance['poa_global'],
                                                temperature,
                                                wind_speed,
                                                u0=faiman_u0,
                                                u1=faiman_u1)
    return total_irradiance, cell_temperature


def generate_pv_power(total_irradiance: pd.Series,
                      cell_temperature: pd.Series,
                      params: dict
                      ) -> pd.Series:

    installed_power = params['installed_power']
    gamma_pdc = params['gamma_pdc']
    eta_env_nom = params['eta_env_nom']
    eta_env_ref = params['eta_env_ref']

    power_dc = pvlib.pvsystem.pvwatts_dc(total_irradiance,
                                         cell_temperature,
                                         installed_power,
                                         gamma_pdc=gamma_pdc,
                                         temp_ref=25.0)

    return pvlib.inverter.pvwatts(power_dc,
                                  installed_power,
                                  eta_inv_nom=eta_env_nom,
                                  eta_inv_ref=eta_env_ref)

def main() -> None:
    config = utils.load_config("config.yaml")

    raw_dir = os.path.join(config['data']['raw_dir'], 'solar')
    synth_dir = os.path.join(config['data']['synth_dir'], 'solar')

    os.makedirs(synth_dir, exist_ok=True)

    features = config['features']
    params = config['pv_params']

    pv_features, _ = clean_data.relevant_features(features=features)

    frames, station_ids = read_dfs(dir=raw_dir, features=pv_features)

    for id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):
        df = frame.copy()
        df = gen_full_dataframe(
                params=params,
                features=features,
                df=df,
        )
        df.to_csv(os.path.join(synth_dir, f'synth_{id}.csv'), sep=";", decimal=".")
    # Save the specs
    #specs_df = pd.DataFrame.from_dict(specs, orient='index')

if __name__ == '__main__':
    main()