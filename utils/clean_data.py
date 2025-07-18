# Clean database data and drop stations with too much missing values

import os
import getpass
import pandas as pd
from tqdm import tqdm
from typing import List
import logging

from utils import tools, get_nwp

def relevant_features(features: dict):
    pv_features = [features['ghi']['name'],
                   features['dhi']['name'],
                   features['temperature']['name'],
                   #features['dewpoint']['name'],
                   features['wind_speed']['name'],
                   #features['pressure']['name'],
                   ]
    wind_features = [features['wind_speed']['name'],
                     features['temperature']['name'],
                     features['relhum']['name'],
                     features['sigma_wind_lon']['name'],
                     features['pressure']['name'],
                     features['d_wind']['name'],
                     ]
    return pv_features, wind_features

def get_station_ids(db_config: dict):
    conn, cursor = tools.connect_db(db_config)
    query = """
        SELECT DISTINCT station_id FROM stations;
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    if rows:
        unique_ids = [row[0] for row in rows]
        return unique_ids
    else:
        print(f"No station ids.")
        return []

def iterate_stations(db_config: dict,
                   station_ids: list):
    conn, cursor = tools.connect_db(db_config)
    frames = []
    for id in tqdm(station_ids, desc='Iterate over stations'):
        query = f"""
            SELECT * FROM stations WHERE station_id = '{id}';
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        if rows:
            # Spaltennamen abrufen
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            #df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Berlin")
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            frames.append(df)
        else:
            logging.warning(f"No data for station_id: {id} found.")
            continue
    return frames

def get_drop_list(frames: List[pd.DataFrame],
                  features: List[str],
                  threshold: float):
    drop_list = []
    for df in frames:
        frame_added = False
        for feature in features:
            if frame_added:
                continue
            share_of_missing_rows = df[feature].isna().sum() / len(df)
            if share_of_missing_rows > threshold:
                station_id = df.station_id.iloc[0]
                drop_list.append(station_id)
                frame_added = True
                continue
    return drop_list

def main():

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        #datefmt=datefmt,
        handlers=[
            #logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
    )

    config = tools.load_config('config.yaml')
    db_config = config['write']['db_conf']
    features = config['features']
    threshold = config['write']['threshold']
    solar_dir = os.path.join(config['data']['raw_dir'], 'solar')
    wind_dir = os.path.join(config['data']['raw_dir'], 'wind')
    passw = getpass.getpass("Enter postgres users password: ")
    config['write']['db_conf']['passw'] = passw

    pv_features, wind_features = relevant_features(features=features)
    master_data = tools.get_master_data(db_config=db_config)

    logging.info('Getting distinct station ids')
    station_ids = get_station_ids(db_config=db_config)
    logging.info('Getting station dataframes')
    frames = iterate_stations(db_config=db_config,
                              station_ids=station_ids)
    logging.info(f'Using an accepted threshold of {threshold*100}% missing rows per column, from {len(frames)} dataframes only remain:')
    if config['write']['clean_pv']:
        drop_pv_stations = get_drop_list(frames=frames,
                                         features=pv_features,
                                         threshold=threshold)
        logging.info(f'{len(frames)-len(drop_pv_stations)} stations for PV power')
        os.makedirs(solar_dir, exist_ok=True)
        pv_features.insert(0, 'station_id')
        for df in frames:
            station_id = df.station_id.iloc[0]
            file_name = f'Station_{station_id}.csv'
            if not station_id in drop_pv_stations:
                df[pv_features].to_csv(os.path.join(solar_dir, file_name))
    if config['write']['clean_wind']:
        drop_wind_stations = get_drop_list(frames=frames,
                                           features=wind_features,
                                           threshold=threshold)
        logging.info(f'{len(frames)-len(drop_wind_stations)} stations for wind power')
        os.makedirs(wind_dir, exist_ok=True)
        wind_features.insert(0, 'station_id')
        for df in frames:
            station_id = df.station_id.iloc[0]
            file_name = f'Station_{station_id}.csv'
            if not station_id in drop_wind_stations:
                df[wind_features].to_csv(os.path.join(wind_dir, file_name))
if __name__ == '__main__':
    main()