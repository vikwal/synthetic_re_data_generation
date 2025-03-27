# Clean database data and drop stations with too much missing values

import os
import getpass
import pandas as pd
from tqdm import tqdm
from typing import List

import utils

def get_station_ids(db_config: dict):
    conn, cursor = utils.connect_db(db_config)
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
    conn, cursor = utils.connect_db(db_config)
    frames = []
    for id in tqdm(station_ids):
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
            print(f"No data for station_id: {id} found.")
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
    
    config = utils.load_config('config.yaml')
    db_config = config['write']['db_conf']
    params = config['synth']
    threshold = config['write']['threshold']
    solar_dir = config['data']['solar_dir']
    wind_dir = config['data']['wind_dir']
    passw = getpass.getpass("Enter postgres users password: ")
    config['write']['db_conf']['passw'] = passw
    
    pv_features = [params['ghi']['param'],
                   params['dhi']['param'],
                   params['temperature']['param'],
                   params['dewpoint']['param'],
                   params['v_wind']['param']]
    wind_features = [params['v_wind']['param'],
                     params['temperature']['param'],
                     params['relhum']['param'],
                     params['sigma_wind_lat']['param'],
                     params['sigma_wind_lon']['param']]
    
    pv_features = [ele.lower() for ele in pv_features]
    wind_features = [ele.lower() for ele in wind_features]
    
    station_ids = get_station_ids(db_config=db_config)
    frames = iterate_stations(db_config=db_config,
                              station_ids=station_ids)
    drop_pv_stations = get_drop_list(frames=frames,
                                 features=pv_features,
                                 threshold=threshold)
    drop_wind_stations = get_drop_list(frames=frames,
                                    features=wind_features,
                                    threshold=threshold)
    print(f'Using an accepted threshold of {threshold*100}% missing rows per column, from {len(frames)} dataframes only remain:')
    print(f'{len(frames)-len(drop_pv_stations)} stations for PV power')
    print(f'{len(frames)-len(drop_wind_stations)} stations for wind power')
    os.makedirs(solar_dir, exist_ok=True)
    os.makedirs(wind_dir, exist_ok=True)
    for df in frames:
        station_id = df.station_id.iloc[0]
        file_name = f'Station_{station_id}.csv'
        if not station_id in drop_pv_stations:
            df.to_csv(os.path.join(solar_dir, file_name))
        if not station_id in drop_wind_stations:
            df.to_csv(os.path.join(wind_dir, file_name))
if __name__ == '__main__':
    main()