# Process raw data and store in data base

import os
import re
import pickle
import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from datetime import datetime
from psycopg2.extras import execute_values

import utils


def get_all_stations_files(dir: str,
                           column_names: list,
                           vars: list):
    # get all station files
    station_files = []
    for root, dirs, files in os.walk(dir):
        if not any(var in root for var in vars):
            continue
        for file in files:
            if 'Stationen' in file:
                station_files.append(os.path.join(root, file))
    # read all station files
    stations = None
    for station in station_files:
        df_station = pd.read_fwf(station,
                                encoding='latin1',
                                skiprows=2,
                                names=column_names)
        df_station['von_datum'] = pd.to_datetime(df_station['von_datum'], format='%Y%m%d')
        df_station['bis_datum'] = pd.to_datetime(df_station['bis_datum'], format='%Y%m%d')
        df_station.drop(['Abgabe'], axis=1, inplace=True)
        table_name = station.split('/')[-2]
        df_station['Table'] = table_name
        if station != None:
            stations = pd.concat([stations, df_station], ignore_index=True)
        else:
            stations = df_station.copy()
    stations['Stations_id'] = stations["Stations_id"].apply(lambda x: f"{x:05d}")
    return stations

def filter_stations(dir: str,
                    stations: pd.DataFrame,
                    to_date: str,
                    from_date: str,
                    vars: list):
    for root, dirs, files in os.walk(dir):
        if not any(var in root for var in vars):
            continue
        print(f'In {root} are {len(files)} files.')
    print('Group sizes of all tables.')
    print('')
    print(stations.groupby('Table').size().sort_index(ascending=False))
    stations.drop(stations[stations.bis_datum < to_date].index, inplace=True)
    stations.drop(stations[stations.von_datum > from_date].index, inplace=True)
    counts = stations.groupby(['Stations_id']).size().value_counts().sort_index()
    print(f'{len(stations.Stations_id.unique())} unique weather stations \n')
    for e, c in enumerate(counts, start=1):
        print(f'{c} stations with values in {e} tables.')
    print('\nAbsolute number of stations for:')
    stations.groupby('Table').size()
    grouped = stations.groupby(['Stations_id'])['Table'].apply(list)
    combination_counts = {}
    for categories in grouped:
        for pair in combinations(categories, 2):
            pair = tuple(sorted(pair))
            combination_counts[pair] = combination_counts.get(pair, 0) + 1
    grouped_counts = stations.groupby(["Stations_id", "Table"]).size().unstack(fill_value=0)
    single_entries = grouped_counts[grouped_counts.sum(axis=1) == 1].sum()
    matrix = pd.DataFrame(0, index=vars, columns=vars)
    for (cat1, cat2), count in combination_counts.items():
        matrix.loc[cat1, cat2] += count
        matrix.loc[cat2, cat1] += count
    for category in vars:
        if category in single_entries:
            matrix.loc[category, category] = single_entries[category]
    print(matrix)
    temp_temp = str(matrix['air_temperature'].loc['air_temperature'])
    #temp_solar = str(matrix['air_temperature'].loc['solar'])
    print('')
    print(f'There are {temp_temp} stations where air_temperature was the only measurement.')
    #print(f'There are {temp_solar} stations where air_temperature and solar are measured.')
    return stations

def get_valid_stations(stations: pd.DataFrame,
                       vars: list):
    valid_stations = (stations.groupby("Stations_id")["Table"].apply(lambda x: set(x) == set(vars)))
    valid_stations_ids = valid_stations[valid_stations].index
    stations = stations[stations["Stations_id"].isin(valid_stations_ids)]
    print('')
    print(f'{len(stations.Stations_id.unique())} unique weather stations after cleaning. \n')
    return stations

def get_file_paths(dir: str,
                   stations: pd.DataFrame,
                   to_date: str,
                   vars: list):
    stations_ids = stations.Stations_id.unique()
    # get list of relevant station files
    file_paths = []
    for root, dirs, files in os.walk(dir):
        if not any(var in root for var in vars):
            continue
        for file in files:
            if 'produkt' in file:
                file_split = re.split(r'[_\.]', file)
                to_date_file = datetime.strptime(file_split[5], '%Y%m%d')
                in_date = to_date_file == datetime.strptime(to_date, '%Y-%m-%d')
                in_ids = file_split[-2] in stations_ids
                if in_date and in_ids:
                    file_paths.append(os.path.join(root, file))
    # check if the list of files contains all relevant stations
    extracted_data = []
    for path in file_paths:
        parts = re.split(r'[\/_.]', path)
        category = parts[-9]
        if category == 'temperature':
            category = 'air_temperature'
        if category == 'test':
            category = 'wind_test'
        date = parts[-3]
        station_id = parts[-2]
        extracted_data.append([category, date, station_id])
    df_proof = pd.DataFrame(extracted_data, columns=["Category", "Date", "Stations_ID"])
    proof = [id for id in df_proof.Stations_ID.unique() if not id in stations_ids]
    if proof:
        print(f'There are inconsistent stations ({len(stations_ids)}) in your dataframe for to_date={to_date}.')
    else:
        print(f'The stations ({len(stations_ids)}) are consistent with your stations_ids list for to_date={to_date}.')
    grouped = defaultdict(lambda: {var: None for var in vars})
    for path in file_paths:
        parts = re.split(r'[\/_.]', path)
        category = parts[-9]
        station_id = parts[-2]
        if category == 'temperature':
            category = 'air_temperature'
        if category == 'test':
            category = 'wind_test'
        grouped[station_id][category] = path
    structured_list = []
    for station_id, paths in grouped.items():
        if all(paths[cat] for cat in vars):
            structured_list.append([paths[var] for var in vars])
    return structured_list

def rename_columns(data: pd.DataFrame,
                   features: dict) -> pd.DataFrame:
    df = data.copy()
    mapping = {}
    for key, value in features.items():
        if 'old_name' not in value.keys():
            continue
        if value['old_name'] in df.columns:
            mapping[value['old_name']] = value['name']
    df.rename(columns=mapping, inplace=True)
    return df

def make_final_frames(path_list: list,
                      stations: pd.DataFrame,
                      from_date: str,
                      features: dict):
    final_df = []
    master_data = []
    for ele in tqdm(path_list, desc='Creating final dataframes'):
        station_df = None
        for file in ele:
            station_id = re.search(r'_(\d{5})\.txt$', file).group(1)
            raw = pd.read_csv(file, sep=';')
            raw['timestamp'] = pd.to_datetime(raw['MESS_DATUM'], format='%Y%m%d%H%M', utc=True)
            raw.set_index('timestamp', inplace=True)
            raw.drop(['MESS_DATUM', '  QN', 'eor'], axis=1, inplace=True)
            if 'RWS_IND_10' in raw.columns:
                raw.drop(['RWS_IND_10'], axis=1, inplace=True)
            raw['STATIONS_ID'] = station_id
            if station_df is not None:
                station_df = pd.merge(station_df, raw, how='inner', on=['timestamp', 'STATIONS_ID'])
            else:
                station_df = raw.copy()
        station_df.replace({-999.0: np.nan, -9999.0: np.nan}, inplace=True)
        station_id = station_df.STATIONS_ID.iloc[0]
        #station_df.drop(['STATIONS_ID'], axis=1, inplace=True)
        station_df.rename(columns={'STATIONS_ID': 'station_id'}, inplace=True)
        cols = ['Stationshoehe', 'geoBreite', 'geoLaenge']
        col_vals = stations[stations.Stations_id == station_id][cols].drop_duplicates().values
        master_cols = col_vals.reshape(-1).tolist()
        master_cols.insert(0, station_id)
        station_df = rename_columns(data=station_df, features=features)
        final_df.append(station_df[from_date:])
        master_data.append(master_cols)
    return final_df, master_data

def create_stations_table(db_config: dict,
                          column_names: list):
    conn, cursor = utils.connect_db(db_config)
    columns = ', '.join([f'{col} REAL' for col in column_names if col != 'station_id'])
    query = f"""
        CREATE TABLE stations (
            timestamp TIMESTAMP WITH TIME ZONE,
            station_id VARCHAR(5),
            {columns},
            PRIMARY KEY (timestamp, station_id)
        );
        """
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    print('Stations table created.')

def write_tables(db_config: dict,
                 df_list: list,
                 master_data=None):
    conn, cursor = utils.connect_db(db_config)
    if master_data:
        query = f"""
                INSERT INTO MasterData (station_id, station_height, latitude, longitude)
                VALUES %s
                ON CONFLICT (station_id) DO UPDATE
                SET station_height = EXCLUDED.station_height,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude;
            """
        execute_values(cursor, query, master_data)
        conn.commit()
        print('Master data written to db.')
    primary_key = ['timestamp', 'station_id']
    for df in tqdm(df_list, desc='Writing data to db'):
        df.reset_index(inplace=True)
        columns = ', '.join(df.columns)
        update_columns = [col for col in df.columns if col not in primary_key]
        if db_config['do_update']:
            update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
            conflict_sql_part = f"DO UPDATE SET {update_clause}"
        else:
            conflict_sql_part = "DO NOTHING"
        # in case of updating conflicting columns DO UPDATE SET {update_clause}
        query = f"""
            INSERT INTO stations ({columns})
            VALUES %s
            ON CONFLICT ({', '.join(primary_key)}) {conflict_sql_part}
        """
        data = [x.tolist() for x in df.to_numpy()]
        execute_values(cursor, query, data)
        conn.commit()
    cursor.close()
    conn.close()

def main() -> None:
    config = utils.load_config("config.yaml")
    directory = config['data']['dir']
    target_dir = config['data']['pkl_dir']
    os.makedirs(target_dir, exist_ok=True)
    features = config['features']
    vars = config['scraping']['vars']
    passw = getpass.getpass("Enter postgres users password: ")
    config['write']['db_conf']['passw'] = passw
    db_config = config['write']['db_conf']
    create_table = config['write']['create_table']
    recent_pkl = config['write']['recent_pkl']
    master_pkl = config['write']['master_pkl']
    historical_pkl = config['write']['historical_pkl']
    write_historical = config['write']['write_historical']
    write_recent = config['write']['write_recent']

    # no data older than this date should be considered (because nwp data is only available from here)
    from_date = config['data']['from_date']
    hist_date = config['data']['hist_date']
    column_names = [
        "Stations_id", "von_datum", "bis_datum", "Stationshoehe",
        "geoBreite", "geoLaenge", "Stationsname", "Bundesland", "Abgabe"
    ]
    stations = get_all_stations_files(dir=directory,
                                      column_names=column_names,
                                      vars=vars)
    to_date = str(stations['bis_datum'].max().date())
    to_date = utils.days_timedelta(date=to_date, days=-1)
    stations = filter_stations(dir=directory,
                               stations=stations,
                               to_date=to_date,
                               from_date=from_date,
                               vars=vars)
    stations = get_valid_stations(stations=stations,
                                  vars=vars)
    if write_recent:
        recent_paths = get_file_paths(dir=directory,
                                    stations=stations,
                                    to_date=to_date,
                                    vars=vars)
    if write_historical:
        historical_paths = get_file_paths(dir=directory,
                                      stations=stations,
                                      to_date=hist_date,
                                      vars=vars)
    recent_dfs_path = os.path.join(target_dir, recent_pkl)
    master_pkl_path = os.path.join(target_dir, master_pkl)
    if os.path.exists(recent_dfs_path) & os.path.exists(master_pkl_path):
        with open(recent_dfs_path, 'rb') as file:
            recent_dfs = pickle.load(file)
        with open(master_pkl_path, 'rb') as file:
            master_data = pickle.load(file)
    else:
        recent_from_date = utils.days_timedelta(date=hist_date,
                                                days=1)
        if write_recent:
            print('Create recent dfs.')
            recent_dfs, master_data = make_final_frames(path_list=recent_paths,
                                                        stations=stations,
                                                        from_date=recent_from_date,
                                                        features=features)
            utils.to_pickle(path=target_dir,
                            name=recent_pkl,
                            obj=recent_dfs)
            utils.to_pickle(path=target_dir,
                            name=master_pkl,
                            obj=master_data)
    historical_dfs_path = os.path.join(target_dir, historical_pkl)
    if os.path.exists(historical_dfs_path) & os.path.exists(master_pkl_path):
        with open(historical_dfs_path, 'rb') as file:
            historical_dfs = pickle.load(file)
        with open(master_pkl_path, 'rb') as file:
            master_data = pickle.load(file)
    else:
        if write_historical:
            print('Create historical dfs.')
            historical_dfs, master_data = make_final_frames(path_list=historical_paths,
                                                            stations=stations,
                                                            from_date=from_date,
                                                            features=features)
            utils.to_pickle(path=target_dir,
                    name=historical_pkl,
                    obj=historical_dfs)
            utils.to_pickle(path=target_dir,
                    name=master_pkl,
                    obj=master_data)
    if create_table:
        column_names = recent_dfs[0].columns
        create_stations_table(db_config=db_config,
                              column_names=column_names)
    if write_historical:
        print('Write historical data to db.')
        write_tables(db_config=db_config,
                    df_list=historical_dfs,
                    master_data=master_data if config['write']['write_master'] else None)
    if write_recent:
        print('Write recent data to db.')
        write_tables(db_config=db_config,
                    df_list=recent_dfs)

if __name__ == '__main__':
    main()