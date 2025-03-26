# Process raw data and store in data base

import os
import re
import yaml
import pickle
import getpass
import psycopg2
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import combinations
from shapely.geometry import Point
from collections import defaultdict
from datetime import datetime, timedelta


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def connect_db(conf):
    conn = psycopg2.connect(
        host=conf['host'],
        database=conf['database'],
        user=conf['user'],  
        password=conf['passw'] 
    )
    return conn, conn.cursor()
   
def execute_query(cursor, 
                  query: str):
    try:
        cursor.execute(query)
        print(f'Query succeed.')
    except psycopg2.Error as ex:
        print('Query failed.')

def add_one_day(date: str):
    date_object = datetime.strptime(date, "%Y-%m-%d")
    new_date = date_object + timedelta(days=1)
    new_date = new_date.strftime("%Y-%m-%d")
    return new_date  

    
def get_all_stations_files(dir: str,
                           params: dict,
                           column_names: list,
                           vars: list):
    features = []
    for feature in params:
        if not feature == 'threshold':
            features.append(params[feature]['param'])
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
    table_counts = stations.groupby(['Stations_id', 'Table']).size()
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
    temp_solar = str(matrix['air_temperature'].loc['solar'])
    print('')
    print(f'There are {temp_temp} stations where air_temperature was the only measurement.')
    print(f'There are {temp_solar} stations where air_temperature and solar are measured.')
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
                in_date = not to_date_file < datetime.strptime(to_date, '%Y-%m-%d')  
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
        print(f'There are inconsistent stations in your dataframe for to_date={to_date}.')
    else:
        print(f'The stations are consistent with your stations_ids list for to_date={to_date}.')
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

def make_final_frames(path_list: list,
                      stations: pd.DataFrame,
                      from_date: str):
    final_df = []
    master_data = []
    for ele in tqdm(path_list):
        station_df = None
        for file in ele:
            station_id = re.search(r'_(\d{5})\.txt$', file).group(1)
            raw = pd.read_csv(file, sep=';')
            raw['timestamp'] = pd.to_datetime(raw['MESS_DATUM'], format='%Y%m%d%H%M')
            raw.set_index('timestamp', inplace=True)
            raw.drop(['MESS_DATUM', '  QN', 'eor'], axis=1, inplace=True)
            raw['STATIONS_ID'] = station_id
            if station_df is not None:
                station_df = pd.merge(station_df, raw, how='inner', on=['timestamp', 'STATIONS_ID'])
            else:
                station_df = raw.copy()
        station_df.replace({-999.0: np.nan, -9999.0: np.nan}, inplace=True)
        
        station_id = station_df.STATIONS_ID.iloc[0]
        cols = ['Stationshoehe', 'geoBreite', 'geoLaenge']
        col_vals = stations[stations.Stations_id == station_id][cols].drop_duplicates().values
        master_cols = col_vals.reshape(-1).tolist()
        master_cols.insert(0, station_id)
        final_df.append(station_df[from_date:])
        master_data.append(master_cols)
    return final_df, master_data

def create_tables(stations: pd.DataFrame,
                  db_config: dict,
                  columns: list):
    stations_ids = stations.Stations_id.unique()
    conn, cursor = connect_db(db_config)
    for station_id in stations_ids:
        table_name = f'Station_{station_id}'
        columns = ', '.join([f'{col} REAL' for col in columns])
        query = f'CREATE TABLE {table_name} (timestamp TIMESTAMP WITH TIME ZONE PRIMARY KEY, {columns})'
        execute_query(cursor=cursor, query=query)
    conn.commit()
    cursor.close()
    conn.close()

def write_tables(db_config: dict,
                 df_list: list,
                 master_data=None):
    conn, cursor = connect_db(db_config)
    if master_data:
        query = f"""
            INSERT INTO MasterData (stations_id, stations_height, latitude, longitude)
            VALUES (%s, %s, %s, %s)
        """
        for row in master_data:
            cursor.execute(query, (row[0], row[1], row[2], row[3]))
    
    conn.commit()
    cursor.close()
    conn.close()

def main() -> None:
    config = load_config("config.yaml")

    directory = config['data']['dir']
    target_dir = config['data']['final_data']
    params = config['synth']
    vars = config['scraping']['vars']
    passw = getpass.getpass("Enter postgres users password: ")
    config['write']['db_conf']['passw'] = passw
    db_config = config['write']['db_conf']
    
    # no data older than this date should be considered (because nwp data is only available from here)
    from_date = config['data']['from_date']
    to_date = config['data']['to_date']
    hist_date = config['data']['hist_date']
    column_names = [
    "Stations_id", "von_datum", "bis_datum", "Stationshoehe",
    "geoBreite", "geoLaenge", "Stationsname", "Bundesland", "Abgabe"
    ]
    stations = get_all_stations_files(dir=directory,
                                      params=params,
                                      column_names=column_names,
                                      vars=vars)
    stations = filter_stations(dir=directory,
                               stations=stations,
                               to_date=to_date,
                               from_date=from_date,
                               vars=vars)
    stations = get_valid_stations(stations=stations,
                                  vars=vars)
    recent_paths = get_file_paths(dir=directory,
                                  stations=stations,
                                  to_date=to_date,
                                  vars=vars)
    historical_paths = get_file_paths(dir=directory,
                                      stations=stations,
                                      to_date=hist_date,
                                      vars=vars)
    recent_from_date = add_one_day(hist_date)
    #recent_dfs, master_data = make_final_frames(path_list=recent_paths,
    #                               stations=stations,
    #                               from_date=recent_from_date)
    #historical_dfs, master_data = make_final_frames(path_list=historical_paths,
    #                                   stations=stations,
    #                                   from_date=from_date)
    columns = ['PP_10', 'TT_10', 'TM5_10', 'RF_10', 'TD_10', 'DS_10', 'GS_10', 'SD_10', 'LS_10', 'SLA_10', 'SLO_10', 'FF_10', 'DD_10']#recent_dfs[0].columns
    create_tables(stations=stations,
                  db_config=db_config,
                  columns=columns)
    
    
if __name__ == '__main__':
    main()