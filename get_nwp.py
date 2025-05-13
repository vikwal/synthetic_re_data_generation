import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional

import utils


def get_nearest_point(db_config: dict,
                      table: str,
                      latitude: float,
                      longitude: float) -> Tuple[float, float]:
    """
    Get the next point in the grid.
    :param grid: List of grid points [(latitude, longitude), ...].
    :param longitude: Longitude of the current point.
    :param latitude: Latitude of the current point.
    :return: Next point in the grid.
    """
    db_config['database'] = db_config['database_frcst']
    conn, cursor = utils.connect_db(db_config)
    query = f"""
        SELECT DISTINCT ST_X(geom), ST_Y(geom)
        FROM {table}
        WHERE starttime = (SELECT MIN(starttime) FROM {table});
    """
    cursor.execute(query)
    data = cursor.fetchall()
    min_distance = float('inf')
    closest_point = None
    for lat, lon in data:
        distance = ((lat - latitude) ** 2 + (lon - longitude) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_point = (lat, lon)
    conn.close()
    return closest_point

def get_level_from_height(db_config: dict,
                          height: float) -> Tuple[float, float]:
    """
    Get the toplevel and bottomlevel from the database.
    :param db_config: Database configuration dictionary.
    :return: List of tuples with toplevel and bottomlevel.
    """
    db_config['database'] = db_config['database_frcst']
    conn, cursor = utils.connect_db(db_config)
    query = f"""
        SELECT DISTINCT toplevel, bottomlevel
        FROM multilevelfields;
    """
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(data, columns=columns)
    level = data.loc[data['toplevel'] <= height & data['bottomlevel'] >= height]
    toplevel = level['toplevel'].values[0]
    bottomlevel = level['bottomlevel'].values[0]
    conn.close()
    return toplevel, bottomlevel

def query_for_location(config: dict,
                       table: str,
                       latitude: float,
                       longitude: float) -> pd.DataFrame:
    """
    Query the database for a specific location.
    :param
    cursor: Database cursor object.
    :param latitude: Latitude of the location.
    :param longitude: Longitude of the location.
    :return: DataFrame with the queried data.
    """
    db_config = config['write']['db_conf']
    db_config['database'] = db_config['database_frcst']
    if table == 'singlelevelfields':
        vars = config['data']['hor_vars'].copy()
        vars[:0] = ['starttime', 'forecasttime']
        columns = ", ".join(vars)
        query = f"""
            SELECT {columns} FROM {table}
            WHERE ST_X(geom) = {latitude} AND ST_Y(geom) = {longitude}
            ORDER BY starttime, forecasttime;
        """
    elif table == 'multilevelfields':
        vars = config['data']['vert_vars'].copy()
        vars[:0] = ['starttime', 'forecasttime', 'toplevel', 'bottomlevel']
        columns = ", ".join(vars)
        query = f"""
            SELECT {columns} FROM {table}
            WHERE ST_X(geom) = {latitude} AND ST_Y(geom) = {longitude}
            ORDER BY starttime, forecasttime, toplevel;
        """
    elif table == 'analysisfields':
        vars = ['w_vert'].copy()
        vars[:0] = ['starttime', 'forecasttime', 'toplevel', 'bottomlevel']
        columns = ", ".join(vars)
        query = f"""
            SELECT {columns} FROM {table}
            WHERE ST_X(geom) = {latitude} AND ST_Y(geom) = {longitude}
            ORDER BY starttime, forecasttime, toplevel;
        """
    conn, cursor = utils.connect_db(db_config)
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        if data:
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)
        else:
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame([], columns=columns)
        conn.close()
        return df
    except Exception as e:
        print(f"Error occured when executing query: {e}")
        return pd.DataFrame()

def get_vertical_wind(config: dict,
                      latitude: float,
                      longitude: float) -> pd.DataFrame:
    w_vert = query_for_location(config=config,
                                table="analysisfields",
                                latitude=latitude,
                                longitude=longitude)
    w_vert['starttime'] = pd.to_datetime(w_vert['starttime'], utc=True)
    w_vert['timestamp'] = w_vert['starttime'] + pd.to_timedelta(w_vert['forecasttime'], unit='h')
    w_vert.set_index('timestamp', inplace=True)
    w_vert.drop(['starttime', 'forecasttime', 'toplevel', 'bottomlevel'], axis=1, inplace=True)
    return w_vert

def get_data_from_db(config: dict,
                     stations: List[str],
                     master_data: pd.DataFrame,
                     table: str):
    db_config = config['write']['db_conf']
    ids = []
    for id in tqdm(stations, desc='Extracting station ids'):
        match = re.split(r'_|\.csv', id)
        if len(match) > 1:
            ids.append(match[1])
    ids = list(set(ids))
    # Get the station ids from the master data
    stations_of_interest = master_data[master_data['station_id'].isin(ids)][['latitude', 'longitude']].values
    ids = list(master_data[master_data['station_id'].isin(ids)]['station_id'])

    nearest_points = []
    # obtaining the nearest grids in analysisfields is sufficient, because the grid is the same for both tables single and multilevelfields
    for lat, lon in tqdm(stations_of_interest, desc='Finding nearest grid points'):
        nearest_point = get_nearest_point(db_config=db_config,
                                          table="singlelevelfields", # maybe in future change to analysisfields
                                          latitude=lat,
                                          longitude=lon)
        nearest_points.append(nearest_point)
    forecasts = []
    vertical_winds = []
    for lat, lon in tqdm(nearest_points, desc=f'Get {table} forecasts'):
        if config['write']['get_wind']:
            forecast = query_for_location(config=config,
                                            table=table,
                                            latitude=lat,
                                            longitude=lon)
            forecasts.append(forecast)
        if config['write']['get_vertical_wind']:
            w_vert = get_vertical_wind(config=config,
                                    latitude=lat,
                                    longitude=lon)
            vertical_winds.append(w_vert)
    if forecasts:
        os.makedirs(os.path.join('data', table), exist_ok=True)
        for id, fc in tqdm(zip(ids, forecasts), desc=f'Saving {table} forecasts'):
            fc.to_csv(os.path.join('data', table, f'ML_Station_{id}.csv'), index=False)
    if vertical_winds:
        os.makedirs(os.path.join('data', 'vertical_wind'), exist_ok=True)
        for id, w_vert in tqdm(zip(ids, vertical_winds), desc=f'Saving vertical wind data'):
            w_vert.to_csv(os.path.join('data', 'vertical_wind', f'w_vert_{id}.csv'))

def main():
    config = utils.load_config("config.yaml")
    db_config = config['write']['db_conf']
    parser = argparse.ArgumentParser(description="Extract and save NWP forecasts for PV and wind stations.")
    parser.add_argument('--get_pv', action='store_true', help='Extract PV forecast data')
    parser.add_argument('--get_wind', action='store_true', help='Extract wind forecast data')
    parser.add_argument('--get_wind_vert', action='store_true', help='Include vertical wind variable in export')
    args = parser.parse_args()
    config['write']['get_pv'] = args.get_pv
    config['write']['get_wind'] = args.get_wind
    config['write']['get_wind_vertical'] = args.get_wind_vert

    logging.info(f'Get singlelevelfields: {config["write"]["get_pv"]}')
    logging.info(f'Get multilevelfields: {config["write"]["get_wind"]}')
    logging.info(f'Get vertical wind: {config["write"]["get_wind_vertical"]}')

    master_data = utils.get_master_data(db_config=db_config)
    if config['write']['get_pv']:
        pv_dir = os.path.join(config['data']['raw_dir'], 'solar')
        pv_stations = os.listdir(pv_dir)
        get_data_from_db(config=config,
                         stations=pv_stations,
                         master_data=master_data,
                         table='singlelevelfields')
    if config['write']['get_wind'] | config['write']['get_vertical_wind']:
        wind_dir = os.path.join(config['data']['raw_dir'], 'wind')
        wind_stations = os.listdir(wind_dir)
        get_data_from_db(config=config,
                         stations=wind_stations,
                         master_data=master_data,
                         table='multilevelfields')

if __name__ == '__main__':
    main()