import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import utils

def get_master_data(db_config: dict):
    """
    Get the master data from the database.
    :param db_config: Database configuration dictionary.
    :return: DataFrame with the master data.
    """
    db_config['database'] = db_config['database_obs']
    conn, cursor = utils.connect_db(db_config)
    query = f"""
            SELECT * FROM masterdata;
        """
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    conn.close()
    return df

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
        WHERE starttime = (SELECT MIN(starttime) FROM singlelevelfields);
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
        vars = config['data']['hor_vars']
        vars[:0] = ['starttime', 'forecasttime']
        columns = ", ".join(vars)
        query = f"""
            SELECT {columns} FROM {table}
            WHERE ST_X(geom) = {latitude} AND ST_Y(geom) = {longitude}
            ORDER BY starttime, forecasttime;
        """
    elif table == 'multilevelfields':
        vars = config['data']['vert_vars']
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

def main():

    config = utils.load_config("config.yaml")
    db_config = config['write']['db_conf']
    master_data = get_master_data(db_config=db_config)

    pv_stations = os.listdir(config['data']['solar_dir'])
    wind_stations = os.listdir(config['data']['wind_dir'])
    pv_ids, wind_ids = [], []
    for pv_id in tqdm(pv_stations, desc='Extracting PV station ids'):
        match = re.split(r'_|\.csv', pv_id)
        if len(match) > 1:
            pv_ids.append(match[1])
    pv_ids = list(set(pv_ids))
    for wind_id in tqdm(wind_stations, desc='Extracting PV station ids'):
        match = re.split(r'_|\.csv', wind_id)
        if len(match) > 1:
            wind_ids.append(match[1])
    wind_ids = list(set(wind_ids))
    # Get the station ids from the master data
    pv_stations_of_interest = master_data[master_data['station_id'].isin(pv_ids)][['latitude', 'longitude']].values
    wind_stations_of_interest = master_data[master_data['station_id'].isin(wind_ids)][['latitude', 'longitude']].values

    pv_ids = list(master_data[master_data['station_id'].isin(pv_ids)]['station_id'])
    wind_ids = list(master_data[master_data['station_id'].isin(wind_ids)]['station_id'])

    nearest_pv_grids = []
    # obtaining the nearest grids in singlelevelfields is sufficient, because the grid is the same for both tables single and multilevelfields
    for lat, lon in tqdm(pv_stations_of_interest, desc='Finding nearest grid points for PV'):
        nearest_point = get_nearest_point(db_config=db_config,
                                          table="singlelevelfields",
                                          latitude=lat,
                                          longitude=lon)
        nearest_pv_grids.append(nearest_point)
    nearest_wind_grids = []
    for lat, lon in tqdm(wind_stations_of_interest, desc='Finding nearest grid points for Wind'):
        nearest_point = get_nearest_point(db_config=db_config,
                                          table="singlelevelfields",
                                          latitude=lat,
                                          longitude=lon)
        nearest_wind_grids.append(nearest_point)

    sl_forecasts = []
    for lat, lon in tqdm(nearest_pv_grids, desc='Get single level forecasts'):
        singlelevel_fc = query_for_location(config=config,
                                            table="singlelevelfields",
                                            latitude=lat,
                                            longitude=lon)
        sl_forecasts.append(singlelevel_fc)
    ml_forecasts = []
    for lat, lon in tqdm(nearest_pv_grids, desc='Get multi level forecasts'):
        multilevel_fc = query_for_location(config=config,
                                           table="multilevelfields",
                                           latitude=lat,
                                           longitude=lon)
        ml_forecasts.append(multilevel_fc)

    os.makedirs(config['data']['sl_forecasts_dir'], exist_ok=True)
    os.makedirs(config['data']['ml_forecasts_dir'], exist_ok=True)
    # Save the forecasts to CSV files
    for id, singlelevel_fc in tqdm(zip(pv_ids, sl_forecasts), desc='Saving single level forecasts'):
        singlelevel_fc.to_csv(os.path.join(config['data']['sl_forecasts_dir'], f'SL_Station_{id}.csv'), index=False)
    for id, multilevel_fc in tqdm(zip(wind_ids, ml_forecasts), desc='Saving multi level forecasts'):
        multilevel_fc.to_csv(os.path.join(config['data']['ml_forecasts_dir'], f'ML_Station_{id}.csv'), index=False)

if __name__ == '__main__':
    main()