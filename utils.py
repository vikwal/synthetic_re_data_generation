import os
import yaml
import pickle
import getpass
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_config(config_path: str) -> dict:
    """
    Load the configuration file.
    :param config_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_master_data(db_config: dict):
    """
    Get the master data from the database.
    :param db_config: Database configuration dictionary.
    :return: DataFrame with the master data.
    """
    conn, cursor = connect_db(db_config)
    query = f"""
            SELECT * FROM masterdata;
        """
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    conn.close()
    return df

def to_pickle(path: str,
              name: str,
              obj):
    """
    Save an object to a pickle file.
    :param path: Directory to save the pickle file.
    :param name: Name of the pickle file.
    :param obj: Object to be saved.
    """
    dir = os.path.join(path, name)
    with open(dir, "wb") as file:
        pickle.dump(obj, file)

def connect_db(conf: dict):
    """
    Connect to the PostgreSQL database using the provided configuration.
    :param conf: Configuration dictionary.
    :return: Connection and cursor objects.
    """
    if conf['passw'] == '':
        passw = getpass.getpass("Enter postgres users password: ")
        conf['passw'] = passw
    conn = psycopg2.connect(
        host=conf['host'],
        database=conf['database'],
        port=conf['port'],
        user=conf['user'],
        password=conf['passw']
    )
    return conn, conn.cursor()

def days_timedelta(date: str,
                   days: int):
    """
    Calculate a new date by adding a number of days to the given date.
    :param date: Original date in the format "YYYY-MM-DD".
    :param days: Number of days to add.
    :return: New date in the format "YYYY-MM-DD".
    """
    date_object = datetime.strptime(date, "%Y-%m-%d")
    new_date = date_object + timedelta(days=days)
    new_date = new_date.strftime("%Y-%m-%d")
    return new_date

def knn_imputer(data: pd.DataFrame,
               n_neighbors: int = 5):
    # To help KNNImputer estimating the temporal saisonalities we add encoded temporal features.
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df_scaled)), columns=data.columns, index=data.index)
    df.drop(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'], axis=1, inplace=True)
    return df