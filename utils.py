import os
import yaml
import pickle
import psycopg2
from datetime import datetime, timedelta

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def to_pickle(path: str,
              name: str,
              obj):
    dir = os.path.join(path, name)
    with open(dir, "wb") as file:
        pickle.dump(obj, file)

def connect_db(conf):
    conn = psycopg2.connect(
        host=conf['host'],
        database=conf['database'],
        user=conf['user'],  
        password=conf['passw'] 
    )
    return conn, conn.cursor()    

def days_timedelta(date: str,
                   days: int):
    date_object = datetime.strptime(date, "%Y-%m-%d")
    new_date = date_object + timedelta(days=days)
    new_date = new_date.strftime("%Y-%m-%d")
    return new_date  