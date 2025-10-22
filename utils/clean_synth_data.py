# Clean database data and drop stations with too much missing values

import os
import getpass
import pandas as pd
from tqdm import tqdm
from typing import List
import logging
import argparse

from . import tools



def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        #datefmt=datefmt,
        handlers=[
            #logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
    )

    parser = argparse.ArgumentParser(description="Clean Synthetic Data")
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=str,
                        help='Please select config (like config_wind)')
    args = parser.parse_args()

    config = tools.load_config(f'./configs/{args.config}.yaml')
    threshold = config['write']['threshold']
    params = config['params']
    ageing_flag = 'noage'
    if params['apply_ageing']:
        ageing_flag = 'age'
    if params['hourly_resolution']:
        resolution = 'hourly'
    else:
        resolution = '10min'
    if 'wind' in args.config:
        energy = 'wind'
    elif 'pv' in args.config:
        energy = 'solar'

    synth_dir = os.path.join(config['data']['synth_dir'],
                             f'{energy}',
                             f'{energy}_{resolution}_{ageing_flag}')
    target_dir

    missing_analysis_path = config['data']['missing_analysis']
    missing_analysis = pd.read_csv(missing_analysis_path, dtype={'station_id': str})
    missing_analysis.set_index('station_id', inplace=True)

    station_ids = []
    for index, row in tqdm(missing_analysis.iterrows(), desc=f'Cleaning data for threshold {threshold}'):
        skip_station = False
        for feature in missing_analysis.columns:
            if row[feature] > (threshold * 100):
                skip_station = True
                break
        if skip_station:
            continue
        station_ids.append(index)

    logging.info(f'Stations to keep: {len(station_ids)}')

if __name__ == '__main__':
    main()