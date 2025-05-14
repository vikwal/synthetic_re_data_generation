import os
import re
import pandas as pd

import utils

def get_turbine_master(turbine_path: str,
                 cp_path: str,
                 specs_path: str,
                 params: dict):
    turbines = params['turbines']
    # read power curves
    power_curves = pd.read_csv(turbine_path, sep=";", decimal=".")
    power_curves.set_index('wind_speed', inplace=True)
    power_curves = power_curves[turbines] * 1000
    # read cp curves
    cp_curves = pd.read_csv(cp_path, sep=";", decimal=".")
    cp_curves.set_index('wind_speed', inplace=True)
    cp_curves = cp_curves[turbines]
    # read turbine specs
    turbine_specs = pd.read_csv(specs_path)
    masterdata = []
    for height, turbine in enumerate(turbines):
        entry = {
            'turbine': turbine,
            'diameter': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Rotordurchmesser'].values[0]),
            'height': params['hub_heights'][height],
            #'cut_in_speed': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Einschaltgeschwindigkeit'].values[0]),
            #'cut_out_speed': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Abschaltgeschwindigkeit'].values[0]),
            #'rated_speed': float(turbine_specs.loc[turbine_specs['Turbine'] == turbine, 'Nenngeschwindigkeit'].values[0]),
        }
        masterdata.append(entry)
    return power_curves, cp_curves, pd.DataFrame(masterdata)

def main() -> None:
    config = utils.load_config('config.yaml')
    synth_dir = config['data']['synth_dir']

    wind_dir = os.path.join(synth_dir, 'wind')
    pv_dir = os.path.join(synth_dir, 'solar')

    masterdata = utils.get_master_data(db_config=config['write']['db_conf'])
    # get station master data
    if os.path.exists(pv_dir):
        target_dir = os.path.join(pv_dir, 'masterdata')
        os.makedirs(target_dir, exist_ok=True)
        pv_data = os.listdir(pv_dir)
        pv_stations = [re.search(r'synth_(\d{5})\.csv', name).group(1) for name in pv_data]
        masterdata.loc[masterdata.station_id.isin(pv_stations)]

    if os.path.exists(wind_dir):
        target_dir = os.path.join(wind_dir, 'masterdata')
        os.makedirs(target_dir, exist_ok=True)
        wind_data = os.listdir(wind_dir)
        wind_stations = [re.search(r'synth_(\d{5})\.csv', name).group(1) for name in wind_data if name != 'masterdata']
        stations_master = masterdata.loc[masterdata.station_id.isin(wind_stations)]

        # get turbine masterdata
        power_curve_path = os.path.join(config['data']['turbine_dir'], config['data']['turbine_power'])
        cp_path = os.path.join(config['data']['turbine_dir'], config['data']['turbine_cp'])
        specs_path = os.path.join(config['data']['turbine_dir'], config['data']['turbine_specs'])

        power_curves, cp_curves, turbine_master = get_turbine_master(
            turbine_path=power_curve_path,
            cp_path=cp_path,
            specs_path=specs_path,
            params=config['wind_params']
        )
        power_curves.to_csv(os.path.join(target_dir, 'power_curves.csv'))
        #cp_curves.to_csv(os.path.join(target_dir, 'cp_curves.csv'))
        turbine_master.to_csv(os.path.join(target_dir, 'turbine_master.csv'), index=False)
        stations_master.to_csv(os.path.join(target_dir, 'stations_master.csv'), index=False)


if __name__ == '__main__':
    main()