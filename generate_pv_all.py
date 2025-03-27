import os
import pandas as pd
import numpy as np
from generate_utils import load_config
from generate_pv_utils import get_features,generate_pv_power

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def main() -> None:

    config_path = "config.yaml"
    config = load_config(config_path)

    dir = config['data']['solar_dir']
    params = config['synth']
    target_dir = 'data/pickle/solar' #Can be added to the config.yaml
    adj_params = config['adjustable_pv_params']

    files = os.listdir(dir)

    for file in files:

        data = pd.read_csv(os.path.join(dir, file))
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    
        data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
        data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)

        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(data)
        df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df_scaled)), columns=data.columns, index=data.index)
        df.drop(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'], axis=1, inplace=True)

        total_irradiance, cell_temperature = get_features(data=df,
                                                  params=params,
                                                  adj_params=adj_params)
        total = total_irradiance['poa_global']
        direct = total_irradiance['poa_direct']
        diffuse = total_irradiance['poa_diffuse']

        power = generate_pv_power(total_irradiance=total,
                                cell_temperature=cell_temperature,
                                adj_params=adj_params)
        
        df['Power'] = power
        df['Total'] = total
        df['Direct'] = direct
        df['Diffuse'] = diffuse

        os.makedirs(target_dir, exist_ok=True)
        file_name = f'Station_{str(int(df.STATIONS_ID.unique()[0]))}.pkl'
        new_dir = os.path.join(target_dir, file_name)
        df.to_pickle(new_dir)

if __name__ == "__main__":
    main()