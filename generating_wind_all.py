import os
import pandas as pd
import numpy as np
from generate_utils import load_config,get_features,generate_wind_power

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler



def main() -> None:

    config_path = "config.yaml"
    config = load_config(config_path)

    dir = config['data']['wind_dir']
    target_dir = 'data/pickle/wind' #Can be added to the config.yaml
    params = config['synth']

    adj_params = config['adjustable_wind_params']


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

        df = get_features(data=df,
                  params=params,
                  adj_params=adj_params)

        power = generate_wind_power(data=df,
                            params=params, 
                            adj_params=adj_params)
        
        df['Power'] = power

        os.makedirs(target_dir, exist_ok=True)
        file_name = f'Station_{str(int(df.STATIONS_ID.unique()[0]))}.pkl'
        new_dir = os.path.join(target_dir, file_name)
        df.to_pickle(new_dir)

if __name__ == "__main__":

    main()