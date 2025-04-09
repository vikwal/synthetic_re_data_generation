import os
import yaml
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

def get_all_stations(config,column_names):

    directory = 'data'
    params = config['synth']

    features = []
    for feature in params:
        if not feature == 'threshold':
            features.append(params[feature]['param'])


    station_files = []
    for root, dirs, files in os.walk(directory):
        if ('data/Archiv' in root) | ('data/final' in root):
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
        table_name = station.split('/')[1]
        df_station['Table'] = table_name
        if station != None:
            stations = pd.concat([stations, df_station], ignore_index=True)
        else:
            stations = df_station.copy()
    stations['Stations_id'] = stations["Stations_id"].apply(lambda x: f"{x:05d}")


    return stations

def get_station_list(dir):

    validStations = []

    files = os.listdir(dir)
    for file in files:
        validStations.append((file.split('_')[1]).split('.')[0])

    return validStations


def get_windspeed_at_height(data: pd.DataFrame,
           params: dict,
           adj_params: dict
          ):
    
    """
    Calculate the wind speed (v2) using different methods.

    Parameters:
    -----------
    data : pd.DataFrame
    params: dict
        Dictionary with parameters names
        params['v_wind']['param']: Wind Speed
        params['d_wind']['param']: Wind Direction
        params['w_wind']['param']: Vertical Wind Speed
        params['sigma_wind_lon']['param']: Standard Deviation longitudinal wind speed
    adj_params : dict
        Dictionary with neccesary parameters
        adj_params['h1']: Height of measurements in m
        adj_params['h2']: Hub height in m
        adj_params['karman']: von Karmans constant
        adj_params['method']: Height of measurement in m
            The method used to calculate v2. Options are:
            - 'alphaI': Uses linear relationship between turbulence intensity and alpha (Ishizaki, 1983) 
            - 'seven_power': A default method using the 1/7 power law.
            If no method is provided, the default 'seven_power' is used.
    Returns:
    --------
    v2 : float
        The calculated wind speed at height h2.
    """
    v1 = data[params['v_wind']['param']]
    
    method = adj_params['v2_method']
    h1 = adj_params['h1']
    h2 = adj_params['hub_height']
    
    if method == 'alphaI':
        direction = data[params['d_wind']['param']]
        sigma_u = data[params['sigma_wind_lon']['param']]
        w = 0.3 #data[params['w_wind']['param']] as long as not vertical wind data ist available
        k = adj_params['karman']
        
        theta = (270 - direction).apply(math.radians)
        u = v1 * theta.apply(math.cos)
        v = v1 * theta.apply(math.sin)
        
        I = sigma_u / u
        u_star = ( (w*u)**2 + (w*v)**2 ) ** 1/4
        
        Au = sigma_u / u_star
        b = 1/(k*Au)
        alpha = b * I
    elif method == 'seven_power':
        alpha = 1/7
    else:
        alpha = 1/7 # 'seven_power'
    v2 = v1 * (h2 / h1) ** alpha
    return v2


def get_temperature_at_height(data: pd.DataFrame,
                              params: dict,
                              adj_params: dict) -> pd.Series:
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h1 = 2 # in meters
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    delta_h = h2 - h1
    t2 = t1 - temp_gradient * delta_h
    t2 = t2 - 273.15
    return t2

    
def get_pressure_at_height(data: pd.DataFrame,
                           params: dict,
                           adj_params: dict,
                           h1: float) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    g = 9.81
    p1 = data[params['pressure']['param']]
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    p2 = p1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) )
    return p2

def get_density_at_height(data: pd.DataFrame,
                          params: dict,
                          adj_params: dict) -> pd.Series:
    R = 8.31451
    M_air = 0.028949 # dry air
    M_h20 = 0.018015 # water 
    g = 9.81
    rho1 = data[params['density']['param']]
    t1 = data[params['temperature']['param']]
    t1 = t1 + 273.15
    h1 = 2 # because of temperature measured at 2 m
    h2 = adj_params['hub_height']
    temp_gradient = adj_params['temp_gradient']
    M = M_air # molar mass of air (including water vapor) is less than that of dry air
    delta_h = h2 - h1
    rho2 = rho1 * ( 1 - (temp_gradient * delta_h) / t1 ) ** ( (M * g) / (temp_gradient * R) - 1)
    return rho2

def get_power_curve(data: pd.DataFrame,
                    power_curve: pd.Series,
                    params: dict) -> pd.Series:
    wind = data[[params["v_wind_hub"]["param"]]]
    power = pd.merge(wind, power_curve,left_on=params["v_wind_hub"]["param"], right_on='wind_speed', how="left")
    power.index = wind.index
    #C_p = (power["power"]/1000) / (0.5 * rho * rotor_area * wind["v_wind_hub"]**3)
    return power['power']

def get_Cp(data: pd.DataFrame,
           cp_curves: pd.Series,
           params: dict,
           adj_params: dict):
    wind = data[[params["v_wind_hub"]["param"]]]
    cp_curve = cp_curves[[adj_params["turbine"]]]
    ticks = np.arange(0, 2501, 1) / 100.0
    wind_speed_index = pd.DataFrame(ticks, columns=['wind_speed'])
    cp_curve = pd.merge(wind_speed_index, cp_curve, how='left', right_index=True, left_on='wind_speed')
    if adj_params['interpol_method'] == 'linear':
        cp_curve = cp_curve.interpolate(method='linear', axis=0)
    elif adj_params['interpol_method'] == 'polynomial':
        cp_curve = cp_curve.interpolate(method='polynomial', order=adj_params['polynom_grad'], axis=0)
    cp_curve.fillna(0, inplace=True)
    Cp = pd.merge(wind, cp_curve,left_on=params["v_wind_hub"]["param"], right_on='wind_speed', how="left")[adj_params["turbine"]]
    Cp.index = wind.index
    return Cp
    
def get_saturated_vapor_pressure(temperature: pd.Series,
                                 model: str = 'improved magnus') -> pd.Series:
    if model == 'huang':
        p_s = np.where(
            temperature > 0,
            np.exp(34.494 - (4924.99 / (temperature + 237.1))) / (temperature + 105) ** 1.57,
            np.exp(43.494 - (6545.8 / (temperature + 278))) / (temperature + 868) ** 2
        )
    elif model == 'improved_magnus':
        p_s = np.where(
            temperature > 0,
            610.94 * np.exp((17.625 * temperature) / (temperature + 243.04)),
            611.21 * np.exp((22.587 * temperature) / (temperature + 273.86))
    )
    return p_s
    
def get_rho(data: pd.DataFrame,
            params: dict) -> pd.Series:
    R_dry = 287.05  # Specific gas constant dry air (J/(kg·K))
    R_w = 461.5  # Specific gas constaint water vapor (J/(kg·K))
    air_pressure = data[params['pressure']['param']]
    temperature = data[params['temperature']['param']]
    relhum = data[params['relhum']['param']]
    p_s = data[params['sat_vap_pressure']['param']]
    # check if relative humidity is in the range between 0 and 1
    if relhum.max() > 1:
        relhum /= 100
    temperature_kelvin = temperature + 273.15 
    p_w = relhum * p_s
    p_g = air_pressure - p_w
    rho_g = p_g / (R_dry * temperature_kelvin)
    rho_w = p_w / (R_w * temperature_kelvin)
    rho = rho_g + rho_w
    return rho

def get_cut_in_cut_out_speeds(power_curve: pd.DataFrame):
    power_curve['wind_speed'] = pd.to_numeric(power_curve['wind_speed'])
    cut_in = power_curve[power_curve['power'] > 0]['wind_speed'].min()
    cut_out = next((power_curve['wind_speed'].iloc[i] for i in range(1, len(power_curve)) if power_curve['power'].iloc[i] == 0 and power_curve['power'].iloc[i - 1] > 0), None)
    return cut_in, cut_out

def get_features_wind(data: pd.DataFrame,
                      cp_curves: pd.DataFrame,
                      power_curve: pd.DataFrame,
                      params: dict,
                      adj_params: dict) -> pd.DataFrame:
    data[params['v_wind_hub']['param']] = round(get_windspeed_at_height(data=data,
                                                                  params=params,
                                                                  adj_params=adj_params), 2)
    data[params['temperature_hub']['param']] = get_temperature_at_height(data,
                                                                         params=params,
                                                                         adj_params=adj_params)
    temperature = data[params['temperature']['param']]
    data[params['sat_vap_pressure']['param']] = get_saturated_vapor_pressure(temperature=temperature,
                                                                             model=adj_params['p_s_model'])
    temperature_hub = data[params['temperature_hub']['param']]
    data[params['sat_vap_pressure_hub']['param']] = get_saturated_vapor_pressure(temperature=temperature_hub,
                                                                                 model=adj_params['p_s_model'])
    data[params['density']['param']] = get_rho(data,
                                               params=params)
    data[params['density_hub']['param']] = get_density_at_height(data,
                                                                 params=params,
                                                                 adj_params=adj_params)
    data[params["cp_curve"]["param"]] = get_Cp(data,
                                               cp_curves=cp_curves,
                                               params=params,
                                               adj_params=adj_params)
    
    data[params["power_curve"]["param"]] = get_power_curve(data,
                                                           power_curve=power_curve,
                                                           params=params)
    return data


def get_wind_power_coefficients(data: pd.DataFrame,
                                power_curve: pd.DataFrame,
                                params: dict,
                                adj_params: dict):
    rho = data[params['rho_hub']['param']]
    rotor_diameter = adj_params['rotor_diameter']
    v = power_curve['wind_speed']
    power = power_curve['power']
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    Cp_vector = power / (0.5 * rho * rotor_area * v**3)
    return Cp_vector


def generate_wind_power(data: pd.DataFrame,
                        power_curve: pd.DataFrame,
                        params: dict,
                        adj_params: dict
                        ) -> pd.Series:
    rated_power = data[params['power_curve']['param']].max()
    Cp = data[params['cp_curve']['param']]
    rotor_diameter = adj_params['rotor_diameter']
    cut_in, cut_out = get_cut_in_cut_out_speeds(power_curve=power_curve)
    rho = data[params['density_hub']['param']]
    wind_speed_hub = data[params['v_wind_hub']['param']]
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    wind_power = np.where(
        wind_speed_hub < cut_in, 0,
        np.where(
            wind_speed_hub <= cut_out,
            np.minimum(rated_power, 0.5 * rho * rotor_area * Cp * wind_speed_hub ** 3),
            0
        )
    )
    data[params['power']['param']] = wind_power
    return data

def plot_power_and_feature(data: pd.DataFrame,
                           params: dict,
                           day: str, 
                           feature: dict,
                           power: pd.Series,
                           save_fig=False,
                           streamlit=False): 
    day = pd.Timestamp(day)
    index_0 = power.index.get_loc(day)
    index_1 = power.index.get_loc(day + pd.Timedelta(days=1))
    feature_specs = params[feature]
    series = data[feature_specs['param']]
    feature_name = feature_specs['name']
    unit = feature_specs['unit']
    date = str(series.index[index_0:index_1][0].date())
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.set_alpha(0)
    fontsize = 14
    lines = []
    # plot power
    line1, = ax1.plot(
    power[index_0:index_1],
    label="Power Output (W)",
    color="black",
    linewidth=2.0
    )
    lines.append(line1)
    # configure secondary y-axis
    ax1.set_xlabel("Time", fontsize=fontsize)
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize-2)
    ax2 = ax1.twinx()
    # plot feature
    line, = ax2.plot(
        series[index_0:index_1],
        label=f"{feature_name} {unit}",
        linestyle='--',
        #color='blue',
        linewidth=2.0
    )
    lines.append(line)
    # configure primary y-axis
    ax2.set_ylabel(f"{feature_name} ({unit})", fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    # Format x-axis to show only hours (HH)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[1:-1])
    # legend
    lines.append(lines.pop(0))
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=fontsize)
    plt.title(f"{feature_name} and Power Output on {date}", fontsize=fontsize)
    fig.tight_layout()
    if(streamlit):
        return fig
    else:
        #plt.grid(True)
        if save_fig:
            save_path = f'figs/{feature_name}'
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f'{date}.png')
            plt.savefig(save_file, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_power_and_feature_wind_streamlit(data: pd.DataFrame,
                           params: dict,
                           day: str, 
                           feature: dict,
                           power: pd.Series): 
    
    day = pd.Timestamp(day)
    tz = power.index.tz
    if tz is not None:
        day = day.tz_localize(tz) 
    index_0 = power.index.get_loc(day)
    index_1 = power.index.get_loc(day + pd.Timedelta(days=1))
    feature_specs = params[feature]
    series = data[feature_specs['param']]
    feature_name = feature_specs['name']
    unit = feature_specs['unit']
    date = str(series.index[index_0:index_1][0].date())

    fig, ax1 = plt.subplots(figsize=(8, 4))

    fontsize = 8
    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 6}  
    color = '#999999'
    line_colors = ['#be95c4']
    lines = []

    # plot power
    line1, = ax1.plot(
    power[index_0:index_1],
    label="Power Output (W)",
    color="#231942",
    linewidth=1.0
    )
    lines.append(line1)

    # configure secondary y-axis
    ax1.set_xlabel("Time", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize-2)
    ax2 = ax1.twinx()

    # plot feature
    line, = ax2.plot(
        series[index_0:index_1],
        label=f"{feature_name} {unit}",
        linestyle='-',
        color=line_colors[0],
        linewidth=1.0
    )
    lines.append(line)

    # configure primary y-axis
    ax2.set_ylabel(f"{feature_name} ({unit})", fontsize=fontsize,color=color, family='DejaVu Sans')
    ax2.tick_params(axis='y', labelsize=fontsize)

    # Format x-axis to show only hours (HH)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[1:-1])

    #Format chart to look similar to the streamlit charts
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)
    labels = [line.get_label() for line in lines]
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)
            
    # legend
    ax1.legend(
        lines,
        labels,
        loc=10,
        bbox_to_anchor=(0.4, -0.2),
        ncol=2,
        frameon=False,  
        fontsize=8,
        labelcolor=color, 
        edgecolor='none'
    )
    ax1.grid(True, axis='y', color='gray', linewidth=0.5)
     
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0) 

    plt.title(f"{feature_name} and Power Output on {date}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    fig.tight_layout()

    return fig

def plot_wind_power_scatter(wind_speed: pd.Series,
                    power: pd.Series,
                    station_id=''):
    
    fontsize = 8
    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 6}  
    color = '#999999'
    point_colors = ['#be95c4']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=wind_speed,y=power,c=point_colors[0])

    for spine in ax.spines.values():
        spine.set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    ax.grid(True, axis='y', color='gray', linewidth=0.5)    
    ax.tick_params(axis='both', which='both', length=0)

    # Add title and labels
    ax.set_title(f"Wind Speed and Power Output for station {station_id}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    ax.set_xlabel('Wind Speed',fontsize=fontsize,color=color, family='DejaVu Sans')
    ax.set_ylabel('Power Output',fontsize=fontsize,color=color, family='DejaVu Sans')

    return fig

def plot_wind_power_line(wind_speed: pd.Series,
                    power: pd.Series,
                    turbine_id=''):
    
    fontsize = 12
    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 10}  
    color = '#999999'
    line_colors = ['#be95c4']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x=wind_speed,y=power,c=line_colors[0])

    for spine in ax.spines.values():
        spine.set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    ax.grid(True, axis='y', color='gray', linewidth=0.5)    
    ax.tick_params(axis='both', which='both', length=0)

    # Add title and labels
    ax.set_title(f"Wind Speed and Power Output for Turbine {turbine_id}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    ax.set_xlabel('Wind Speed',fontsize=fontsize,color=color, family='DejaVu Sans')
    ax.set_ylabel('Power Output',fontsize=fontsize,color=color, family='DejaVu Sans')

    return fig

def plot_wind_power_scatter_line(wind_speed_st: pd.Series,
                    power_st: pd.Series,
                    wind_speed_trb: pd.Series,
                    power_trb: pd.Series,
                    station_id='',
                    turbine_id=''):

    fontsize = 8
    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 6}  
    color = '#999999'
    point_color = '#be95c4'
    line_color = '#231942'
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x=wind_speed_st,y=power_st,c=point_color)
    #ax.plot(wind_speed_trb,power_trb,line_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    ax.grid(True, axis='y', color='gray', linewidth=0.5)    
    ax.tick_params(axis='both', which='both', length=0)

    # Add title and labels
    ax.set_title(f"Wind Speed and Power Output for station {station_id} and {turbine_id}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    ax.set_xlabel('Wind Speed',fontsize=fontsize,color=color, family='DejaVu Sans')
    ax.set_ylabel('Power Output',fontsize=fontsize,color=color, family='DejaVu Sans')

    return fig