import pandas as pd
import numpy as np
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from generate_utils import load_config,get_station_list,plot_power_and_feature_wind_streamlit,plot_wind_power_scatter_line,get_features_wind,generate_wind_power
from generate_pv_utils import plot_power_and_features_pv_streamlit,get_features_pv,generate_pv_power

wind_dir = 'wind'
solar_dir = 'solar'


# Function to display wind plot based on the marker selected
def show_wind_plot(marker_name):

    turbine_dir = config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)
    params = config['synth']
    adj_params = config['adjustable_wind_params']

    # power curve & cp curve data
    power_curve = pd.read_csv(turbine_path)
    power_curve = power_curve[["wind_speed",adj_params["turbine"]]]
    power_curve.rename(columns={adj_params["turbine"]: "power"} ,inplace=True)
    power_curve['power'] = power_curve['power'] * 1000

    cp_curves = pd.read_csv(cp_path, sep=";", decimal=".")
    cp_curves = cp_curves.set_index("Turbine").T
    cp_curves.index = pd.to_numeric(cp_curves.index, errors='coerce')

    turbine_specs = pd.read_csv(specs_path)
    turbine_row = turbine_specs[turbine_specs['Turbine'] == adj_params['turbine']]
    rotor_diameter = float(turbine_row["Rotordurchmesser"].iloc[0].strip("'"))
    hub_height = float(turbine_row['Nabenhöhe'].iloc[0])
    adj_params['rotor_diameter'] = rotor_diameter
    adj_params['hub_height'] = hub_height

    #fileName = f'Station_{str(marker_name)}.pkl'
    fileName = f'Station_{str(marker_name)}.csv'
    new_dir = os.path.join(wind_dir, fileName)

    if(os.path.exists(new_dir)):

        #dataSelectedStation = pd.read_pickle(new_dir)
        dataSelectedStation = pd.read_csv(new_dir)

        dataSelectedStation['timestamp'] = pd.to_datetime(dataSelectedStation['timestamp'])
        dataSelectedStation.set_index('timestamp', inplace=True)

        dataSelectedStation = get_features_wind(data=dataSelectedStation,
                                                cp_curves=cp_curves,
                                                power_curve=power_curve,
                                                params=params,
                                                adj_params=adj_params)

        dataSelectedStation = generate_wind_power(data=dataSelectedStation,
                                    power_curve=power_curve,
                                    params=params, 
                                    adj_params=adj_params)
        

        with st.container(border=True):

            selectedTurbineData = turbineData[["wind_speed",selectedTurbine]]

            fig = plot_wind_power_scatter_line(wind_speed_st=dataSelectedStation['v_wind_hub'],
                                            power_st=dataSelectedStation['power'],
                                            wind_speed_trb=selectedTurbineData['wind_speed'].values,
                                            power_trb=selectedTurbineData[selectedTurbine].values,
                                            station_id=str(marker_name),
                                            turbine_id=selectedTurbine)
            
            st.pyplot(fig=fig,use_container_width=True)

        #dates_arr = dataSelectedStation.index.date
        #formatted_dates = np.array([date.strftime('%Y-%m-%d') for date in dates_arr])
        #formatted_dates = np.unique(formatted_dates)
        #dates_list = formatted_dates.tolist()

        dates_arr = dataSelectedStation.index.strftime('%Y-%m-%d')
        formatted_dates = np.unique(dates_arr)
        dates_list = formatted_dates.tolist()

        with st.container(border=True):

            selected_date = st.selectbox("Select a date",dates_list)
            selected_feature = st.selectbox(label="Select a feature",options=["v_wind_hub","density_hub"])
        
            fig = plot_power_and_feature_wind_streamlit(data=dataSelectedStation,
                                params=config['synth'],
                                day=selected_date,
                                feature=selected_feature,
                                power=dataSelectedStation['power']) 

            st.pyplot(fig=fig,use_container_width=True)

# Function to display pv plot based on the marker selected
def show_pv_plot(marker_name):

    params = config['synth']
    adj_params = config['adjustable_pv_params']

    #fileName = f'Station_{str(marker_name)}.pkl'

    fileName = f'Station_{str(marker_name)}.csv'
    new_dir = os.path.join(solar_dir, fileName)

    if(os.path.exists(new_dir)):

        dataSelectedStation = pd.read_csv(new_dir)

        dataSelectedStation['timestamp'] = pd.to_datetime(dataSelectedStation['timestamp'])
        dataSelectedStation.set_index('timestamp', inplace=True)

        total_irradiance, cell_temperature = get_features_pv(data=dataSelectedStation,
                                                  params=params,
                                                  adj_params=adj_params,
                                                  latitude=stations.loc[stations['station_id'] == marker_name,'latitude'].iloc[0],
                                                  longitude=stations.loc[stations['station_id'] == marker_name,'longitude'].iloc[0],
                                                  elevation=stations.loc[stations['station_id'] == marker_name,'station_height'].iloc[0])
        
        total = total_irradiance['poa_global']
        direct = total_irradiance['poa_direct']
        diffuse = total_irradiance['poa_diffuse']

        power = generate_pv_power(total_irradiance=total,
                                cell_temperature=cell_temperature,
                                adj_params=adj_params)
        
        dataSelectedStation['Power'] = power
        dataSelectedStation['Total'] = total
        dataSelectedStation['Direct'] = direct
        dataSelectedStation['Diffuse'] = diffuse

        dates_arr = dataSelectedStation.index.strftime('%Y-%m-%d')
        formatted_dates = np.unique(dates_arr)
        dates_list = formatted_dates.tolist()

        selected_date = st.selectbox("Select a date",dates_list)
        selected_feature = st.multiselect(label="Select features",options=["Total","Direct","Diffuse"],default="Total")
        features = []

        for feature in selected_feature:
            features.append(dataSelectedStation[feature])

        if (len(selected_feature) != 0):

            fig = plot_power_and_features_pv_streamlit(day=selected_date,
                                plot_names=selected_feature,
                                features=features,
                                power=dataSelectedStation['Power'])
    
            st.pyplot(fig=fig,use_container_width=True)

#For creating a dashboard canvas
st.set_page_config(
    page_title="Power generation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

#Creating a sidebar to choose solar/wind time series
with st.sidebar:

    st.title("Power generation Dashboard")
    typeInput = st.selectbox(label="Type of Energy resource",options=["Solar", "Wind"],index=None)


#Loading all the station data and plotting it on the map    
config_path = "config.yaml"
config = load_config(config_path)


column_names = ["Stations_id", "von_datum", "bis_datum", "Stationshoehe",
                "geoBreite", "geoLaenge", "Stationsname", "Bundesland", "Abgabe"]

#Only execute the code when something is selected in selectbox 
if typeInput is not None:

    #PV part
    if typeInput.lower() == "solar":

        with st.sidebar:

            colsd1,colsd2 = st.columns([0.5,0.5])

            with colsd1:

                installed_power = st.number_input("Installed Power", value=None)
                surface_tilt = st.number_input("Surface Tilt", value=None)
                surface_azimuth = st.number_input("Surface Azimuth", value=None)

            with colsd2:

                gamma_pdc = st.number_input("Gamma PDC", value=None)
                albedo = st.number_input("Albedo", value=None)
                eta_inv_nom = st.number_input("Eta Inverted Nom", value=None)

        #stations = get_all_stations(config,column_names)
        stations= pd.read_csv("masterdata.csv",dtype={'station_id':str})

        stations_list = get_station_list(solar_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['station_id'].astype(int).isin(stations_list)]

        #stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        #stations = stations[stations['Table'] == typeInput.lower()]

        markers = {row['station_id']: [row['latitude'], row['longitude']] for _, row in stations.iterrows()}
        #markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}

        #For centering the content in the dashboard canvas
        col1, col2, col3 = st.columns([0.1,0.8,0.1])

        with col2:

            with st.container(border=True):
                # Create a map centered around a specific location
                m = folium.Map(location=[51.4, 10.4515], zoom_start=7)

                # Add markers to the map
                for name, location in markers.items():
                    folium.Marker(location, popup=name, icon=folium.Icon(color="darkpurple")).add_to(m)

                solar_map_data = st_folium(m, use_container_width=True, height=1200)

                # Store the selected marker name in session state if it's not already set
                if "selected_marker" not in st.session_state:
                    st.session_state.selected_marker = None

                # Update session state based on the map click
                if solar_map_data and "last_clicked" in solar_map_data and solar_map_data["last_object_clicked"] is not None:
                    lat = solar_map_data["last_object_clicked"]["lat"]
                    lon = solar_map_data["last_object_clicked"]["lng"]
    
                    # Determine which marker was clicked based on coordinates
                    for marker_name, coords in markers.items():
                        if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                            st.session_state.selected_marker = marker_name
                            break

            with st.container(border=True):
                # Show the plot for the selected marker
                if st.session_state.selected_marker:
                    show_pv_plot(st.session_state.selected_marker)
                    del st.session_state.selected_marker #Clearing the session state to show no plots when switched from pv to wind

    #Wind part
    else:

        turbineData = pd.read_csv("power_curves/turbine_power.csv")

        with st.sidebar:
            selectedTurbine = st.selectbox(label="Select a turbine", options=turbineData.columns[1:])

        stations= pd.read_csv("masterdata.csv",dtype={'station_id':str})

        stations_list = get_station_list(wind_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['station_id'].astype(int).isin(stations_list)]

        #stations = get_all_stations(config,column_names)
        
        #stations_list = get_station_list(wind_dir)
        #stations_list = [int(station) for station in stations_list]

        #stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        #stations = stations[stations['Table'] == 'wind_test']

        markers = {row['station_id']: [row['latitude'], row['longitude']] for _, row in stations.iterrows()}
        #markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}
        
        #For centering the content in the dashboard canvas
        col1, col2, col3 = st.columns([0.1,0.8,0.1])

        with col2:

            with st.container(border=True):

                # Create a map centered around a specific location
                m = folium.Map(location=[51.4, 10.4515], zoom_start=7)

                # Add markers to the map
                for name, location in markers.items():
                    folium.Marker(location, popup=name, icon=folium.Icon(color="darkpurple")).add_to(m)

                wind_map_data = st_folium(m, use_container_width=True, height=1200)

                # Store the selected marker name in session state if it's not already set
                if "selected_marker" not in st.session_state:
                    st.session_state.selected_marker = None

                # Update session state based on the map click
                if wind_map_data and "last_clicked" in wind_map_data and wind_map_data["last_object_clicked"] is not None:
                    lat = wind_map_data["last_object_clicked"]["lat"]
                    lon = wind_map_data["last_object_clicked"]["lng"]
    
                    # Determine which marker was clicked based on coordinates
                    for marker_name, coords in markers.items():
                        if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                            st.session_state.selected_marker = marker_name
                            break
            
            with st.container(border=True):

                # Show the plot for the selected marker
                if st.session_state.selected_marker:
                    show_wind_plot(st.session_state.selected_marker)
                    del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv
