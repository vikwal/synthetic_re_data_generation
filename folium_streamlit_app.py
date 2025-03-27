import pandas as pd
import numpy as np
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from generate_utils import load_config,get_all_stations,get_station_list,plot_power_and_feature_wind_streamlit,plot_wind_power_scatter_line
from generate_pv_utils import plot_power_and_features_pv_streamlit

wind_dir = 'data/pickle/wind'
solar_dir = 'data/pickle/solar'


# Function to display wind plot based on the marker selected
def show_wind_plot(marker_name):

    fileName = f'Station_{str(marker_name)}.pkl'
    new_dir = os.path.join(wind_dir, fileName)

    if(os.path.exists(new_dir)):

        dataSelectedStation = pd.read_pickle(new_dir)

        with st.container(border=True):

            selectedTurbineData = turbineData[["wind_speed",selectedTurbine]]

            fig = plot_wind_power_scatter_line(wind_speed_st=dataSelectedStation['v_wind_hub'],
                                            power_st=dataSelectedStation['Power'],
                                            wind_speed_trb=selectedTurbineData['wind_speed'].values,
                                            power_trb=selectedTurbineData[selectedTurbine].values,
                                            station_id=str(marker_name),
                                            turbine_id=selectedTurbine)
            
            st.pyplot(fig=fig,use_container_width=True)

        dates_arr = dataSelectedStation.index.date
        formatted_dates = np.array([date.strftime('%Y-%m-%d') for date in dates_arr])
        formatted_dates = np.unique(formatted_dates)
        dates_list = formatted_dates.tolist()

        with st.container(border=True):

            selected_date = st.selectbox("Select a date",dates_list)
            selected_feature = st.selectbox(label="Select a feature",options=["v_wind_hub","density_hub"])
        
            fig = plot_power_and_feature_wind_streamlit(data=dataSelectedStation,
                                params=config['synth'],
                                day=selected_date,
                                feature=selected_feature,
                                power=dataSelectedStation['Power']) 

            st.pyplot(fig=fig,use_container_width=True)


        #with st.container(border=True):
            #selected_turbine = st.selectbox(label="Select a turbine", options=turbineData.columns[1:])
            #selected_turbine = st.selectbox(label="Select a turbine", options=["T1","T2","T3","T4","T5"])

            #turbineFile = f'{selected_turbine}.csv'
            #dir = os.path.join('data/turbine',turbineFile)
            #dataSelectedTurbine = pd.read_csv(dir) 

            #st.line_chart(data=dataSelectedTurbine,x='Wind Speed (m/s)',y='Power Output (W)',x_label='Wind Speed',y_label='Power',color='#be95c4',use_container_width=True)

            #st.line_chart(data=turbineData[["wind_speed",selected_turbine]],x='wind_speed',y=selected_turbine,x_label='Wind Speed',y_label='Power',color='#be95c4',use_container_width=True)

# Function to display pv plot based on the marker selected
def show_pv_plot(marker_name):

    fileName = f'Station_{str(marker_name)}.pkl'
    new_dir = os.path.join(solar_dir, fileName)

    if(os.path.exists(new_dir)):

        dataSelectedStation = pd.read_pickle(new_dir)

        dates_arr = dataSelectedStation.index.date
        formatted_dates = np.array([date.strftime('%Y-%m-%d') for date in dates_arr])
        formatted_dates = np.unique(formatted_dates)
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

        stations = get_all_stations(config,column_names)

        stations_list = get_station_list(solar_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        stations = stations[stations['Table'] == typeInput.lower()]

        markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}

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
                    del st.session_state.selected_marker #Clearing the session state to show no plots when the switched from pv to wind

    #Wind part
    else:

        turbineData = pd.read_csv("power_curves/turbine_power.csv")

        with st.sidebar:
            selectedTurbine = st.selectbox(label="Select a turbine", options=turbineData.columns[1:])

        stations = get_all_stations(config,column_names)
        
        stations_list = get_station_list(wind_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        stations = stations[stations['Table'] == 'wind_test']

        markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}
        
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
                    del st.session_state.selected_marker #Clearing the session state to show no plots when the switched from wind to pv
