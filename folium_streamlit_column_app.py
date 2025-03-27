import pandas as pd
import numpy as np
import os
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium
from generate_utils import load_config,get_all_stations,get_station_list,plot_power_and_feature_wind_streamlit
from generate_pv_utils import plot_power_and_features_pv_streamlit

wind_dir = 'data/pickle/wind'
solar_dir = 'data/pickle/solar'


# Function to display wind plot based on the marker selected
def show_wind_plot(marker_name, column_name):

    fileName = f'Station_{str(marker_name)}.pkl'
    new_dir = os.path.join(wind_dir, fileName)

    if(os.path.exists(new_dir)):

        dataSelectedStation = pd.read_pickle(new_dir)

        if column_name == 'col1':
            st.scatter_chart(data=dataSelectedStation,x='v_wind_hub',y='Power',x_label='Wind Speed',y_label='Power',color='#be95c4',width=700,height=500)

        elif column_name == 'col2':
            dates_arr = dataSelectedStation.index.date
            formatted_dates = np.array([date.strftime('%Y-%m-%d') for date in dates_arr])
            formatted_dates = np.unique(formatted_dates)
            dates_list = formatted_dates.tolist()

            selected_date = st.selectbox("Select a date",dates_list)
            selected_feature = st.selectbox(label="Select a feature",options=["v_wind_hub","density_hub"])
        

            fig = plot_power_and_feature_wind_streamlit(data=dataSelectedStation,
                                params=config['synth'],
                                day=selected_date,
                                feature=selected_feature,
                                power=dataSelectedStation['Power']) 

            st.pyplot(fig)

# Function to display wind plot based on the marker selected
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
        #selected_feature = st.selectbox(label="Select a feature",options=["v_wind_hub","density_hub"])
        plot_names = ['Total', 'Direct', 'Diffuse']
        features = [dataSelectedStation['Total'],dataSelectedStation['Direct'],dataSelectedStation['Diffuse']]
        

        fig = plot_power_and_features_pv_streamlit(day=selected_date,
                            plot_names=plot_names,
                            features=features,
                            power=dataSelectedStation['Power'])
    
        st.pyplot(fig)

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


column_names = [
    "Stations_id", "von_datum", "bis_datum", "Stationshoehe",
    "geoBreite", "geoLaenge", "Stationsname", "Bundesland", "Abgabe"
]


if typeInput is not None:
    if typeInput.lower() == "solar":

        stations = get_all_stations(config,column_names)

        stations_list = get_station_list(solar_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        stations = stations[stations['Table'] == typeInput.lower()]

        markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}

        col1, col2 = st.columns([0.6,0.4])

        with col1:

            # Create a map centered around a specific location
            m = folium.Map(location=[51.1657, 10.4515], zoom_start=7)

            # Add markers to the map
            for name, location in markers.items():
                folium.Marker(location, popup=name, icon=folium.Icon(color="darkpurple")).add_to(m)

            map_data = st_folium(m, width=1500, height=1000)

            selectedStations = stations[stations['Table'] == typeInput.lower()]
            solar_stations = selectedStations['Stations_id'].unique()
            solar_stations = [int(station) for station in solar_stations]
            solar_stations = [station for station in solar_stations if station in stations_list]


            # Store the selected marker name in session state if it's not already set
            if "selected_marker" not in st.session_state:
                st.session_state.selected_marker = None

            # Update session state based on the map click
            if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                lat = map_data["last_object_clicked"]["lat"]
                lon = map_data["last_object_clicked"]["lng"]
    
                # Determine which marker was clicked based on coordinates
                for marker_name, coords in markers.items():
                    if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                        st.session_state.selected_marker = marker_name
                        break
        
        with col2:
            # Show the plot for the selected marker
            if st.session_state.selected_marker:
                show_pv_plot(st.session_state.selected_marker)

    else:

        stations = get_all_stations(config,column_names)
        
        stations_list = get_station_list(wind_dir)
        stations_list = [int(station) for station in stations_list]

        stations = stations[stations['Stations_id'].astype(int).isin(stations_list)]
        stations = stations[stations['Table'] == 'wind_test']

        markers = {int(row['Stations_id']): [row['geoBreite'], row['geoLaenge']] for _, row in stations.iterrows()}

        col1, col2 = st.columns([0.6,0.4])

        with col1:

            # Create a map centered around a specific location
            m = folium.Map(location=[51.1657, 10.4515], zoom_start=7)

            # Add markers to the map
            for name, location in markers.items():
                folium.Marker(location, popup=name, icon=folium.Icon(color="darkpurple")).add_to(m)

            map_data = st_folium(m, width=1500, height=500)

            selectedStations = stations[stations['Table'] == 'wind_test']
            wind_stations = selectedStations['Stations_id'].unique()
            wind_stations = [int(station) for station in wind_stations]
            wind_stations = [station for station in wind_stations if station in stations_list]


            # Store the selected marker name in session state if it's not already set
            if "selected_marker" not in st.session_state:
                st.session_state.selected_marker = None

            # Update session state based on the map click
            if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                lat = map_data["last_object_clicked"]["lat"]
                lon = map_data["last_object_clicked"]["lng"]
    
                # Determine which marker was clicked based on coordinates
                for marker_name, coords in markers.items():
                    if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                        st.session_state.selected_marker = marker_name
                        break

            # Show the plot for the selected marker
            if st.session_state.selected_marker:
                show_wind_plot(st.session_state.selected_marker,'col1')

        with col2:

            selectedStations = stations[stations['Table'] == 'wind_test']
            wind_stations = selectedStations['Stations_id'].unique()
            wind_stations = [int(station) for station in wind_stations]
            wind_stations = [station for station in wind_stations if station in stations_list]


            # Store the selected marker name in session state if it's not already set
            if "selected_marker" not in st.session_state:
                st.session_state.selected_marker = None

            # Update session state based on the map click
            if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                lat = map_data["last_object_clicked"]["lat"]
                lon = map_data["last_object_clicked"]["lng"]
    
                # Determine which marker was clicked based on coordinates
                for marker_name, coords in markers.items():
                    if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                        st.session_state.selected_marker = marker_name
                        break

            # Show the plot for the selected marker
            if st.session_state.selected_marker:
                show_wind_plot(st.session_state.selected_marker,'col2')

