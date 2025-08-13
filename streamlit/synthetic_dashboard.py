from pathlib import Path
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tools import load_config_ruamel, write_config
from plots_utils import get_station_list,create_map,show_pv_plot,show_wind_plot,find_nearest_station_for_park

# --- Page Configuration ---
st.set_page_config(page_title="Power generation Dashboard")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Center the main title */
        h1 {
            text-align: center;
            color: #2F80ED; /* A nice blue color for the title */
            margin-bottom: 2rem; /* Add some space below the title */
        }

        /* Style for the radio button labels (the question text) */
        .stRadio > label {
            font-size: 3.5em; /* Set to be roughly half the size of the main title */
            font-weight: 500; /* Medium font weight */
            color: #333; /* Darker text color */
            text-align: center; /* Center the label text itself */
            display: block; /* Ensure label takes full width to center its text */
            margin-bottom: 20px; /* Increased space between the question and the options */
        }

        /* Style for the individual radio button options (the circles and their text) */
        div[role="radiogroup"] {
            display: flex;
            justify-content: center; /* Center the radio options horizontally */
            flex-wrap: wrap; /* Allow options to wrap if space is limited */
            gap: 30px; /* Increased space between individual radio options */
            margin-bottom: 2.5rem; /* Increased space below the radio buttons */
        }

        /* Style for the text content of each radio option */
        div[role="radiogroup"] label span {
            font-size: 3.5em; /* Set to be roughly half the size of the main title */
        }

        /* Style for buttons */
        .stButton > button {
            display: block; /* Make button a block element to control margin */
            margin-left: auto; /* Auto margin on left to center */
            margin-right: auto; /* Auto margin on right to center */
            font-size: 1.5em; /* Set to be roughly half the size of the main title */
            padding: 18px 35px; /* More padding for a bigger click target */
            border-radius: 10px; /* Slightly more rounded corners for buttons */
            border: none; /* Remove default border */
            background-color: #28A745; /* Green background for action buttons */
            color: white; /* White text color */
            cursor: pointer; /* Pointer cursor on hover */
            transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth transition for hover effect and slight scale */
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15); /* More prominent shadow */
        }

        .stButton > button:hover {
            background-color: #218838; /* Darker green on hover */
            transform: translateY(-2px); /* Slight lift effect on hover */
        }

        /* Style for selectbox labels (if used elsewhere) */
        .stSelectbox > label {
            display: block;
            text-align: center;
            font-size: 1em;
            color: #555;
        }
            
        /* Outer sidebar container (dark grey background) */
    section[data-testid="stSidebar"] {
        width: 400px !important;
        min-width: 400px !important;
    }

    /* Inner block that holds all widgets/content */
    section[data-testid="stSidebar"] > div:first-child {
        width: 400px !important;
    }

    /* Shift the main content accordingly */
    div[data-testid="stSidebarContent"] {
        width: 100% !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Initialize 'current_view' to control which part of the app is shown
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "home"

# Initialize 'action_type' for the radio button selection on the home page
if "action_type" not in st.session_state:
    st.session_state["action_type"] = "Visualize Synthetic Plant" # Default selection

if "selected_marker" not in st.session_state:
                    st.session_state.selected_marker = None

# --- Data Generation relevant tasks ---

config_path = "./config.yaml"
config = load_config_ruamel(config_path)

masterdata_path = "./data/masterdata.csv"
masterdata = pd.read_csv(masterdata_path,sep=',',dtype={'station_id':str})

turbine_dir = Path.cwd().parent / config['data']['turbine_dir']
turbine_power = config['data']['turbine_power']
turbine_filepath = os.path.join(turbine_dir, turbine_power)
turbine_types = pd.read_csv(turbine_filepath,sep=';').columns.tolist()

pv_dir = "./data/solar"
wind_dir = "./data/wind"

# --- Main Application Logic based on current_view ---
if st.session_state["current_view"] == "home":
    # --- Home Page Content ---

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    logo_left_col, logo_center_col, logo_right_col = st.columns([4, 2, 6])

    with logo_left_col:
        st.image("./logos/logo.svg", width=300)
        
    with logo_right_col:
        st.image("./logos/HKA_Logo.png", width=600)        

    st.markdown("<h1>Welcome to the Synthetic Data Generation Dashboard</h1>", unsafe_allow_html=True)

    # Centering the Radio Button using st.columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2: 
        # Radio button for selecting an action
        action_options = ["Visualize Synthetic Plant", "Compare Your Plant"]
        selected_action = st.radio(
            "What would you like to do?",
            action_options,
            index=action_options.index(st.session_state["action_type"])
        )
    
    st.session_state["action_type"] = selected_action

    # Simulate redirection by clicking a button
    # These buttons will change the 'current_view' and trigger a rerun
    if st.session_state["action_type"] == "Visualize Synthetic Plant":
        if st.button("Go to Synthetic Plant Visualization"):
            st.session_state["current_view"] = "visualize synthetic plant"
            st.rerun() 

    elif st.session_state["action_type"] == "Compare Your Plant":
        if st.button("Go to Plant Comparison"):
            st.session_state["current_view"] = "compare your plant"
            st.rerun() 

    st.markdown("<div style='margin-top: 300px;'></div>", unsafe_allow_html=True)
    ##logo_left_col, logo_right_col = st.columns([100, 0.5])

    ##with logo_left_col:
    ##    st.caption("This work is done by Meghna Negi in the research project FederatedForecasts,funded by the German Federal Ministry of Research, Technology and Space [Grant 13FH587KX1]")

elif st.session_state["current_view"] == "visualize synthetic plant":
    # --- Visualize Synthetic Plant Page logic ---

    #Creating a sidebar to choose solar/wind application
    with st.sidebar:
        typeInput = st.selectbox(label="Type of Energy resource",options=["Solar", "Wind"],index=None)

    if typeInput is not None:
        
        if typeInput == "Solar":

            with st.sidebar:
                colsd1,colsd2 = st.columns([0.5,0.5])

                with colsd1:

                    config['pv_params']['dc_rating_watts'] = st.number_input("Installed Power", value=10000)
                    config['pv_params']['surface_tilt'] = st.number_input("Surface Tilt", value=35)
                    config['pv_params']['surface_azimuth'] = st.number_input("Surface Azimuth", value=180)

                with colsd2:

                    config['pv_params']['gamma_pdc'] = st.number_input("Gamma PDC", value=-0.0035,format='%0.5f')
                    config['pv_params']['albedo'] = st.number_input("Albedo", value=0.25)
                    config['pv_params']['eta_inv_nom'] = st.number_input("Eta Inverted Nom", value=0.96)

                write_config(config,config_path)

            stations_list = get_station_list(pv_dir)
            stations_list = [int(station) for station in stations_list]

            solardata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            markers = {row['station_id']: [float(row['latitude']), float(row['longitude'])] for _, row in solardata.iterrows()}

            #For centering the content in the dashboard canvas
            col1, col2, col3 = st.columns([0.1,0.8,0.1])

            with col2:
                st.info("Click on the station marker you want to generate the data for!!", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers)
                
                    # Update session state based on the map click
                    if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                        lat = map_data["last_object_clicked"]["lat"]
                        lon = map_data["last_object_clicked"]["lng"]
    
                        # Determine which marker was clicked based on coordinates
                        for marker_name, coords in markers.items():
                            if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                                st.session_state.selected_marker = marker_name
                                break

                with st.container(border=True):
                    # Show the plot for the selected marker
                    if st.session_state.selected_marker:
                        park_data = pd.DataFrame
                        latitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['latitude'].iloc[0]
                        longitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['longitude'].iloc[0]
                        elevation = solardata.loc[solardata.station_id == st.session_state.selected_marker]['station_height'].iloc[0]
                        show_pv_plot(config,st.session_state.selected_marker,pv_dir,latitude,longitude,elevation,park_data)
                        del st.session_state.selected_marker #Clearing the session state to show no plots when switched from pv to wind

        if typeInput == "Wind":

            turbine_list = []
            height_list = []
            commissioning_date = None

            with st.sidebar:

                ageing = st.toggle("Enable Ageing")
                config['wind_params']['apply_ageing'] = ageing

                if ageing:

                    commissioning_date = st.date_input("Commissioning Date", value="2013-01-01", format= "YYYY-MM-DD")


                turbines_number = st.number_input("Number of turbines in park",value=1,min_value=1)

                col1, col2 = st.columns([0.5,0.5])

                with col1:

                    selected_turbine = st.selectbox(label="Select a type of turbine", options=turbine_types[1:], key=f'turbine0')
                    turbine_list.append(selected_turbine)

                with col2:

                    hub_height = st.number_input("Hub Height", value=5, key=f'height0')
                    height_list.append(hub_height)

                for number in range(turbines_number-1):

                    with col1:
                        selected_turbine = st.selectbox(label="Select a type of turbine", label_visibility="hidden", options=turbine_types[1:], key=f'turbine{number+1}')
                        turbine_list.append(selected_turbine)

                    with col2:
                        hub_height = st.number_input(label="Hub Height", label_visibility="hidden", value=5, key=f'height{number+1}')
                        height_list.append(hub_height)

                config['wind_params']['turbines'] = turbine_list
                config['wind_params']['hub_heights'] = height_list

                write_config(config,config_path)
            
            stations_list = get_station_list(wind_dir)
            stations_list = [int(station) for station in stations_list]

            winddata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            markers = {row['station_id']: [row['latitude'], row['longitude']] for _, row in winddata.iterrows()}

            #For centering the content in the dashboard canvas
            col1, col2, col3 = st.columns([0.1,0.8,0.1])

            with col2:
                st.info("Click on the station marker you want to generate the data for!!", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers)
                
                    # Update session state based on the map click
                    if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                        lat = map_data["last_object_clicked"]["lat"]
                        lon = map_data["last_object_clicked"]["lng"]
    
                        # Determine which marker was clicked based on coordinates
                        for marker_name, coords in markers.items():
                            if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                                st.session_state.selected_marker = marker_name
                                break

                with st.container(border=True):

                    # Show the plot for the selected marker
                    if st.session_state.selected_marker:
                        park_data = pd.DataFrame
                        show_wind_plot(config,st.session_state.selected_marker,wind_dir,winddata,park_data,commissioning_date)
                        del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv


elif st.session_state["current_view"] == "compare your plant":
    # --- Compare Your Plant Page logic ---

    #Creating a sidebar to choose solar/wind application
    with st.sidebar:
        typeInput = st.selectbox(label="Type of Energy resource",options=["Solar", "Wind"],index=None)

    if typeInput is not None:
        
        if typeInput == "Solar":

            real_park = None
            my_location = None

            with st.sidebar:

                info_icon = "ℹ️" 

                # Use st.expander to create a collapsible section that looks like a pop-up
                with st.expander(info_icon):
                    st.write(
                        "Wiki for data schema is located in the `docs` subfolder "
                    )

                uploaded_file = st.file_uploader("Upload you park data", type="csv", accept_multiple_files=False)
                if uploaded_file is not None:
                    real_park = pd.read_csv(uploaded_file, sep=';')

                colsd1,colsd2 = st.columns([0.5,0.5])

                with colsd1:

                    config['pv_params']['dc_rating_watts'] = st.number_input("Installed Power", value=10000)
                    config['pv_params']['surface_tilt'] = st.number_input("Surface Tilt", value=35)
                    config['pv_params']['surface_azimuth'] = st.number_input("Surface Azimuth", value=180)
                    latitude = st.number_input("Park latitude", value=51.4)

                with colsd2:

                    config['pv_params']['gamma_pdc'] = st.number_input("Gamma PDC", value=-0.0035,format='%0.5f')
                    config['pv_params']['albedo'] = st.number_input("Albedo", value=0.25)
                    config['pv_params']['eta_inv_nom'] = st.number_input("Eta Inverted Nom", value=0.96)
                    longitude = st.number_input("Park longitude", value=10.4515)
                
                write_config(config,config_path)
                my_location = (latitude, longitude)

            stations_list = get_station_list(pv_dir)
            stations_list = [int(station) for station in stations_list]

            solardata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            result = find_nearest_station_for_park(latitude,longitude,solardata)

            markers = {row['station_id']: [float(row['latitude']), float(row['longitude'])] for _, row in solardata.iterrows()}
            nearest_station = {result['park_id']: result}

            #For centering the content in the dashboard canvas
            col1, col2, col3 = st.columns([0.1,0.8,0.1])

            with col2:
                st.info("Click on the station marked in green!!", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers,my_location,nearest_station)

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

                with st.container(border=True):

                    # Show the plot for the selected marker
                    if st.session_state.selected_marker:
                        latitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['latitude'].iloc[0]
                        longitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['longitude'].iloc[0]
                        elevation = solardata.loc[solardata.station_id == st.session_state.selected_marker]['station_height'].iloc[0]

                        if real_park is not None and not real_park.empty:
                            if not isinstance(real_park.index, pd.DatetimeIndex):
                                try:
                                    real_park.index = pd.to_datetime(real_park.index)
                                except Exception as e:
                                    st.error(f"Error converting secondary DataFrame index to DatetimeIndex: {e}")

                            show_pv_plot(config,st.session_state.selected_marker,pv_dir,latitude,longitude,elevation,real_park)
                            del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv
                        else:
                            st.warning(body="Upload the park file",icon="⚠️")

        elif typeInput == "Wind":

            real_park = None
            my_location = None
            commissioning_date = None

            turbine_list = []
            height_list = []

            with st.sidebar:

                info_icon = "ℹ️" 

                # Use st.expander to create a collapsible section that looks like a pop-up
                with st.expander(info_icon):
                    st.write(
                        "Wiki for data schema is located in the `docs` subfolder "
                    )
                
                uploaded_file = st.file_uploader("Upload you park data", type="csv", accept_multiple_files=False)
                if uploaded_file is not None:
                    real_park = pd.read_csv(uploaded_file, sep=';')
                
                    power_t_columns = [col for col in real_park.columns if col.startswith('power_t')]
                    speed_t_columns = [col for col in real_park.columns if col.startswith('wind_speed_t')]
                    real_park['power'] = real_park[power_t_columns].sum(axis=1)
                    real_park['wind_speed_mean'] = real_park[speed_t_columns].mean(axis=1)

                col1, col2 = st.columns([0.5,0.5])

                with col1:
                    latitude = st.number_input("Park latitude", value=51.4)

                with col2:
                    longitude = st.number_input("Park longitude", value=10.4515)

                ageing = st.toggle("Enable Ageing")
                config['wind_params']['apply_ageing'] = ageing

                if ageing:

                    commissioning_date = st.date_input("Commissioning Date", value="2013-01-01", format= "YYYY-MM-DD")

                my_location = (latitude, longitude)

                turbines_number = st.number_input("Number of turbines in park",value=1,min_value=1)

                col1, col2 = st.columns([0.5,0.5])

                with col1:

                    selected_turbine = st.selectbox(label="Select a type of turbine", options=turbine_types[1:], key=f'turbine0')
                    turbine_list.append(selected_turbine)

                with col2:

                    hub_height = st.number_input("Hub Height", value=5, key=f'height0')
                    height_list.append(hub_height)

                for number in range(turbines_number-1):

                    with col1:
                        selected_turbine = st.selectbox(label="Select a type of turbine", label_visibility="hidden", options=turbine_types[1:], key=f'turbine{number+1}')
                        turbine_list.append(selected_turbine)

                    with col2:
                        hub_height = st.number_input(label="Hub Height", label_visibility="hidden", value=5, key=f'height{number+1}')
                        height_list.append(hub_height)

                config['wind_params']['turbines'] = turbine_list
                config['wind_params']['hub_heights'] = height_list

                write_config(config,config_path)
            
            stations_list = get_station_list(wind_dir)
            stations_list = [int(station) for station in stations_list]

            winddata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            result = find_nearest_station_for_park(latitude,longitude,winddata)
            nearest_station = {result['park_id']: result}

            markers = {row['station_id']: [row['latitude'], row['longitude']] for _, row in winddata.iterrows()}

            #For centering the content in the dashboard canvas
            col1, col2, col3 = st.columns([0.1,0.8,0.1])

            with col2:

                st.info("Click on the station marked in green!!", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers,my_location,nearest_station)

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
                with st.container(border=True):

                    # Show the plot for the selected marker
                    if st.session_state.selected_marker:
                        if real_park is not None and not real_park.empty:
                            if not isinstance(real_park.index, pd.DatetimeIndex):
                                try:
                                    real_park.index = pd.to_datetime(real_park.index)
                                except Exception as e:
                                    st.error(f"Error converting secondary DataFrame index to DatetimeIndex: {e}")
                            show_wind_plot(config,st.session_state.selected_marker,wind_dir,winddata,real_park,commissioning_date)
                            del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv
                        else:
                            st.warning(body="Upload the park file",icon="⚠️")