from pathlib import Path
import re
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tools import load_config_ruamel, write_config
from plots_utils import (get_station_list, create_map, show_pv_plot, show_wind_plot,
                        find_nearest_station_for_park, build_grid_index, find_nearest_grid_point,
                        show_multi_turbine_wind_plot, search_location)


def panel_compensated_center(lat, lon, zoom, panel_half_px=190):
    """
    Shift a map center's longitude west so a target point renders in the middle
    of the space actually visible next to the floating options panel (~380px
    wide, so half of that on either side of the point) instead of being covered
    by it — the map itself spans the full width, so centering on the point's
    true coordinates would place it behind the panel.
    """
    degrees_per_pixel = 360.0 / (256 * (2 ** zoom))
    return [lat, lon - panel_half_px * degrees_per_pixel]


def render_results_view():
    """
    Full, fresh screen (no navbar, no tabs, no map/options panel) showing just the
    plots and data table for whichever section's "📊 View Generated Data" button was
    clicked. Relies on a snapshot of the data captured in session_state at the moment
    the button was shown, since the widgets that produced that data (file uploaders,
    turbine list, etc.) aren't re-rendered on this screen.
    """
    view = st.session_state.get("active_results_view")

    # Small top spacer so the back button isn't flush against the very top of the
    # browser, where Streamlit's own toolbar can overlap/clip it.
    st.markdown("<div style='padding-top: 2.5rem;'></div>", unsafe_allow_html=True)

    if st.button("← Back to Dashboard", key="results_view_back_btn"):
        st.session_state["active_results_view"] = None
        st.rerun()

    def render_title(text):
        # Match the main landing-page title's size (2.6rem) for consistency, with
        # some breathing room below before the success/info boxes that follow.
        st.markdown(
            f"<h2 style='text-align: center; font-size: 2.6rem; margin: 0 0 1.5rem 0;'>{text}</h2>",
            unsafe_allow_html=True,
        )

    if view == "viz_solar":
        snap = st.session_state.get("viz_solar_results_snapshot")
        if not snap:
            st.warning("No generated data to show yet.")
            return
        render_title("Generated Solar Data")
        show_pv_plot(
            snap["config"], snap["selected_marker"], snap["pv_dir"],
            snap["latitude"], snap["longitude"], snap["elevation"], pd.DataFrame,
        )

    elif view == "viz_wind":
        snap = st.session_state.get("viz_wind_results_snapshot")
        if not snap:
            st.warning("No generated data to show yet.")
            return
        render_title("Generated Wind Data")
        show_multi_turbine_wind_plot(
            snap["wind_turbines"], snap["grid_points"], snap["kdtree"], snap["filenames"],
            commissioning_date=snap["commissioning_date"],
            apply_wake=snap.get("apply_wake", True),
            wake_decay=snap.get("wake_decay", 0.075),
        )

    elif view == "compare_solar":
        snap = st.session_state.get("compare_solar_results_snapshot")
        if not snap:
            st.warning("No generated data to show yet.")
            return
        render_title("Generated vs Real Solar Data")
        show_pv_plot(
            snap["config"], snap["selected_marker"], snap["pv_dir"],
            snap["latitude"], snap["longitude"], snap["elevation"], snap["real_park"],
        )

    elif view == "compare_wind":
        snap = st.session_state.get("compare_wind_results_snapshot")
        if not snap:
            st.warning("No generated data to show yet.")
            return
        render_title("Generated vs Real Wind Data")
        show_wind_plot(
            snap["config"], snap["selected_marker"], snap["wind_dir"],
            snap["winddata"], snap["real_park"], snap["commissioning_date"],
        )

# --- Page Configuration ---
# No Streamlit sidebar drawer is used anymore: options live in a column directly
# beside the map (like renewables.ninja), always visible, not a collapsible panel.
st.set_page_config(page_title="Synthetic Renewables Data Generation Dashboard", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Center the main title */
        h1 {
            text-align: center;
            color: #2F80ED; /* A nice blue color for the title */
            margin-bottom: 2rem; /* Add some space below the title */
        }

        /* Note: the old landing-page radio ("What would you like to do?") used to be
           styled here at a huge font size. That page no longer exists — the same
           radio widget is now the compact mode toggle in the top navbar, so those
           oversized rules were removed rather than left to blow up the navbar. */

        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            font-size: 1.5em;
            padding: 18px 35px;
            border-radius: 10px;
            border: none;
            background-color: #28A745;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }

        .stButton > button:hover {
            background-color: #218838;
            transform: translateY(-2px);
        }

        /* ⬇ New style for smaller secondary buttons */
        div[class*="st-key-back_to_home_visualise"] .stButton button{
            font-size: 1em !important;
            padding: 8px 20px !important;
            background-color: #6c757d !important;
            color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin-top: 10px !important;
        }

        div[class*="st-key-back_to_home_visualise"] .stButton button:hover {
            background-color: #5a6268 !important;
        }

        /* Smaller button — back_to_home_compare */
        div[class*="st-key-back_to_home_compare"] .stButton button {
            font-size: 1em !important;
            padding: 8px 20px !important;
            background-color: #6c757d !important;
            color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin-top: 10px !important;
        }

        div[class*="st-key-back_to_home_compare"] .stButton button:hover {
            background-color: #5a6268 !important;
        }

        /* Compact "data ready" popup card — floats over the MAP (top-right),
           outside the options panel entirely, so it doesn't compete for space
           with the input fields, but still doesn't require scrolling to see. */
        div[class*="st-key-viz_solar_data_ready_popup"],
        div[class*="st-key-viz_wind_data_ready_popup"],
        div[class*="st-key-compare_solar_data_ready_popup"],
        div[class*="st-key-compare_wind_data_ready_popup"] {
            position: absolute !important;
            top: 0.5rem !important;
            right: 1.5rem !important;
            z-index: 1000 !important;
            width: 260px !important;
            padding: 10px 12px !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18) !important;
            background: #ffffff !important;
            color: #1a1a1a !important;
        }

        /* This card has a hardcoded white background regardless of theme, so its
           text must be forced dark too — otherwise Streamlit's dark-mode text
           color (light gray/white) becomes invisible against the white card. */
        div[class*="data_ready_popup"] label,
        div[class*="data_ready_popup"] p,
        div[class*="data_ready_popup"] span,
        div[class*="data_ready_popup"] div,
        div[class*="data_ready_popup"] h1,
        div[class*="data_ready_popup"] h2,
        div[class*="data_ready_popup"] h3,
        div[class*="data_ready_popup"] h4 {
            color: #1a1a1a !important;
        }

        div[class*="data_ready_popup"] .stAlert {
            padding: 6px 10px !important;
            font-size: 0.85em !important;
        }

        div[class*="data_ready_popup"] .stButton button {
            display: block !important;
            width: 100% !important;
            margin: 6px 0 0 0 !important;
            font-size: 0.85em !important;
            padding: 6px 10px !important;
            box-shadow: none !important;
        }

        /* Keep the paired chart cards on the results screen the same height even
           when one card has an extra selector (e.g. a feature dropdown) that the
           other doesn't — otherwise the row looks uneven/mismatched. */
        div[class*="results_chart_card"] {
            min-height: 620px;
            max-width: 780px;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* These cards have no selector above their plot, so with the shared
           min-height they end up top-aligned with a lot of empty space below.
           Center their content vertically instead. */
        div[class*="st-key-results_chart_card_total"],
        div[class*="st-key-results_chart_card_degradation_summary"],
        div[class*="st-key-results_chart_card_degradation_timeline"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        /* Back button on the full-screen "View Generated Data" page — small,
           left-aligned, secondary style, not the big green primary button. */
        div[class*="st-key-results_view_back_btn"] .stButton button {
            display: inline-block !important;
            width: auto !important;
            margin: 0 !important;
            font-size: 0.95em !important;
            padding: 8px 18px !important;
            background-color: #6c757d !important;
            color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }

        div[class*="st-key-results_view_back_btn"] .stButton button:hover {
            background-color: #5a6268 !important;
        }

        /* Location search buttons — sit next to the search text input, so keep
           them compact (no huge padding/font) and prevent the label from
           wrapping vertically. */
        div[class*="st-key-viz_solar_search_btn"] .stButton button,
        div[class*="st-key-viz_wind_search_btn"] .stButton button,
        div[class*="st-key-compare_solar_search_btn"] .stButton button,
        div[class*="st-key-compare_wind_search_btn"] .stButton button {
            display: inline-block !important;
            width: 100% !important;
            white-space: nowrap !important;
            font-size: 0.9em !important;
            padding: 8px 10px !important;
            margin: 0 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }

        /* Add turbine button — same compact style as the search buttons, but
           only half-width instead of filling the whole options panel, and
           centered. */
        div[class*="st-key-viz_wind_add_turbine_btn"] .stButton button {
            display: block !important;
            width: 50% !important;
            margin: 0 auto !important;
            white-space: nowrap !important;
            font-size: 0.9em !important;
            padding: 8px 10px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }

        /* Small icon-only delete buttons (e.g. turbine list rows) — plain,
           unobtrusive, not the big green primary-action style. */
        div[class*="st-key-viz_wind_remove_turbine_"] .stButton button {
            display: inline-block !important;
            width: auto !important;
            min-width: 0 !important;
            font-size: 0.95em !important;
            line-height: 1 !important;
            padding: 4px 8px !important;
            margin: 0 !important;
            background-color: transparent !important;
            color: inherit !important;
            border: 1px solid #ddd !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }

        div[class*="st-key-viz_wind_remove_turbine_"] .stButton button:hover {
            background-color: #f5f5f5 !important;
            border-color: #ccc !important;
            transform: none !important;
        }

        /* Style for selectbox labels (if used elsewhere) */
        .stSelectbox > label {
            display: block;
            text-align: center;
            font-size: 1em;
            color: #555;
        }

        /* Cap the page at a comfortable fixed width and center it, instead of
           stretching to fill whatever monitor it's opened on — a laptop and an
           external monitor otherwise render wildly different proportions
           (oversized maps/plots, cramped text) because "full browser width"
           means something different on each screen. */
        .block-container {
            max-width: 1600px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            padding-top: 2.5rem !important;
        }

        /* Floating options panel over a full-bleed map, renewables.ninja style.
           This is a CSS hack (position: absolute pulled out of normal flow),
           not a native Streamlit layout — may need tuning across screen sizes. */
        div[class*="st-key-viz_floating_panel"],
        div[class*="st-key-compare_floating_panel"] {
            position: absolute;
            top: 0.5rem;
            left: 0;
            z-index: 999;
            width: 380px;
            max-width: 90vw;
            max-height: 78vh;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.97);
            border-radius: 10px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.18);
            padding: 16px 18px;
            color: #1a1a1a !important;
        }

        /* Same reasoning as the popup card above — this panel is always a
           near-opaque white regardless of light/dark theme, so force its text
           dark so labels/inputs stay legible in dark mode. */
        div[class*="st-key-viz_floating_panel"] label,
        div[class*="st-key-viz_floating_panel"] p,
        div[class*="st-key-viz_floating_panel"] span,
        div[class*="st-key-viz_floating_panel"] div,
        div[class*="st-key-viz_floating_panel"] h1,
        div[class*="st-key-viz_floating_panel"] h2,
        div[class*="st-key-viz_floating_panel"] h3,
        div[class*="st-key-viz_floating_panel"] h4,
        div[class*="st-key-compare_floating_panel"] label,
        div[class*="st-key-compare_floating_panel"] p,
        div[class*="st-key-compare_floating_panel"] span,
        div[class*="st-key-compare_floating_panel"] div,
        div[class*="st-key-compare_floating_panel"] h1,
        div[class*="st-key-compare_floating_panel"] h2,
        div[class*="st-key-compare_floating_panel"] h3,
        div[class*="st-key-compare_floating_panel"] h4 {
            color: #1a1a1a !important;
        }

        /* Text inputs/selects inside the panel also need an explicit light
           background — in dark mode Streamlit's own input backgrounds turn
           dark, which combined with the dark text above would be unreadable. */
        div[class*="st-key-viz_floating_panel"] input,
        div[class*="st-key-viz_floating_panel"] textarea,
        div[class*="st-key-viz_floating_panel"] div[data-baseweb="select"] > div,
        div[class*="st-key-compare_floating_panel"] input,
        div[class*="st-key-compare_floating_panel"] textarea,
        div[class*="st-key-compare_floating_panel"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }

        /* Tab panel needs to be a positioning context for the floating panel above */
        .stTabs [data-baseweb="tab-panel"] {
            position: relative;
        }

        /* Position the footer */
        .fixed-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f0f2f6; /* A light gray background, similar to Streamlit's default */
                color: #888;
                text-align: center;
                padding: 10px 0;
                font-size: 0.8em;
                z-index: 9999; /* Ensure it's on top of other elements */
                border-top: 1px solid #e6e6e6;
            }        
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "selected_marker" not in st.session_state:
                    st.session_state.selected_marker = None

if 'selected_grid_point' not in st.session_state:
    st.session_state.selected_grid_point = None

# --- Grid Point Setup ---
# Build grid index once at startup for performance
if 'grid_points' not in st.session_state:
    with st.spinner("Loading grid points..."):
        grid_points, kdtree, filenames = build_grid_index()
        st.session_state['grid_points'] = grid_points
        st.session_state['kdtree'] = kdtree  
        st.session_state['filenames'] = filenames

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

#pv_columns = ["global_horizontal_irradiance","diffuse_horizontal_irradiance","direct_normal_irradiance"]

if "active_results_view" not in st.session_state:
    st.session_state["active_results_view"] = None

# --- Results view: a fresh, standalone screen (no navbar/tabs/map) showing the
# plots and data table for whichever section's "View Generated Data" button was
# clicked. When active, skip rendering the rest of the dashboard entirely.
if st.session_state["active_results_view"]:
    render_results_view()
    st.stop()

# --- Top Navbar: logo and title (always visible, no landing page) ---
navbar_logo_col, navbar_title_col, navbar_spacer_col = st.columns([1.8, 3, 1.2])

with navbar_logo_col:
    # Small top spacer so the logo isn't flush against the very top of the
    # browser (where Streamlit's own toolbar can overlap it) and sits level
    # with the title next to it.
    # The source file is a wide 2459x568 lockup with small subtext, so it
    # needs a bigger render width than before or the "University of Applied
    # Sciences" line becomes unreadable.
    st.markdown("<div style='padding-top: 18px;'></div>", unsafe_allow_html=True)
    st.image("./logos/HKA_Logo.png", width=320)

with navbar_title_col:
    st.markdown(
        "<div style='padding-top: 14px; text-align: center;'><h1 style='font-size: 2.6rem; margin: 0;'>Synthetic Renewables Data Generation Dashboard</h1></div>",
        unsafe_allow_html=True,
    )

# --- Mode tabs: Visualize Your Plant / Compare Your Plant ---
# Each tab's own map (Germany-centered by default in create_map) is the main visual.
tab_visualise, tab_compare = st.tabs(["🔆 Visualise Your Plant", "🔍 Compare Your Plant"])

# --- Visualise Your Plant tab ---
with tab_visualise:

    if "active_resource_visualise" not in st.session_state:
        # No resource chosen yet: the map still shows on load, but with no
        # markers at all, until Solar or Wind is picked via its Run button.
        st.session_state["active_resource_visualise"] = None

    # Reserved slot for the "Data ready / View Generated Data" popup. Created first
    # (before options_col/map_col) so it's a sibling near the top of the DOM, not
    # nested inside the options panel — its own CSS position takes it out of flow
    # to float over the map, but creating it first keeps it above everything else
    # if that positioning doesn't apply for any reason.
    viz_results_placeholder = st.empty()

    # Floating options panel over a full-bleed map (CSS overlay), like
    # renewables.ninja — not a side-by-side column. map_col is a plain
    # full-width container; options_col is pulled out of normal flow by
    # CSS (position: absolute) so the map behind it spans the full width.
    options_col = st.container(key="viz_floating_panel")
    map_col = st.container()

    with options_col:

        resource_options = ["Solar", "Wind"]
        current_resource = st.session_state["active_resource_visualise"]

        chosen_resource = st.radio(
            "Resource",
            resource_options,
            index=resource_options.index(current_resource) if current_resource in resource_options else None,
            horizontal=True,
            key="viz_resource_picker",
        )

        if chosen_resource != st.session_state["active_resource_visualise"]:
            st.session_state["active_resource_visualise"] = chosen_resource
            st.rerun()

        if chosen_resource == "Solar":

            # --- Location search ---
            st.markdown("**📍 Search location**")
            search_input_col, search_btn_col = st.columns([0.68, 0.32], vertical_alignment="bottom")
            with search_input_col:
                search_query = st.text_input(
                    "Search for a place",
                    placeholder="e.g. Berlin, Munich...",
                    key="viz_solar_location_search",
                    label_visibility="collapsed",
                )
            with search_btn_col:
                search_clicked = st.button("Search", key="viz_solar_search_btn")

            if search_clicked and search_query:
                st.session_state["viz_solar_last_search"] = search_query
                st.session_state["viz_solar_search_results"] = search_location(search_query)

            results = st.session_state.get("viz_solar_search_results", [])
            if results:
                match_labels = [r["display_name"] for r in results]
                picked_label = st.radio(
                    "Matches", match_labels, key="viz_solar_search_pick", label_visibility="collapsed"
                )
                picked = next((r for r in results if r["display_name"] == picked_label), None)
                if picked is not None:
                    new_loc = (picked["lat"], picked["lon"])
                    if st.session_state.get("viz_solar_searched_location") != new_loc:
                        st.session_state["viz_solar_searched_location"] = new_loc
                        st.session_state["viz_solar_lat_input"] = str(new_loc[0])
                        st.session_state["viz_solar_lon_input"] = str(new_loc[1])
                        st.rerun()
            elif search_clicked and search_query:
                st.caption("No matches found.")

            # --- Manual latitude / longitude entry (mirrors Compare Your Plant) ---
            colsd_lat, colsd_lon = st.columns([0.5, 0.5])
            with colsd_lat:
                manual_solar_lat_str = st.text_input("Park latitude", key="viz_solar_lat_input", placeholder="e.g. 51.4")
            with colsd_lon:
                manual_solar_lon_str = st.text_input("Park longitude", key="viz_solar_lon_input", placeholder="e.g. 10.4515")

            manual_solar_lat, manual_solar_lon = None, None
            try:
                if manual_solar_lat_str.strip():
                    manual_solar_lat = float(manual_solar_lat_str)
                if manual_solar_lon_str.strip():
                    manual_solar_lon = float(manual_solar_lon_str)
            except ValueError:
                st.caption("⚠️ Latitude/longitude must be numbers.")

            if manual_solar_lat is not None and manual_solar_lon is not None:
                new_manual_loc = (manual_solar_lat, manual_solar_lon)
                if new_manual_loc != st.session_state.get("viz_solar_searched_location"):
                    st.session_state["viz_solar_searched_location"] = new_manual_loc
            else:
                st.session_state.pop("viz_solar_searched_location", None)

            st.markdown("---")

            # --- Solar photovoltaic power (PV) parameters ---
            st.markdown("**☀️ Solar photovoltaic power (PV)**")

            colsd1,colsd2 = st.columns([0.5,0.5])

            with colsd1:

                config['pv_params']['dc_rating_watts'] = st.number_input("Installed Capacity", value=10000, key="viz_pv_capacity")
                config['pv_params']['surface_tilt'] = st.number_input("Surface Tilt", value=35, key="viz_pv_tilt")
                config['pv_params']['surface_azimuth'] = st.number_input("Surface Azimuth", value=180, key="viz_pv_azimuth")

            with colsd2:

                config['pv_params']['gamma_pdc'] = st.number_input("Temperature Coefficient", value=-0.0035,format='%0.5f', key="viz_pv_gamma")
                config['pv_params']['albedo'] = st.number_input("Albedo", value=0.25, key="viz_pv_albedo")
                config['pv_params']['eta_inv_nom'] = st.number_input("Inverter Efficiency", value=0.96, key="viz_pv_eta")

            write_config(config,config_path)

        elif chosen_resource == "Wind":

            # --- Location search (helps find coordinates for a turbine) ---
            st.markdown("**📍 Search location**")
            wind_search_input_col, wind_search_btn_col = st.columns([0.68, 0.32], vertical_alignment="bottom")
            with wind_search_input_col:
                wind_search_query = st.text_input(
                    "Search for a place",
                    placeholder="e.g. Berlin, Munich...",
                    key="viz_wind_location_search",
                    label_visibility="collapsed",
                )
            with wind_search_btn_col:
                wind_search_clicked = st.button("Search", key="viz_wind_search_btn")

            if wind_search_clicked and wind_search_query:
                st.session_state["viz_wind_last_search"] = wind_search_query
                st.session_state["viz_wind_search_results"] = search_location(wind_search_query)

            wind_results = st.session_state.get("viz_wind_search_results", [])
            if wind_results:
                wind_match_labels = [r["display_name"] for r in wind_results]
                wind_picked_label = st.radio(
                    "Matches", wind_match_labels, key="viz_wind_search_pick", label_visibility="collapsed"
                )
                wind_picked = next((r for r in wind_results if r["display_name"] == wind_picked_label), None)
                if wind_picked is not None:
                    picked_loc = (wind_picked["lat"], wind_picked["lon"])
                    if st.session_state.get("viz_wind_last_picked_loc") != picked_loc:
                        st.session_state["viz_wind_last_picked_loc"] = picked_loc
                        st.session_state["viz_wind_add_lat"] = str(wind_picked["lat"])
                        st.session_state["viz_wind_add_lon"] = str(wind_picked["lon"])
                        # Also mark this as the map's current focus point, so the
                        # map auto-pans/zooms to it (same mechanism as clicking).
                        st.session_state["viz_wind_pending_click"] = {"lat": wind_picked["lat"], "lng": wind_picked["lon"]}
                        st.rerun()
            elif wind_search_clicked and wind_search_query:
                st.caption("No matches found.")

            st.markdown("---")

            # --- Wind park parameters ---
            st.markdown("**💨 Wind power**")

            ageing = st.toggle("Enable Ageing", key="viz_wind_ageing")
            config['wind_params']['apply_ageing'] = ageing

            commissioning_date = None
            if ageing:
                commissioning_date = st.date_input("Commissioning Date", value="2013-01-01", format="YYYY-MM-DD", key="viz_wind_commissioning")

            st.session_state["viz_wind_commissioning_date"] = commissioning_date

            # Wake losses: reduces each turbine's output based on how much it
            # sits in another turbine's wake (only matters with 2+ turbines).
            apply_wake = st.toggle("Apply wake losses between turbines", value=True, key="viz_wind_apply_wake")
            if apply_wake:
                wake_decay = st.slider(
                    "Wake decay coefficient (k)", min_value=0.03, max_value=0.15,
                    value=0.075, step=0.005, key="viz_wind_wake_decay",
                    help="How fast a turbine's wake widens/weakens with distance. "
                         "Lower (~0.04) for open/offshore-like terrain, higher (~0.1) "
                         "for onshore/complex terrain.",
                )
            else:
                wake_decay = 0.075
            st.session_state["viz_wind_wake_decay_value"] = wake_decay

            write_config(config,config_path)

            st.markdown("---")

            # --- Add a turbine: location, type and hub height together ---
            st.markdown("**Add a turbine**")

            # Sync in any pending map click BEFORE the lat/lon widgets below are
            # instantiated — Streamlit forbids writing to a widget's session_state
            # key after that widget has already been created in the current run,
            # which is exactly what happens if the map (rendered later, in map_col)
            # tries to set these same keys after this run already drew the widgets.
            # Doing it here, pre-widget, on the rerun the click triggers, avoids that.
            pending_loc_for_sync = st.session_state.get("viz_wind_pending_click")
            if pending_loc_for_sync:
                synced_loc = (pending_loc_for_sync["lat"], pending_loc_for_sync["lng"])
                if st.session_state.get("viz_wind_add_synced_for") != synced_loc:
                    st.session_state["viz_wind_add_lat"] = str(pending_loc_for_sync["lat"])
                    st.session_state["viz_wind_add_lon"] = str(pending_loc_for_sync["lng"])
                    st.session_state["viz_wind_add_synced_for"] = synced_loc

            add_lat_col, add_lon_col = st.columns([0.5, 0.5])
            with add_lat_col:
                add_turbine_lat_str = st.text_input("Turbine latitude", key="viz_wind_add_lat", placeholder="e.g. 51.4")
            with add_lon_col:
                add_turbine_lon_str = st.text_input("Turbine longitude", key="viz_wind_add_lon", placeholder="e.g. 10.4515")

            add_type_col, add_height_col = st.columns([0.5, 0.5])
            with add_type_col:
                add_turbine_type = st.selectbox(label="Select a type of turbine", options=turbine_types[1:], key="viz_wind_add_type")
            with add_height_col:
                add_turbine_height = st.number_input("Hub Height", value=10, key="viz_wind_add_height")

            # Per-turbine commissioning date — only shown when ageing is
            # enabled overall. Each turbine can have its own install date, so
            # a park with a mix of old and recently-repowered turbines can be
            # modeled instead of forcing every turbine to age identically.
            add_turbine_commissioning = None
            if ageing:
                add_turbine_commissioning = st.date_input(
                    "Turbine commissioning date", value="2013-01-01", format="YYYY-MM-DD",
                    key="viz_wind_add_commissioning",
                    help="Used to age this specific turbine's output — different turbines "
                         "in the same park can have different install dates.",
                )

            add_turbine_clicked = st.button("Add turbine", key="viz_wind_add_turbine_btn")

            if "viz_wind_turbines" not in st.session_state:
                st.session_state["viz_wind_turbines"] = []

            if add_turbine_clicked:
                try:
                    add_lat_val = float(add_turbine_lat_str)
                    add_lon_val = float(add_turbine_lon_str)
                    st.session_state["viz_wind_turbines"].append({
                        "lat": add_lat_val,
                        "lon": add_lon_val,
                        "turbine": add_turbine_type,
                        "hub_height": add_turbine_height,
                        "commissioning_date": add_turbine_commissioning,
                    })
                    st.rerun()
                except (TypeError, ValueError):
                    st.warning("⚠️ Enter valid latitude and longitude before adding a turbine.")

            st.markdown("---")

            # --- List of turbines added so far ---
            st.markdown("**Turbines list**")
            wind_turbines = st.session_state.get("viz_wind_turbines", [])
            if wind_turbines:
                for t_idx, t in enumerate(wind_turbines):
                    list_col, remove_col = st.columns([0.85, 0.15], vertical_alignment="center")
                    with list_col:
                        age_note = ""
                        if ageing and t.get("commissioning_date"):
                            age_note = f", commissioned {t['commissioning_date']}"
                        st.caption(f"📍 ({t['lat']:.4f}, {t['lon']:.4f}) — {t['turbine']}, hub {t['hub_height']}m{age_note}")
                    with remove_col:
                        if st.button("🗑️", key=f"viz_wind_remove_turbine_{t_idx}"):
                            st.session_state["viz_wind_turbines"].pop(t_idx)
                            st.rerun()
            else:
                st.caption("No turbines added yet.")

        else:
            st.caption("Choose Solar or Wind to see locations on the map.")

    if chosen_resource == "Solar":

            stations_list = get_station_list(pv_dir)
            stations_list = [int(station) for station in stations_list]

            solardata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            markers = {row['station_id']: [float(row['latitude']), float(row['longitude'])] for _, row in solardata.iterrows()}

            my_location = None
            nearest_station = {}
            searched_loc = st.session_state.get("viz_solar_searched_location")
            if searched_loc:
                my_location = searched_loc
                search_result = find_nearest_station_for_park(searched_loc[0], searched_loc[1], solardata)
                nearest_station = {search_result['park_id']: search_result}
                st.session_state.selected_marker = search_result['park_id']

            with map_col:
                # st.info("Click on the station marker you want to generate the data for.", icon="ℹ️")

                # Auto-pan/zoom the map to the searched/entered location instead of
                # leaving it at the default Germany-wide view.
                map_zoom = 11 if my_location else 7
                map_center = panel_compensated_center(my_location[0], my_location[1], map_zoom) if my_location else None

                map_data = create_map(markers, my_location, nearest_station, key="viz_solar_map", center=map_center, zoom_start=map_zoom)

                # Update session state based on the map click
                if map_data and "last_clicked" in map_data and map_data["last_object_clicked"] is not None:
                    lat = map_data["last_object_clicked"]["lat"]
                    lon = map_data["last_object_clicked"]["lng"]

                    # Determine which marker was clicked based on coordinates
                    for marker_name, coords in markers.items():
                        if abs(coords[0] - lat) < 0.01 and abs(coords[1] - lon) < 0.01:
                            st.session_state.selected_marker = marker_name
                            break

                # Data is ready once a marker is selected — offer a button, rendered
                # up in the floating panel (not below the tall map), to view the
                # plots/table on a fresh screen instead of showing them inline.
                if st.session_state.selected_marker:
                    latitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['latitude'].iloc[0]
                    longitude = solardata.loc[solardata.station_id == st.session_state.selected_marker]['longitude'].iloc[0]
                    elevation = solardata.loc[solardata.station_id == st.session_state.selected_marker]['station_height'].iloc[0]

                    st.session_state["viz_solar_results_snapshot"] = {
                        "config": config,
                        "selected_marker": st.session_state.selected_marker,
                        "pv_dir": pv_dir,
                        "latitude": latitude,
                        "longitude": longitude,
                        "elevation": elevation,
                    }

                    with viz_results_placeholder.container():
                        with st.container(key="viz_solar_data_ready_popup"):
                            st.success(f"✅ Data ready for station {st.session_state.selected_marker}.")
                            if st.button("View Generated Data", key="viz_solar_view_data_btn"):
                                st.session_state["active_results_view"] = "viz_solar"
                                st.rerun()

                    del st.session_state.selected_marker #Clearing the session state to show no plots when switched from pv to wind

    if chosen_resource == "Wind":

            commissioning_date = st.session_state.get("viz_wind_commissioning_date")
            wind_turbines = st.session_state.get("viz_wind_turbines", [])

            # Note: each turbine can sit at its own location (independent NWP grid
            # point), so the map shows one pin per added turbine, plus a pending pin
            # for the last clicked spot (used to prefill the "Add a turbine" fields).
            with map_col:
                grid_points = st.session_state.get("grid_points")
                kdtree = st.session_state.get("kdtree")
                filenames = st.session_state.get("filenames")

                markers = {}
                for t_idx, t in enumerate(wind_turbines):
                    markers[f"Turbine {t_idx + 1}: {t['turbine']}"] = [t['lat'], t['lon']]
                    if kdtree is not None:
                        nearest = find_nearest_grid_point(t['lat'], t['lon'], grid_points, kdtree, filenames)
                        if nearest:
                            markers[f"grid_point_turbine_{t_idx}"] = [nearest['grid_lat'], nearest['grid_lon']]

                pending_loc = st.session_state.get("viz_wind_pending_click")
                if pending_loc:
                    markers['📍 Click location (not yet added)'] = [pending_loc['lat'], pending_loc['lng']]
                    if kdtree is not None:
                        nearest_pending = find_nearest_grid_point(pending_loc['lat'], pending_loc['lng'], grid_points, kdtree, filenames)
                        if nearest_pending:
                            markers['grid_point_pending'] = [nearest_pending['grid_lat'], nearest_pending['grid_lon']]

                # Auto-pan/zoom to whichever location is currently "in focus" —
                # a pending click/search takes priority, otherwise the most
                # recently added turbine — instead of leaving the default view.
                if pending_loc:
                    focus_lat, focus_lon = pending_loc['lat'], pending_loc['lng']
                elif wind_turbines:
                    focus_lat, focus_lon = wind_turbines[-1]['lat'], wind_turbines[-1]['lon']
                else:
                    focus_lat, focus_lon = None, None

                if focus_lat is not None:
                    wind_map_zoom = 11
                    wind_map_center = panel_compensated_center(focus_lat, focus_lon, wind_map_zoom)
                else:
                    wind_map_zoom = 7
                    wind_map_center = None

                map_data = create_map(markers, key="viz_wind_map", center=wind_map_center, zoom_start=wind_map_zoom)

                if map_data and "last_clicked" in map_data and map_data["last_clicked"] is not None:
                    clicked_lat = map_data["last_clicked"]["lat"]
                    clicked_lng = map_data["last_clicked"]["lng"]

                    current_pending = st.session_state.get("viz_wind_pending_click")
                    is_new_click = (
                        not current_pending
                        or abs(current_pending["lat"] - clicked_lat) > 0.0001
                        or abs(current_pending["lng"] - clicked_lng) > 0.0001
                    )
                    if is_new_click:
                        # Only stage the click here — the "Add a turbine" lat/lon
                        # widgets already rendered earlier in options_col this run,
                        # so writing to their session_state keys directly at this
                        # point would raise a StreamlitAPIException. The sync step
                        # right before those widgets (in options_col) picks this up
                        # on the rerun, before they're instantiated again.
                        st.session_state["viz_wind_pending_click"] = {"lat": clicked_lat, "lng": clicked_lng}
                        st.rerun()

            if wind_turbines:
                st.session_state["viz_wind_results_snapshot"] = {
                    "wind_turbines": wind_turbines,
                    "grid_points": st.session_state.get("grid_points"),
                    "kdtree": st.session_state.get("kdtree"),
                    "filenames": st.session_state.get("filenames"),
                    "commissioning_date": commissioning_date,
                    "apply_wake": st.session_state.get("viz_wind_apply_wake", True),
                    "wake_decay": st.session_state.get("viz_wind_wake_decay_value", 0.075),
                }
                with viz_results_placeholder.container():
                    with st.container(key="viz_wind_data_ready_popup"):
                        st.success(f"✅ Data ready for {len(wind_turbines)} turbine(s).")
                        if st.button("View Generated Data", key="viz_wind_view_data_btn"):
                            st.session_state["active_results_view"] = "viz_wind"
                            st.rerun()
            # No turbines yet: nothing floats over the map — the "No turbines
            # added yet" hint already lives in the options panel next to the
            # turbine list, so no need to duplicate it here (it was rendering
            # in normal document flow above the map, overlapping the floating
            # options panel).

    if chosen_resource is None:
        # Nothing chosen yet: map spans the full page width, no markers.
        with map_col:
            create_map({}, key="viz_empty_map", zoom_start=6, center=[47.0, 6.0])


# --- Compare Your Plant tab ---
with tab_compare:

    if "active_resource_compare" not in st.session_state:
        # No resource chosen yet: the map still shows on load, but with no
        # markers at all, until Solar or Wind is picked.
        st.session_state["active_resource_compare"] = None

    solar_real_park = None
    solar_latitude, solar_longitude = 51.4, 10.4515
    wind_real_park = None
    wind_latitude, wind_longitude = 51.4, 10.4515
    wind_commissioning_date = None

    # Reserved slot for the "Data ready / View Generated Data" popup. Created first
    # (before options_col/map_col) so it's a sibling near the top of the DOM, not
    # nested inside the options panel — its own CSS position takes it out of flow
    # to float over the map, but creating it first keeps it above everything else
    # if that positioning doesn't apply for any reason.
    compare_results_placeholder = st.empty()

    # Floating options panel over a full-bleed map (CSS overlay), like
    # renewables.ninja — not a side-by-side column. map_col is a plain
    # full-width container; options_col is pulled out of normal flow by
    # CSS (position: absolute) so the map behind it spans the full width.
    options_col = st.container(key="compare_floating_panel")
    map_col = st.container()

    with options_col:

        resource_options = ["Solar", "Wind"]
        current_resource = st.session_state["active_resource_compare"]

        chosen_resource = st.radio(
            "Resource",
            resource_options,
            index=resource_options.index(current_resource) if current_resource in resource_options else None,
            horizontal=True,
            key="compare_resource_picker",
        )

        if chosen_resource != st.session_state["active_resource_compare"]:
            st.session_state["active_resource_compare"] = chosen_resource
            st.rerun()

        if chosen_resource == "Solar":

            # --- Solar photovoltaic power (PV) parameters ---
            st.markdown("**☀️ Solar photovoltaic power (PV)**")

            uploaded_file = st.file_uploader(
                "Upload you park data",
                type="csv",
                accept_multiple_files=False,
                key="compare_pv_upload",
                help="Wiki for data schema is located in the `docs` subfolder.",
            )
            #st.info("Upload the correct file", icon="ℹ️")
            if uploaded_file is not None:

                solar_real_park = pd.read_csv(uploaded_file, sep=';')
                st.info("Please ensure that PV file is uploaded.", icon="ℹ️")

                #pattern = re.compile(r'(wind_speed|power)_t\d+', re.IGNORECASE)
                #matched_columns = []

                #for col in real_park.columns:
                #    if pattern.search(col):
                #        matched_columns.append(col)

                #if matched_columns:
                #    st.error("The format of file is incorrect", icon="🚨")

            colsd1,colsd2 = st.columns([0.5,0.5])

            with colsd1:

                config['pv_params']['dc_rating_watts'] = st.number_input("Installed Capacity", value=10000, key="compare_pv_capacity")
                config['pv_params']['surface_tilt'] = st.number_input("Surface Tilt", value=35, key="compare_pv_tilt")
                config['pv_params']['surface_azimuth'] = st.number_input("Surface Azimuth", value=180, key="compare_pv_azimuth")
                solar_latitude = st.number_input("Park latitude", value=51.4, key="compare_pv_lat")

            with colsd2:

                config['pv_params']['gamma_pdc'] = st.number_input("Temperature Coefficient", value=-0.0035,format='%0.5f', key="compare_pv_gamma")
                config['pv_params']['albedo'] = st.number_input("Albedo", value=0.25, key="compare_pv_albedo")
                config['pv_params']['eta_inv_nom'] = st.number_input("Inverter Efficiency", value=0.96, key="compare_pv_eta")
                solar_longitude = st.number_input("Park longitude", value=10.4515, key="compare_pv_lon")

            write_config(config,config_path)

        elif chosen_resource == "Wind":

            # --- Wind power parameters ---
            st.markdown("**💨 Wind power**")

            uploaded_file = st.file_uploader(
                "Upload you park data",
                type="csv",
                accept_multiple_files=False,
                key="compare_wind_upload",
                help="Wiki for data schema is located in the `docs` subfolder.",
            )
            #st.info("Upload the correct file", icon="ℹ️")
            if uploaded_file is not None:

                wind_real_park = pd.read_csv(uploaded_file, sep=';')

                st.info("Please ensure that Wind file is uploaded.", icon="ℹ️")

                #column_names = set(real_park.columns.tolist())

                # Check if the column names match the expected list
                #matched_columns = column_names.intersection(pv_columns)

                #if matched_columns:
                #    st.error("The format of file is incorrect", icon="🚨")
                #else:
                power_t_columns = [col for col in wind_real_park.columns if col.startswith('power_t')]
                speed_t_columns = [col for col in wind_real_park.columns if col.startswith('wind_speed_t')]

                if power_t_columns and speed_t_columns:

                    wind_real_park['power'] = wind_real_park[power_t_columns].sum(axis=1)
                    wind_real_park['wind_speed_mean'] = wind_real_park[speed_t_columns].mean(axis=1)

                else:

                    if 'wind_speed' in wind_real_park.columns:
                        print("No 't' columns found. Renaming 'wind_speed' to 'wind_speed_mean'...")
                        wind_real_park.rename(columns={'wind_speed': 'wind_speed_mean'}, inplace=True)
                    else:
                        print("No suitable columns found for processing. Returning original DataFrame.")

            colw1, colw2 = st.columns([0.5,0.5])

            with colw1:
                wind_latitude = st.number_input("Park latitude", value=51.4, key="compare_wind_lat")

            with colw2:
                wind_longitude = st.number_input("Park longitude", value=10.4515, key="compare_wind_lon")

            ageing = st.toggle("Enable Ageing", key="compare_wind_ageing")
            config['wind_params']['apply_ageing'] = ageing

            if ageing:
                wind_commissioning_date = st.date_input("Commissioning Date", value="2013-01-01", format="YYYY-MM-DD", key="compare_wind_commissioning")

            turbines_number = st.number_input("Number of turbines in park", value=1, min_value=1, key="compare_wind_turbines_number")

            wind_turbine_list = []
            wind_height_list = []

            colw1, colw2 = st.columns([0.5,0.5])

            with colw1:
                selected_turbine = st.selectbox(label="Select a type of turbine", options=turbine_types[1:], key="compare_turbine0")
                wind_turbine_list.append(selected_turbine)

            with colw2:
                hub_height = st.number_input("Hub Height", value=10, key="compare_height0")
                wind_height_list.append(hub_height)

            for number in range(turbines_number-1):

                with colw1:
                    selected_turbine = st.selectbox(label="Select a type of turbine", label_visibility="hidden", options=turbine_types[1:], key=f"compare_turbine{number+1}")
                    wind_turbine_list.append(selected_turbine)

                with colw2:
                    hub_height = st.number_input(label="Hub Height", label_visibility="hidden", value=10, key=f"compare_height{number+1}")
                    wind_height_list.append(hub_height)

            config['wind_params']['turbines'] = wind_turbine_list
            config['wind_params']['hub_heights'] = wind_height_list

            write_config(config,config_path)

        else:
            st.caption("Choose Solar or Wind to see locations on the map.")

    if chosen_resource == "Solar":

            real_park = solar_real_park
            latitude, longitude = solar_latitude, solar_longitude
            my_location = (latitude, longitude)

            stations_list = get_station_list(pv_dir)
            stations_list = [int(station) for station in stations_list]

            solardata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            result = find_nearest_station_for_park(latitude,longitude,solardata)

            markers = {row['station_id']: [float(row['latitude']), float(row['longitude'])] for _, row in solardata.iterrows()}
            nearest_station = {result['park_id']: result}

            with map_col:
                st.info("Station marked in green is the nearest station to your location, use of this station is suggested.", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers,my_location,nearest_station, key="compare_solar_map")

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

                # Data ready check — rendered up in the floating panel (not below
                # the tall map) so the person doesn't need to scroll to find it.
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

                        st.session_state["compare_solar_results_snapshot"] = {
                            "config": config,
                            "selected_marker": st.session_state.selected_marker,
                            "pv_dir": pv_dir,
                            "latitude": latitude,
                            "longitude": longitude,
                            "elevation": elevation,
                            "real_park": real_park,
                        }
                        with compare_results_placeholder.container():
                            with st.container(key="compare_solar_data_ready_popup"):
                                st.success(f"✅ Data ready for station {st.session_state.selected_marker}.")
                                if st.button("View Generated Data", key="compare_solar_view_data_btn"):
                                    st.session_state["active_results_view"] = "compare_solar"
                                    st.rerun()

                        del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv
                    else:
                        with compare_results_placeholder.container():
                            st.warning(body="Upload the park file",icon="⚠️")

    if chosen_resource == "Wind":

            real_park = wind_real_park
            my_location = (wind_latitude, wind_longitude)
            commissioning_date = wind_commissioning_date
            latitude, longitude = my_location

            stations_list = get_station_list(wind_dir)
            stations_list = [int(station) for station in stations_list]

            winddata = masterdata[masterdata['station_id'].astype(int).isin(stations_list)]

            result = find_nearest_station_for_park(latitude,longitude,winddata)
            nearest_station = {result['park_id']: result}

            markers = {row['station_id']: [row['latitude'], row['longitude']] for _, row in winddata.iterrows()}

            with map_col:

                st.info("Station marked in green is the nearest station to your location, use of this station is suggested.", icon="ℹ️")
                with st.container(border=True):

                    map_data = create_map(markers,my_location,nearest_station, key="compare_wind_map")

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

                # Data ready check — rendered up in the floating panel (not below
                # the tall map) so the person doesn't need to scroll to find it.
                if st.session_state.selected_marker:
                    if real_park is not None and not real_park.empty:
                        if not isinstance(real_park.index, pd.DatetimeIndex):
                            try:
                                real_park.index = pd.to_datetime(real_park.index)
                            except Exception as e:
                                st.error(f"Error converting secondary DataFrame index to DatetimeIndex: {e}")

                        st.session_state["compare_wind_results_snapshot"] = {
                            "config": config,
                            "selected_marker": st.session_state.selected_marker,
                            "wind_dir": wind_dir,
                            "winddata": winddata,
                            "real_park": real_park,
                            "commissioning_date": commissioning_date,
                        }
                        with compare_results_placeholder.container():
                            with st.container(key="compare_wind_data_ready_popup"):
                                st.success(f"✅ Data ready for station {st.session_state.selected_marker}.")
                                if st.button("View Generated Data", key="compare_wind_view_data_btn"):
                                    st.session_state["active_results_view"] = "compare_wind"
                                    st.rerun()

                        del st.session_state.selected_marker #Clearing the session state to show no plots when switched from wind to pv
                    else:
                        with compare_results_placeholder.container():
                            st.warning(body="Upload the park file",icon="⚠️")

    if chosen_resource is None:
        # Nothing chosen yet: map spans the full page width, no markers.
        with map_col:
            create_map({}, key="compare_empty_map", zoom_start=6, center=[47.0, 6.0])


# --- Persistent footer (shown regardless of mode) ---
footer_left_col, footer_center_col, footer_right_col = st.columns([4, 2, 2])

with footer_right_col:
    st.caption("Funded by")
    st.image("./logos/logo.svg", width=150)

st.markdown("""
    <div class="fixed-footer">
        This work is done by Meghna Negi in the research project FederatedForecasts, funded by the German Federal Ministry of Research, Technology and Space [Grant 13FH587KX1]
    </div>
""", unsafe_allow_html=True)