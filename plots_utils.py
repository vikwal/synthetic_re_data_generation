import datetime
from streamlit_folium import st_folium
import streamlit as st
from pathlib import Path
import folium
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
from collections import defaultdict
from generate_pv import get_features,generate_pv_power
from generate_wind import read_dfs, get_turbines, gen_full_dataframe, get_ageing_degradation, get_park_params
from utils.clean_data import relevant_features

from geopy.distance import geodesic
import requests

def get_elevation(lat, lon):
    url = 'https://api.open-elevation.com/api/v1/lookup'
    params = {'locations': f'{lat},{lon}'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        elevation = response.json()['results'][0]['elevation']
        return elevation
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")

def find_nearest_station_for_park(park_lat, park_lon, all_wind_params):
    """
    Finds the nearest weather station for a single wind park.

    Args:
        park_lat (float): The latitude of the park.
        park_lon (float): The longitude of the park.
        all_wind_params (pd.DataFrame): A DataFrame of all available weather stations.

    Returns:
        dict: A dictionary with the park's data and information about the nearest station.
    """
    park_coords = (park_lat, park_lon)

    # Calculate distances to all stations
    distances = all_wind_params.apply(
        lambda row: geodesic(park_coords, (row['latitude'], row['longitude'])).km, axis=1)

    # Find the nearest station
    nearest_station = all_wind_params.loc[distances.idxmin()]

    # Get park elevation (uncomment to enable)
    # park_elevation = get_elevation(lat=park_lat, lon=park_lon)

    # Construct the result dictionary
    result = {
        'park_id': nearest_station['station_id'],
        'latitude': park_coords[0],
        'longitude': park_coords[1],
        'distance_km': distances.min(),
        'altitude': nearest_station['station_height'],
        # 'park_elevation': park_elevation
    }
    
    return result

def get_station_list(dir):

    validStations = []

    files = os.listdir(dir)
    for file in files:
        validStations.append((file.split('_')[1]).split('.')[0])

    return validStations

def process_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame index is a timezone-naive DatetimeIndex.
    """
    # Drop any unnamed columns that may have been created during CSV export
    unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols, inplace=False)

    # Convert the index to datetime objects if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.error(f"Error converting DataFrame index to DatetimeIndex: {e}")
            return df # Return the unprocessed dataframe on failure

    # Ensure the index is timezone-naive for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df

def create_map(markers, my_location = None, nearest_station = {}):

    # Create a map centered around a specific location
    m = folium.Map(location=[51.4, 10.4515], zoom_start=7)

    # Add markers to the map
    for name, location in markers.items():

        if name in nearest_station:
            color = "green"
            popup_text = f"{name} Distance: {nearest_station[name]['distance_km']:.2f} km"
        else:
            color = "darkpurple"
            popup_text = name

        folium.Marker(location, popup=popup_text, icon=folium.Icon(color=color)).add_to(m)

    if my_location:

        folium.Marker(location=my_location,popup="My Plant",icon=folium.Icon(color='red', icon='glyphicon-user')).add_to(m)


    map_data = st_folium(m, use_container_width=True, height=1200)

    return map_data

def plot_power_and_features_pv_streamlit(day: str, 
                            plot_names: list,
                            features: list,
                            power: pd.Series,
                            ##feature_name: str,
                            synchronize_axes=True,
                            ):
    
    day = pd.Timestamp(day)
    tz = power.index.tz
    if tz is not None:
        day = day.tz_localize(tz)
    index_0 = power.index.get_loc(day)
    index_1 = power.index.get_loc(day + pd.Timedelta(days=1))
    date = str(features[0].index[index_0:index_1][0].date())

    fig, ax1 = plt.subplots(figsize=(8, 4))

    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 6}  
    color = '#999999'
    line_colors = ['#3772ff','#df2935','#fdca40']  
    fontsize = 8
    lines = []
    title_suffix = ''
    
    # plot power
    line1, = ax1.plot(
    power[index_0:index_1],
    label="Power Output (W)",
    color="#080708",
    linewidth=1.0
    )
    lines.append(line1)

    # configure secondary y-axis
    ax1.set_xlabel("Time", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize-2) 
    ax2 = ax1.twinx()

    color_index = 0

    # plot irradiance components
    for name, series in zip(plot_names, features):

        line, = ax2.plot(
            series[index_0:index_1],
            label=f"{name} (W/m$^2$)",
            linestyle='-',
            color=line_colors[color_index],
            linewidth=1.0
        )
        lines.append(line)
        color_index += 1

    # configure primary y-axis
    ax2.set_ylabel("Energy flux density (W/m$^2$)", fontsize=fontsize, color=color, family='DejaVu Sans')
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

    # Synchronize y-axes
    if synchronize_axes:
        title_suffix = '(synched axes)'
        all_ghi_min = min([series[index_0:index_1].min() for series in features])
        all_ghi_max = max([series[index_0:index_1].max() for series in features])
        y_min = min(all_ghi_min, power[index_0:index_1].min())
        y_max = max(all_ghi_max, power[index_0:index_1].max())
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

    # legend
    lines.append(lines.pop(0))
    labels = [line.get_label() for line in lines]
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

    plt.title(f"Irradiance and Power Output on {date} {title_suffix}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    fig.tight_layout()

    return fig

def plot_quarterly_boxplot(data_for_year: pd.DataFrame, selected_year: int, real_data: pd.DataFrame = None):

    def get_quarter(month):
        if month in [1, 2, 3]:
            return "Jan - Mar"
        elif month in [4, 5, 6]:
            return "Apr - Jun"
        elif month in [7, 8, 9]:
            return "Jul - Sep"
        else:
            return "Oct - Dec"

    # Define the canonical order for the quarters
    quarter_order = ["Jan - Mar", "Apr - Jun", "Jul - Sep", "Oct - Dec"]

    # Check if a second dataset is provided for comparison
    if real_data is not None and not real_data.empty:
        # Filter both datasets for the selected year
        data_for_year = data_for_year[data_for_year.index.year == selected_year].copy()
        real_data = real_data[real_data.index.year == selected_year].copy()

        # Add a source column to differentiate the data
        data_for_year['source'] = 'Synthetic'
        real_data['source'] = 'Real'

        # Combine the dataframes
        combined_df = pd.concat([real_data,data_for_year])
        combined_df['quarter'] = combined_df.index.month.map(get_quarter)
        combined_df['quarter'] = pd.Categorical(combined_df['quarter'], categories=quarter_order, ordered=True)

        # Create the boxplot with a 'hue' for the source
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=combined_df, x='quarter', y='power', hue='source', ax=ax, palette="pastel", showfliers=False)
        ax.set_title(f"Quaterly Power Distribution for {selected_year} (Real vs. Synthetic)", fontsize=16)
        #ax.legend(title="Data Series")
        if ax.get_legend():
            ax.get_legend().set_title("Data Series")

    else:
        # Fallback to the original behavior if no real data is provided
        data_for_year = data_for_year[data_for_year.index.year == selected_year].copy()
        data_for_year['quarter'] = data_for_year.index.month.map(get_quarter)
        data_for_year['quarter'] = pd.Categorical(data_for_year['quarter'], categories=quarter_order, ordered=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data_for_year, x='quarter', y='power', ax=ax, palette="viridis", showfliers=False)
        ax.set_title(f"Quaterly Power Distribution for {selected_year}", fontsize=16)

    # Common plot settings
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig


def plot_power_histogram(data_for_year: pd.DataFrame, selected_year: int, real_data: pd.DataFrame = None):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    if real_data is not None and not real_data.empty:
        # Filter both dataframes for the selected year
        synthetic_data = data_for_year[data_for_year.index.year == selected_year].copy()
        real_data_filtered = real_data[real_data.index.year == selected_year].copy()

        synthetic_data['source'] = 'Synthetic'
        real_data_filtered['source'] = 'Real'

        # Concatenate the dataframes for plotting with Seaborn's 'hue' parameter
        combined_df = pd.concat([ real_data_filtered,synthetic_data])

        sns.histplot(
            data=combined_df,
            x='power',
            hue='source',
            ax=ax,
            kde=False,
            palette="pastel",
            stat='density',
            multiple='layer',
            alpha=0.6         
        )
        ax.set_title(f"Power Distribution for {selected_year} (Real vs. Synthetic)", fontsize=16)
        #ax.legend(title="Data Series")
        if ax.get_legend():
            ax.get_legend().set_title("Data Series")
    else:
        
        data_for_year = data_for_year[data_for_year.index.year == selected_year].copy()
        sns.histplot(
            data=data_for_year,
            x='power',
            ax=ax,
            kde=False,
            color='skyblue',
            stat='density'
        )
        ax.set_title(f"Power Histogram for {selected_year}", fontsize=16)

    max_power = data_for_year['power'].max()
    step = 500
    new_ticks = list(range(0, int(max_power) + step, step))
    ax.set_xticks(new_ticks)
    ax.set_xticklabels([f'{tick}' for tick in new_ticks])
    ax.set_xlabel("Power (W)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=10))
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    return fig


def plot_multi_year_power_production(data_for_all_years: pd.DataFrame, real_data: pd.DataFrame = None):

    fig, ax = plt.subplots(figsize=(10, 6))

    if real_data is not None and not real_data.empty:
        # Calculate yearly power sum for synthetic data
        synthetic_yearly_power_sum = data_for_all_years.resample('Y').sum(numeric_only=True)['power']
        
        # Calculate yearly power sum for real data
        real_yearly_power_sum = real_data.resample('Y').sum(numeric_only=True)['power']
        
        plot_data = pd.DataFrame({
            'Year': synthetic_yearly_power_sum.index.year,
            'Total Power (W)': synthetic_yearly_power_sum.values,
            'Data Series': 'Synthetic'
        })
        
        real_plot_data = pd.DataFrame({
            'Year': real_yearly_power_sum.index.year,
            'Total Power (W)': real_yearly_power_sum.values,
            'Data Series': 'Real'
        })

        # Concatenate the dataframes
        combined_plot_data = pd.concat([real_plot_data,plot_data])

        # Create the bar plot using Seaborn with the 'hue' parameter
        sns.barplot(data=combined_plot_data, x='Year', y='Total Power (W)', 
                    hue='Data Series', ax=ax, palette="coolwarm")
        
        ax.set_title("Total Power Production per Year (Real vs. Synthetic)", fontsize=16)
        
    else:

        yearly_power_sum = data_for_all_years.resample('Y').sum(numeric_only=True)['power']
        plot_data = pd.DataFrame({
            'Year': yearly_power_sum.index.year,
            'Total Power (W)': yearly_power_sum.values
        })
        sns.barplot(data=plot_data, x='Year', y='Total Power (W)', ax=ax, palette="coolwarm")
        ax.set_title("Total Power Production per Year", fontsize=16)


    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Total Power (W)", fontsize=12)
    ax.grid(True, axis='y', linestyle='-', alpha=0.7)
    
    ax.set_xticks(range(len(plot_data['Year'])))
    ax.set_xticklabels(plot_data['Year'], rotation=45)
    
    def y_formatter(x, pos):
        return f'{int(x):,}'
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(y_formatter))
    
    fig.tight_layout()

    return fig


def plot_daily_power_global_irradiance(power_series: pd.Series, ghi_series: pd.Series, selected_date: str):
    day = pd.Timestamp(selected_date)
    # Assume both series have the same timezone info as their index
    tz = power_series.index.tz
    if tz is not None:
        day = day.tz_localize(tz)
    
    start_of_day = day.floor('D')
    end_of_day = start_of_day + pd.Timedelta(days=1)
    
    # Slice the series for the selected day
    daily_power_data = power_series.loc[start_of_day:end_of_day].iloc[:-1]
    daily_ghi_data = ghi_series.loc[start_of_day:end_of_day].iloc[:-1]
    
    if daily_power_data.empty or daily_ghi_data.empty:
        st.warning(f"No Power or GHI data available for {selected_date}.")
        return None

    date_str = str(daily_power_data.index[0].date())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 8}  
    color = '#999999'
    fontsize = 8
    
    # Plot Power on ax1
    line1, = ax1.plot(
        daily_power_data.index, daily_power_data, # Plotting the series directly
        label="Power Output (W)",
        color="#080708",
        linewidth=1.5
    )

    ax1.set_xlabel("Time", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize) 

    # Create twinx for GHI (Total)
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        daily_ghi_data.index, daily_ghi_data, # Plotting the series directly
        label="Global Horizontal Irradiance (W/m$^2$)",
        color="#df2935", # A distinct color for GHI
        linewidth=1.5,
        linestyle='--' # Different linestyle to distinguish
    )
    ax2.set_ylabel("Global Horizontal Irradiance (W/m$^2$)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax2.tick_params(axis='y', labelsize=fontsize)

    # --- Synchronize Y-axes ---
    # Find the overall min and max across both series
    all_values = pd.concat([daily_power_data, daily_ghi_data])
    y_min = all_values.min() * 0.9 # Add a little padding
    y_max = all_values.max() * 1.1 # Add a little padding
    
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    # --- End Synchronize Y-axes ---

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    
    # Hide spines for a cleaner look
    for ax_obj in [ax1, ax2]:
        for spine in ax_obj.spines.values():
            spine.set_visible(False)
            
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc='lower center', # Adjusted legend location
        bbox_to_anchor=(0.5, -0.25), # Adjust position
        ncol=2,
        frameon=False,  
        fontsize=8,
        labelcolor=color, 
        edgecolor='none'
    )
    ax1.grid(True, axis='y', color='gray', linewidth=0.5)
     
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)

    plt.title(f"Globak Horizontal Irradiance and Power Output {date_str} (Synced Axes)", 
              fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    fig.tight_layout()
    return fig

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
    series = data[feature_specs['name']]
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

def plot_hourly_real_vs_synth_boxplots(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, date_range: tuple):
    """
    Generates a side-by-side box plot for two dataframes for a given date range,
    grouped by hour of the day.
    """
    start_date = pd.to_datetime(date_range[0]).tz_localize(None)
    end_date = pd.to_datetime(date_range[1]).tz_localize(None)
    
    # Filter data for the specified date range
    df1_filtered = df1[(df1.index.normalize() >= start_date) & (df1.index.normalize() <= end_date)].copy()
    df2_filtered = df2[(df2.index.normalize() >= start_date) & (df2.index.normalize() <= end_date)].copy()

    # Check for empty dataframes
    if df1_filtered.empty or df2_filtered.empty:
        st.warning(f"No data found in the specified date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
        return None

    # Add a 'hour' column to both dataframes
    df1_filtered['hour'] = df1_filtered.index.hour
    df2_filtered['hour'] = df2_filtered.index.hour

    # Combine the dataframes into a single one for plotting
    df1_filtered['series'] = label1
    df2_filtered['series'] = label2
    combined_df = pd.concat([df1_filtered, df2_filtered])

    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        data=combined_df,
        x='hour',
        y='power', # Changed from 'power' to 'Power' to match existing code
        hue='series',
        ax=ax,
        palette="pastel"
    )

    ax.set_title(f"Hourly Power Distribution for {label1} and {label2} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                 fontsize=14)
    ax.set_xlabel("Hour of the Day", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.legend(title="Data Series")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_daily_power_comparison_for_day(synthetic_data: pd.DataFrame, real_data: pd.DataFrame, selected_date: str):
    
    if real_data.empty:
        st.warning("Real data is not available for comparison.")
        return

    # Filter data for the selected day
    synthetic_for_day = synthetic_data.loc[selected_date]
    real_for_day = real_data.loc[selected_date]
    
    if synthetic_for_day.empty or real_for_day.empty:
        st.warning(f"No data available for the selected date: {selected_date}.")
        return

    # Combine data into a single DataFrame for plotting
    synthetic_for_day['Data Series'] = 'Synthetic'
    real_for_day['Data Series'] = 'Real'
    combined_data = pd.concat([real_for_day,synthetic_for_day])
    combined_data['Hour'] = combined_data.index.hour
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=combined_data, x='Hour', y='power', hue='Data Series', ax=ax)
    
    ax.set_title(f"Power Output on {selected_date} (Real vs. Synthetic)", fontsize=16)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.grid(True, axis='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_wind_speed_vs_power_scatter( real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    
    if real_data is None or real_data.empty or 'wind_speed_mean' not in real_data.columns or 'power' not in real_data.columns:
        st.warning("Real data is missing 'wind_speed_mean' or 'power' column for comparison.")
        return
    
    if 'wind_speed_mean' not in synthetic_data.columns or 'power' not in synthetic_data.columns:
        st.warning("Synthetic data is missing 'wind_speed_mean' or 'power' column.")
        return
        
    # Prepare data for plotting
    synthetic_data_plot = synthetic_data[['wind_speed_mean', 'power']].copy()
    real_data_plot = real_data[['wind_speed_mean', 'power']].copy()
    
    synthetic_data_plot['Data Series'] = 'Synthetic'
    real_data_plot['Data Series'] = 'Real'
    
    combined_data = pd.concat([real_data_plot,synthetic_data_plot])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the scatter plot using seaborn
    sns.scatterplot(data=combined_data, x='wind_speed_mean', y='power', hue='Data Series', ax=ax, alpha=0.5, s=20)
    
    ax.set_title("Wind Speed vs. power Output", fontsize=16)
    ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("power (W)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_monthly_power_comparison(synthetic_data: pd.DataFrame, real_data: pd.DataFrame, selected_year: int):
    
    if real_data is None or real_data.empty:
        st.warning("Real data is not available for comparison.")
        return

    # Filter data for the selected year
    synthetic_data_for_year = synthetic_data[synthetic_data.index.year == selected_year]
    real_data_for_year = real_data[real_data.index.year == selected_year]

    if synthetic_data_for_year.empty or real_data_for_year.empty:
        st.warning(f"No data available for the selected year: {selected_year}.")
        return

    # Resample to monthly frequency and sum the power
    synthetic_monthly = synthetic_data_for_year['power'].resample('M').sum().reset_index()
    real_monthly = real_data_for_year['power'].resample('M').sum().reset_index()
    
    synthetic_monthly['Data Series'] = 'Synthetic'
    real_monthly['Data Series'] = 'Real'
    
    combined_df = pd.concat([real_monthly,synthetic_monthly])
    combined_df['Month'] = combined_df['timestamp'].dt.strftime('%b')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(data=combined_df, x='Month', y='power', hue='Data Series', ax=ax, palette="coolwarm")
    
    ax.set_title(f"Monthly Total Power Production Comparison ({selected_year})", fontsize=16)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Power (W)", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def show_pv_plot(config,marker_name,dir,latitude,longitude,elevation,park_data):

    dataSelectedStation = pd.DataFrame

    params = config['pv_params']
    features = config['features']

    params['latitude'] = latitude
    params['longitude'] = longitude
    params['altitude'] = elevation

    fileName = f'Station_{str(marker_name)}.csv'
    new_dir = os.path.join(dir, fileName)

    if(os.path.exists(new_dir)):

        dataSelectedStation = pd.read_csv(new_dir,sep=',')

        dataSelectedStation['timestamp'] = pd.to_datetime(dataSelectedStation['timestamp'])
        dataSelectedStation.set_index('timestamp', inplace=True)

        total_irradiance, cell_temperature = get_features(data=dataSelectedStation,
                                                          features=features,
                                                          params=params)
        
        total = total_irradiance['poa_global']
        direct = total_irradiance['poa_direct']
        diffuse = total_irradiance['poa_diffuse']

        power = generate_pv_power(total_irradiance=total,
                                cell_temperature=cell_temperature,
                                params=params)
        
        dataSelectedStation['power'] = power
        dataSelectedStation['Total'] = total
        dataSelectedStation['Direct'] = direct
        dataSelectedStation['Diffuse'] = diffuse

        dataSelectedStation = dataSelectedStation.resample('H').mean()

        dates_arr = dataSelectedStation.index.strftime('%Y-%m-%d')
        formatted_dates = np.unique(dates_arr)

        if not isinstance(dataSelectedStation.index, pd.DatetimeIndex):
            try:
                dataSelectedStation.index = pd.to_datetime(dataSelectedStation.index)
            except Exception as e:
                st.error(f"Error converting primary DataFrame index to DatetimeIndex: {e}")
                return
            
        dates_list = formatted_dates.tolist()
        years_list = dataSelectedStation.index.year.unique().tolist()

        if not park_data.empty:
            if 'timestamp' in park_data.columns:
                park_data = park_data.set_index('timestamp')

            park_data = process_dataframe_index(park_data)
            if not isinstance(park_data.index, pd.DatetimeIndex):
                try:
                    park_data.index = pd.to_datetime(park_data.index)
                except Exception as e:
                    st.error(f"Error converting secondary DataFrame index to DatetimeIndex: {e}")
                    return
            
            years_list_real = park_data.index.year.unique().tolist()
            dates_list_real = park_data.index.strftime('%Y-%m-%d').unique().tolist()

            years_list = sorted(list(set(years_list).intersection(years_list_real)))
            dates_list = sorted(list(set(dates_list).intersection(dates_list_real)))

            if dataSelectedStation.index.tz is not None:
                dataSelectedStation.index = dataSelectedStation.index.tz_localize(None)

            if park_data.index.tz is not None:
                park_data.index = park_data.index.tz_localize(None)

            with st.container(border=True):
                col1,col2 = st.columns([0.5,0.5])
                with col1:
                    start_date = st.selectbox("Select a start date",dates_list)
                try:
                    start_date_idx = dates_list.index(start_date)
                except ValueError:
                    start_date_idx = 0 

                end_dates_list = dates_list[start_date_idx:]
                
                with col2:
                    end_date = st.selectbox("Select a end date", end_dates_list)

                date_range = (start_date,end_date)

                fig = plot_hourly_real_vs_synth_boxplots(park_data,dataSelectedStation,'Real','Synthetic',date_range)
                st.pyplot(fig=fig,use_container_width=True)

            
            selected_year = st.selectbox("Select a year",years_list)
            data_for_year = dataSelectedStation[dataSelectedStation.index.year == selected_year].copy()

            with st.container(border=True):

                fig_boxplot = plot_quarterly_boxplot(data_for_year=data_for_year,selected_year=selected_year,real_data=park_data)
                st.pyplot(fig=fig_boxplot,use_container_width=True)

                fig_hist = plot_power_histogram(data_for_year=data_for_year,selected_year=selected_year,real_data=park_data)
                st.pyplot(fig=fig_hist,use_container_width=True)

                fig_yearly = plot_monthly_power_comparison(synthetic_data=dataSelectedStation,real_data=park_data,selected_year=selected_year)
                st.pyplot(fig=fig_yearly,use_container_width=True)

            with st.container(border=True):

                fig_total = plot_multi_year_power_production(data_for_all_years=dataSelectedStation,real_data=park_data)
                st.pyplot(fig=fig_total,use_container_width=True)

            with st.container(border=True):
                st.dataframe(dataSelectedStation)

        else:

            selected_date = st.selectbox("Select a date",dates_list)
            selected_feature = st.multiselect(label="Select features",options=["Total","Direct","Diffuse"],default="Total")
            features = []
            ghi = dataSelectedStation["ghi"]

            with st.container(border=True):
                for feature in selected_feature:
                    features.append(dataSelectedStation[feature])

                if (len(selected_feature) != 0):

                    fig_feature = plot_power_and_features_pv_streamlit(day=selected_date,
                                        plot_names=selected_feature,
                                        features=features,
                                        power=dataSelectedStation['power'])
            
                    st.pyplot(fig=fig_feature,use_container_width=True)

                fig_global = plot_daily_power_global_irradiance(power_series=dataSelectedStation['power'],
                                                                ghi_series=ghi,
                                                                selected_date=selected_date)
                st.pyplot(fig=fig_global,use_container_width=True)
            
            selected_year = st.selectbox("Select a year",years_list)
            data_for_year = dataSelectedStation[dataSelectedStation.index.year == selected_year].copy()

            with st.container(border=True):

                fig_boxplot = plot_quarterly_boxplot(data_for_year=data_for_year,selected_year=selected_year)
                st.pyplot(fig=fig_boxplot,use_container_width=True)

                fig_hist = plot_power_histogram(data_for_year=data_for_year,selected_year=selected_year)
                st.pyplot(fig=fig_hist,use_container_width=True)

            with st.container(border=True):

                fig_total = plot_multi_year_power_production(data_for_all_years=dataSelectedStation)
                st.pyplot(fig=fig_total,use_container_width=True)

            with st.container(border=True):
                st.dataframe(dataSelectedStation)

def show_wind_plot(config: dict,
                   marker_name: str,
                   dir: str,
                   masterdata: pd.DataFrame,
                   park_data: pd.DataFrame,
                   commissioning_date: datetime.date = None):

    dataSelectedStation = pd.DataFrame
    turbine_dir = Path.cwd().parent / config['data']['turbine_dir']
    turbine_power = config['data']['turbine_power']
    turbine_path = os.path.join(turbine_dir, turbine_power)
    specs_path = config['data']['turbine_specs']
    specs_path = os.path.join(turbine_dir, specs_path)
    cp_path = config["data"]["turbine_cp"]
    cp_path = os.path.join(turbine_dir, cp_path)
    wind_ages_path = config['data']['wind_ages'] #Get the data
    wind_ages = np.load(wind_ages_path)
    w_vert_dir = config['data']['w_vert_dir'] #Get the data
    features = config['features']
    params = config['wind_params']

    commissioning_date = commissioning_date

    #fileName = f'Station_{str(marker_name)}.pkl'
    fileName = f'Station_{str(marker_name)}.csv'
    new_dir = os.path.join(dir, fileName)

    if(os.path.exists(new_dir)):

        _, wind_features = relevant_features(features=features)

        #dataSelectedStation = pd.read_pickle(new_dir)
        frames, station_ids = read_dfs(path=dir,
                                      w_vert_dir=w_vert_dir,
                                      features=wind_features,
                                      hourly_resolution=params['hourly_resolution'],
                                      specific_id=marker_name)
        
        power_curves, cp_curves, specs = get_turbines(turbine_path=turbine_path,
                                                    cp_path=cp_path,
                                                    specs_path=specs_path,
                                                    params=params)

        power_master = {}
        turbine_master = defaultdict(dict)
        for station_id, frame in tqdm(zip(station_ids, frames), desc="Processing stations"):

            df = frame.copy()
            degradation_vector, commissioning_date = get_ageing_degradation(time_vector=df.index,
                                                                            real_ages=wind_ages, commissioning_date=commissioning_date)

            if not params['apply_ageing']:
                degradation_vector = None
                commissioning_date = '-'
                print("commissioning_date",commissioning_date)
            specific_params = get_park_params(station_id=station_id,
                                            masterdata=masterdata,
                                            params=params,
                                            commissioning_date=commissioning_date)
            power_master[station_id] = specific_params
            for turbine_id, turbine in enumerate(params['turbines'], start=1):
                hub_height = params['hub_heights'][turbine_id-1]
                dataSelectedStation = gen_full_dataframe(
                        power_curves=power_curves,
                        turbine=turbine,
                        params=params,
                        features=features,
                        df=df,
                        hub_height=hub_height,
                        rated_power=None, # only needed when to curtail rated power
                        specs=specs,
                        degradation_vector=degradation_vector,
                        suffix_for_turbine_cols=f'_t{turbine_id}'
                )
                turbine_master[f't{turbine_id}']['diameter'] = specs[turbine]['diameter']
                turbine_master[f't{turbine_id}']['hub_height'] = hub_height
                turbine_master[f't{turbine_id}']['cut_in'] = specs[turbine]['cut_in']
                turbine_master[f't{turbine_id}']['cut_out'] = specs[turbine]['cut_out']
                turbine_master[f't{turbine_id}']['rated'] = specs[turbine]['rated']
                turbine_master[f't{turbine_id}']['turbine_name'] = turbine
                turbine_master[f't{turbine_id}']['park_id'] = station_id

        power_t_columns = [col for col in dataSelectedStation.columns if col.startswith('power_t')]
        speed_t_columns = [col for col in dataSelectedStation.columns if col.startswith('wind_speed_t')]
        dataSelectedStation['power'] = dataSelectedStation[power_t_columns].sum(axis=1)
        dataSelectedStation['wind_speed_mean'] = dataSelectedStation[speed_t_columns].mean(axis=1)

        dates_arr = dataSelectedStation.index.strftime('%Y-%m-%d')
        formatted_dates = np.unique(dates_arr)

        if not isinstance(dataSelectedStation.index, pd.DatetimeIndex):
            try:
                dataSelectedStation.index = pd.to_datetime(dataSelectedStation.index)
            except Exception as e:
                st.error(f"Error converting primary DataFrame index to DatetimeIndex: {e}")
                return

        dates_list = formatted_dates.tolist()
        years_list = dataSelectedStation.index.year.unique().tolist()

        if not park_data.empty:
            if 'timestamp' in park_data.columns:
                park_data = park_data.set_index('timestamp')

            park_data = process_dataframe_index(park_data)
            if not isinstance(park_data.index, pd.DatetimeIndex):
                try:
                    park_data.index = pd.to_datetime(park_data.index)
                except Exception as e:
                    st.error(f"Error converting secondary DataFrame index to DatetimeIndex: {e}")
                    return
            
            years_list_real = park_data.index.year.unique().tolist()
            dates_list_real = park_data.index.strftime('%Y-%m-%d').unique().tolist()

            years_list = sorted(list(set(years_list).intersection(years_list_real)))
            dates_list = sorted(list(set(dates_list).intersection(dates_list_real)))

            if dataSelectedStation.index.tz is not None:
                dataSelectedStation.index = dataSelectedStation.index.tz_localize(None)

            if park_data.index.tz is not None:
                park_data.index = park_data.index.tz_localize(None)

            with st.container(border=True):

                selected_date = st.selectbox("Select a date",dates_list)

                fig = plot_daily_power_comparison_for_day(real_data=park_data,synthetic_data=dataSelectedStation,selected_date=selected_date)
                st.pyplot(fig=fig,use_container_width=True) 
            
            selected_year = st.selectbox("Select a year",years_list)
            data_for_year = dataSelectedStation[dataSelectedStation.index.year == selected_year].copy()

            with st.container(border=True):

                fig_boxplot = plot_quarterly_boxplot(data_for_year=data_for_year,selected_year=selected_year,real_data=park_data)
                st.pyplot(fig=fig_boxplot,use_container_width=True)

                fig_hist = plot_power_histogram(data_for_year=data_for_year,selected_year=selected_year,real_data=park_data)
                st.pyplot(fig=fig_hist,use_container_width=True)

            with st.container(border=True):

                fig_wind_power = plot_wind_speed_vs_power_scatter(real_data=park_data,synthetic_data=dataSelectedStation)
                st.pyplot(fig=fig_wind_power,use_container_width=True) ##How to use windspeed

                fig_total = plot_multi_year_power_production(data_for_all_years=dataSelectedStation,real_data=park_data)
                st.pyplot(fig=fig_total,use_container_width=True)
            
            with st.container(border=True):
                st.dataframe(dataSelectedStation)

        else:

            with st.container(border=True):

                selected_date = st.selectbox("Select a date",dates_list)
                selected_feature = st.selectbox(label="Select a feature",options=["wind_speed_hub","density_hub"])
            
                fig = plot_power_and_feature_wind_streamlit(data=dataSelectedStation,
                                    params=config['features'],
                                    day=selected_date,
                                    feature=selected_feature,
                                    power=dataSelectedStation['power']) 

                st.pyplot(fig=fig,use_container_width=True)

            selected_year = st.selectbox("Select a year",years_list)
            data_for_year = dataSelectedStation[dataSelectedStation.index.year == selected_year].copy()

            with st.container(border=True):

                fig_boxplot = plot_quarterly_boxplot(data_for_year=data_for_year,selected_year=selected_year)
                st.pyplot(fig=fig_boxplot,use_container_width=True)

                fig_hist = plot_power_histogram(data_for_year=data_for_year,selected_year=selected_year)
                st.pyplot(fig=fig_hist,use_container_width=True)

            with st.container(border=True):

                fig_total = plot_multi_year_power_production(data_for_all_years=dataSelectedStation)
                st.pyplot(fig=fig_total,use_container_width=True)

            with st.container(border=True):
                st.dataframe(dataSelectedStation)