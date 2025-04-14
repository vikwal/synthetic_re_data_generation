import logging
import time
import json5
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

log = logging.getLogger("windmodel")
log.setLevel(logging.INFO)

metadata_info = {
    "schema_name": "windmodel",
    "data_date": "2024-06-12",
    "data_source": "https://www.wind-turbine-models.com/powercurves",
    "license": "https://www.wind-turbine-models.com/terms",
    "description": "Wind turbine performance. Wind turbine test performance data by model.",
}

desired_ranges = [
    (0, 4), (4, 6), (6, 8), (8, 10), (10, 13),
    (13, 16), (16, 19), (19, 22), (22, 25)
]
default_output_filename = "turbine_power.csv"


def get_turbines_with_power_curve():
    """Fetches a list of turbine IDs with available power curves."""
    try:
        page = requests.get(metadata_info["data_source"])
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        name_list = soup.find(class_="chosen-select")
        wind_turbines_with_curve = []
        if name_list:
            for i in name_list.find_all("option"):
                value = i.get("value")
                if value:
                    wind_turbines_with_curve.append(value)
        else:
            log.error("Could not find turbine list (CSS class 'chosen-select').")
        return wind_turbines_with_curve
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching turbine list: {e}")
        return []


def _fetch_single_range_data(turbine_id, start, stop):
    """Helper: Fetches data for ONE turbine and ONE range. Returns raw DataFrame."""
    url = metadata_info["data_source"]
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"_action": "compare", "turbines[]": turbine_id, "windrange[]": [start, stop]}

    try:
        resp = requests.post(url, headers=headers, data=data)
        resp.raise_for_status()
        json_data = resp.json()

        if "result" not in json_data:
            log.warning(f"No 'result' in JSON for turbine {turbine_id}, range {start}-{stop}.")
            return pd.DataFrame()

        strings = json_data["result"]
        begin = strings.find("data:")
        end_marker = '"}]'
        end = strings.find(end_marker, begin)

        if begin == -1 or end == -1:
            log.warning(f"Could not find 'data:' or '{end_marker}' for turbine {turbine_id}, range {start}-{stop}.")
            return pd.DataFrame()

        relevant_js = "{" + strings[begin : end + len(end_marker)] + "}}"
        curve_as_dict = json5.loads(relevant_js)

        if not (
            "data" in curve_as_dict and "labels" in curve_as_dict["data"] and
            "datasets" in curve_as_dict["data"] and len(curve_as_dict["data"]["datasets"]) > 0 and
            "data" in curve_as_dict["data"]["datasets"][0] and "label" in curve_as_dict["data"]["datasets"][0]
        ):
            log.warning(f"Unexpected JSON structure for turbine {turbine_id}, range {start}-{stop}.")
            return pd.DataFrame()

        x = curve_as_dict["data"]["labels"]
        y = curve_as_dict["data"]["datasets"][0]["data"]
        label = curve_as_dict["data"]["datasets"][0]["label"]

        try:
            x_float = [float(speed) for speed in x]
        except ValueError:
            log.warning(f"Could not convert wind speeds to float for turbine {turbine_id}, range {start}-{stop}: {x}")
            return pd.DataFrame()

        df = pd.DataFrame(np.asarray(y, dtype=float),
                          index=pd.Index(x_float, name="wind_speed"),
                          columns=[label])
        return df

    except requests.exceptions.RequestException as e:
        log.error(f"Network error fetching turbine {turbine_id}, range {start}-{stop}: {e}")
    except json5.JSONDecodeError as e:
        log.error(f"JSON parsing error for turbine {turbine_id}, range {start}-{stop}: {e}")
    except Exception as e:
        log.error(f"Unexpected error in _fetch_single_range_data for {turbine_id}, range {start}-{stop}: {e}")
    return pd.DataFrame()


def download_turbine_curve(turbine_id, ranges):
    """
    Downloads, combines, and interpolates curve data for one turbine across multiple ranges.
    Zusätzlich wird anhand der rohen, gescrapten Daten ermittelt, ab welcher Windgeschwindigkeit interpoliert werden soll.
    """
    all_range_dfs = []
    for start, stop in ranges:
        df_range = _fetch_single_range_data(turbine_id, start, stop)
        if not df_range.empty:
            all_range_dfs.append(df_range)
        time.sleep(0.1) 

    if not all_range_dfs:
        log.warning(f"No data received for turbine {turbine_id} in specified ranges.")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_range_dfs, axis=0)
    combined_df = combined_df.sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    raw_df = combined_df.copy()

    try:
        combined_df.index = pd.to_numeric(combined_df.index)
        combined_df = combined_df.sort_index()
        order = 3
        non_nan_count = combined_df.iloc[:, 0].notna().sum()

        if non_nan_count > order:
            combined_df = combined_df.interpolate(method="polynomial", order=order, limit_direction='both', axis=0)
        elif non_nan_count > 1:
            log.warning(f"Using linear interpolation for turbine {turbine_id} due to insufficient points ({non_nan_count}).")
            combined_df = combined_df.interpolate(method="linear", limit_direction='both', axis=0)

        combined_df = combined_df.fillna(0)

    except Exception as e:
        log.error(f"Interpolation failed for turbine {turbine_id}: {e}. Filling NaNs with 0.")
        combined_df = combined_df.fillna(0)

    try:
        col = combined_df.columns[0]
        nonzero_raw = raw_df.loc[raw_df[col] > 0]
        if not nonzero_raw.empty:
            first_nonzero = nonzero_raw.index.min()
            zeros_below = raw_df.loc[(raw_df[col] == 0) & (raw_df.index < first_nonzero)]
            if not zeros_below.empty:
                threshold = zeros_below.index.max()
            else:
                threshold = first_nonzero - 1

            combined_df.loc[combined_df.index <= threshold, col] = 0

    except Exception as e:
        log.error(f"Error during threshold correction for turbine {turbine_id}: {e}")

    combined_df.index.name = "wind_speed"
    return combined_df


def download_all_turbines(ranges_to_download):
    """Downloads curves for all turbines across the specified ranges."""
    wind_turbines = get_turbines_with_power_curve()
    if not wind_turbines:
        log.error("No turbine IDs found for download.")
        return pd.DataFrame()

    curves = []
    log.info(f"Starting download for {len(wind_turbines)} turbines.")
    for turbine_id in tqdm(wind_turbines, desc="Processing turbines"):
        curve = download_turbine_curve(turbine_id, ranges=ranges_to_download)
        if not curve.empty:
            if curve.columns[0] is None or str(curve.columns[0]).strip() == "":
                curve.columns = [f"Turbine_{turbine_id}"]
                log.warning(f"Using fallback name for turbine {turbine_id} due to missing label.")
            curves.append(curve)

    if not curves:
        log.error("Could not download data for any turbine.")
        return pd.DataFrame()

    log.info("Combining data from all turbines...")
    df = pd.concat(curves, axis=1, join='outer')
    df = df.sort_index()
    df = df.fillna(0)
    df = df[df.any(axis=1)]  
    df[df < 0] = 0           
    log.info("Download and processing complete.")
    return df


def main(output_filename=default_output_filename):
    """Main function: downloads data using global ranges and saves to file."""
    log.info(f"Starting data download using ranges: {desired_ranges}")
    turbine_data = download_all_turbines(ranges_to_download=desired_ranges)

    if not turbine_data.empty:
        log.info(f"Saving data to {output_filename}")
        try:
            with open(output_filename, "w", encoding='utf-8') as f:
                turbine_data.to_csv(f, sep=";", decimal=".")
            log.info("Data saved successfully.")
        except IOError as e:
            log.error(f"Error writing CSV file: {e}")
    else:
        log.warning("No data available to save.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(output_filename=default_output_filename)
