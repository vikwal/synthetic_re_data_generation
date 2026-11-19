import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict
from geopy.distance import geodesic

from . import tools


def parse_csv_filename(filename: str) -> Tuple[float, float]:
    """
    Parse CSV filename to extract latitude and longitude.
    Format: XX_XXXX_YY_YYYY_ML.csv
    where XX = latitude integer part, XXXX = latitude decimal part (4 digits)
          YY = longitude integer part, YYYY = longitude decimal part (4 digits)

    :param filename: Name of the CSV file (without path)
    :return: Tuple of (latitude, longitude)
    """
    # Remove file extension
    base_name = filename.replace('_ML.csv', '').replace('_SL.csv', '')
    parts = base_name.split('_')

    if len(parts) == 4:
        lat_int = int(parts[0])
        lat_dec = int(parts[1])
        lon_int = int(parts[2])
        lon_dec = int(parts[3])

        # Reconstruct coordinates
        latitude = lat_int + lat_dec / 10000.0
        longitude = lon_int + lon_dec / 10000.0

        return latitude, longitude
    else:
        raise ValueError(f"Cannot parse filename: {filename}")


def find_closest_grid_points(csv_files: List[str],
                             park_lat: float,
                             park_lon: float,
                             n: int = 1) -> List[Tuple[str, float]]:
    """
    Find the n closest grid points to the park location.

    :param csv_files: List of CSV file paths
    :param park_lat: Park latitude
    :param park_lon: Park longitude
    :param n: Number of closest points to return
    :return: List of tuples (csv_file_path, distance_in_km)
    """
    distances = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            grid_lat, grid_lon = parse_csv_filename(filename)
            distance = geodesic((park_lat, park_lon), (grid_lat, grid_lon)).km
            distances.append((csv_file, distance))
        except ValueError as e:
            logging.warning(f"Skipping file {filename}: {e}")
            continue

    # Sort by distance and return the n closest points
    distances.sort(key=lambda x: x[1])
    return distances[:n]


def merge_nwp_forecasts(csv_files: List[str], park_id: str) -> pd.DataFrame:
    """
    Merge multiple NWP forecast CSV files into a single DataFrame.

    :param csv_files: List of CSV file paths to merge
    :param park_id: Park ID for logging purposes
    :return: Merged and sorted DataFrame
    """
    dataframes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Error reading {csv_file} for park {park_id}: {e}")
            continue

    if not dataframes:
        logging.warning(f"No data found for park {park_id}")
        return pd.DataFrame()

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Sort by starttime, forecasttime, toplevel
    if 'starttime' in merged_df.columns and 'forecasttime' in merged_df.columns:
        sort_columns = ['starttime', 'forecasttime']
        if 'toplevel' in merged_df.columns:
            sort_columns.append('toplevel')
        merged_df = merged_df.sort_values(by=sort_columns)
        merged_df = merged_df.reset_index(drop=True)

    return merged_df


def process_wind_nwp(config: dict, n_points: int = 1, from_date: str = None, to_date: str = None):
    """
    Process wind NWP data from CSV files.

    :param config: Configuration dictionary
    :param n_points: Number of closest grid points to consider
    :param from_date: Optional start date filter for starttime (format: 'YYYY-MM-DD')
    :param to_date: Optional end date filter for starttime (format: 'YYYY-MM-DD')
    """
    # Read wind parameters
    wind_param_file = os.path.join('data', 'wind_parameter.csv')

    if not os.path.exists(wind_param_file):
        logging.error(f"Wind parameter file not found: {wind_param_file}")
        return

    wind_params = pd.read_csv(wind_param_file, sep=';')
    logging.info(f"Loaded {len(wind_params)} wind parks from {wind_param_file}")

    # Base directory for wind NWP data
    base_dir = '/mnt/nas/icon-d2/csv/ML'
    forecast_hours = ['09']

    # Output directory
    output_dir = os.path.join('data', 'ML')
    os.makedirs(output_dir, exist_ok=True)

    # Process each park
    for idx, row in tqdm(wind_params.iterrows(), total=len(wind_params), desc='Processing wind parks'):
        park_id = str(row['park_id']).zfill(5)  # Ensure 5-digit format
        park_lat = row['latitude']
        park_lon = row['longitude']

        logging.info(f"Processing park {park_id} at ({park_lat}, {park_lon})")

        all_csv_files = []

        # Collect CSV files from all forecast hour directories
        for hour in forecast_hours:
            park_dir = os.path.join(base_dir, hour, park_id)

            if not os.path.exists(park_dir):
                logging.warning(f"Directory not found: {park_dir}")
                continue

            # Get all ML CSV files in this directory
            csv_pattern = os.path.join(park_dir, '*_ML.csv')
            csv_files = glob.glob(csv_pattern)

            if not csv_files:
                logging.warning(f"No CSV files found in {park_dir}")
                continue

            # Find closest grid points for this forecast hour
            closest_points = find_closest_grid_points(csv_files, park_lat, park_lon, n_points)

            # Add the closest CSV files to our collection
            for csv_file, distance in closest_points:
                all_csv_files.append(csv_file)
                logging.debug(f"Selected {os.path.basename(csv_file)} (distance: {distance:.2f} km)")

        # Merge all selected CSV files
        if all_csv_files:
            merged_df = merge_nwp_forecasts(all_csv_files, park_id)

            if not merged_df.empty:
                # Filter by date range if specified
                if from_date is not None or to_date is not None:
                    if 'starttime' in merged_df.columns:
                        merged_df['starttime'] = pd.to_datetime(merged_df['starttime'], utc=True)

                        if from_date is not None:
                            from_datetime = pd.to_datetime(from_date, utc=True)
                            merged_df = merged_df[merged_df['starttime'] >= from_datetime]
                            logging.debug(f"Filtered from {from_date}: {len(merged_df)} rows remaining")

                        if to_date is not None:
                            to_datetime = pd.to_datetime(to_date, utc=True)
                            merged_df = merged_df[merged_df['starttime'] <= to_datetime]
                            logging.debug(f"Filtered to {to_date}: {len(merged_df)} rows remaining")
                    else:
                        logging.warning(f"Column 'starttime' not found for date filtering")

                if not merged_df.empty:
                    output_file = os.path.join(output_dir, f'nwp_{park_id}.csv')
                    merged_df.to_csv(output_file, index=False)
                    logging.info(f"Saved NWP data for park {park_id} to {output_file} ({len(merged_df)} rows)")
                else:
                    logging.warning(f"No data remaining after date filtering for park {park_id}")
            else:
                logging.warning(f"No data to save for park {park_id}")
        else:
            logging.warning(f"No CSV files collected for park {park_id}")


def process_solar_nwp(config: dict, n_points: int = 1, from_date: str = None, to_date: str = None):
    """
    Process solar NWP data from CSV files.
    (To be implemented if needed)

    :param config: Configuration dictionary
    :param n_points: Number of closest grid points to consider
    :param from_date: Optional start date filter for starttime (format: 'YYYY-MM-DD')
    :param to_date: Optional end date filter for starttime (format: 'YYYY-MM-DD')
    """
    logging.info("Solar NWP processing not yet implemented")
    pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Load configuration
    config = tools.load_config("configs/config.yaml")

    # Check config for what to process
    get_wind_nwp = config.get('write', {}).get('get_wind_nwp', False)
    get_solar_nwp = config.get('write', {}).get('get_solar_nwp', False)
    n_points = config.get('nwp', {}).get('n_closest_points', 1)
    from_date = config.get('nwp', {}).get('from_date', None)
    to_date = config.get('nwp', {}).get('to_date', None)

    logging.info(f"Configuration:")
    logging.info(f"  - Get Wind NWP: {get_wind_nwp}")
    logging.info(f"  - Get Solar NWP: {get_solar_nwp}")
    logging.info(f"  - Number of closest points: {n_points}")
    logging.info(f"  - From date: {from_date if from_date else 'None (no filter)'}")
    logging.info(f"  - To date: {to_date if to_date else 'None (no filter)'}")

    if get_wind_nwp:
        logging.info("=" * 50)
        logging.info("Processing Wind NWP data")
        logging.info("=" * 50)
        process_wind_nwp(config, n_points=n_points, from_date=from_date, to_date=to_date)

    if get_solar_nwp:
        logging.info("=" * 50)
        logging.info("Processing Solar NWP data")
        logging.info("=" * 50)
        process_solar_nwp(config, n_points=n_points, from_date=from_date, to_date=to_date)

    if not get_wind_nwp and not get_solar_nwp:
        logging.warning("Neither wind nor solar NWP processing is enabled in config!")
        logging.warning("Set 'write.get_wind_nwp' or 'write.get_solar_nwp' to True in config.yaml")


if __name__ == '__main__':
    main()