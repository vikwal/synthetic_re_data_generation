import os
import pygrib
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import time
import re
import glob
from collections import defaultdict

def find_grib_files(grib_dir, pattern="era5_*_*.grib"):
    """
    Find all GRIB files matching the monthly pattern era5_MM_YYYY.grib
    
    Args:
        grib_dir: Directory containing GRIB files
        pattern: File pattern to match (default: era5_*_*.grib)
    
    Returns:
        list: Sorted list of GRIB file paths
    """
    grib_files = []
    search_pattern = os.path.join(grib_dir, pattern)
    
    for file_path in glob.glob(search_pattern):
        filename = os.path.basename(file_path)
        # Extract year and month from filename like era5_01_2023.grib
        match = re.match(r'era5_(\d{2})_(\d{4})\.grib', filename)
        if match:
            month, year = match.groups()
            grib_files.append({
                'path': file_path,
                'year': int(year),
                'month': int(month),
                'filename': filename
            })
    
    # Sort by year, then month
    grib_files.sort(key=lambda x: (x['year'], x['month']))
    
    print(f"Found {len(grib_files)} GRIB files:")
    for gf in grib_files:
        print(f"  {gf['filename']} ({gf['year']}-{gf['month']:02d})")
    
    return grib_files

def extract_multiple_grib_to_parquet_by_grid(grib_dir, output_dir="data/parquet/grid_points", germany_bounds=None):
    """
    Extract multiple monthly GRIB files and create one parquet file per unique grid point
    
    Args:
        grib_dir: Directory containing era5_MM_YYYY.grib files
        output_dir: Directory to save individual grid point parquet files
        germany_bounds: (min_lat, max_lat, min_lon, max_lon) to filter for Germany
    
    Returns:
        dict: {grid_key: parquet_file_path} mapping
    """
    
    # Variable mapping
    var_mapping = {
        '10 metre U wind component': 'u_wind_10m',
        '10 metre V wind component': 'v_wind_10m',
        '100 metre U wind component': 'u_wind_100m',
        '100 metre V wind component': 'v_wind_100m',
        'Instantaneous 10 metre wind gust': 'wind_gust_10m',
        'Friction velocity': 'friction_wind',
        '2 metre temperature': 'temp_2m',
        'Surface pressure': 'pressure',
        '2 metre dewpoint temperature': 'dew_point_2m'
    }
    
    # Default Germany bounds
    if germany_bounds is None:
        germany_bounds = (47.0, 55.5, 5.5, 15.5)  # (min_lat, max_lat, min_lon, max_lon)
    
    min_lat, max_lat, min_lon, max_lon = germany_bounds
    
    # Find all GRIB files
    grib_files = find_grib_files(grib_dir)
    
    if not grib_files:
        print(f"No GRIB files found in {grib_dir} matching pattern era5_*_*.grib")
        return {}
    
    print(f"\nExtracting data from {len(grib_files)} GRIB files")
    print(f"Germany bounds: lat {min_lat}-{max_lat}, lon {min_lon}-{max_lon}")
    print("Creating separate parquet files for each grid point...")
    
    # Dictionary to store data by grid point: {(lat, lon): {timestamp: {variables}}}
    grid_data = defaultdict(lambda: defaultdict(dict))
    
    start_time = time.time()
    total_messages = 0
    
    # Process each GRIB file
    for file_idx, grib_info in enumerate(grib_files, 1):
        grib_file_path = grib_info['path']
        filename = grib_info['filename']
        
        print(f"\n[{file_idx}/{len(grib_files)}] Processing: {filename}")
        
        try:
            grbs = pygrib.open(grib_file_path)
            file_messages = 0
            
            for grb in grbs:
                var_name = grb.name
                file_messages += 1
                total_messages += 1
                
                if var_name in var_mapping:
                    if file_messages % 100 == 0:  # Progress every 100 messages
                        print(f"  Processed {file_messages} messages from {filename}")
                    
                    # Get timestamp
                    starttime = grb.validDate
                    horizon = grb.startStep
                    timestamp = starttime + timedelta(hours=horizon)
                    
                    # Extract data values and coordinates
                    values, lats, lons = grb.data()
                    variable_name = var_mapping[var_name]
                    
                    # Process all grid points within Germany bounds
                    rows, cols = values.shape
                    for i in range(rows):
                        for j in range(cols):
                            lat = lats[i, j]
                            lon = lons[i, j]
                            
                            # Filter for Germany bounds
                            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                                # Round coordinates to avoid floating point precision issues
                                lat_rounded = round(lat, 4)
                                lon_rounded = round(lon, 4)
                                grid_key = (lat_rounded, lon_rounded)
                                
                                # Store data organized by grid point -> timestamp -> variable
                                grid_data[grid_key][timestamp][variable_name] = float(values[i, j])
            
            grbs.close()
            print(f"  ✓ {filename}: {file_messages} messages processed")
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            continue
    
    extraction_time = time.time() - start_time
    print(f"\n=== Extraction Summary ===")
    print(f"✓ Processed {len(grib_files)} GRIB files")
    print(f"✓ Total messages: {total_messages}")
    print(f"✓ Unique grid points: {len(grid_data)}")
    print(f"✓ Extraction time: {extraction_time:.1f}s")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create parquet file for each grid point
    print(f"\nCreating parquet files...")
    grid_files = {}
    
    for grid_idx, (grid_key, timestamps_data) in enumerate(grid_data.items(), 1):
        lat, lon = grid_key
        
        if grid_idx % 100 == 0:  # Progress every 100 grid points
            print(f"Processing grid point {grid_idx}/{len(grid_data)}")
        
        # Convert this grid point's data to records
        records = []
        for timestamp, variables in timestamps_data.items():
            record = {
                'timestamp': timestamp,
                'latitude': lat,
                'longitude': lon,
                'u_wind_10m': variables.get('u_wind_10m'),
                'v_wind_10m': variables.get('v_wind_10m'),
                'u_wind_100m': variables.get('u_wind_100m'),
                'v_wind_100m': variables.get('v_wind_100m'),
                'wind_gust_10m': variables.get('wind_gust_10m'),
                'friction_wind': variables.get('friction_wind'),
                'temp_2m': variables.get('temp_2m'),
                'pressure': variables.get('pressure'),
                'dew_point_2m': variables.get('dew_point_2m')
            }
            
            # Calculate derived wind variables
            if record['u_wind_10m'] is not None and record['v_wind_10m'] is not None:
                record['wind_speed_10m'] = np.sqrt(
                    record['u_wind_10m']**2 + record['v_wind_10m']**2
                )
                record['wind_direction_10m'] = np.arctan2(
                    record['v_wind_10m'], record['u_wind_10m']
                ) * 180 / np.pi
            
            records.append(record)
        
        # Create DataFrame for this grid point
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        # Create filename: grid_LAT_LON.parquet
        filename = f"grid_{lat:07.4f}_{lon:07.4f}.parquet"
        file_path = output_dir / filename
        
        # Save to parquet
        df.to_parquet(
            file_path,
            compression='snappy',
            index=False,
            engine='pyarrow'
        )
        
        # Track the file
        grid_files[grid_key] = str(file_path)
    
    total_time = time.time() - start_time
    
    # Summary statistics
    total_files = len(grid_files)
    total_size_mb = sum(Path(f).stat().st_size for f in grid_files.values()) / (1024 * 1024)
    
    # Sample a few files to get average records
    sample_files = list(grid_files.values())[:min(10, len(grid_files))]
    total_sample_records = sum(len(pd.read_parquet(f)) for f in sample_files)
    avg_records = total_sample_records / len(sample_files) if sample_files else 0
    
    print(f"\n=== Final Summary ===")
    print(f"✓ Created {total_files} parquet files")
    print(f"✓ Total size: {total_size_mb:.1f} MB")
    print(f"✓ Average records per grid: ~{avg_records:.0f}")
    print(f"✓ Total processing time: {total_time:.1f}s")
    print(f"✓ Files saved to: {output_dir}")
    
    # Show time range
    if grid_files:
        sample_df = pd.read_parquet(list(grid_files.values())[0])
        min_time = sample_df['timestamp'].min()
        max_time = sample_df['timestamp'].max()
        print(f"✓ Time range: {min_time} to {max_time}")
    
    return grid_files

def create_grid_index(grid_files_dict, output_dir="data/parquet/grid_points"):
    """
    Create an index file mapping coordinates to parquet files for fast lookup
    """
    print("\nCreating grid coordinate index...")
    
    index_records = []
    for (lat, lon), file_path in grid_files_dict.items():
        index_records.append({
            'latitude': lat,
            'longitude': lon,
            'parquet_file': Path(file_path).name  # Just the filename, not full path
        })
    
    index_df = pd.DataFrame(index_records)
    index_file = Path(output_dir) / "grid_index.parquet"
    
    index_df.to_parquet(index_file, compression='snappy', index=False)
    
    print(f"✓ Grid index created: {index_file}")
    print(f"  Contains {len(index_df)} grid point references")
    
    return index_file

if __name__ == "__main__":
    # Process all monthly ERA5 GRIB files
    grib_directory = "/Users/meghna/Downloads/wind"  # Directory containing era5_*_*.grib files
    output_directory = "/Users/meghna/synthetic_re_data_generation/streamlit/data/parquet/grid_points"
    
    print("=== ERA5 Monthly GRIB to Individual Grid Parquet Files ===")
    
    # Extract and convert all monthly files
    grid_files = extract_multiple_grib_to_parquet_by_grid(
        grib_dir=grib_directory,
        output_dir=output_directory
    )
    
    if grid_files:
        # Create index for fast lookup
        index_file = create_grid_index(grid_files, output_directory)
        
        # Show some sample files
        print(f"\nSample files created:")
        for i, (grid_key, file_path) in enumerate(list(grid_files.items())[:5]):
            lat, lon = grid_key
            file_size = Path(file_path).stat().st_size / 1024
            print(f"  {Path(file_path).name} ({file_size:.1f} KB)")
    else:
        print("No grid files were created. Check your GRIB directory and file naming.")