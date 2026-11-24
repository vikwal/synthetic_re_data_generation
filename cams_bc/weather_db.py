"""Shared weather access for CAMS correction; meteorology also serves PV and wind."""
import pandas as pd
from parks.era5_db import connect, load_grid_points, fetch_cell, nearest_cells, check_complete, RAW_COLS


def load_solar_points(conn):
    with conn.cursor() as c:
        c.execute('SELECT cell_id,lat,lon FROM cams_weather.solar_points ORDER BY cell_id')
        rows=c.fetchall()
    if not rows:
        raise RuntimeError('cams_weather.solar_points is empty')
    return pd.DataFrame(rows,columns=['cell_id','lat','lon']).astype({'lat':float,'lon':float})
