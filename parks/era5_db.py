"""ERA5 single-level input read directly from Postgres (public.era5_wind_grid).

The full 0.25 deg field (1,221 cells, 47.25-55.25 N / 6-15 E) exists only in
the database; no point files are written. Each turbine is driven by its
geodesically nearest cell (pyproj WGS84 -- Euclidean degrees are anisotropic
at 50 N). The frame returned by fetch_cell() has exactly the column schema the
chain reads from the station CSVs (generate_wind.read_dfs), minus the three
round-2 NetCDF extras sshf/blh/gwd, which the power-law chain does not use.

Timestamps in the table are naive UTC; the session time zone is forced to UTC
(the server default is Europe/Berlin). A SHA256 over the fetched values goes
into every manifest so a later re-ingest of the table is detectable.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from pyproj import Geod

TABLE = "public.era5_wind_grid"
POINTS = "public.era5_wind_grid_points"
RAW_COLS = ["u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m", "wind_gust_10m",
            "friction_wind", "temp_2m", "pressure", "dew_point_2m"]

_GEOD = Geod(ellps="WGS84")


def connect(url: str = None):
    import psycopg2
    url = url or os.environ.get("WEATHER_DB_URL")
    if not url:
        raise RuntimeError("WEATHER_DB_URL not set (export it from ~/.bashrc)")
    conn = psycopg2.connect(url)
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC';")
    conn.commit()
    return conn


def load_grid_points(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT cell_id, lat, lon FROM {POINTS} ORDER BY cell_id;")
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"{POINTS} is empty")
    return pd.DataFrame(rows, columns=["cell_id", "lat", "lon"])


def nearest_cells(points: pd.DataFrame, lat, lon) -> tuple:
    """Geodesically nearest grid cell per (lat, lon): (cell_ids, distances_km)."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    plon, plat = points["lon"].values, points["lat"].values
    ids, dist = np.empty(len(lat), dtype=int), np.empty(len(lat))
    for i, (a, b) in enumerate(zip(lat, lon)):
        _, _, d = _GEOD.inv(np.full(len(plat), b), np.full(len(plat), a), plon, plat)
        j = int(np.argmin(d))
        ids[i], dist[i] = int(points["cell_id"].iloc[j]), d[j] / 1000.0
    return ids, dist


def fetch_cell(conn, cell_id: int, start: str, end: str) -> pd.DataFrame:
    """Hourly frame (UTC index 'timestamp') of RAW_COLS for one cell, [start, end]."""
    sql = (f"SELECT g.timestamp, {', '.join('g.' + c for c in RAW_COLS)} FROM {TABLE} g "
           f"JOIN {POINTS} p ON p.geom = g.geom "
           f"WHERE p.cell_id = %s AND g.timestamp >= %s AND g.timestamp <= %s ORDER BY g.timestamp;")
    with conn.cursor() as cur:
        cur.execute(sql, (int(cell_id), start, end))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["timestamp"] + RAW_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df = df.set_index("timestamp").astype("float64")
    check_complete(df, start, end, cell_id)
    return df


def check_complete(df: pd.DataFrame, start: str, end: str, cell_id) -> None:
    """The chain's KNN imputer must stay a no-op: hourly, gap-free, no NaN."""
    expect = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h")
    if len(df) != len(expect) or not df.index.equals(expect):
        missing = expect.difference(df.index)
        raise ValueError(f"cell {cell_id}: {len(missing)} missing hours "
                         f"(first {list(missing[:3])}) in [{start}, {end}]")
    if df.isna().any().any():
        raise ValueError(f"cell {cell_id}: NaN values in {df.columns[df.isna().any()].tolist()}")


def frame_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(df.index.asi8).tobytes())
    h.update(np.ascontiguousarray(df[RAW_COLS].values.astype("float64")).tobytes())
    return h.hexdigest()


def coverage(conn, cell_id: int) -> tuple:
    """First and last timestamp stored for one cell (the field is filled
    uniformly, so one cell stands for the table)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT min(g.timestamp), max(g.timestamp) FROM {TABLE} g "
                    f"JOIN {POINTS} p ON p.geom = g.geom WHERE p.cell_id = %s;", (int(cell_id),))
        lo, hi = cur.fetchone()
    return pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
