"""Irradiance and meteorological drivers on the common hourly frame.

Every driver returns a DataFrame indexed by hourly UTC interval START with the columns
  ghi, dhi            W/m2, hourly means
  bhi                 W/m2 direct horizontal (where the source has it; else ghi - dhi)
  ghi_clear           W/m2 clear-sky GHI of the source where available (CAMS McClear, ERA5 ssrdc)
  src                 'station' | 'cams' | 'era5'
  imputed_share       fraction of the hour that is model-imputed (station) / unreliable (CAMS)
Met variables (temp_air degC, wind_speed m/s at 10 m, pressure Pa, albedo) come from
era5_met()/era5_solar() and are merged by the chain.

Sources and their timestamp conventions (pv.timebase):
  imputed DWD station  ${DATA_ROOT}/synthetic/interpol/solar/Station_<id>.parquet, 30 min, label END
  CAMS radiation       ADS CSV (Wh/m2 per 15 min, 'Observation period' start/end) or
                       WeatherDB public.cams_radiation (valid_time = END)
  ERA5                 WeatherDB public.era5_solar (J/m2 accumulated over [t-1h, t], label END),
                       public.era5_wind_grid (instantaneous at t)
"""

import glob
import gzip
import io
import os

import numpy as np
import pandas as pd

from pv import timebase

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
IMPUTED_DIR = os.path.join(DATA_ROOT, "synthetic", "interpol", "solar")
RAW_SOLAR_DIR = os.path.join(DATA_ROOT, "synthetic", "raw", "solar")
CAMS_PM_DIR = "/mnt/nas/cams/sqltocsv"

ERA5_SOLAR = "public.era5_solar"
ERA5_SOLAR_POINTS = "public.era5_solar_grid_points"
CAMS_TABLE = "public.cams_radiation"


# ------------------------------------------------------------------ helpers

def _clip_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _finish(df: pd.DataFrame, src: str) -> pd.DataFrame:
    df["ghi"] = df["ghi"].clip(lower=0)
    df["dhi"] = df["dhi"].clip(lower=0)
    df["dhi"] = np.minimum(df["dhi"], df["ghi"])
    if "bhi" not in df:
        df["bhi"] = df["ghi"] - df["dhi"]
    df["src"] = src
    return df


# ------------------------------------------------------------- DWD station

def imputed_stations(root: str = IMPUTED_DIR) -> list:
    return sorted(os.path.basename(p)[8:-8] for p in glob.glob(os.path.join(root, "Station_*.parquet")))


def nearest_station(lat: float, lon: float, master: pd.DataFrame, ids: list = None) -> tuple:
    """(station_id, distance_km) of the nearest station in `master` (columns station_id, latitude,
    longitude), optionally restricted to `ids`."""
    m = master if ids is None else master[master.station_id.isin(ids)]
    d = haversine_km(lat, lon, m.latitude.values, m.longitude.values)
    j = int(np.argmin(d))
    return str(m.station_id.iloc[j]), float(d[j])


def station_imputed(station_id: str, start=None, end=None, root: str = IMPUTED_DIR) -> pd.DataFrame:
    """Hourly GHI/DHI of one DWD station: observed where available, else the transformer
    imputation; night-time gaps are zero. imputed_share = fraction of the hour taken from the
    imputation (0 = fully observed)."""
    p = os.path.join(root, f"Station_{station_id}.parquet")
    df = pd.read_parquet(p).set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)
    for v in ("ghi", "dhi"):
        obs, imp = df[f"{v}_observed"], df[f"{v}_imputed"]
        val = obs.where(obs.notna(), imp)
        val = val.where(df["ist_tag"], 0.0)          # night: nothing to fill
        out[v] = val.astype("float64")
        out[f"{v}_imputed_flag"] = (obs.isna() & imp.notna() & df["ist_tag"]).astype(float)
        out[f"{v}_missing_flag"] = (val.isna() & df["ist_tag"]).astype(float)
    h = timebase.to_hourly_mean(out, label="end", step="30min")
    res = h[["ghi", "dhi"]].copy()
    res["imputed_share"] = h[["ghi_imputed_flag", "dhi_imputed_flag"]].max(axis=1)
    res["missing_share"] = h[["ghi_missing_flag", "dhi_missing_flag"]].max(axis=1)
    res["station_id"] = station_id
    return _clip_window(_finish(res, "station"), start, end)


def station_raw_met(station_id: str, start=None, end=None, root: str = RAW_SOLAR_DIR) -> pd.DataFrame:
    """Hourly temp_air (degC), wind_speed (m/s), precipitation (mm/h) from the raw 10-min DWD file
    (label END). Gaps stay NaN."""
    df = pd.read_parquet(os.path.join(root, f"Station_{station_id}.parquet"))
    cols = {"temperature_2m": "temp_air", "wind_speed": "wind_speed", "precipitation_rate": "precip_mm"}
    d = df[list(cols)].apply(pd.to_numeric, errors="coerce").rename(columns=cols)
    h = d.resample("1h", closed="right", label="right").agg({"temp_air": "mean", "wind_speed": "mean", "precip_mm": "sum"})
    h.index = h.index - pd.Timedelta("1h")
    return _clip_window(h, start, end)


# -------------------------------------------------------------------- CAMS

CAMS_COLS = {"TOA": "toa", "Clear sky GHI": "ghi_clear", "Clear sky BHI": "bhi_clear",
             "Clear sky DHI": "dhi_clear", "Clear sky BNI": "bni_clear", "GHI": "ghi", "BHI": "bhi",
             "DHI": "dhi", "BNI": "bni", "Reliability": "reliability"}


def read_cams_csv(path: str) -> pd.DataFrame:
    """Raw ADS CSV (optionally .gz): 15-min sums in Wh/m2 -> 15-min means in W/m2, index = START."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
        header = None
        with op(path, "rt", encoding="utf-8", errors="replace") as g:
            for ln in g:
                if ln.startswith("#") and "Observation period" in ln:
                    header = ln.lstrip("# ").strip().split(";")
    if header is None:
        raise ValueError(f"{path}: no column header line")
    df = pd.read_csv(io.StringIO("".join(lines)), sep=";", names=header)
    per = df["Observation period"].str.split("/", expand=True)
    df.index = pd.to_datetime(per[0], utc=True)
    step_h = (pd.to_datetime(per[1].iloc[0], utc=True) - df.index[0]) / pd.Timedelta("1h")
    df = df.drop(columns=["Observation period"]).rename(columns=CAMS_COLS)
    for c in df.columns:
        if c != "reliability":
            df[c] = df[c] / step_h                   # Wh per interval -> W/m2 mean
    return df


def cams_from_csv(paths: list, start=None, end=None) -> pd.DataFrame:
    """Hourly frame from one or more yearly ADS CSV files of one point."""
    parts = [read_cams_csv(p) for p in sorted(paths)]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    h = timebase.to_hourly_mean(df, label="start")
    h["imputed_share"] = 1.0 - h["reliability"].clip(0, 1)
    return _clip_window(_finish(h, "cams"), start, end)


def cams_station(conn, station_id: str, start=None, end=None) -> pd.DataFrame:
    """Hourly frame from WeatherDB public.cams_radiation (raw Wh per 15 min, valid_time = END)."""
    sql = (f"SELECT valid_time, toa, ghi_clear, bhi_clear, dhi_clear, bni_clear, ghi, bhi, dhi, bni, "
           f"reliability FROM {CAMS_TABLE} WHERE station_id = %s ORDER BY valid_time;")
    with conn.cursor() as cur:
        cur.execute(sql, (station_id,))
        rows = cur.fetchall()
    cols = ["timestamp", "toa", "ghi_clear", "bhi_clear", "dhi_clear", "bni_clear", "ghi", "bhi", "dhi", "bni", "reliability"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").astype("float64")
    for c in cols[1:-1]:
        df[c] = df[c] * 4.0                      # Wh per 15 min -> W/m2
    h = timebase.to_hourly_mean(df, label="end", step="15min")
    h["imputed_share"] = 1.0 - h["reliability"].clip(0, 1)
    return _clip_window(_finish(h, "cams"), start, end)


# -------------------------------------------------------------------- ERA5

def era5_solar_points(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT cell_id, lat, lon FROM {ERA5_SOLAR_POINTS} ORDER BY cell_id;")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["cell_id", "lat", "lon"]).astype({"lat": float, "lon": float})


def era5_solar(conn, lat: float, lon: float, start=None, end=None) -> tuple:
    """(hourly frame, meta) of the nearest ERA5 cell: ghi, bhi, dhi (= ghi - bhi), ghi_clear,
    bhi_clear, toa, albedo (fal), cloud_total. Accumulations J/m2 over [t-1h, t] -> W/m2, START."""
    pts = era5_solar_points(conn)
    d = haversine_km(lat, lon, pts.lat.values, pts.lon.values)
    j = int(np.argmin(d))
    cell = int(pts.cell_id.iloc[j])
    sql = (f"SELECT s.valid_time, s.ssrd, s.fdir, s.ssrdc, s.cdir, s.tisr, s.fal, s.tcc FROM {ERA5_SOLAR} s "
           f"JOIN {ERA5_SOLAR_POINTS} p ON p.geom = s.geom WHERE p.cell_id = %s ORDER BY s.valid_time;")
    with conn.cursor() as cur:
        cur.execute(sql, (cell,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["timestamp", "ssrd", "fdir", "ssrdc", "cdir", "tisr", "fal", "tcc"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").astype("float64").sort_index()
    df.index = df.index - pd.Timedelta("1h")                 # END -> START
    out = pd.DataFrame(index=df.index)
    out["ghi"] = df["ssrd"] / 3600.0
    out["bhi"] = df["fdir"] / 3600.0
    out["dhi"] = out["ghi"] - out["bhi"]
    out["ghi_clear"] = df["ssrdc"] / 3600.0
    out["bhi_clear"] = df["cdir"] / 3600.0
    out["toa"] = df["tisr"] / 3600.0
    out["albedo"] = df["fal"]
    out["cloud_total"] = df["tcc"]
    out["imputed_share"] = 0.0
    meta = {"era5_solar_cell": cell, "era5_solar_cell_lat": float(pts.lat.iloc[j]),
            "era5_solar_cell_lon": float(pts.lon.iloc[j]), "era5_solar_cell_dist_km": float(d[j])}
    return _clip_window(_finish(out, "era5"), start, end), meta


def era5_met(conn, lat: float, lon: float, start, end) -> tuple:
    """Hourly temp_air (degC), wind_speed (10 m, m/s), pressure (Pa), dew_point (degC) of the nearest
    ERA5 cell. Instantaneous values at t and t+1h are averaged for the interval [t, t+1h)."""
    from parks import era5_db
    pts = era5_db.load_grid_points(conn)
    ids, dist = era5_db.nearest_cells(pts, [lat], [lon])
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    raw = era5_db.fetch_cell(conn, int(ids[0]), s.strftime("%Y-%m-%d %H:%M"), e.strftime("%Y-%m-%d %H:%M"))
    out = pd.DataFrame(index=raw.index)
    out["temp_air"] = raw["temp_2m"] - 273.15
    out["wind_speed"] = np.hypot(raw["u_wind_10m"], raw["v_wind_10m"])
    out["pressure"] = raw["pressure"]
    out["dew_point"] = raw["dew_point_2m"] - 273.15
    out = (out + out.shift(-1)) / 2.0
    out.iloc[-1] = out.iloc[-2]
    meta = {"era5_met_cell": int(ids[0]), "era5_met_cell_dist_km": float(dist[0])}
    return _clip_window(out, start, end), meta


# ----------------------------------------------------------- CAMS PM (soiling)

def cams_pm(station_id: str, start=None, end=None, root: str = CAMS_PM_DIR) -> pd.DataFrame:
    """Hourly pm2_5 / pm10 in g/m3 from the CAMS air-quality CSV of a station (stored in kg/m3)."""
    df = pd.read_csv(os.path.join(root, f"cams_{station_id}.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()[["pm2_5", "pm10"]].astype(float) * 1e3
    h = df.resample("1h", closed="left", label="left").mean().interpolate(limit=6)
    return _clip_window(h, start, end)


# ---------------------------------------------------------------- assembly

def merge_driver(irr: pd.DataFrame, met: pd.DataFrame, index: pd.DatetimeIndex, albedo_default: float = 0.2) -> pd.DataFrame:
    """Join irradiance and met frames on the chain index; fill albedo and small met gaps."""
    df = irr.reindex(index).join(met.reindex(index), rsuffix="_met")
    if "albedo" not in df or df["albedo"].isna().all():
        df["albedo"] = albedo_default
    df["albedo"] = df["albedo"].fillna(albedo_default).clip(0.05, 0.9)
    for c in ("temp_air", "wind_speed", "pressure"):
        if c in df:
            df[c] = df[c].interpolate(limit=3)
    return df
