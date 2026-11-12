"""CAMS request points for PV parks: the park centre plus the 8-point ring (3 and 8 km, N/E/S/W, geodesic
WGS84 as for the DWD station rings) and one multi-year ADS request per point.

One request for 2023-07..2026-09 returns the same numbers as yearly chunks (checked 2026-10-13 on a station
ring point: max |diff| 6e-5 Wh/m², float rounding), takes ~90 s and counts once against the ADS daily
quota (~500 requests), so 90 parks x 9 points fit into two quota days instead of seven.

File names follow scripts/cams_bc/apply_parks.py: centre CAMS_rad_<park>_<tag>.csv in cams_dir, ring
CAMS_rad_<park>_r<3|8>_<N|E|S|W>_<tag>.csv in ring_dir."""
import os
import time

import numpy as np
import pandas as pd
from pyproj import Geod

import settings

GEOD = Geod(ellps="WGS84")
RING = [(3.0, "N", 0.0), (3.0, "E", 90.0), (3.0, "S", 180.0), (3.0, "W", 270.0),
        (8.0, "N", 0.0), (8.0, "E", 90.0), (8.0, "S", 180.0), (8.0, "W", 270.0)]
DEM_MOSAIC = settings.get("dem_mosaic_90m")
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"


def dem_altitudes(lats, lons, mosaic: str = DEM_MOSAIC) -> np.ndarray:
    """GLO-30 (90 m mosaic) surface height at the points, clipped at 0 m (McClear expects >= 0)."""
    import rasterio
    with rasterio.open(mosaic) as d:
        z = np.array([v[0] for v in d.sample(list(zip(lons, lats)))], float)
    return np.clip(np.where(np.isfinite(z), z, 0.0), 0.0, None)


def ring_points(park_id: str, lat: float, lon: float) -> list:
    pts = [{"point_id": park_id, "park_id": park_id, "kind": "centre", "radius_km": 0.0, "azimuth": "", "lat": lat, "lon": lon}]
    for r, name, az in RING:
        lo, la, _ = GEOD.fwd(lon, lat, az, r * 1000.0)
        pts.append({"point_id": f"{park_id}_r{int(r)}_{name}", "park_id": park_id, "kind": "ring", "radius_km": r, "azimuth": name,
                    "lat": round(la, 4), "lon": round(lo, 4)})
    return pts


def point_table(parks: pd.DataFrame) -> pd.DataFrame:
    """parks: frame with park_id, lat, lon (4 decimals used). Adds DEM altitude."""
    rows = []
    for r in parks.itertuples():
        rows += ring_points(r.park_id, round(float(r.lat), 4), round(float(r.lon), 4))
    t = pd.DataFrame(rows)
    t["altitude_m"] = np.round(dem_altitudes(t.lat.values, t.lon.values)).astype(int)
    return t


def target_path(cams_dir: str, ring_dir: str, pt, tag: str) -> str:
    d = cams_dir if pt.kind == "centre" else ring_dir
    return os.path.join(d, f"CAMS_rad_{pt.point_id}_{tag}.csv")


def fetch_point(pt, start: str, end: str, target: str, retries: int = 3, timeout: int = 1800) -> str:
    """One ADS request (15 min, observed_cloud + McClear, UT). Returns 'ok', 'skip' or 'fail'."""
    import cdsapi
    if os.path.exists(target) and os.path.getsize(target) > 1000:
        return "skip"
    req = {"sky_type": "observed_cloud", "location": {"latitude": float(pt.lat), "longitude": float(pt.lon)},
           "altitude": str(int(pt.altitude_m)), "date": f"{start}/{end}", "time_step": "15minute",
           "time_reference": "universal_time", "format": "csv"}
    c = cdsapi.Client(url=ADS_URL, timeout=timeout, quiet=True, progress=False)
    for attempt in range(retries):
        try:
            t0 = time.time()
            c.retrieve("cams-solar-radiation-timeseries", req, target + ".part")
            os.replace(target + ".part", target)
            print(f"{pt.point_id} ok {time.time() - t0:.0f}s", flush=True)
            return "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f"{pt.point_id} attempt {attempt} failed: {msg[:160]}", flush=True)
            if "daily" in msg.lower() or "quota" in msg.lower() or "limit" in msg.lower():
                return "quota"
            time.sleep(30 * 2 ** attempt)
    return "fail"
