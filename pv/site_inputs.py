"""External site inputs per park: OpenStreetMap solar plant footprints (Overpass) and PVGIS horizon
profiles. Both are small public requests (one per park) with mirrors and retries; results are cached as
JSON keyed by park_id so the run is resumable."""
import json
import os
import time

import pandas as pd
import requests

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://lz4.overpass-api.de/api/interpreter",
                 "https://z.overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/printhorizon"
UA = {"User-Agent": "synthetic-re-data-generation/0.1 (research)"}


def overpass_query(lat: float, lon: float, radius_m: int = 3000) -> str:
    return f"""[out:json][timeout:90];
(
  way["power"="plant"]["plant:source"="solar"](around:{radius_m},{lat},{lon});
  relation["power"="plant"]["plant:source"="solar"](around:{radius_m},{lat},{lon});
  way["power"="generator"]["generator:source"="solar"](around:{radius_m},{lat},{lon});
  relation["power"="generator"]["generator:source"="solar"](around:{radius_m},{lat},{lon});
  way["landuse"="industrial"]["plant:source"="solar"](around:{radius_m},{lat},{lon});
);
out tags geom;"""


def fetch_overpass(lat: float, lon: float, radius_m: int = 3000, attempts: int = 8) -> dict:
    q = overpass_query(lat, lon, radius_m)
    for i in range(attempts):
        url = OVERPASS_URLS[i % len(OVERPASS_URLS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=150)
            if r.status_code == 200:
                return r.json()
            print("overpass status", r.status_code, url, flush=True)
        except Exception as e:  # noqa: BLE001
            print("overpass error", str(e)[:100], flush=True)
        time.sleep(15 + 10 * i)
    return {"elements": [], "error": "no response"}


def fetch_pvgis_horizon(lat: float, lon: float, attempts: int = 5) -> dict:
    """Horizon height (deg) by azimuth clockwise from north in 7.5 deg steps (PVGIS A: 0 = south,
    positive west -> converted)."""
    for i in range(attempts):
        try:
            r = requests.get(PVGIS_URL, params={"lat": lat, "lon": lon, "outputformat": "json"}, headers=UA, timeout=60)
            if r.status_code == 200:
                prof = r.json()["outputs"]["horizon_profile"]
                return {str((float(p["A"]) + 180.0) % 360.0): float(p["H_hor"]) for p in prof}
            print("pvgis status", r.status_code, flush=True)
        except Exception as e:  # noqa: BLE001
            print("pvgis error", str(e)[:100], flush=True)
        time.sleep(10 + 10 * i)
    return {}


def fetch_all(parks: pd.DataFrame, osm_path: str, horizon_path: str, radius_m: int = 3000, pause: float = 3.0) -> None:
    """parks: park_id, lat, lon. Appends to the two JSON caches, skipping parks already present."""
    osm = json.load(open(osm_path)) if os.path.exists(osm_path) else {}
    hz = json.load(open(horizon_path)) if os.path.exists(horizon_path) else {}
    for r in parks.itertuples():
        if r.park_id not in osm or "error" in osm[r.park_id]:
            osm[r.park_id] = fetch_overpass(r.lat, r.lon, radius_m)
            json.dump(osm, open(osm_path, "w"))
            print(r.park_id, "osm elements", len(osm[r.park_id].get("elements", [])), flush=True)
            time.sleep(pause)
        if not hz.get(r.park_id):
            hz[r.park_id] = fetch_pvgis_horizon(r.lat, r.lon)
            json.dump(hz, open(horizon_path, "w"))
            print(r.park_id, "horizon points", len(hz[r.park_id]), flush=True)
            time.sleep(1.0)
