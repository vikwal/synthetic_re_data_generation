#!/usr/bin/env python3
"""WP0.6 — park layouts and Malo-ID <-> park_id mapping.

Reads data/wind_farm_meterdata/masterdata_wind_20240601.csv (per-turbine WGS84
coordinates, model, hub height, rotor, installed kW, night curtailment) and
reconstructs the mapping between the benchmark Malo-IDs and the
config park_ids (nearest DWD station logic from evaluation_real.ipynb).

Outputs (data/round2/):
  park_layouts.csv  one row per turbine incl. UTM32 x,y
  park_mapping.csv  park_id (config) <-> malo_id <-> Parkname + distance
"""

import glob
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MASTER = os.path.join(REPO, "data", "wind_farm_meterdata", "masterdata_wind_20240601.csv")
METER = os.path.join(REPO, "data", "wind_farm_meterdata", "meterdata_wind_20240601.csv")
COMM = os.path.join(REPO, "data", "comm_dates.csv")
CONFIG_DIR = os.path.join(REPO, "configs", "real_wind_parks_era5")
OUT_DIR = os.path.join(REPO, "data", "round2")

LAT, LON = "Koordinaten Breite (WGS 84)", "Koordinaten Länge (WGS 84)"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    master = pd.read_csv(MASTER, sep=";")
    meter_cols = pd.read_csv(METER, sep=";", nrows=0).columns.drop("Zeit").tolist()

    # config park ids ("00198_1", "07374", ...) and their station coordinates
    park_ids = sorted(os.path.basename(p)[len("config_"):-len(".yaml")]
                      for p in glob.glob(os.path.join(CONFIG_DIR, "config_*.yaml")))
    comm = pd.read_csv(COMM, sep=";", dtype={"park_id": str}).set_index("park_id")
    stations = pd.DataFrame({"park_id": park_ids})
    stations["station_id"] = stations["park_id"].str[:5]
    stations = stations.join(comm[["latitude", "longitude"]], on="station_id")
    assert stations[["latitude", "longitude"]].notna().all().all(), "station coords missing"

    # park centroids by Parkname (same grouping as evaluation_real.ipynb)
    centroids = master.groupby("Parkname")[["Malo-ID", LAT, LON]].mean().reset_index()
    centroids["Malo-ID"] = centroids["Malo-ID"].astype("int64").astype(str)

    # Match configs to benchmark parks by TURBINE SIGNATURE (model histogram),
    # with geographic distance as tiebreak. Pure nearest-distance mapping
    # mismatches co-located parks (e.g. Issum/Kengen/Rheurdt at station 01303).
    import yaml

    def norm_model(s: str) -> str:
        # extract the family token: MM100, M104, E-82, V126, N117, ...
        import re
        s = str(s).upper().replace("-", "").replace(".", "")
        m = re.search(r"(MM\d+|M\d{3}|E\d{2,3}|V\d{2,3}|N\d{2,3})", s)
        return m.group(1) if m else s.split()[0]

    def config_signature(pid: str):
        with open(os.path.join(CONFIG_DIR, f"config_{pid}.yaml")) as f:
            cfg = yaml.safe_load(f)
        models = [norm_model(t) for t in cfg["params"]["turbines"]]
        return pd.Series(models).value_counts().to_dict()

    def master_signature(grp: pd.DataFrame):
        models = [norm_model(t) for t in grp["Typ"]]
        return pd.Series(models).value_counts().to_dict()

    master_sigs = {name: master_signature(grp)
                   for name, grp in master.groupby("Parkname")}

    def sig_score(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
        union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
        return inter / union if union else 0.0

    cand = []
    cent = centroids.set_index("Parkname")
    for pid in park_ids:
        sig = config_signature(pid)
        srow = stations[stations["park_id"] == pid].iloc[0]
        for name, msig in master_sigs.items():
            d = haversine_km(cent.loc[name, LAT], cent.loc[name, LON],
                             srow["latitude"], srow["longitude"])
            cand.append((1.0 - sig_score(sig, msig), d, pid, name))
    taken_pid, taken_name, mapping = set(), set(), []
    for score, dist, pid, name in sorted(cand):
        if pid in taken_pid or name in taken_name or score > 0.5 or dist > 60:
            continue
        taken_pid.add(pid)
        taken_name.add(name)
        mapping.append({"park_id": pid, "malo_id": cent.loc[name, "Malo-ID"],
                        "park_name": name, "distance_km": round(float(dist), 2),
                        "signature_match": round(1.0 - score, 2)})
    mapping = pd.DataFrame(mapping).sort_values("park_id").reset_index(drop=True)

    missing_meter = set(mapping["malo_id"]) - set(meter_cols)
    assert not missing_meter, f"mapped malo ids missing in meterdata: {missing_meter}"

    # per-turbine layout with UTM32 coordinates
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = tf.transform(master[LON].values, master[LAT].values)
    layouts = master.rename(columns={
        "Parkname": "park_name", "ParkID": "operator_park_id", "Typ": "model",
        "Hersteller": "manufacturer", "Nabenhöhe (m)": "hub_height",
        "Rotordurchmesser (m)": "rotor_diameter",
        "Installierte Leistung (kW)": "rated_kw", "Malo-ID": "malo_id",
        LAT: "latitude", LON: "longitude",
        "Nachtabregelung": "night_curtailment",
        "Nachtabregelung von (CET)": "night_curt_from",
        "Nachtabregelung bis (CET)": "night_curt_to",
        "Nachtabregelung max kW": "night_curt_max_kw",
    })
    layouts["malo_id"] = layouts["malo_id"].astype("int64").astype(str)
    layouts["x_utm32"], layouts["y_utm32"] = x, y
    layouts = layouts.merge(mapping[["park_id", "malo_id"]], on="malo_id", how="left")

    mapping.to_csv(os.path.join(OUT_DIR, "park_mapping.csv"), index=False)
    layouts.to_csv(os.path.join(OUT_DIR, "park_layouts.csv"), index=False)

    n_mapped = layouts["park_id"].notna().sum()
    print(f"configs: {len(park_ids)} | centroids: {len(centroids)} | mapped pairs: {len(mapping)}")
    print(f"turbines: {len(layouts)} (mapped to a config park: {n_mapped})")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
