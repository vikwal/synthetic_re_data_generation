"""Grid areas and grid nodes.

Area of a park: column 'area' of FL_Contribution/pipeline/curtailment/
park_grid_areas.csv (Bundesland by point-in-polygon, NI split at 8.7 E).
For points without a table entry (DWD station runs) the same rule is applied
here on the Bundeslaender polygons.

Nodes: single-linkage clustering of the park centroids with threshold
node_radius_km (haversine); parks of a node share all grid events. Node id =
smallest park id of the cluster, so ids do not depend on the input order.
"""

import json

import numpy as np
import pandas as pd

from curtailment.config import area_key

STATE_AREA = {"DE-SH": "A1_SH", "DE-HH": "A1_SH", "DE-HB": "A2_NI_NW", "DE-ST": "A3_NI_O_ST",
              "DE-BB": "A4_NO", "DE-MV": "A4_NO", "DE-BE": "A4_NO",
              "DE-NW": "A5_MITTE_W", "DE-HE": "A5_MITTE_W", "DE-RP": "A5_MITTE_W", "DE-SL": "A5_MITTE_W",
              "DE-TH": "A5_MITTE_W", "DE-SN": "A5_MITTE_W", "DE-BY": "A6_SUED", "DE-BW": "A6_SUED"}
NI_SPLIT_LON = 8.7
EARTH_R_KM = 6371.0088


def load_area_table(path: str) -> pd.Series:
    """park id -> area key."""
    t = pd.read_csv(path)
    return pd.Series([area_key(a) for a in t["area"]], index=t["lokation"].astype(str), name="area")


def _rings(geom: dict) -> list:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.asarray(p[0]) for p in polys]


def _inside(lon, lat, ring) -> np.ndarray:
    x, y = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    res = np.zeros(len(lon), bool)
    for xi, yi, xk, yk in zip(x, y, xj, yj):
        res ^= ((yi > lat) != (yk > lat)) & (lon < (xk - xi) * (lat - yi) / (yk - yi + 1e-15) + xi)
    return res


def state_of(lon, lat, geojson: str) -> np.ndarray:
    """Bundesland id (e.g. 'DE-SH'); enclaves (BE, HB, HH) first; points just outside
    the simplified polygons get the state of the nearest vertex."""
    lon, lat = np.atleast_1d(np.asarray(lon, float)), np.atleast_1d(np.asarray(lat, float))
    with open(geojson) as f:
        feats = json.load(f)["features"]
    feats = sorted(feats, key=lambda f: f["properties"]["id"] not in ("DE-BE", "DE-HB", "DE-HH"))
    out = np.full(len(lon), "", dtype=object)
    for f in feats:
        hit = np.zeros(len(lon), bool)
        for r in _rings(f["geometry"]):
            hit |= _inside(lon, lat, r)
        out[(out == "") & hit] = f["properties"]["id"]
    miss = np.flatnonzero(out == "")
    if len(miss):
        verts = [(f["properties"]["id"], r) for f in feats for r in _rings(f["geometry"])]
        ids = np.concatenate([[i] * len(r) for i, r in verts])
        xy = np.concatenate([r for _, r in verts])
        for j in miss:
            out[j] = ids[np.argmin((xy[:, 0] - lon[j]) ** 2 + ((xy[:, 1] - lat[j]) * 1.6) ** 2)]
    return out


def area_of_point(lon, lat, geojson: str) -> np.ndarray:
    lon = np.atleast_1d(np.asarray(lon, float))
    st = state_of(lon, lat, geojson)
    a = np.array([STATE_AREA.get(s, "") for s in st], dtype=object)
    ni = st == "DE-NI"
    a[ni] = np.where(lon[ni] < NI_SPLIT_LON, "A2_NI_NW", "A3_NI_O_ST")
    return a


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def cluster_nodes(ids, lat, lon, radius_km: float) -> pd.Series:
    """Single linkage (union-find over all pairs with d <= radius): id -> node id."""
    ids = [str(i) for i in ids]
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    for i, j in zip(*np.nonzero(np.triu(d <= radius_km, k=1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = [find(i) for i in range(len(ids))]
    members = pd.Series(ids).groupby(root).transform("min")
    return pd.Series(members.values, index=ids, name="node")


def node_table(parks: pd.DataFrame, area: pd.Series, radius_km: float) -> pd.DataFrame:
    """parks: park_id, latitude, longitude, capacity_kw. Returns one row per park with
    node, area (own), node_area (capacity-weighted majority within the node) and
    the node size."""
    p = parks[["park_id", "latitude", "longitude", "capacity_kw"]].copy()
    p["park_id"] = p["park_id"].astype(str)
    missing = sorted(set(p["park_id"]) - set(area.index))
    if missing:
        raise KeyError(f"no grid area for parks {missing}")
    p["area"] = p["park_id"].map(area)
    p["node"] = cluster_nodes(p["park_id"], p["latitude"], p["longitude"], radius_km).values
    cap = p.groupby(["node", "area"])["capacity_kw"].sum().reset_index()
    cap = cap.sort_values(["node", "capacity_kw", "area"], ascending=[True, False, True])
    p["node_area"] = p["node"].map(cap.drop_duplicates("node").set_index("node")["area"])
    p["node_size"] = p.groupby("node")["park_id"].transform("size")
    return p.sort_values("park_id").reset_index(drop=True)
