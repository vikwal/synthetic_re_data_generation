"""Per-turbine layouts and turbine groups of the real parks (park = MaStR Lokation).

Inputs (all produced in FL_Contribution, read-only here):
  selection   wind_park_selection_v1_1.csv   90 parks: client, class, ...
  turbines    wind_turbines_matched.csv      one MaStR unit per row, lib_name
  osm         osm_corrections_v1_1.csv       coordinate to use per unit
                                             (MaStR or OSM-corrected)

A group is a set of identical turbines that the chain can simulate once and
multiply by n: same power curve (lib_name), hub height, commissioning year,
MaStR net rating and ERA5 cell. Group ids t1..tG follow (year, type, hub,
rating, cell). The group's commissioning date is the mean of its members'.
Rating cap = min(MaStR net rating, curve maximum): the curve itself is never
rescaled (real curves only), so units registered above their curve maximum
(e.g. V136-3.45 at 3600 kW) cannot exceed the curve.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer

from parks import era5_db

GROUP_KEY = ["lib_name", "hub_height", "commissioning_year", "rated_kw", "era5_cell_id"]
LAYOUT_COLS = ["park_id", "park_name", "client", "cls", "turbine_id", "operator", "manufacturer",
               "model", "lib_name", "rated_kw", "rated_cap_kw", "hub_height", "rotor_diameter",
               "commissioning_date", "commissioning_year", "latitude", "longitude",
               "x_utm32", "y_utm32", "coord_source", "osm_status", "d_osm_m",
               "era5_cell_id", "era5_cell_dist_km", "group_id"]

_TF = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def build_turbine_table(selection: pd.DataFrame, turbines: pd.DataFrame,
                        osm: pd.DataFrame, curve_max_kw: dict) -> pd.DataFrame:
    """One row per unit of the selected parks with the coordinate to use."""
    t = turbines[turbines["lokation"].isin(selection["lokation"])].copy()
    missing = set(selection["lokation"]) - set(t["lokation"])
    if missing:
        raise ValueError(f"parks without units in the turbine table: {sorted(missing)}")
    if t["lib_name"].isna().any():
        raise ValueError("units without power-curve match (lib_name) in the selection")
    o = osm.set_index("EinheitMastrNummer")
    if not set(t["EinheitMastrNummer"]) <= set(o.index):
        raise ValueError("OSM correction table does not cover all selected units")
    o = o.loc[t["EinheitMastrNummer"]]
    sel = selection.set_index("lokation")
    out = pd.DataFrame({
        "park_id": t["lokation"].values,
        "park_name": sel.loc[t["lokation"], "name"].values,
        "client": sel.loc[t["lokation"], "client"].values,
        "cls": sel.loc[t["lokation"], "cls"].values,
        "turbine_id": t["EinheitMastrNummer"].values,
        "operator": t["operator"].values,
        "manufacturer": t["Hersteller"].values,
        "model": t["lib_name"].values,
        "lib_name": t["lib_name"].values,
        "rated_kw": t["Nettonennleistung"].astype(float).values,
        "hub_height": t["Nabenhoehe"].astype(float).values,
        "rotor_diameter": t["Rotordurchmesser"].astype(float).values,
        "commissioning_date": pd.to_datetime(t["Inbetriebnahmedatum"]).dt.strftime("%Y-%m-%d").values,
        "commissioning_year": pd.to_datetime(t["Inbetriebnahmedatum"]).dt.year.values,
        "latitude": o["lat"].astype(float).values,
        "longitude": o["lon"].astype(float).values,
        "coord_source": o["coord_source"].values,
        "osm_status": o["status"].values,
        "d_osm_m": o["d_osm_m"].astype(float).values,
    })
    out["rated_cap_kw"] = np.minimum(out["rated_kw"], out["lib_name"].map(curve_max_kw))
    out["x_utm32"], out["y_utm32"] = _TF.transform(out["longitude"].values, out["latitude"].values)
    return out


def assign_cells(layout: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    ids, dist = era5_db.nearest_cells(points, layout["latitude"], layout["longitude"])
    layout = layout.copy()
    layout["era5_cell_id"], layout["era5_cell_dist_km"] = ids, dist
    return layout


def assign_groups(layout: pd.DataFrame) -> pd.DataFrame:
    """Adds group_id t1..tG per park (requires era5_cell_id)."""
    layout = layout.copy()
    layout["group_id"] = ""
    for pid, g in layout.groupby("park_id"):
        keys = g[GROUP_KEY].drop_duplicates().sort_values(
            ["commissioning_year", "lib_name", "hub_height", "rated_kw", "era5_cell_id"])
        gid = {tuple(k): f"t{i}" for i, k in enumerate(keys.itertuples(index=False), start=1)}
        layout.loc[g.index, "group_id"] = [gid[tuple(r)] for r in g[GROUP_KEY].itertuples(index=False)]
    return layout[LAYOUT_COLS]


def group_table(layout: pd.DataFrame) -> pd.DataFrame:
    """One row per (park, group) with n, mean commissioning date, cell."""
    rows = []
    for (pid, gid), g in layout.groupby(["park_id", "group_id"], sort=False):
        dates = pd.to_datetime(g["commissioning_date"])
        rows.append({"park_id": pid, "group_id": gid, "lib_name": g["lib_name"].iloc[0],
                     "n_turbines": len(g), "hub_height_m": float(g["hub_height"].iloc[0]),
                     "rotor_diameter_m": float(g["rotor_diameter"].mean()),
                     "rated_kw": float(g["rated_kw"].iloc[0]),
                     "rated_cap_kw": float(g["rated_cap_kw"].iloc[0]),
                     "commissioning_year": int(g["commissioning_year"].iloc[0]),
                     "commissioning_date": str(pd.Timestamp(dates.astype("int64").mean()).date()),
                     "era5_cell_id": int(g["era5_cell_id"].iloc[0]),
                     "era5_cell_dist_km": float(g["era5_cell_dist_km"].mean())})
    gt = pd.DataFrame(rows)
    gt["_order"] = gt["group_id"].str[1:].astype(int)
    return gt.sort_values(["park_id", "_order"]).drop(columns="_order").reset_index(drop=True)


def primary_cell(groups: pd.DataFrame) -> int:
    """ERA5 cell carrying the largest share of the park's rating (ties: lowest id)."""
    cap = (groups["n_turbines"] * groups["rated_kw"]).groupby(groups["era5_cell_id"]).sum()
    return int(cap[cap == cap.max()].index.min())


def min_spacing_d(g: pd.DataFrame) -> float:
    """Smallest pairwise spacing in rotor diameters (larger rotor of the pair)."""
    if len(g) < 2:
        return np.inf
    x, y, r = g["x_utm32"].values, g["y_utm32"].values, g["rotor_diameter"].values
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    return float((d / np.maximum(r[:, None], r[None, :])).min())
