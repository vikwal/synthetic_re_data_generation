"""Metadata tables of the release (FL_Contribution/federated_setting.md section 7).

parks.csv        park_id, site_id, technology, archetype, capacity_kw, n_groups,
                 commissioning_first/last, availability_mean, dc_ac_ratio,
                 history_start, client_id + real-park extras
wind_groups.csv  park_id, group_id, turbine_type, n_turbines, hub_height_m,
                 rotor_diameter_m, rated_kw, commissioning_year, power_curve_id + extras
clients.csv      client_id, archetype, n_parks, region_id, capacity_kw
sites.csv        one site per park (real parks are their own site): coordinates,
                 terrain metrics, ERA5 cell
Capacity = sum of MaStR net ratings (capacity_kw); capacity_cap_kw = sum of
min(net rating, curve maximum), the most the simulated park can deliver.
"""

import pandas as pd


def parks_table(selection: pd.DataFrame, layout: pd.DataFrame, groups: pd.DataFrame,
                gate: pd.DataFrame, run_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in selection.iterrows():
        lk = s["lokation"]
        lay, grp = layout[layout.park_id == lk], groups[groups.park_id == lk]
        cap_by_cell = (grp.n_turbines * grp.rated_kw).groupby(grp.era5_cell_id).sum()
        rows.append({
            "park_id": lk, "site_id": lk, "technology": "wind", "archetype": s["cls"],
            "capacity_kw": float(lay.rated_kw.sum()), "n_groups": len(grp),
            "commissioning_first": lay.commissioning_date.min(),
            "commissioning_last": lay.commissioning_date.max(),
            "availability_mean": 1.0, "dc_ac_ratio": float("nan"), "history_start": "",
            "client_id": s["client"], "client_kind": s["kind"], "region": int(s["region"]),
            "name": s["name"], "n_turbines": len(lay), "k_types": int(lay.lib_name.nunique()),
            "n_operators": int(lay.operator.nunique()),
            "capacity_cap_kw": float(lay.rated_cap_kw.sum()),
            "latitude": float(lay.latitude.mean()), "longitude": float(lay.longitude.mean()),
            "primary_era5_cell": int(cap_by_cell[cap_by_cell == cap_by_cell.max()].index.min()),
            "n_era5_cells": int(grp.era5_cell_id.nunique()),
            "correction_branch": "C",
            "osm_corrected_units": int(lay.osm_status.astype(str).str.startswith("corrected").sum()),
            "osm_unresolved_units": int((lay.osm_status == "unresolved").sum()),
            "replaced_v1_park": s.get("replaced") if isinstance(s.get("replaced"), str) else "",
        })
    out = pd.DataFrame(rows)
    if gate is not None and len(gate):
        g = gate.rename(columns={"lokation": "park_id", "branch": "gate_branch_hybrid",
                                 "reason": "gate_reason", "station_id": "gate_station_id",
                                 "dist_km": "gate_station_dist_km"})
        out = out.merge(g[["park_id", "gate_branch_hybrid", "gate_reason", "gate_station_id",
                           "gate_station_dist_km"]], on="park_id", how="left")
    if run_summary is not None and len(run_summary):
        out = out.merge(run_summary, on="park_id", how="left")
    return out


def wind_groups_table(groups: pd.DataFrame, ct_info: pd.DataFrame = None,
                      aging: pd.DataFrame = None) -> pd.DataFrame:
    g = groups.rename(columns={"lib_name": "turbine_type"}).copy()
    g["power_curve_id"] = g["turbine_type"]
    cols = ["park_id", "group_id", "turbine_type", "n_turbines", "hub_height_m", "rotor_diameter_m",
            "rated_kw", "commissioning_year", "power_curve_id", "rated_cap_kw", "commissioning_date",
            "era5_cell_id", "era5_cell_dist_km"]
    g = g[cols]
    if ct_info is not None and len(ct_info):
        ct = ct_info.groupby("lib_name")["generic_ct"].max().rename("ct_generic").reset_index()
        g = g.merge(ct.rename(columns={"lib_name": "turbine_type"}), on="turbine_type", how="left")
    if aging is not None and len(aging):
        g = g.merge(aging, on=["park_id", "group_id"], how="left")
    return g


def clients_table(parks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, p in parks.groupby("client_id", sort=False):
        kind = p["client_kind"].iloc[0]
        rows.append({"client_id": cid,
                     "archetype": "holdout" if cid == "holdout" else kind,
                     "n_parks": len(p), "region_id": int(p["region"].iloc[0]) if kind == "regional" else "",
                     "capacity_kw": float(p["capacity_kw"].sum()), "n_turbines": int(p["n_turbines"].sum())})
    return pd.DataFrame(rows)


def sites_table(parks: pd.DataFrame, topo: pd.DataFrame, cell_dist: pd.Series) -> pd.DataFrame:
    s = parks[["park_id", "longitude", "latitude", "primary_era5_cell", "correction_branch"]].rename(
        columns={"park_id": "site_id", "longitude": "lon", "latitude": "lat",
                 "primary_era5_cell": "era5_cell_id"})
    s["era5_cell_dist_km"] = s["site_id"].map(cell_dist)
    t = topo.rename(columns={"location_id": "site_id", "elevation": "altitude"})
    keep = ["site_id", "altitude", "slope", "aspect", "tpi5", "tpi75", "tdi", "elev_std", "z0", "clc_class"]
    return s.merge(t[[c for c in keep if c in t.columns]], on="site_id", how="left")
