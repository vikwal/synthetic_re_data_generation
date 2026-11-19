"""Release tables: parks.csv, clients.csv (sites.csv, pv_arrays.csv and pv_masterdata_view.csv come from
the sites stage and are copied)."""
import pandas as pd


def parks_table(sel: pd.DataFrame, par: pd.DataFrame, sites: pd.DataFrame, run_summary: pd.DataFrame, dates: pd.DataFrame = None) -> pd.DataFrame:
    p = par.set_index("park_id")
    s = sites.set_index("site_id")
    rows = []
    for r in sel.itertuples():
        rows.append({
            "park_id": r.park_id, "site_id": r.park_id, "technology": "pv", "archetype": r.cls,
            "capacity_kw": float(p.loc[r.park_id, "ac_kw"]), "capacity_kwp_dc": float(p.loc[r.park_id, "kwp_dc"]),
            "dc_ac_ratio": float(p.loc[r.park_id, "dc_ac"]), "n_arrays": int(s.loc[r.park_id, "n_arrays"]),
            "commissioning_first": f"{dates.loc[r.park_id, 'commissioning_first']:%Y-%m-%d}" if dates is not None else f"{int(r.year)}-07-01",
            "commissioning_last": f"{dates.loc[r.park_id, 'commissioning_full']:%Y-%m-%d}" if dates is not None else f"{int(r.year_last)}-07-01",
            "availability_mean": 0.98, "history_start": "",
            "client_id": r.client, "client_kind": r.kind, "region": int(r.region) if pd.notna(r.region) else None, "cohort": r.cohort, "size_bin": r.size_bin,
            "terrain_cls": r.terrain_cls, "n_units_mastr": int(r.n_units), "n_lokationen": int(r.n_lokationen),
            "n_operators": int(r.n_operators), "bundesland": r.bundesland,
            "latitude": float(r.lat), "longitude": float(r.lon),
            "orientation_class_register": r.orientation_class, "tilt_class_register": r.tilt_class,
            "register_deviation": s.loc[r.park_id, "register_deviation"] if isinstance(s.loc[r.park_id, "register_deviation"], str) else "",
            "footprint_src": s.loc[r.park_id, "footprint_src"], "horizon_src": s.loc[r.park_id, "horizon_src"],
            "power_limit_register": r.power_limit if isinstance(r.power_limit, str) else "",
            "era5_solar_cell": "", "era5_met_cell": "",
        })
    out = pd.DataFrame(rows)
    if run_summary is not None and len(run_summary):
        out = out.merge(run_summary, on="park_id", how="left")
        if "series_start" in out:
            out["history_start"] = out.pop("series_start")
    return out


def clients_table(parks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, p in parks.groupby("client_id", sort=False):
        kind = p["client_kind"].iloc[0]
        rows.append({"client_id": cid, "archetype": "holdout" if cid == "holdout" else kind, "n_parks": len(p),
                     "region_id": int(p["region"].iloc[0]) if kind == "regional" else "",
                     "capacity_kw": float(p["capacity_kw"].sum()), "capacity_kwp_dc": float(p["capacity_kwp_dc"].sum()),
                     "n_arrays": int(p["n_arrays"].sum())})
    return pd.DataFrame(rows)
