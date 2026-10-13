"""Feature frames per station (30 min, daytime rows only) from the solar_bc station frames and the
CAMS ring parquets. Everything a model may see at an unseen location must come from CAMS, ERA5, the
sun and the coordinates - never from the station's own measurement."""
import os

import numpy as np
import pandas as pd

from . import features as F
from .config import AZIMUTHS, DATASET, IMPUTED, PROCESSED, RADII, RING_DIR, RING_POINTS, SOLAR_CONSTANT, SPLITS, STEP
from .splits import station_table

PROCESSED_COLS = ["ghi_obs", "dhi_obs", "y_valid_ghi", "y_valid_dhi", "day_mask", "cams_ghi", "cams_bhi", "cams_dhi",
                  "cams_bni", "cams_ghi_clear", "cams_dhi_clear", "cams_bhi_clear", "cams_toa", "cams_reliability",
                  "ghi", "bhi", "ghi_clear", "toa", "albedo", "cloud_total", "cloud_low", "cloud_mid", "cloud_high",
                  "water_vapour", "k_c", "f_dir", "lw_cre", "temp_2m", "pressure", "dew_dep"]
ERA5_RENAME = {c: f"era5_{c}" for c in ["ghi", "bhi", "ghi_clear", "toa", "albedo", "cloud_total", "cloud_low", "cloud_mid",
                                         "cloud_high", "water_vapour", "k_c", "f_dir", "lw_cre", "temp_2m", "pressure", "dew_dep"]}


def station_ids() -> list:
    return sorted(f[8:13] for f in os.listdir(PROCESSED) if f.startswith("Station_") and f.endswith(".parquet"))


def ring_points() -> pd.DataFrame:
    rp = pd.read_csv(RING_POINTS, dtype={"station_id": str})
    rp["punkt_id"] = rp["punkt_id"].astype(float).astype(int)
    return rp[rp["radius_km"].isin(RADII)]


def to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """15-min CAMS (mean power, interval end labels) -> 30-min means; both quarters required, as in
    solar_bc.data.cams_auf_raster."""
    r = df.resample(STEP, label="right", closed="right")
    return r.mean().where(r.count() == 2)


def ring_features(ring: dict, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """ring: {(radius, azimuth): 30-min frame with ghi, dhi, ghi_clear (interval END labels)}."""
    out = pd.DataFrame(index=idx)
    for r in RADII:
        for a in AZIMUTHS:
            d = ring[(r, a)].reindex(idx)
            out[f"ring{r:g}{a}_kc"] = d["ghi"] / np.maximum(d["ghi_clear"], 1.0)
            out[f"ring{r:g}{a}_kd"] = d["dhi"] / np.maximum(d["ghi"], 1.0)
    for r in RADII:
        kc = out[[f"ring{r:g}{a}_kc" for a in AZIMUTHS]]
        out[f"ring{r:g}_kc_mean"] = kc.mean(axis=1)
        out[f"ring{r:g}_kc_std"] = kc.std(axis=1)
        out[f"ring{r:g}_kc_ns"] = out[f"ring{r:g}N_kc"] - out[f"ring{r:g}S_kc"]
        out[f"ring{r:g}_kc_ew"] = out[f"ring{r:g}O_kc"] - out[f"ring{r:g}W_kc"]
        out[f"ring{r:g}_kd_mean"] = out[[f"ring{r:g}{a}_kd" for a in AZIMUTHS]].mean(axis=1)
    out["ring_all_kc_mean"] = out[[f"ring{r:g}_kc_mean" for r in RADII]].mean(axis=1)
    return out


def ring_frame(sid: str, idx: pd.DatetimeIndex, rp: pd.DataFrame) -> pd.DataFrame:
    pts = rp[rp.station_id == sid].set_index(["radius_km", "azimut"])
    ring = {}
    for r in RADII:
        for a in AZIMUTHS:
            pid = int(pts.loc[(r, a), "punkt_id"])
            d = pd.read_parquet(os.path.join(RING_DIR, f"Punkt_{pid}.parquet"), columns=["ghi", "dhi", "ghi_clear"])
            ring[(r, a)] = to_30min(d)
    return ring_features(ring, idx)


def assemble(d: pd.DataFrame, ring: pd.DataFrame, lat: float, lon: float, elev: float, nb: pd.DataFrame = None) -> pd.DataFrame:
    """Full feature frame from a 30-min frame `d` (regular index, interval END labels) holding the
    cams_* columns and the era5_* columns (ERA5_RENAME values), plus the ring features. Used for the
    stations (dataset) and for the park points (application) alike, so both see identical features."""
    d = d.copy()
    idx = d.index
    d["kc_cams"] = d["cams_ghi"] / np.maximum(d["cams_ghi_clear"], 1.0)
    d["kd_cams"] = d["cams_dhi"] / np.maximum(d["cams_ghi"], 1.0)
    d["kt_cams"] = d["cams_ghi"] / np.maximum(d["cams_toa"], 1.0)
    d["fdir_cams"] = d["cams_bhi"] / np.maximum(d["cams_ghi"], 1.0)
    # sun geometry from the CAMS TOA irradiance (toa = E0n cos z) and the longitude (solar local time)
    doy = idx.dayofyear.to_numpy(float)
    e0n = SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0))
    d["cos_zen"] = np.clip(d["cams_toa"].to_numpy(float) / e0n, 0.0, 1.0)
    d["airmass"] = np.minimum(1.0 / np.maximum(d["cos_zen"], 0.05), 20.0)
    hour = (idx.hour.to_numpy(float) + idx.minute.to_numpy(float) / 60.0 + lon / 15.0) % 24.0
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["lat"], d["lon"], d["elev"] = lat, lon, elev
    d = d.join(ring)
    if nb is not None:
        d = d.join(nb)
    d["centre_minus_ring3"] = d["kc_cams"] - d["ring3_kc_mean"]
    # temporal context of the satellite product (past AND future: this is reanalysis-style use, the
    # synthetic series are produced after the fact)
    kc = d["kc_cams"]
    for n, name in ((1, "m1"), (-1, "p1"), (2, "m2"), (-2, "p2")):
        d[f"kc_lag_{name}"] = kc.shift(n)
    d["kc_std5"] = kc.rolling(5, center=True, min_periods=3).std()
    d["kc_mean5"] = kc.rolling(5, center=True, min_periods=3).mean()
    if "day_mask" not in d:
        d["day_mask"] = d["cams_ghi_clear"] > 10.0
    day_kc = kc.where(d["day_mask"])
    grp = day_kc.groupby(idx.date)
    d["kc_daymean"] = grp.transform("mean")
    d["kc_daystd"] = grp.transform("std")
    for r in RADII:
        d[f"ring{r:g}_kc_lag_m1"] = d[f"ring{r:g}_kc_mean"].shift(1)
        d[f"ring{r:g}_kc_lag_p1"] = d[f"ring{r:g}_kc_mean"].shift(-1)
    d["kd_lag_m1"], d["kd_lag_p1"] = d["kd_cams"].shift(1), d["kd_cams"].shift(-1)
    for c in F.columns(F.FULL):
        d[c] = d[c].astype(np.float32)
    return d


def add_imputed_targets(d: pd.DataFrame, sid: str) -> pd.DataFrame:
    """OPTIONAL and OFF: fill measurement gaps with the transformer imputation as additional training
    targets. Rejected 2026-10-13 (imputed values are model output, not measurement); the imputed
    product is used as INPUT only, see neighbour_features. Kept for an ablation."""
    p = os.path.join(IMPUTED, f"Station_{sid}.parquet")
    for v in ("ghi", "dhi"):
        d[f"src_{v}"] = np.int8(0)
    if not os.path.exists(p):
        return d
    imp = pd.read_parquet(p, columns=["timestamp", "ist_tag", "ghi_imputed", "dhi_imputed"]).set_index("timestamp").sort_index()
    imp = imp.reindex(d.index)
    for v in ("ghi", "dhi"):
        fill = d[f"{v}_obs"].isna() & imp[f"{v}_imputed"].notna() & d["day_mask"].astype(bool) & imp["ist_tag"].fillna(False).astype(bool)
        d.loc[fill, f"{v}_obs"] = imp.loc[fill, f"{v}_imputed"].to_numpy()
        d.loc[fill, f"y_valid_{v}"] = True
        d.loc[fill, f"src_{v}"] = np.int8(1)
    d["dhi_obs"] = np.minimum(d["dhi_obs"], d["ghi_obs"])
    return d


def build_station(sid: str, rp: pd.DataFrame = None, coords: pd.DataFrame = None) -> pd.DataFrame:
    rp = ring_points() if rp is None else rp
    coords = station_table([sid]) if coords is None else coords
    d = pd.read_parquet(os.path.join(PROCESSED, f"Station_{sid}.parquet"), columns=PROCESSED_COLS)
    d = d.rename(columns=ERA5_RENAME)
    for v in ("ghi", "dhi"):
        d[f"src_{v}"] = np.int8(0)                     # targets are observed only (decision 2026-10-13)
    lat, lon, elev = (float(coords.loc[sid, c]) for c in ("lat", "lon", "elev"))
    nb = neighbour_features(lat, lon, d.index, neighbour_table(), coords_all(), exclude=[sid])
    d = assemble(d, ring_frame(sid, d.index, rp), lat, lon, elev, nb)
    d.insert(0, "station_id", sid)
    d = d[d["day_mask"]]
    return d[["station_id"] + F.TARGET_COLS + F.columns(F.FULL)].copy()


def refresh_neighbours(sid: str, table: pd.DataFrame = None, coords: pd.DataFrame = None) -> pd.DataFrame:
    """Rewrite only the neighbour columns of an existing station frame (after a change of the
    neighbour source or feature list) instead of rebuilding ring and ERA5 features."""
    table = neighbour_table() if table is None else table
    coords = coords_all() if coords is None else coords
    d = pd.read_parquet(dataset_path(sid))
    lat, lon = (float(coords.loc[sid, c]) for c in ("lat", "lon"))
    nb = neighbour_features(lat, lon, d.index, table, coords, exclude=[sid])
    d = d.drop(columns=[c for c in d.columns if c.startswith("nb")])
    d = d.join(nb[F.NEIGHBOURS])
    return d[["station_id"] + F.TARGET_COLS + F.columns(F.FULL)].copy()


# --------------------------------------------------------------- neighbour stations as INPUT

def coords_all() -> pd.DataFrame:
    """Coordinates of all stations. Served from the split table (lat/lon/elev are stored there), so
    the park application needs no station list from another repo; the DWD list is the fallback."""
    if os.path.exists(SPLITS):
        sp = pd.read_csv(SPLITS, dtype={"station_id": str}).set_index("station_id")
        if {"lat", "lon", "elev"} <= set(sp.columns):
            return sp[["lat", "lon", "elev"]].astype(float)
    return station_table(station_ids())


def neighbour_table(path: str = None, source: str = "raw") -> pd.DataFrame:
    """Wide 30-min table of all stations' measured clear-sky index, diffuse fraction and CAMS clear-sky
    index: columns <sid>_kc, <sid>_kd, <sid>_kcc. Built once and cached next to the dataset.
    source='raw' (default, decision 2026-10-13): the pure QC-passed observation, NaN at gaps - the
    imputer used other stations' measurements as input, so filled values could carry a test station's
    own measurement into its neighbour features; neighbour_source_study.py showed the filled share is
    worth nothing. source='filled': observed, gaps filled with the transformer imputation (kept for
    that comparison; the imputation product's ghi_observed is already observed-else-imputed)."""
    if source not in ("filled", "raw"):
        raise ValueError(source)
    suffix = "" if source == "filled" else "_raw"
    path = path or os.path.join(os.path.dirname(DATASET), f"neighbours{suffix}_30min.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    cols = {}
    for sid in station_ids():
        p = pd.read_parquet(os.path.join(PROCESSED, f"Station_{sid}.parquet"),
                            columns=["cams_ghi", "cams_ghi_clear", "day_mask", "y_valid_ghi", "y_valid_dhi"])
        if source == "filled":
            imp = pd.read_parquet(os.path.join(IMPUTED, f"Station_{sid}.parquet"),
                                  columns=["timestamp", "ghi_observed", "ghi_imputed", "dhi_observed", "dhi_imputed"]).set_index("timestamp").sort_index().reindex(p.index)
            ghi = imp["ghi_observed"].where(imp["ghi_observed"].notna(), imp["ghi_imputed"])
            dhi = imp["dhi_observed"].where(imp["dhi_observed"].notna(), imp["dhi_imputed"])
        else:
            imp = pd.read_parquet(os.path.join(IMPUTED, f"Station_{sid}.parquet"),
                                  columns=["timestamp", "ghi_raw", "dhi_raw"]).set_index("timestamp").sort_index().reindex(p.index)
            # same BSRN QC as the targets, so an outlier never enters as a neighbour value
            ghi = imp["ghi_raw"].where(p["y_valid_ghi"].astype(bool))
            dhi = imp["dhi_raw"].where(p["y_valid_dhi"].astype(bool) & p["y_valid_ghi"].astype(bool))
        clear = np.maximum(p["cams_ghi_clear"], 1.0)
        day = p["day_mask"].astype(bool)
        # kc clipped at 2: beyond that only twilight ratios of two small numbers (q99.9 of the raw
        # table is 1.8, max 4.5), which would dominate the MLP's standardisation
        cols[f"{sid}_kc"] = (ghi / clear).clip(0.0, 2.0).where(day).astype(np.float32)
        cols[f"{sid}_kd"] = (dhi / np.maximum(ghi, 1.0)).clip(0.0, 1.0).where(day).astype(np.float32)
        cols[f"{sid}_kcc"] = (p["cams_ghi"] / clear).where(day).astype(np.float32)
    t = pd.DataFrame(cols)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.to_parquet(path)
    return t


def nearest_stations(lat: float, lon: float, coords: pd.DataFrame, exclude: list = (), k: int = None) -> pd.DataFrame:
    k = k or F.N_NEIGHBOURS
    c = coords.drop(index=[s for s in exclude if s in coords.index])
    d = haversine_km(lat, lon, c["lat"].to_numpy(), c["lon"].to_numpy())
    order = np.argsort(d)[:k]
    return pd.DataFrame({"station_id": c.index[order], "dist_km": d[order]})


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def neighbour_features(lat: float, lon: float, idx: pd.DatetimeIndex, table: pd.DataFrame, coords: pd.DataFrame,
                       exclude: list = (), k: int = None) -> pd.DataFrame:
    """Measurement-network input at a point: the k nearest stations' measured kc and kd, their CAMS
    error (kc_obs - kc_cams) and distance, plus inverse-distance-weighted means. The target station
    itself is excluded (the model must never see its own measurement)."""
    k = k or F.N_NEIGHBOURS
    nb = nearest_stations(lat, lon, coords, exclude, k)
    out = pd.DataFrame(index=idx)
    errs, kcs, w = [], [], []
    for i, r in enumerate(nb.itertuples(index=False)):
        kc = table[f"{r.station_id}_kc"].reindex(idx)
        kcc = table[f"{r.station_id}_kcc"].reindex(idx)
        out[f"nb{i}_kc"] = kc
        out[f"nb{i}_kd"] = table[f"{r.station_id}_kd"].reindex(idx)
        out[f"nb{i}_err"] = kc - kcc
        out[f"nb{i}_dist_km"] = np.float32(r.dist_km)
        errs.append(out[f"nb{i}_err"])
        kcs.append(kc)
        w.append(1.0 / max(r.dist_km, 1.0))
    w = np.array(w) / np.sum(w)
    E, K = pd.concat(errs, axis=1), pd.concat(kcs, axis=1)
    out["nb_err_wmean"] = (E * w).sum(axis=1, min_count=1) / (E.notna() * w).sum(axis=1).replace(0, np.nan)
    out["nb_kc_wmean"] = (K * w).sum(axis=1, min_count=1) / (K.notna() * w).sum(axis=1).replace(0, np.nan)
    out["nb_err_std"] = E.std(axis=1)
    return out.astype(np.float32)


def dataset_path(sid: str) -> str:
    return os.path.join(DATASET, f"Station_{sid}.parquet")


def load_stations(ids: list, columns: list = None) -> pd.DataFrame:
    parts = [pd.read_parquet(dataset_path(s), columns=columns) for s in ids]
    return pd.concat(parts)


def valid_mask(df: pd.DataFrame, var: str, feature_cols: list) -> np.ndarray:
    """Target present and plausible (solar_bc QC), CAMS and ring present."""
    m = df[f"y_valid_{var}"].to_numpy(bool) & df["cams_ghi"].notna().to_numpy() & df["cams_ghi_clear"].notna().to_numpy()
    ring = [c for c in feature_cols if c.startswith("ring")]
    if ring:
        m &= df[ring[0]].notna().to_numpy()
    return m
