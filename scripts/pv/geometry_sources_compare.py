#!/usr/bin/env python
"""Geometry sources per park for the C1 benchmark (PV_Sim2Real, plan 2026-11-05, step 1) - EXPLORATORY.

For the six metered parks (pv1, pv2, pv3, pv5, pv6, pv7) and the 13 fleet parks of the paper table (accepted laser
measurement, list of 2026-10-19) every available geometry source is written into one table (park x source):
  profile              folded cross-row profile of the window method (read: v6 table / release CSV, not recomputed)
  table_plane          one plane per table on the flank (second estimator of the window method; the per-table planes
                       are recomputed with pv.geodata_windows.measure_window(detail=True) on the stored window cells)
  array_plane_huber    robust plane (Huber) through all surface cells 0.5-6 m above ground in the OSM outline (box
  array_plane_ransac   parks: union of the accepted windows); Lingfors 2025 / Lukac 2025
  hough_axis_a/_b      row axis from a Hough transform of the row-mask edges (numpy; skimage is not installed in
                       synthre); azimuth = axis +- 90 deg, both candidates; Edun 2021
  mbr_axis_a/_b        row axis from the minimum rotated rectangle of every row component (long edge), area-weighted
                       circular mean; azimuth = axis +- 90 deg; Stid 2025
  fixed_30_180         constant 30 deg / 180 deg
  pvlib_opt_clearsky   tilt of the maximum annual POA irradiance at azimuth 180 (Ineichen clear sky, year 2024, hourly)
  pvlib_opt_cams       the same with the corrected CAMS of the park (ghi_corr/dhi_corr, 30 min; fleet: calendar year
                       2024; metered parks: 2023-07-24..2024-05-31, the available window)
  register_class_mid   MaStR classes -> midpoint of parks_pv.geometry.TILT_RANGE (class, cohort), class-centre azimuth;
                       Ost-West -> pair at the midpoint of EW_TILT
  register_v1_nominal  fleet only: the nominal mounts of release v1 (pv_arrays.csv, share-weighted)
  operator             metered parks only: operator master data (tilt/azimuth per row, kW-weighted)
  meter_estimator_312d metered parks only: estimator from the measured power, 312-day column of
                       FL_Contribution/reports/pv_angle_decomposition/tables/tab_angles.tex (read, not recomputed)

Nothing is written outside --out (default results/geometry_sources_2026-11-05/): geometry_sources.csv, per-park JSON
details, twins/arrays_<source>.csv (input tables for scripts/pv/trianel_twins.py --geodata-csv) and, with
--twins-summary, twins/summary.csv. Tiles are read from the existing caches; the window CSVs of the stored runs are
read, not rewritten. No coordinates in the CSV outputs.

    synthre/bin/python scripts/pv/geometry_sources_compare.py --parks pv2
    synthre/bin/python scripts/pv/geometry_sources_compare.py --all
    synthre/bin/python scripts/pv/geometry_sources_compare.py --twins-summary
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks_pv import paths  # noqa: E402
from parks_pv.geometry import EW_TILT, TILT_RANGE  # noqa: E402
from pv import geodata_tiles as GT, geodata_windows as GW  # noqa: E402
from pv.plausibility import AZ_CENTRE  # noqa: E402
from parks_pv import geometry_estimate as R  # noqa: E402   (fleet helpers, used read-only)
import geometry_trianel_windows as TW  # noqa: E402  (prepare(pv))

OUT_DEFAULT = os.path.join(REPO, "results", "geometry_sources_2026-11-05")
TRIANEL = ["pv1", "pv2", "pv3", "pv5", "pv6", "pv7"]
# the 13 fleet parks of the paper table (pv_sim2real/results/geodata_release_2026-10-19/accepted_laser.csv)
FLEET13 = ["SEL981208943378", "SEL966692449001", "SEL981828226431", "SEL946560347417", "SEL913725723355", "SEL913435527427",
           "SEL920468626324", "SEL969089055946", "SEL983009290071", "SEL946686841516", "SEL984175261216", "SEL954870838563",
           "SEL963887911169"]
FLEET_RUN_DEFAULT = "_v2e"                                   # window-method run the profile is read from
FLEET_RUN = {"SEL946560347417": "_v2", "SEL969089055946": "_v2"}   # not in _v2e (left the selection); _v2 is the last window run with them
V6_TABLE = os.path.join(TW.FIGS, "geodata_arrays_trianel_windows_v6.csv")
TRIANEL_WIN_DIR = os.path.join(REPO, "results", "pv", "geodata_trianel_windows")
SELECTION_V1 = os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv")
ESTIMATOR_312D = {"pv1": (20.0, 180.0), "pv2": (25.0, 180.0), "pv3": (15.0, 180.0), "pv5": (10.0, 170.0), "pv6": (10.0, 180.0), "pv7": (10.0, 180.0)}
HAG_MIN, HAG_MAX = 0.5, 6.0
CAMS_YEAR_FLEET = ("2024-01-01", "2025-01-01")
CAMS_WINDOW_TRIANEL = ("2023-07-24", "2024-06-01")
TWIN_SOURCES = ["profile", "table_plane", "array_plane_huber", "fixed_30_180", "pvlib_opt_cams", "pvlib_opt_clearsky",
                "register_class_mid", "operator", "meter_estimator_312d"]


# ------------------------------------------------------------------------------------------------ small helpers
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cohort(year: int) -> str:
    return "<=2012" if year <= 2012 else "2013-20" if year <= 2020 else "2021+"


def az_diff(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def circ_mean(deg, w=None, period: float = 360.0) -> float:
    deg = np.asarray(deg, float)
    w = np.ones(len(deg)) if w is None else np.asarray(w, float)
    k = 2 * np.pi / period                                                  # degrees -> radians of the (possibly doubled) angle
    return float((np.arctan2((w * np.sin(k * deg)).sum(), (w * np.cos(k * deg)).sum()) / k) % period)


def plane_angles(a: float, b: float, conv: float) -> tuple:
    """z = a x + b y + c on the grid (x east, y north) -> (tilt deg, true azimuth of the downslope direction)."""
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    az = float((np.degrees(np.arctan2(-a, -b)) + conv) % 360.0)
    return tilt, az


def layout_of(arrays: list) -> str:
    """'east-west pair' when two arrays face opposite ways (180 +- 25 deg), else 'single'."""
    az = [float(x["azimuth"]) for x in arrays if np.isfinite(x["azimuth"])]
    for i in range(len(az)):
        for k in range(i + 1, len(az)):
            if abs(abs(az_diff(az[i], az[k])) - 180.0) <= 25.0:
                return "east-west pair"
    return "single" if az else "n/a"


def grid_xy(shape: tuple, ii: np.ndarray, jj: np.ndarray, res: float) -> tuple:
    return (jj + 0.5) * res, (shape[0] - ii - 0.5) * res


def true_bool(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().eq("true").values


# ------------------------------------------------------------------------------------------------ new estimators
def array_plane(hag: np.ndarray, dgm: np.ndarray, mask: np.ndarray, conv: float, res: float, max_pts: int = 300_000) -> dict:
    """Robust planes through the surface cells 0.5-6 m above ground inside mask: Huber and RANSAC on the absolute
    surface (tilt against the horizontal), plus the Huber plane on the height above ground and the terrain plane."""
    from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor
    m = mask & np.isfinite(hag) & (hag > HAG_MIN) & (hag < HAG_MAX) & np.isfinite(dgm)
    ii, jj = np.nonzero(m)
    out = dict(n_cells=int(len(ii)), area_m2=float(len(ii) * res * res))
    if len(ii) < 500:
        out["error"] = "fewer than 500 cells above ground"
        return out
    rng = np.random.default_rng(0)
    if len(ii) > max_pts:
        s = rng.choice(len(ii), max_pts, replace=False)
        ii, jj = ii[s], jj[s]
    x, y = grid_xy(hag.shape, ii, jj, res)
    A = np.column_stack([x - x.mean(), y - y.mean()])
    z_surf = (hag[ii, jj] + dgm[ii, jj]).astype(float)
    z_hag = hag[ii, jj].astype(float)
    hub = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(A, z_surf - z_surf.mean())
    t, az = plane_angles(hub.coef_[0], hub.coef_[1], conv)
    out.update(huber_tilt=t, huber_az=az, huber_rmse=float(np.sqrt(np.mean((hub.predict(A) - (z_surf - z_surf.mean())) ** 2))))
    ran = RANSACRegressor(LinearRegression(), residual_threshold=0.30, min_samples=50, max_trials=300, random_state=0).fit(A, z_surf)
    t, az = plane_angles(ran.estimator_.coef_[0], ran.estimator_.coef_[1], conv)
    out.update(ransac_tilt=t, ransac_az=az, ransac_inlier_share=float(ran.inlier_mask_.mean()))
    hub2 = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(A, z_hag - z_hag.mean())
    t, az = plane_angles(hub2.coef_[0], hub2.coef_[1], conv)
    out.update(huber_hag_tilt=t, huber_hag_az=az)
    # terrain plane over the same cells (DGM1)
    coef, *_ = np.linalg.lstsq(np.column_stack([A, np.ones(len(A))]), dgm[ii, jj].astype(float), rcond=None)
    t, az = plane_angles(coef[0], coef[1], conv)
    out.update(terrain_tilt=t, terrain_az=az)
    return out


def hough_axis(mask: np.ndarray, conv: float, res: float, step_deg: float = 0.5, rho_m: float = 0.5, max_pts: int = 80_000) -> dict:
    """Row axis from a Hough transform (numpy) of the edges of the row mask: for every direction theta (compass of
    the line direction, 0-180) the points are projected on the normal, rho = x cos(theta) - y sin(theta) (the
    convention of measure_window); the direction with the strongest accumulator peaks (mean of the 10 fullest rho
    bins) is the row axis. ratio2 = best score of a direction >= 10 deg away / best score."""
    from scipy.ndimage import binary_erosion
    edge = mask & ~binary_erosion(mask, iterations=1)
    ii, jj = np.nonzero(edge)
    out = dict(n_edge=int(len(ii)))
    if len(ii) < 200:
        out["error"] = "fewer than 200 edge cells"
        return out
    rng = np.random.default_rng(0)
    if len(ii) > max_pts:
        s = rng.choice(len(ii), max_pts, replace=False)
        ii, jj = ii[s], jj[s]
    x, y = grid_xy(mask.shape, ii, jj, res)
    thetas = np.arange(0.0, 180.0, step_deg)
    scores = np.zeros(len(thetas))
    for k, t in enumerate(thetas):
        tr = np.radians(t)
        rho = x * np.cos(tr) - y * np.sin(tr)
        h = np.bincount(np.floor((rho - rho.min()) / rho_m).astype(int))
        scores[k] = np.sort(h)[-10:].mean()
    k0 = int(np.argmax(scores))
    far = np.abs(((thetas - thetas[k0]) + 90.0) % 180.0 - 90.0) >= 10.0
    out.update(axis_grid=float(thetas[k0]), axis_true=float((thetas[k0] + conv) % 180.0), score=float(scores[k0]),
               ratio2=float(scores[far].max() / scores[k0]) if far.any() else np.nan, n_used=int(len(ii)))
    return out


def mbr_axis(mask: np.ndarray, conv: float, res: float, min_m2: float = 40.0, max_pts: int = 4000) -> dict:
    """Row axis from the minimum rotated rectangle of every connected row component (>= min_m2): the long edge gives
    the axis, the area-weighted circular mean over the components the park axis; the short edge is the row width."""
    from scipy.ndimage import find_objects, label
    from shapely.geometry import MultiPoint
    lab, n = label(mask, structure=np.ones((3, 3), bool))
    out = dict(n_components_all=int(n))
    if n == 0:
        out["error"] = "no component"
        return out
    rng = np.random.default_rng(0)
    rows = []
    for c, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == c
        area = float(sub.sum() * res * res)
        if area < min_m2:
            continue
        ii, jj = np.nonzero(sub)
        ii, jj = ii + sl[0].start, jj + sl[1].start
        if len(ii) > max_pts:
            s = rng.choice(len(ii), max_pts, replace=False)
            ii, jj = ii[s], jj[s]
        x, y = grid_xy(mask.shape, ii, jj, res)
        rect = MultiPoint(list(zip(x, y))).convex_hull.minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            continue
        p = np.asarray(rect.exterior.coords)[:4]
        e1, e2 = p[1] - p[0], p[2] - p[1]
        l1, l2 = np.hypot(*e1), np.hypot(*e2)
        long_e, long_l, short_l = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
        rows.append(dict(area_m2=area, axis_grid=float(np.degrees(np.arctan2(long_e[0], long_e[1])) % 180.0), long_m=float(long_l), short_m=float(short_l)))
    if not rows:
        out["error"] = f"no component >= {min_m2:.0f} m2"
        return out
    df = pd.DataFrame(rows)
    ax = circ_mean(df.axis_grid, df.area_m2, period=180.0)
    dev = np.abs(((df.axis_grid - ax) + 90.0) % 180.0 - 90.0)
    out.update(axis_grid=ax, axis_true=float((ax + conv) % 180.0), n_components=int(len(df)), axis_dev_median=float(np.median(dev)),
               axis_dev_p90=float(np.percentile(dev, 90)), short_m_median=float(df.short_m.median()), long_m_median=float(df.long_m.median()),
               elongation_median=float((df.long_m / df.short_m.clip(lower=0.5)).median()))
    return out


def table_planes(hag: np.ndarray, dgm: np.ndarray, win: pd.DataFrame, nw: int, conv: float, source: str, res: float, tent: bool) -> dict:
    """Per-table planes of the accepted windows, recomputed with measure_window(detail=True) on the stored cells.
    Returns the planes (window, row, cells, tilt vs ground, true azimuth), the horizontal per-window medians
    (tilt_rows_h of the stored table) and a consistency check (recomputed vs stored tilt_profile)."""
    ok = win[true_bool(win.ok)]
    planes, checks, tilt_h = [], [], []
    for r in ok.itertuples():
        cell = (int(r.i0), int(r.j0), float(r.f_in), float(r.f_tab))
        try:
            res_w, det = GW.measure_window(hag, dgm, cell, nw, conv, source, res, w=int(r.w), detail=True, tent=tent)
        except Exception as ex:  # noqa: BLE001
            checks.append(dict(w=int(r.w), error=f"{type(ex).__name__}: {str(ex)[:60]}"))
            continue
        if np.isfinite(res_w.get("tilt_profile", np.nan)) and np.isfinite(r.tilt_profile):
            checks.append(dict(w=int(r.w), d_tilt_profile=float(res_w["tilt_profile"] - r.tilt_profile), d_axis=float(az_diff(res_w.get("axis_true", np.nan), r.axis_true))))
        Rdf = det.get("R")
        if Rdf is not None and len(Rdf):
            for p in Rdf.itertuples():
                planes.append(dict(w=int(r.w), row=int(p.row), cells=int(p.cells), tilt_ground=float(p.tilt), az_true=float(p.az_true)))
        if np.isfinite(r.tilt_rows_h):
            tilt_h.append(float(r.tilt_rows_h))
    out = dict(n_windows_ok=int(len(ok)), n_planes=len(planes), tilt_rows_h_median=float(np.median(tilt_h)) if tilt_h else np.nan)
    chk = pd.DataFrame([c for c in checks if "d_tilt_profile" in c])
    out["check_max_abs_d_tilt_profile"] = float(chk.d_tilt_profile.abs().max()) if len(chk) else np.nan
    out["check_max_abs_d_axis"] = float(chk.d_axis.abs().max()) if len(chk) else np.nan
    out["errors"] = [c for c in checks if "error" in c][:5]
    if planes:
        P = pd.DataFrame(planes)
        east = P[(P.az_true % 360.0) < 180.0]
        west = P[(P.az_true % 360.0) >= 180.0]
        she = float(east.cells.sum() / P.cells.sum())
        out.update(tilt_ground_median=float(np.average(P.tilt_ground, weights=P.cells)), share_east=she,
                   az_all=circ_mean(P.az_true, P.cells),
                   az_east=circ_mean(east.az_true, east.cells) if len(east) else np.nan, tilt_east=float(np.median(east.tilt_ground)) if len(east) else np.nan,
                   az_west=circ_mean(west.az_true, west.cells) if len(west) else np.nan, tilt_west=float(np.median(west.tilt_ground)) if len(west) else np.nan)
        out["pair"] = bool(0.25 <= she <= 0.75 and np.isfinite(out["az_east"]) and abs(abs(az_diff(out["az_east"], out["az_west"])) - 180.0) <= 35.0)
    return out


def pvlib_optimum(lat: float, lon: float, alt: float, times: pd.DatetimeIndex, ghi: np.ndarray, dhi: np.ndarray, dni: np.ndarray = None) -> dict:
    """Tilt (0..60 deg, 1-deg steps) of the maximum POA irradiance sum at azimuth 180 (Hay-Davies, pvlib)."""
    import pvlib
    loc = pvlib.location.Location(lat, lon, altitude=alt or 0.0)
    sp = loc.get_solarposition(times)
    ghi, dhi = pd.Series(ghi, index=times).clip(lower=0.0), pd.Series(dhi, index=times).clip(lower=0.0)
    if dni is None:                                                         # GHI/DHI -> DNI, capped by the clear-sky DNI (low sun)
        dni_clear = loc.get_clearsky(times, model="ineichen").dni
        dni = pvlib.irradiance.dni(ghi, dhi, sp.zenith, dni_clear=dni_clear, clearsky_tolerance=1.1).fillna(0.0).clip(lower=0.0)
    else:
        dni = pd.Series(dni, index=times).clip(lower=0.0)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    day = sp.apparent_zenith < 90.0
    sums = {}
    for tilt in range(0, 61):
        poa = pvlib.irradiance.get_total_irradiance(tilt, 180.0, sp.apparent_zenith[day], sp.azimuth[day], dni[day], ghi[day], dhi[day],
                                                    dni_extra=dni_extra[day], model="haydavies")
        sums[tilt] = float(poa["poa_global"].sum())
    best = max(sums, key=sums.get)
    return dict(tilt=float(best), gain_vs_0=float(sums[best] / sums[0] - 1.0), gain_vs_30=float(sums[best] / sums[30] - 1.0),
                within_1pct=[t for t in sums if sums[t] >= 0.99 * sums[best]], n_steps=int(len(times)))


def pvlib_rows(park: str, lat: float, lon: float, alt: float, cams_path: str, window: tuple) -> tuple:
    rows, det = [], {}
    import pvlib
    times = pd.date_range("2024-01-01", "2025-01-01", freq="1h", tz="UTC", inclusive="left")
    cs = pvlib.location.Location(lat, lon, altitude=alt or 0.0).get_clearsky(times, model="ineichen")
    o = pvlib_optimum(lat, lon, alt, times, cs.ghi.values, cs.dhi.values, cs.dni.values)
    det["clearsky"] = o
    rows.append(dict(park=park, source="pvlib_opt_clearsky", array="a0", tilt_deg=o["tilt"], azimuth_deg=180.0, layout="single", share=1.0,
                     note=f"Ineichen clear sky 2024 hourly; +{o['gain_vs_30'] * 100:.1f} % vs 30 deg; within 1 %: {min(o['within_1pct'])}-{max(o['within_1pct'])} deg"))
    if cams_path and os.path.exists(cams_path):
        c = pd.read_parquet(cams_path)
        c = c[(c.index >= pd.Timestamp(window[0], tz="UTC")) & (c.index < pd.Timestamp(window[1], tz="UTC"))]
        c = c[["ghi_corr", "dhi_corr"]].dropna()
        if len(c) > 1000:
            t_mid = c.index + pd.Timedelta(minutes=15)          # START labels of 30-min means -> interval centre
            o = pvlib_optimum(lat, lon, alt, t_mid, c.ghi_corr.values, c.dhi_corr.values)
            det["cams"] = dict(o, window=list(window), n=int(len(c)))
            rows.append(dict(park=park, source="pvlib_opt_cams", array="a0", tilt_deg=o["tilt"], azimuth_deg=180.0, layout="single", share=1.0,
                             note=f"corrected CAMS xgb_v5 {window[0]}..{window[1]} ({len(c)} steps); +{o['gain_vs_30'] * 100:.1f} % vs 30 deg; "
                                  f"within 1 %: {min(o['within_1pct'])}-{max(o['within_1pct'])} deg"))
        else:
            rows.append(dict(park=park, source="pvlib_opt_cams", array="a0", tilt_deg=np.nan, azimuth_deg=np.nan, layout="n/a", share=1.0, note="corrected CAMS too short"))
    else:
        rows.append(dict(park=park, source="pvlib_opt_cams", array="a0", tilt_deg=np.nan, azimuth_deg=np.nan, layout="n/a", share=1.0, note="no corrected CAMS file"))
    return rows, det


def register_class_arrays(orient_cls: str, tilt_cls: str, coh: str) -> tuple:
    lo, hi = TILT_RANGE.get((tilt_cls, coh), TILT_RANGE[("20 - 40 Grad", coh)])
    if orient_cls == "Ost-West":
        t = 0.5 * (EW_TILT[0] + EW_TILT[1])
        return [dict(name="e", tilt=t, azimuth=90.0, share=0.5), dict(name="w", tilt=t, azimuth=270.0, share=0.5)], f"MaStR {orient_cls} / {tilt_cls}; EW_TILT midpoint"
    az = AZ_CENTRE.get(orient_cls, np.nan)
    return [dict(name="a0", tilt=0.5 * (lo + hi), azimuth=az, share=1.0)], f"MaStR {orient_cls} / {tilt_cls}, cohort {coh}: TILT_RANGE {lo:g}-{hi:g} midpoint, class centre"


# ------------------------------------------------------------------------------------------------ inputs per park
def trianel_inputs(pv: str) -> dict:
    """Rasters and metadata of a metered park from geometry_trianel_windows.prepare + the stored window run (v6)."""
    P = TW.prepare(pv, with_rgb=False)
    win = pd.read_csv(os.path.join(TRIANEL_WIN_DIR, f"{pv}_windows.csv"))
    park = json.load(open(os.path.join(TRIANEL_WIN_DIR, f"{pv}_park.json")))
    v6 = pd.read_csv(V6_TABLE)
    v6 = v6[v6.park_id == pv]
    profile = [dict(name=str(a.array_id), tilt=float(a.tilt), azimuth=float(a.azimuth), share=float(a.share)) for a in v6.itertuples()]
    rows = P["rows"]
    operator = []
    for r in rows.itertuples():
        operator.append(dict(tilt=0.5 * (float(r.tilt0) + float(r.tilt1)), azimuth=0.5 * (float(r.az0) + float(r.az1)), kw=float(r.kw)))
    op = pd.DataFrame(operator)
    op = op.groupby(["tilt", "azimuth"], as_index=False).kw.sum()
    op["share"] = op.kw / op.kw.sum()
    try:
        alt = float(TW.TC.dem_altitude(P["lat"], P["lon"]))
    except Exception:  # noqa: BLE001
        alt = 0.0
    return dict(kind="trianel", park=pv, hag=P["hag"], dgm=P["dgm"], inside=P["inside"], conv=float(P["conv"]), source=P["source"],
                win=win, park_json=park, nw=int(round(float(park["win_m"]) / GW.RES)), box=False, footprint=P["fp_src"],
                profile=profile, profile_note=f"v6 table; {P['source']}, survey {P['survey_year']}, park {park['status']}, tilt_med {park['tilt_med']:.1f}",
                tilt_rows_med=float(park.get("tilt_rows_med", np.nan)), register_orientation=P["register_orientation"],
                register_tilt_class=P["register_tilt_class"], year_built=int(P["year_built"]), lat=float(P["lat"]), lon=float(P["lon"]), alt=alt,
                cams_path=os.path.join(TW.B, "cams_corrected", f"{pv}_cams_corrected_30min_xgb_v5.parquet"), cams_window=CAMS_WINDOW_TRIANEL,
                operator=[dict(name=f"op{k}", tilt=float(r.tilt), azimuth=float(r.azimuth), share=float(r.share)) for k, r in enumerate(op.itertuples())],
                tent=bool(GW.RULES["tent_pair"] and P["register_orientation"] == "Ost-West"))


def fleet_inputs(pid: str) -> dict:
    """Rasters and metadata of a fleet park: selection v1 row, cached tiles, the footprint of the stored run and its
    window table (run _v2e, or _v2 for the two parks that left the selection)."""
    from pyproj import Transformer
    from shapely.geometry import Point
    sel = pd.read_csv(SELECTION_V1, dtype={"park_id": str})
    r = next(sel[sel.park_id == pid].itertuples())
    state = r.bundesland
    code, zone = R.STATE[state]
    run = FLEET_RUN.get(pid, FLEET_RUN_DEFAULT)
    rel = pd.read_csv(os.path.join(paths.DATA_DIR, f"geodata_release_{state.lower()}{run}.csv"), dtype={"pv": str})
    rel = rel[rel.pv == pid]
    best = rel[true_bool(rel.best) & true_bool(rel.accept) & true_bool(rel.laser)] if len(rel) else rel
    if not len(best):
        best = rel[rel.source.isin(R.LASER)]
    if not len(best):
        raise RuntimeError(f"{pid}: no laser row in run {run}")
    b = best.iloc[0]
    src = str(b.source)
    outd = os.path.join(REPO, "results", "pv", f"geodata_release{run}", pid)
    win = pd.read_csv(os.path.join(outd, f"windows_{src}.csv"))
    park = json.load(open(os.path.join(outd, f"park_{src}.json")))
    arr_csv = os.path.join(outd, f"arrays_{src}.csv")
    arrays = pd.read_csv(arr_csv) if os.path.exists(arr_csv) else pd.DataFrame()
    profile = [dict(name=f"g{int(a.group)}", tilt=float(a.tilt), azimuth=float(a.azimuth), share=float(a.share)) for a in arrays.itertuples()]
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    pt = Point(*to_m(float(r.lon), float(r.lat)))
    margin = R.fetch_margin(r, osm, zone)
    tiles = R.fetch_tiles(state, float(r.lat), float(r.lon), int(r.year), r.landkreis, None, margin_m=margin)
    paths_ = dict(R.sources_from(tiles)).get(src)
    if not paths_:
        raise RuntimeError(f"{pid}: source {src} not in the tile cache")
    box = bool(park.get("box", False))
    if box:
        s = float(park["box_side_m"])
        bounds, inside = (pt.x - s / 2, pt.y - s / 2, pt.x + s / 2, pt.y + s / 2), None
    else:
        poly, _, _ = R.footprint(r, osm, to_m, pt)
        if poly is None:
            raise RuntimeError(f"{pid}: stored run used an OSM outline, none found now")
        g = GW.analysis_grid(poly, GW.RES)
        bounds, inside = g["bounds"], g["inside"]
    RR = R.rasters(tiles, src, paths_, bounds, zone, float(r.lat), float(r.lon))
    if inside is None:
        inside = np.ones(RR["hag"].shape, bool)
    v1 = pd.read_csv(os.path.join(paths.WORK_ROOT, "data", "parks_pv_v1", "pv_arrays.csv"), dtype={"park_id": str})   # register arrays of parks_pv_v1
    v1 = v1[v1.park_id == pid]
    reg_v1 = [dict(name=str(nom), tilt=float(g.mount_tilt.iloc[0]), azimuth=float(g.mount_azimuth.iloc[0]), share=float(g.share.sum()))
              for nom, g in v1.groupby(v1.array_id.str.rsplit("_", n=1).str[0], sort=False)]     # one entry per nominal mount (terrain clusters summed)
    return dict(kind="fleet", park=pid, hag=RR["hag"], dgm=RR["dgm"], inside=inside, conv=float(RR["conv"]), source=src, win=win, park_json=park,
                nw=int(round(float(park["win_m"]) / GW.RES)), box=box, footprint=park.get("footprint", "box" if box else "osm"), run=run,
                profile=profile, profile_note=f"run {run}, {src}, status {b.status}, accept {b.accept}, survey {b.survey_year}, tilt_measured {b.tilt_measured}, az {b.az_measured}, ew {b.ew_measured}",
                tilt_rows_med=float(park.get("tilt_rows_med", np.nan)), register_orientation=str(r.orientation_class), register_tilt_class=str(r.tilt_class),
                year_built=int(r.year), lat=float(r.lat), lon=float(r.lon), alt=float(r.elevation) if np.isfinite(r.elevation) else 0.0,
                cams_path=os.path.join(paths.CORR_DIR, f"{pid}_cams_corrected_30min_xgb_v5.parquet"), cams_window=CAMS_YEAR_FLEET,
                register_v1=reg_v1, state_code=code, tent=bool(("tent_pair" in str(b.get("rules", ""))) and str(r.orientation_class) == "Ost-West"))


# ------------------------------------------------------------------------------------------------ one park
def run_park(park: str, out_dir: str) -> list:
    t0 = time.time()
    I = trianel_inputs(park) if park.startswith("pv") else fleet_inputs(park)
    hag, dgm, inside, conv, res, nw = I["hag"], I["dgm"], I["inside"], I["conv"], GW.RES, I["nw"]
    win = I["win"]
    det = dict(park=park, kind=I["kind"], source=I["source"], footprint=I["footprint"], box=I["box"], conv=conv, hag_shape=list(hag.shape),
               n_windows=int(len(win)), n_ok=int(true_bool(win.ok).sum()) if len(win) else 0, run=I.get("run"))
    # grid consistency with the stored window table
    if len(win):
        det["grid_fits"] = bool(win.i0.max() + nw <= hag.shape[0] and win.j0.max() + nw <= hag.shape[1])
    rows = []
    prof = I["profile"]
    p_lay = layout_of(prof)
    p_tilt = float(np.average([a["tilt"] for a in prof], weights=[a["share"] for a in prof])) if prof else np.nan
    p_az = [a["azimuth"] for a in prof]

    def add(source, arrays, layout, note):
        for a in arrays:
            t, az = float(a.get("tilt", np.nan)), float(a.get("azimuth", np.nan))
            daz = min((abs(az_diff(az, q)) for q in p_az), default=np.nan) if np.isfinite(az) and p_az else np.nan
            rows.append(dict(park=park, source=source, array=str(a.get("name", "a0")), tilt_deg=round(t, 2) if np.isfinite(t) else np.nan,
                             azimuth_deg=round(az, 1) if np.isfinite(az) else np.nan, layout=layout, share=round(float(a.get("share", 1.0)), 4), note=note,
                             dtilt_profile_deg=round(t - p_tilt, 2) if np.isfinite(t) and np.isfinite(p_tilt) else np.nan,
                             daz_profile_deg=round(daz, 1) if np.isfinite(daz) else np.nan))

    # 1 profile (read)
    add("profile", prof if prof else [dict(name="a0", tilt=np.nan, azimuth=np.nan, share=1.0)], p_lay if prof else "n/a", I["profile_note"])
    # 2 table planes (recomputed per table on the stored cells)
    try:
        tp = table_planes(hag, dgm, win, nw, conv, I["source"], res, I["tent"]) if len(win) else dict(error="no window table")
        det["table_plane"] = tp
        if tp.get("n_planes"):
            note = (f"{tp['n_planes']} table planes in {tp['n_windows_ok']} accepted windows; tilt = median tilt_rows_h (horizontal), vs ground {tp['tilt_ground_median']:.1f}; "
                    f"check vs stored tilt_profile max |d| {tp['check_max_abs_d_tilt_profile']:.2f} deg, axis {tp['check_max_abs_d_axis']:.2f} deg")
            if tp["pair"]:
                add("table_plane", [dict(name="e", tilt=tp["tilt_rows_h_median"], azimuth=tp["az_east"], share=tp["share_east"]),
                                    dict(name="w", tilt=tp["tilt_rows_h_median"], azimuth=tp["az_west"], share=1 - tp["share_east"])], "east-west pair",
                    note + f"; tilt east/west vs ground {tp['tilt_east']:.1f}/{tp['tilt_west']:.1f}")
            elif p_lay == "east-west pair":                      # the per-table plane is fitted on the flank chosen by the profile only
                add("table_plane", [dict(name="e", tilt=tp["tilt_rows_h_median"], azimuth=tp["az_all"] % 360.0, share=0.5),
                                    dict(name="w", tilt=tp["tilt_rows_h_median"], azimuth=(tp["az_all"] + 180.0) % 360.0, share=0.5)],
                    "east-west pair (layout from profile)",
                    note + f"; planes fitted on the chosen flank only (facing {tp['az_all']:.0f}); pair completed from the profile layout")
            else:
                add("table_plane", [dict(name="a0", tilt=tp["tilt_rows_h_median"], azimuth=tp["az_all"], share=1.0)], "single flank",
                    note + f"; share of planes facing 0-180: {tp['share_east']:.2f}")
        else:
            add("table_plane", [dict(name="a0", tilt=I["tilt_rows_med"], azimuth=np.nan, share=1.0)], "n/a", f"no planes recomputed ({tp.get('error', tp.get('errors'))}); tilt_rows_med of the stored run")
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        add("table_plane", [dict(name="a0", tilt=I["tilt_rows_med"], azimuth=np.nan, share=1.0)], "n/a", f"error {type(ex).__name__}: {str(ex)[:80]}; tilt_rows_med of the stored run")
    # mask for the array-level estimators: OSM outline, box parks the union of the accepted windows
    if I["box"] and len(win) and true_bool(win.ok).any():
        mask = np.zeros(hag.shape, bool)
        for r in win[true_bool(win.ok)].itertuples():
            mask[int(r.i0):int(r.i0) + nw, int(r.j0):int(r.j0) + nw] = True
        mask_note = f"mask = union of {int(true_bool(win.ok).sum())} accepted windows (box park)"
    else:
        mask, mask_note = inside.copy(), f"mask = {I['footprint']} outline"
    det["mask_note"] = mask_note
    rowmask = mask & np.isfinite(hag) & (hag > HAG_MIN) & (hag < HAG_MAX)
    det["rowmask_area_m2"] = float(rowmask.sum() * res * res)
    # 3 array plane
    try:
        ap = array_plane(hag, dgm, mask, conv, res)
        det["array_plane"] = ap
        if "error" in ap:
            add("array_plane_huber", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", ap["error"])
            add("array_plane_ransac", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", ap["error"])
        else:
            add("array_plane_huber", [dict(tilt=ap["huber_tilt"], azimuth=ap["huber_az"])], "single plane",
                f"{mask_note}; {ap['n_cells']} cells 0.5-6 m above ground; surface plane, rmse {ap['huber_rmse']:.2f} m; plane on height above ground {ap['huber_hag_tilt']:.1f} deg / {ap['huber_hag_az']:.0f}; terrain plane {ap['terrain_tilt']:.1f} deg / {ap['terrain_az']:.0f}")
            add("array_plane_ransac", [dict(tilt=ap["ransac_tilt"], azimuth=ap["ransac_az"])], "single plane",
                f"{mask_note}; residual 0.30 m, inliers {ap['ransac_inlier_share']:.2f}, seed 0")
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        add("array_plane_huber", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", f"error {type(ex).__name__}: {str(ex)[:80]}")
    # 4 Hough axis
    try:
        ho = hough_axis(rowmask, conv, res)
        det["hough"] = ho
        if "error" in ho:
            add("hough_axis_a", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", ho["error"])
        else:
            note = f"{mask_note}; axis {ho['axis_true']:.1f} deg true from {ho['n_used']} edge cells; score ratio of the next direction >= 10 deg away {ho['ratio2']:.2f}; azimuth = axis +- 90"
            add("hough_axis_a", [dict(name="a", tilt=np.nan, azimuth=(ho["axis_true"] + 90.0) % 360.0)], "axis only", note)
            add("hough_axis_b", [dict(name="b", tilt=np.nan, azimuth=(ho["axis_true"] + 270.0) % 360.0)], "axis only", note)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        add("hough_axis_a", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", f"error {type(ex).__name__}: {str(ex)[:80]}")
    # 5 minimum rotated rectangle per row component
    try:
        mb = mbr_axis(rowmask, conv, res)
        det["mbr"] = mb
        if "error" in mb:
            add("mbr_axis_a", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", mb["error"])
        else:
            note = (f"{mask_note}; {mb['n_components']} components >= 40 m2 (of {mb['n_components_all']}); axis {mb['axis_true']:.1f} deg true, "
                    f"deviation median/p90 {mb['axis_dev_median']:.1f}/{mb['axis_dev_p90']:.1f} deg; short edge median {mb['short_m_median']:.1f} m, elongation {mb['elongation_median']:.1f}; azimuth = axis +- 90")
            add("mbr_axis_a", [dict(name="a", tilt=np.nan, azimuth=(mb["axis_true"] + 90.0) % 360.0)], "axis only", note)
            add("mbr_axis_b", [dict(name="b", tilt=np.nan, azimuth=(mb["axis_true"] + 270.0) % 360.0)], "axis only", note)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        add("mbr_axis_a", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", f"error {type(ex).__name__}: {str(ex)[:80]}")
    # 6 fixed
    add("fixed_30_180", [dict(tilt=30.0, azimuth=180.0)], "single", "constant")
    # 7 pvlib optimum
    try:
        pr, pdet = pvlib_rows(park, I["lat"], I["lon"], I["alt"], I["cams_path"], I["cams_window"])
        det["pvlib"] = pdet
        for x in pr:
            x["dtilt_profile_deg"] = round(x["tilt_deg"] - p_tilt, 2) if np.isfinite(x["tilt_deg"]) and np.isfinite(p_tilt) else np.nan
            x["daz_profile_deg"] = round(min((abs(az_diff(x["azimuth_deg"], q)) for q in p_az), default=np.nan), 1) if np.isfinite(x["azimuth_deg"]) and p_az else np.nan
        rows.extend(pr)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        add("pvlib_opt_clearsky", [dict(tilt=np.nan, azimuth=np.nan)], "n/a", f"error {type(ex).__name__}: {str(ex)[:80]}")
    # 8 register
    arr, note = register_class_arrays(I["register_orientation"], I["register_tilt_class"], cohort(I["year_built"]))
    add("register_class_mid", arr, layout_of(arr), note)
    if I["kind"] == "fleet":
        add("register_v1_nominal", I["register_v1"], layout_of(I["register_v1"]), "release v1 nominal mounts (pv_arrays.csv mount_tilt/mount_azimuth)")
    # 9 operator, 10 estimator
    if I["kind"] == "trianel":
        add("operator", I["operator"], layout_of(I["operator"]), "operator master data, kW-weighted per (tilt, azimuth)")
        t, az = ESTIMATOR_312D[park]
        add("meter_estimator_312d", [dict(tilt=t, azimuth=az)], "single", "tab_angles.tex (pv_angle_decomposition), 312-day column, 5-deg grid")
    det["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(det, open(os.path.join(out_dir, f"{park}_details.json"), "w"), indent=1, default=str)
    log(f"{park}: {len(rows)} rows in {det['seconds']} s; grid_fits {det.get('grid_fits')}; table-plane check {det.get('table_plane', {}).get('check_max_abs_d_tilt_profile')}")
    return rows


# ------------------------------------------------------------------------------------------------ twins input tables
def write_twin_tables(df: pd.DataFrame, out_dir: str) -> None:
    """twins/arrays_<source>.csv in the format of geodata_arrays_trianel_windows_v6.csv (park_id, array_id, tilt,
    azimuth, share, source) for the six metered parks; a park without a value for a source is left out (the twin
    then falls back to the operator data, as trianel_twins.geometry does)."""
    tw = os.path.join(out_dir, "twins")
    os.makedirs(tw, exist_ok=True)
    for s in TWIN_SOURCES:
        d = df[(df.source == s) & df.park.isin(TRIANEL) & df.tilt_deg.notna() & df.azimuth_deg.notna()]
        rows = []
        for pv, g in d.groupby("park", sort=False):
            sh = g.share.astype(float)
            sh = sh / sh.sum()
            for k, (a, s_) in enumerate(zip(g.itertuples(), sh)):
                rows.append(dict(park_id=pv, array_id=f"a{k}", tilt=round(float(a.tilt_deg), 2), azimuth=round(float(a.azimuth_deg), 1), share=round(float(s_), 4),
                                 source=f"geometry_sources_compare {s}"))
        pd.DataFrame(rows, columns=["park_id", "array_id", "tilt", "azimuth", "share", "source"]).to_csv(os.path.join(tw, f"arrays_{s}.csv"), index=False)
    log("twin tables written to", tw)


def twins_summary(out_dir: str) -> None:
    """twins/summary.csv: R2 and nRMSE (daytime, 30-min steps, as in trianel_twins) per park and pooled over the six
    parks (capacity-normalised series) for every twin run under twins/<source>/."""
    tw = os.path.join(out_dir, "twins")
    rows = []
    for d in sorted(glob.glob(os.path.join(tw, "*", "parks.csv"))):
        src = os.path.basename(os.path.dirname(d))
        parks = pd.read_csv(d)
        ys, ps = [], []
        for r in parks.itertuples():
            rel = pd.read_parquet(os.path.join(os.path.dirname(d), f"synth_{r.park_id}.parquet"))
            met = pd.read_parquet(os.path.join(os.path.dirname(d), f"meter_{r.park_id}.parquet"))["power_w"].reindex(rel.index)
            cap = float(r.kwp_dc) * 1000.0
            day = (rel["zenith"] < 90) & met.notna()
            y, p = met[day] / cap, rel.loc[day, "power_park"] / cap
            ys.append(y), ps.append(p)
            rows.append(dict(source=src, park=r.park_id, r2_day=float(r.r2_day), nrmse_day_pct=float(r.nrmse_day) * 100.0, energy_ratio_day=float(r.energy_ratio_day),
                             r2_winter=float(r.r2_winter) if "r2_winter" in parks.columns else np.nan, geometry=r.arrays, n=int(day.sum())))
        if ys:
            y, p = pd.concat(ys), pd.concat(ps)
            sse, sst = float(((p - y) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
            rows.append(dict(source=src, park="pooled", r2_day=1.0 - sse / sst, nrmse_day_pct=float(np.sqrt(sse / len(y))) * 100.0,
                             energy_ratio_day=float(p.sum() / y.sum()), r2_winter=np.nan, geometry="", n=int(len(y))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tw, "summary.csv"), index=False)
    if len(df):
        pd.set_option("display.width", 250)
        print(df.pivot(index="source", columns="park", values="r2_day").round(3).to_string())
        print(df.pivot(index="source", columns="park", values="nrmse_day_pct").round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", default=None, help="comma-separated (pv2 or SEL...)")
    ap.add_argument("--all", action="store_true", help="the six metered parks and the 13 fleet parks")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--twins-summary", action="store_true", help="only summarise twins/<source>/ runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.twins_summary:
        twins_summary(a.out)
        return
    parks = a.parks.split(",") if a.parks else (TRIANEL + FLEET13 if a.all else ["pv2"])
    rows = []
    for p in parks:
        try:
            rows += run_park(p, a.out)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            rows.append(dict(park=p, source="error", array="", tilt_deg=np.nan, azimuth_deg=np.nan, layout="n/a", share=np.nan, note=f"{type(ex).__name__}: {str(ex)[:120]}"))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "geometry_sources.csv"), index=False)   # written after every park
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "geometry_sources.csv"), index=False)
    if a.all or (a.parks and any(p in TRIANEL for p in parks)):
        write_twin_tables(df, a.out)
    pd.set_option("display.width", 250)
    print(df[["park", "source", "array", "tilt_deg", "azimuth_deg", "layout", "share", "dtilt_profile_deg", "daz_profile_deg"]].to_string(index=False))
    print("written", os.path.join(a.out, "geometry_sources.csv"))


if __name__ == "__main__":
    main()
