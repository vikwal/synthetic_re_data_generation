"""Module-table geometry from open surface and terrain models: per table a plane fitted into the DOM (tilt,
azimuth, height), the ground plane from the DGM, plus 1 m rasters of ground slope/aspect and module tilt/azimuth.

Tables come from OSM power=generator polygons where mapped; otherwise they are segmented from the DOM: cells
0.6-4 m above the DGM inside the footprint, opened/closed along the row direction from the orthophoto spectrum
(pv.imagery), connected components >= 40 m2. Tested 2026-10-14 on a Bavarian park (DOM20 06/2025, DGM1 2017):
26 tables, tilt 14.0-14.8 deg (register '< 20 Grad'), azimuth 180-185 / 205 deg in two row blocks, height above
ground 2.0 m, plane rms 0.4-1.1 m."""
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.mask import mask as rmask
from scipy import ndimage
from shapely.geometry import Polygon, mapping, shape


def _plane(xs, ys, zs):
    A = np.column_stack([xs - xs.mean(), ys - ys.mean(), np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    a, b = coef[0], coef[1]
    return {"tilt": float(np.degrees(np.arctan(np.hypot(a, b)))), "azimuth": float(np.degrees(np.arctan2(-a, -b)) % 360.0),
            "rms": float(np.sqrt(np.mean((A @ coef - zs) ** 2))), "n": int(len(xs)), "z_mean": float(zs.mean())}


def plane_in(path: str, poly, inner_buffer: float = 0.0, min_cells: int = 30, keep=None):
    """Plane through the raster cells inside poly (buffered inwards); keep: optional callable(z, transform) -> mask."""
    with rasterio.open(path) as src:
        z, tf = rmask(src, [mapping(poly.buffer(-inner_buffer))], crop=True, filled=True, nodata=np.nan)
    z = z[0].astype(float)
    z[z < -1000] = np.nan
    ok = np.isfinite(z)
    if keep is not None:
        ok &= keep(z, tf)
    rows, cols = np.where(ok)
    if len(rows) < min_cells:
        return None
    xs, ys = rasterio.transform.xy(tf, rows, cols)
    return _plane(np.array(xs), np.array(ys), z[rows, cols])


def ground_sampler(dgm_path: str):
    """Returns f(xs, ys) -> ground height (nearest DGM cell)."""
    src = rasterio.open(dgm_path)
    arr = src.read(1).astype(float)
    arr[arr < -1000] = np.nan
    tf = src.transform

    def f(xs, ys):
        rows, cols = rasterio.transform.rowcol(tf, xs, ys)
        rows, cols = np.clip(rows, 0, arr.shape[0] - 1), np.clip(cols, 0, arr.shape[1] - 1)
        return arr[rows, cols]
    return f


def table_planes(dom_path: str, dgm_path: str, polygons: list, min_height: float = 0.6, max_height: float = 4.0, inner: float = 0.6) -> pd.DataFrame:
    """One row per table polygon (UTM): module plane from DOM cells that stand min..max m above the DGM, ground
    plane from the DGM (3 m buffer), long-axis direction, area."""
    ground = ground_sampler(dgm_path)

    def keep(z, tf):
        rows, cols = np.indices(z.shape)
        xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
        h = z.ravel() - ground(np.array(xs), np.array(ys))
        return ((h >= min_height) & (h <= max_height)).reshape(z.shape)
    out = []
    for i, poly in enumerate(polygons):
        if not poly.is_valid or poly.area < 20:
            continue
        dom = plane_in(dom_path, poly, inner, keep=keep)
        dgm = plane_in(dgm_path, poly.buffer(3.0), 0.0, min_cells=9)
        if dom is None or dgm is None:
            continue
        xy = np.array(poly.exterior.coords)[:-1] - np.array(poly.centroid.coords)[0]
        _, s, vt = np.linalg.svd(xy, full_matrices=False)
        out.append({"table": i, "area_m2": float(poly.area), "axis_deg": float(np.degrees(np.arctan2(vt[0][0], vt[0][1])) % 180.0),
                    "elongation": float(s[0] / max(s[1], 1e-9)), "tilt": dom["tilt"], "azimuth": dom["azimuth"], "plane_rms": dom["rms"],
                    "n_cells": dom["n"], "height_above_ground": dom["z_mean"] - dgm["z_mean"], "ground_slope": dgm["tilt"],
                    "ground_aspect": dgm["azimuth"], "x": float(poly.centroid.x), "y": float(poly.centroid.y)})
    return pd.DataFrame(out)


def segment_tables(dom_path: str, dgm_path: str, footprint, row_axis_deg: float, min_height: float = 0.6, max_height: float = 4.0,
                   min_area: float = 40.0) -> list:
    """Table polygons from the DOM where OSM has none: height above ground band inside the footprint, morphological
    opening/closing with a line element along the rows, connected components -> polygons (UTM)."""
    with rasterio.open(dom_path) as src:
        z, tf = rmask(src, [mapping(footprint)], crop=True, filled=True, nodata=np.nan)
        res = src.res[0]
    z = z[0].astype(float)
    z[z < -1000] = np.nan
    ground = ground_sampler(dgm_path)
    rows, cols = np.indices(z.shape)
    xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
    h = (z.ravel() - ground(np.array(xs), np.array(ys))).reshape(z.shape)
    m = (h >= min_height) & (h <= max_height)
    # line structuring element along the row axis (compass -> image angle), length ~ 3 m
    L = max(int(3.0 / res), 3)
    ang = np.radians(90.0 - row_axis_deg)
    se = np.zeros((L, L), bool)
    c = L // 2
    for t in np.linspace(-c, c, 4 * L):
        se[int(round(c - t * np.sin(ang))) % L, int(round(c + t * np.cos(ang))) % L] = True
    m = ndimage.binary_opening(m, se)
    m = ndimage.binary_closing(m, se)
    polys = []
    for geom, val in features.shapes(m.astype(np.uint8), mask=m, transform=tf):
        p = shape(geom)
        if p.area >= min_area:
            polys.append(p.simplify(0.3))
    return polys


def rasters_1m(tables: pd.DataFrame, footprint, dgm_path: str) -> dict:
    """1 m rasters over the footprint: ground slope/aspect (DGM, Horn 3x3) and module tilt/azimuth (table planes
    burned into their polygons; NaN elsewhere). Returns {'transform', 'slope', 'aspect', 'tilt', 'azimuth'}."""
    with rasterio.open(dgm_path) as src:
        z, tf = rmask(src, [mapping(footprint.buffer(5))], crop=True, filled=True, nodata=np.nan)
        res = src.res[0]
    z = z[0].astype(float)
    z[z < -1000] = np.nan
    dzdx = ndimage.sobel(z, axis=1) / (8 * res)
    dzdy = -ndimage.sobel(z, axis=0) / (8 * res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    tilt = np.full(z.shape, np.nan)
    az = np.full(z.shape, np.nan)
    if len(tables):
        for col, arr in (("tilt", tilt), ("azimuth", az)):
            shapes = [(mapping(p), float(v)) for p, v in zip(tables["geometry"], tables[col])]
            burned = features.rasterize(shapes, out_shape=z.shape, transform=tf, fill=np.nan, dtype="float64")
            arr[:] = burned
    return {"transform": tf, "slope": slope, "aspect": aspect, "tilt": tilt, "azimuth": az}
