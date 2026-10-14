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


def _faces(z, tf, ok, row_axis_deg, res, min_share=0.2):
    """Split the cells of a table into faces by the sign of the DOM gradient along the row normal: a south
    table is one face, an east-west table has an east and a west face. Returns list of boolean masks."""
    zs = ndimage.gaussian_filter(np.where(ok, z, np.nanmean(z[ok]) if ok.any() else 0.0), sigma=max(1.0 / res, 1.0))
    gy, gx = np.gradient(zs, res)                       # gy: image rows (south positive), gx: east
    ang = np.radians(row_axis_deg + 90.0)               # row normal, compass
    comp = gx * np.sin(ang) - gy * np.cos(ang)          # gradient component along the normal (north = -gy)
    pos, neg = ok & (comp > 0), ok & (comp < 0)
    n = ok.sum()
    faces = [m for m in (pos, neg) if m.sum() >= min_share * n]
    return faces if len(faces) == 2 else [ok]


def table_planes(dom_path: str, dgm_path: str, polygons: list, min_height: float = 0.6, max_height: float = 4.0, inner: float = 0.6,
                 row_axis_deg: float = None) -> pd.DataFrame:
    """One row per table FACE: module plane from DOM cells that stand min..max m above the DGM (split into east/west
    faces along the row normal where both exist), ground plane from the DGM (3 m buffer), long-axis direction, area."""
    ground = ground_sampler(dgm_path)
    out = []
    for i, poly in enumerate(polygons):
        if not poly.is_valid or poly.area < 20:
            continue
        inner_poly = poly.buffer(-inner)
        if inner_poly.is_empty or inner_poly.area < 4.0:
            inner_poly = poly
        with rasterio.open(dom_path) as src:
            z, tf = rmask(src, [mapping(inner_poly)], crop=True, filled=True, nodata=np.nan)
            res = src.res[0]
        z = z[0].astype(float)
        z[z < -1000] = np.nan
        rows, cols = np.indices(z.shape)
        xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
        xs, ys = np.array(xs), np.array(ys)
        h = z.ravel() - ground(xs, ys)
        ok = (np.isfinite(z.ravel()) & (h >= min_height) & (h <= max_height)).reshape(z.shape)
        if ok.sum() < 30:
            continue
        dgm = plane_in(dgm_path, poly.buffer(3.0), 0.0, min_cells=9)
        if dgm is None:
            continue
        xy = np.array(poly.exterior.coords)[:-1] - np.array(poly.centroid.coords)[0]
        _, s, vt = np.linalg.svd(xy, full_matrices=False)
        axis = float(np.degrees(np.arctan2(vt[0][0], vt[0][1])) % 180.0)
        faces = _faces(z, tf, ok, row_axis_deg if row_axis_deg is not None else axis, res)
        if len(faces) == 2:                          # accept the split only for two opposite, clearly tilted planes
            pl = [_plane(xs[f.ravel()], ys[f.ravel()], z.ravel()[f.ravel()]) for f in faces]
            daz = abs(((pl[0]["azimuth"] - pl[1]["azimuth"]) + 180.0) % 360.0 - 180.0)
            if not (min(pl[0]["tilt"], pl[1]["tilt"]) >= 5.0 and daz > 120.0):
                faces = [ok]
        for j, fm in enumerate(faces):
            sel = fm.ravel()
            dom = _plane(xs[sel], ys[sel], z.ravel()[sel])
            out.append({"table": i, "face": j, "n_faces": len(faces), "area_m2": float(poly.area * sel.sum() / ok.sum()),
                        "axis_deg": axis, "elongation": float(s[0] / max(s[1], 1e-9)), "tilt": dom["tilt"], "azimuth": dom["azimuth"],
                        "plane_rms": dom["rms"], "n_cells": dom["n"], "height_above_ground": dom["z_mean"] - dgm["z_mean"],
                        "ground_slope": dgm["tilt"], "ground_aspect": dgm["azimuth"], "x": float(poly.centroid.x), "y": float(poly.centroid.y)})
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

def height_mask(dom_path: str, dgm_path: str, footprint, min_height: float = 0.6, max_height: float = 4.0, block: int = 5) -> tuple:
    """(mask of module cells, height-above-ground, transform, res) at block x DOM resolution inside the footprint."""
    with rasterio.open(dom_path) as src:
        z, tf = rmask(src, [mapping(footprint)], crop=True, filled=True, nodata=np.nan)
        res0 = src.res[0]
    z = z[0].astype(float)
    z[z < -1000] = np.nan
    h0, w0 = (z.shape[0] // block) * block, (z.shape[1] // block) * block
    zb = np.nanmean(z[:h0, :w0].reshape(h0 // block, block, w0 // block, block), axis=(1, 3))
    tfb = tf * tf.scale(block, block)
    res = res0 * block
    rows, cols = np.indices(zb.shape)
    xs, ys = rasterio.transform.xy(tfb, rows.ravel(), cols.ravel())
    h = (zb.ravel() - ground_sampler(dgm_path)(np.array(xs), np.array(ys))).reshape(zb.shape)
    m = np.isfinite(h) & (h >= min_height) & (h <= max_height)
    return m, h, tfb, res


def row_axis_from_mask(m: np.ndarray, res: float, pitch_range=(4.0, 20.0)) -> tuple:
    """Dominant stripe direction and pitch of the module mask (2D spectrum): (row_axis_deg, pitch_m, share)."""
    x = m.astype(float) - m.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=res))
    kx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=res))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    band = (K > 1 / pitch_range[1]) & (K < 1 / pitch_range[0])
    ang = np.degrees(np.arctan2(KX, -KY)) % 180.0
    hist = np.zeros(180)
    np.add.at(hist, ang[band].astype(int) % 180, F[band])
    a = int(np.argmax(hist))
    sub = band & (np.abs(((ang - a) + 90) % 180 - 90) < 2)
    kpk = K[sub][np.argmax(F[sub])]
    return float((a + 90) % 180), float(1 / kpk), float(hist[max(a - 2, 0):a + 3].sum() / hist.sum())


def row_bands(m: np.ndarray, tf, res: float, row_axis_deg: float, pitch_m: float) -> list:
    """Cut the module mask into row bands: project cells onto the row normal, find the peaks of the occupancy
    profile (one per row), assign cells to the nearest peak, polygonise each band. Returns UTM polygons."""
    from scipy.signal import find_peaks
    rows, cols = np.where(m)
    if len(rows) < 50:
        return []
    xs, ys = rasterio.transform.xy(tf, rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    a = np.radians(row_axis_deg)
    n_coord = xs * np.cos(a) - ys * np.sin(a)             # coordinate along the row normal (compass axis + 90)
    bins = np.arange(n_coord.min() - res, n_coord.max() + 2 * res, res)
    prof, _ = np.histogram(n_coord, bins)
    prof = ndimage.gaussian_filter1d(prof.astype(float), 1.0)
    peaks, _ = find_peaks(prof, distance=max(int(0.5 * pitch_m / res), 2), prominence=prof.max() * 0.1)
    if len(peaks) == 0:
        return []
    centres = bins[peaks] + res / 2
    lab = np.argmin(np.abs(n_coord[:, None] - centres[None, :]), axis=1)
    label_img = np.zeros(m.shape, np.int32)
    label_img[rows, cols] = lab + 1
    polys = []
    for k in range(1, len(centres) + 1):
        band = label_img == k
        if band.sum() < 20:
            continue
        band = ndimage.binary_closing(band, iterations=1)
        for geom, val in features.shapes(band.astype(np.uint8), mask=band, transform=tf):
            pg = shape(geom)
            if pg.area >= 40:
                polys.append(pg)
    return polys


def footprint_from_dom(dom_path: str, dgm_path: str, x: float, y: float, radius_m: float = 1500.0, min_height: float = 0.6,
                       max_height: float = 4.0, min_area_ha: float = 0.3, expected_module_ha: float = None):
    """Footprint of the module field belonging to a registered coordinate (which may be rounded by up to ~700 m):
    height band inside a search disc, closing over the row gaps (10 m), connected components; the component is
    chosen by distance to the point and, if given, by how well its module area matches the expected one
    (kWp / typical kWp-per-ha). Returns (convex hull + 5 m, module_area_ha) or (None, 0)."""
    from shapely.geometry import Point, MultiPoint
    disc = Point(x, y).buffer(radius_m)
    m, h, tf, res = height_mask(dom_path, dgm_path, disc, min_height, max_height)
    filled = ndimage.binary_closing(m, iterations=max(int(10.0 / res), 1))
    lab, n = ndimage.label(filled)
    if n == 0:
        return None, 0.0
    rows, cols = np.indices(m.shape)
    xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
    d = np.hypot(np.array(xs) - x, np.array(ys) - y).reshape(m.shape)
    best, best_score = None, -1.0
    for k in range(1, n + 1):
        comp = lab == k
        mod_ha = (comp & m).sum() * res * res / 1e4
        density = mod_ha * 1e4 / max(comp.sum() * res * res, 1.0)     # module cells / closed area: rows 0.2-0.75, hedges ~1
        if mod_ha < min_area_ha or not (0.15 <= density <= 0.8):
            continue
        score = 1.0 / (1.0 + d[comp].min() / 500.0)
        if expected_module_ha:
            score *= np.exp(-abs(np.log(mod_ha / expected_module_ha)))
        if score > best_score:
            best, best_score = comp, score
    if best is None:
        return None, 0.0
    r_, c_ = np.where(best & m)
    px, py = rasterio.transform.xy(tf, r_, c_)
    hull = MultiPoint(list(zip(px, py))).convex_hull.buffer(5.0)
    return hull, float(len(r_) * res * res / 1e4)
