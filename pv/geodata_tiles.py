"""Park geometry from open geodata tiles of any German state: orthophoto -> row axis, pitch and blocks
(2D spectrum, windowed local axes), surface model (laser DOM1 raster, LAZ point cloud or image-based DOM) minus
terrain -> tilt per table via folded cross-row profile and flank plane fit. All rasters are mosaicked on the fly
into the park's UTM zone; results are expressed with true-north azimuths (meridian convergence corrected).

Inputs per park: a footprint polygon (lon/lat), lists of tile paths per product (GeoTIFF, zip containing a GeoTIFF
or XYZ grid, LAZ/LAS), the park's UTM zone (32 or 33). See scripts/pv/geometry_trianel_states.py for the driver."""
import os
import re
import zipfile

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, Proj
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shp_transform

from pv.imagery import row_spectrum

CRS = {32: "EPSG:25832", 33: "EPSG:25833"}


def to_utm(poly_lonlat: Polygon, zone: int) -> Polygon:
    tr = Transformer.from_crs("EPSG:4326", CRS[zone], always_xy=True)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), poly_lonlat)


def meridian_convergence(lat: float, lon: float, zone: int) -> float:
    """Degrees to add to a grid-north bearing to obtain a true-north bearing."""
    return float(Proj(CRS[zone]).get_factors(lon, lat).meridian_convergence)


# ------------------------------------------------------------------------------------------------ raster I/O
def _open_any(path: str):
    """rasterio dataset for a GeoTIFF, or the first GeoTIFF inside a zip (via /vsizip/)."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise ValueError(f"no GeoTIFF in {path}")
        return rasterio.open(f"/vsizip/{path}/{tifs[0]}")
    return rasterio.open(path)


def _xyz_grid(path: str) -> tuple:
    """XYZ ASCII grid (optionally inside a zip) -> (array, transform, res). Regular grid assumed."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.lower().endswith((".xyz", ".txt", ".asc"))][0]
            with z.open(name) as f:
                data = np.loadtxt(f, delimiter=None)
    else:
        data = np.loadtxt(path)
    xs, ys = np.unique(np.round(data[:, 0], 3)), np.unique(np.round(data[:, 1], 3))
    res = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    arr = np.full((len(ys), len(xs)), np.nan, np.float32)
    ix = np.round((data[:, 0] - xs[0]) / res).astype(int); iy = np.round((ys[-1] - data[:, 1]) / res).astype(int)
    arr[iy, ix] = data[:, 2]
    return arr, from_bounds(xs[0] - res / 2, ys[0] - res / 2, xs[-1] + res / 2, ys[-1] + res / 2, len(xs), len(ys)), res


def mosaic(paths: list, bounds: tuple, res: float, zone: int, bands=None) -> np.ndarray:
    """Mosaic tiles into a float array covering bounds (minx, miny, maxx, maxy in CRS[zone]) at res metres.
    Reprojects when a tile's CRS differs. Returns (bands, H, W) float32 with NaN where no data."""
    minx, miny, maxx, maxy = bounds
    H, W = int(round((maxy - miny) / res)), int(round((maxx - minx) / res))
    dst_transform = from_bounds(minx, miny, maxx, maxy, W, H)
    nb = None; out = None
    for p in paths:
        if p.lower().endswith(".laz") or p.lower().endswith(".las"):
            continue
        if p.lower().endswith(".zip") and not any(n.lower().endswith((".tif", ".tiff")) for n in zipfile.ZipFile(p).namelist()):
            arr, tr, r0 = _xyz_grid(p); src_crs = CRS[zone]; src_arrays = [arr]; src_transform = tr; nodata = np.nan
        else:
            ds = _open_any(p)
            try:
                tb = ds.bounds
                if ds.crs is None or str(ds.crs) == CRS[zone]:
                    if tb.right <= minx or tb.left >= maxx or tb.top <= miny or tb.bottom >= maxy:
                        ds.close(); continue
            except Exception:
                pass
            idx = bands or list(range(1, ds.count + 1))
            src_arrays = [ds.read(i).astype(np.float32) for i in idx]; src_transform = ds.transform; src_crs = ds.crs or CRS[zone]; nodata = ds.nodata
        if out is None:
            nb = len(src_arrays); out = np.full((nb, H, W), np.nan, np.float32)
        for b, a in enumerate(src_arrays):
            if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                a = np.where(a == nodata, np.nan, a)
            tmp = np.full((H, W), np.nan, np.float32)
            reproject(a, tmp, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs=CRS[zone],
                      resampling=Resampling.bilinear if res < 1.0 else Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            m = np.isfinite(tmp); out[b][m] = tmp[m]
    return out


def dsm_from_laz(paths: list, bounds: tuple, res: float, ground_classes=(2,)) -> tuple:
    """Point cloud -> (surface = max z per cell, ground = mean z of ground-classified points per cell, n points).
    Cells without points are NaN; the ground raster is only meaningful where ground points exist (gaps under the
    tables are filled by the caller from the DGM)."""
    import laspy
    minx, miny, maxx, maxy = bounds
    H, W = int(round((maxy - miny) / res)), int(round((maxx - minx) / res))
    surf = np.full((H, W), -np.inf, np.float32); gsum = np.zeros((H, W), np.float64); gcnt = np.zeros((H, W), np.int32); npts = 0
    for p in paths:
        if not p.lower().endswith((".laz", ".las")):
            continue
        with laspy.open(p) as f:
            h = f.header
            if h.maxs[0] <= minx or h.mins[0] >= maxx or h.maxs[1] <= miny or h.mins[1] >= maxy:
                continue
            for chunk in f.chunk_iterator(5_000_000):
                x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
                cls = np.asarray(chunk.classification) if hasattr(chunk, "classification") else np.zeros(len(x), int)
                m = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
                if not m.any():
                    continue
                j = np.clip(((x[m] - minx) / res).astype(int), 0, W - 1); i = np.clip(((maxy - y[m]) / res).astype(int), 0, H - 1); zz = z[m]; cc = cls[m]; npts += int(m.sum())
                np.maximum.at(surf, (i, j), zz)
                g = np.isin(cc, ground_classes)
                np.add.at(gsum, (i[g], j[g]), zz[g]); np.add.at(gcnt, (i[g], j[g]), 1)
    surf[~np.isfinite(surf)] = np.nan
    ground = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), np.nan).astype(np.float32)
    return surf, ground, npts


def laz_dates(paths: list) -> list:
    """Acquisition dates from LAS headers (creation date), per file."""
    import laspy
    out = []
    for p in paths:
        if p.lower().endswith((".laz", ".las")):
            with laspy.open(p) as f:
                h = f.header; out.append((os.path.basename(p), str(getattr(h, "creation_date", None)), h.point_count))
    return out


# ------------------------------------------------------------------------------------------- row direction
def row_axis_from_image(rgb: np.ndarray, inside: np.ndarray, res: float, win_m: float = 80.0, step_m: float = 40.0) -> dict:
    """rgb (3,H,W) at res m, inside mask. Returns global peaks (grid-north axis, pitch, share) and the window map."""
    g = np.nanmean(rgb, axis=0); blk = max(int(round(1.0 / res)), 1)
    H, W = (g.shape[0] // blk) * blk, (g.shape[1] // blk) * blk
    g1 = np.nanmean(g[:H, :W].reshape(H // blk, blk, W // blk, blk), axis=(1, 3))
    v1 = (inside[:H, :W].reshape(H // blk, blk, W // blk, blk).mean(axis=(1, 3)) > 0.99) & np.isfinite(g1)
    g1 = np.where(np.isfinite(g1), g1, 0.0)
    peaks = row_spectrum(g1, v1, res * blk) if v1.sum() > 400 else []     # true cell size of g1 (2026-10-21)
    win, step = int(win_m), int(step_m); loc = []
    for i in range(0, g1.shape[0] - win + 1, step):
        for j in range(0, g1.shape[1] - win + 1, step):
            sub = g1[i:i + win, j:j + win]; vs = v1[i:i + win, j:j + win]
            if vs.mean() < 0.6 or sub[vs].std() < 3:
                continue
            pk = row_spectrum(sub, vs, res * blk, n_peaks=1)
            if pk and pk[0][2] >= 0.25:
                loc.append((j + win / 2, i + win / 2, pk[0][0], pk[0][1], pk[0][2]))
    loc = np.array(loc) if loc else np.zeros((0, 5))
    blocks = []
    if len(loc):
        ax = loc[:, 2]; used = np.zeros(len(loc), bool)
        for a, p, s in peaks:
            m = (np.abs(((ax - a) + 90) % 180 - 90) <= 6) & ~used
            if m.sum() >= 2:
                blocks.append({"axis_grid": float(np.degrees(np.arctan2(np.sin(2 * np.radians(ax[m])).mean(), np.cos(2 * np.radians(ax[m])).mean())) / 2 % 180), "pitch_m": float(np.median(loc[m, 3])), "n_windows": int(m.sum()), "share_windows": float(m.sum() / len(loc))}); used |= m
    return {"peaks": [(float(a), float(p), float(s)) for a, p, s in peaks], "blocks": blocks, "n_windows": int(len(loc)), "windows": loc}


def ground_from_surface(surf: np.ndarray, res: float, window_m: float = 25.0) -> np.ndarray:
    """Ground proxy when no terrain model exists: 5th percentile of the surface in a sliding window (about two
    pitches), then smoothed. Valid for parks on gentle terrain where the gaps between rows expose the ground."""
    from scipy.ndimage import percentile_filter, uniform_filter
    w = max(int(window_m / res) | 1, 3)
    filled = np.where(np.isfinite(surf), surf, np.nanmedian(surf))
    return uniform_filter(percentile_filter(filled, 5, size=w), size=max(w // 3, 1))


def pitch_from_surface(hag: np.ndarray, inside: np.ndarray, res: float, axis_grid_deg: float, lo_m: float = 4.0, hi_m: float = 20.0) -> tuple:
    """Row pitch from the surface itself: median height above ground along the across-row coordinate (0.25 m bins),
    autocorrelation, first strong peak between lo_m and hi_m. Returns (pitch_m, peak_corr)."""
    H, W = hag.shape; jj, ii = np.meshgrid(np.arange(W), np.arange(H)); X = (jj + 0.5) * res; Y = (H - ii - 0.5) * res
    a = np.radians(axis_grid_deg); s = X * np.cos(a) - Y * np.sin(a)
    ok = inside & np.isfinite(hag) & (hag > -0.5) & (hag < 6)
    if ok.sum() < 1000:
        return np.nan, np.nan
    bw = 0.25; so = s[ok]; k = ((so - so.min()) / bw).astype(int); n = k.max() + 1
    sums = np.bincount(k, weights=hag[ok], minlength=n); cnt = np.bincount(k, minlength=n); prof = np.where(cnt > 5, sums / np.maximum(cnt, 1), np.nan)
    prof = np.where(np.isfinite(prof), prof, np.nanmean(prof)); x = prof - prof.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]; ac /= max(ac[0], 1e-9)
    lag = np.arange(len(ac)) * bw; m = (lag >= lo_m) & (lag <= hi_m)
    if not m.any():
        return np.nan, np.nan
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ac[m], prominence=0.05)
    if len(pk) == 0:
        return np.nan, np.nan
    best = pk[np.argmax(ac[m][pk])]
    return float(lag[m][best]), float(ac[m][best])


def footprint_from_windows(windows: np.ndarray, axis_grid_deg: float, res_img: float, bounds: tuple, tol_deg: float = 8.0, buffer_m: float = 25.0):
    """Hull of the window centres whose local axis matches the row axis: a data-driven footprint for parks without an
    OSM outline. windows rows: (col_m, row_m, axis, pitch, share) in 1 m image coordinates (row from top)."""
    from shapely.geometry import MultiPoint
    if len(windows) == 0:
        return None
    d = np.abs(((windows[:, 2] - axis_grid_deg) + 90) % 180 - 90); sel = windows[d <= tol_deg]
    if len(sel) < 3:
        return None
    minx, miny, maxx, maxy = bounds
    pts = [(minx + c, maxy - r) for c, r, *_ in sel]
    return MultiPoint(pts).convex_hull.buffer(buffer_m)


def footprint_from_hag(hag: np.ndarray, res: float, bounds: tuple, expected_m2: float, point_xy: tuple, pitch_m: float = 10.0,
                       h_lo: float = 0.4, h_hi: float = 4.5, area_tol=(0.25, 4.0)):
    """Table-block footprint from the height above ground: cells between h_lo and h_hi, closed over about one pitch
    (fills the gaps between rows), connected components; keep components whose area is within area_tol x
    expected_m2 (expected block area incl. gaps), nearest first. Returns a shapely polygon in bounds' CRS or None."""
    from scipy import ndimage
    from rasterio import features
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    m = np.isfinite(hag) & (hag > h_lo) & (hag < h_hi)
    k = max(int(round(pitch_m / res)) | 1, 3)
    m = ndimage.binary_closing(m, structure=np.ones((k, k)), iterations=1)
    m = ndimage.binary_opening(m, structure=np.ones((3, 3)))
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * res * res
    H, W = hag.shape; tr = from_bounds(*bounds, W, H); p = Point(point_xy)
    cands = []
    for i, a in enumerate(areas, 1):
        if a < area_tol[0] * expected_m2 or a > area_tol[1] * expected_m2 * 3:
            continue
        polys = [shape(g) for g, v in features.shapes((lab == i).astype(np.uint8), mask=(lab == i), transform=tr) if v == 1]
        poly = unary_union(polys).buffer(0)
        cands.append((poly.distance(p), a, poly))
    if not cands:
        # fall back to the largest component
        i = int(np.argmax(areas)) + 1
        polys = [shape(g) for g, v in features.shapes((lab == i).astype(np.uint8), mask=(lab == i), transform=tr) if v == 1]
        return unary_union(polys).buffer(0)
    cands.sort(key=lambda c: c[0])
    keep = [c[2] for c in cands if c[0] <= cands[0][0] + 150.0]
    return unary_union(keep).buffer(res)


def _fold_contrast(s_vals, h_vals, pitch, bw=0.25):
    nb = max(int(np.ceil(pitch / bw)), 4); idx = np.minimum((np.mod(s_vals, pitch) / bw).astype(int), nb - 1)
    sums = np.bincount(idx, weights=h_vals, minlength=nb); cnt = np.bincount(idx, minlength=nb)
    prof = np.where(cnt > 20, sums / np.maximum(cnt, 1), np.nan)
    return float(np.nanpercentile(prof, 90) - np.nanpercentile(prof, 10)) if np.isfinite(prof).sum() >= 4 else 0.0


def refine_fold(hag: np.ndarray, inside: np.ndarray, res: float, axis0_deg: float, pitch0_m: float, d_axis: float = 4.0, d_pitch: float = 0.12) -> tuple:
    """Refine row axis (grid deg) and pitch on the surface itself by maximising the contrast of the folded
    mean profile (coarse-to-fine grid search). Returns (axis_deg, pitch_m, contrast_m)."""
    H, W = hag.shape; jj, ii = np.meshgrid(np.arange(W), np.arange(H)); X = (jj + 0.5) * res; Y = (H - ii - 0.5) * res
    ok = inside & np.isfinite(hag) & (hag > -0.5) & (hag < 6)
    if ok.sum() < 1000:
        return axis0_deg, pitch0_m, 0.0
    Xo, Yo, h = X[ok], Y[ok], hag[ok]
    if len(h) > 400_000:
        sel = np.random.default_rng(0).choice(len(h), 400_000, replace=False); Xo, Yo, h = Xo[sel], Yo[sel], h[sel]
    def contrast(axis, pitch):
        a = np.radians(axis); return _fold_contrast(Xo * np.cos(a) - Yo * np.sin(a), h, pitch)
    best = (contrast(axis0_deg, pitch0_m), axis0_deg, pitch0_m)
    for step in (0.5, 0.1):
        axes = np.arange(best[1] - (d_axis if step == 0.5 else 0.6), best[1] + (d_axis if step == 0.5 else 0.6) + 1e-9, step)
        for ax in axes:
            c = contrast(ax, best[2])
            if c > best[0]: best = (c, float(ax), best[2])
        pitches = np.arange(best[2] * (1 - d_pitch), best[2] * (1 + d_pitch), 0.05 if step == 0.5 else 0.01)
        for pm in pitches:
            c = contrast(best[1], pm)
            if c > best[0]: best = (c, best[1], float(pm))
    return best[1], best[2], best[0]


# -------------------------------------------------------------------------------------------------- tilt
def tilt_from_surface(hag: np.ndarray, inside: np.ndarray, res: float, axis_grid_deg: float, pitch_m: float, min_h: float = 0.45) -> dict:
    """Folded cross-row profile and flank plane fit per table. hag = height above ground (H,W) at res m, with NaN
    outside / no data. Row axis in grid-north degrees. Returns tilt estimates, the profile, flank width, side."""
    H, W = hag.shape; jj, ii = np.meshgrid(np.arange(W), np.arange(H)); X = (jj + 0.5) * res; Y = (H - ii - 0.5) * res
    a = np.radians(axis_grid_deg); n_across = np.array([np.cos(a), -np.sin(a)])
    s = X * n_across[0] + Y * n_across[1]
    ok = inside & np.isfinite(hag) & (hag > -0.5) & (hag < 6.0)
    if ok.sum() < 2000:
        return {"ok": False, "reason": "too few cells"}
    bin_w = max(res, 0.2); nb = int(np.ceil(pitch_m / bin_w))
    ph = np.mod(s[ok], pitch_m); idx = np.minimum((ph / bin_w).astype(int), nb - 1); h = hag[ok]
    prof = np.array([np.median(h[idx == k]) if (idx == k).sum() > 30 else np.nan for k in range(nb)]); cen = (np.arange(nb) + 0.5) * bin_w
    if np.all(np.isnan(prof)):
        return {"ok": False, "reason": "empty profile"}
    shift = cen[np.nanargmax(prof)]; ph2 = np.mod(ph - shift + pitch_m / 2, pitch_m); idx2 = np.minimum((ph2 / bin_w).astype(int), nb - 1)
    prof2 = np.array([np.median(h[idx2 == k]) if (idx2 == k).sum() > 30 else np.nan for k in range(nb)])
    k_r = int(np.nanargmax(prof2)); h_ridge = float(prof2[k_r]); above = prof2 > min_h
    lo = k_r
    while lo - 1 >= 0 and above[lo - 1]: lo -= 1
    hi = k_r
    while hi + 1 < nb and above[hi + 1]: hi += 1
    wl, wr = cen[k_r] - cen[lo], cen[hi] - cen[k_r]; side = "left" if wl > wr else "right"; k_f = lo if side == "left" else hi
    h_foot = float(prof2[k_f]); flank = float(abs(cen[k_r] - cen[k_f]))
    tilt_profile = float(np.degrees(np.arctan2(h_ridge - h_foot, max(flank, 0.1))))
    tilt_left = float(np.degrees(np.arctan2(h_ridge - float(prof2[lo]), max(wl, 0.1)))); tilt_right = float(np.degrees(np.arctan2(h_ridge - float(prof2[hi]), max(wr, 0.1))))
    two_flank = bool(min(wl, wr) / max(wl, wr, 0.1) > 0.6 and min(wl, wr) >= 1.5)   # ridge in the middle: east-west pair
    dir_across = float(np.degrees(np.arctan2(n_across[0], n_across[1])) % 360)          # compass direction in which s grows
    flank_dir = dir_across if side == "right" else (dir_across + 180) % 360               # modules fall towards this direction
    # plane fit per table on the flank cells only
    tab = np.floor((s[ok] - shift + pitch_m / 2) / pitch_m).astype(int)
    on_flank = ((ph2 >= cen[k_r] - 0.3) & (ph2 <= cen[k_f] + 0.3)) if side == "right" else ((ph2 >= cen[k_f] - 0.3) & (ph2 <= cen[k_r] + 0.3))
    rows = []
    Xo, Yo = X[ok], Y[ok]
    for t in np.unique(tab):
        m = (tab == t) & on_flank & (h > min_h)
        if m.sum() < max(200, int(60 / res)):
            continue
        A = np.column_stack([Xo[m], Yo[m], np.ones(m.sum())]); coef, *_ = np.linalg.lstsq(A, h[m], rcond=None)
        ga, gb = coef[0], coef[1]
        rows.append((int(t), int(m.sum()), float(np.degrees(np.arctan(np.hypot(ga, gb)))), float(np.degrees(np.arctan2(-ga, -gb)) % 360), float(np.sqrt(np.mean((A @ coef - h[m]) ** 2)))))
    tilts = np.array([r[2] for r in rows]) if rows else np.array([])
    return {"ok": True, "h_ridge": h_ridge, "h_foot": h_foot, "flank_m": flank, "tilt_profile": tilt_profile, "flank_dir_grid": flank_dir,
            "two_flank": two_flank, "tilt_left": tilt_left, "tilt_right": tilt_right, "flank_left_m": float(wl), "flank_right_m": float(wr),
            "tilt_plane_median": float(np.median(tilts)) if len(tilts) else np.nan, "tilt_plane_p10": float(np.percentile(tilts, 10)) if len(tilts) else np.nan,
            "tilt_plane_p90": float(np.percentile(tilts, 90)) if len(tilts) else np.nan, "azimuth_plane_grid": float(np.median([r[3] for r in rows])) if rows else np.nan,
            "n_tables": len(rows), "profile": (cen.tolist(), prof2.tolist()), "ridge_sym": float(min(wl, wr) / max(wl, wr, 0.1))}
