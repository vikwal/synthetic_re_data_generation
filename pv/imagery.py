"""Module-row geometry from orthophotos: dominant row direction and row pitch of a park footprint via the
2D power spectrum of the 1 m grey image (module tables are a periodic stripe pattern with 6-15 m pitch).
Tested 2026-10-14 on a Bavarian DOP20 tile (0.2 m): two blocks resolved (row axes 90 and 117 deg, pitch 12 m),
where the OSM polygons gave only the mean axis (104 deg). Compass convention: row axis 0-180 deg from north;
south-facing module azimuth = axis + 90 (90 < az < 270), north-south rows (axis < 30 or > 150) = east-west park."""
import numpy as np
import rasterio
from rasterio.mask import mask as rmask
from shapely.geometry import mapping


def clip_grey(tile_path: str, poly_crs, block: int = 5, inner_buffer_m: float = 5.0) -> tuple:
    """Grey image of the polygon (in the tile CRS) at block x native resolution; (image, valid mask, res_m)."""
    with rasterio.open(tile_path) as src:
        img, _ = rmask(src, [mapping(poly_crs.buffer(-inner_buffer_m))], crop=True, filled=True, nodata=0)
        res = src.res[0] * block
    g = img.astype(float).mean(0)
    valid = g > 0
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    g1 = g[:h, :w].reshape(h // block, block, w // block, block).mean((1, 3))
    v1 = valid[:h, :w].reshape(h // block, block, w // block, block).mean((1, 3)) > 0.99
    return g1, v1, res


def row_spectrum(g1: np.ndarray, v1: np.ndarray, res_m: float, pitch_range=(4.0, 15.0), n_peaks: int = 3,
                 pad: int = 4, taper_m: float = 15.0, harmonic_ratio: float = 0.5) -> list:
    """Peaks of the 2D power spectrum within the pitch band: [(row_axis_deg, pitch_m, power_share), ...].
    res_m = cell size of g1 in metres. pad: zero-padding factor of the FFT (finer k grid). taper_m: cosine taper
    width at the mask edge (0 = hard edge). harmonic_ratio: a peak whose wedge carries at least this share of its
    energy at k/2 is taken for the 2nd harmonic and the pitch is doubled (0 = off). Share and suppression windows
    wrap around 0/180 deg; directions are rounded to the nearest degree. (2026-10-21)"""
    x = np.where(v1, g1 - g1[v1].mean(), 0.0)
    if taper_m > 0 and v1.any() and not v1.all():
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(v1) * res_m                                   # distance to the mask edge (m)
        w = np.where(d < taper_m, 0.5 * (1 - np.cos(np.pi * np.clip(d, 0, taper_m) / taper_m)), 1.0)
        x = x * w
    ny, nx = x.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(x, s=(pad * ny, pad * nx)))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(pad * ny, d=res_m))
    kx = np.fft.fftshift(np.fft.fftfreq(pad * nx, d=res_m))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    band = (K > 1 / pitch_range[1]) & (K < 1 / pitch_range[0])
    ang = np.degrees(np.arctan2(KX, -KY)) % 180.0            # compass of the wave vector (image y points south)
    abin = np.round(ang).astype(int) % 180                    # nearest degree; 179.6 and 0.3 fall into bin 0
    hist = np.zeros(180)
    np.add.at(hist, abin[band], F[band])
    if hist.sum() <= 0:
        return []
    out, used = [], np.zeros(180, bool)
    for a in np.argsort(hist)[::-1]:
        if used[a] or len(out) >= n_peaks:
            continue
        wedge = np.abs(((ang - a) + 90) % 180 - 90) < 2
        sub = band & wedge
        if not sub.any():
            continue
        kpk = K[sub][np.argmax(F[sub])]
        if harmonic_ratio > 0 and kpk / 2 >= 1 / pitch_range[1]:   # 2nd harmonic? only if the fundamental is still a plausible pitch
            e_pk = F[wedge & (np.abs(K - kpk) < 0.1 * kpk)].sum()
            e_half = F[wedge & (K > 0) & (np.abs(K - kpk / 2) < 0.1 * kpk / 2)].sum()
            if e_half >= harmonic_ratio * e_pk:
                kpk = kpk / 2
        share = hist[np.arange(a - 2, a + 3) % 180].sum() / hist.sum()
        out.append((float((a + 90) % 180), float(1 / kpk), float(share)))
        used[np.arange(a - 5, a + 6) % 180] = True
    return out

def row_geometry(tile_path: str, poly_crs, block: int = 5) -> dict:
    g1, v1, res = clip_grey(tile_path, poly_crs, block)
    peaks = row_spectrum(g1, v1, res)
    axis, pitch, share = peaks[0]
    ns = axis < 30.0 or axis > 150.0
    return {"row_axis_deg": axis, "pitch_m": pitch, "power_share": share, "peaks": peaks,
            "module_azimuth": None if ns else (axis + 90.0) % 360.0, "east_west": bool(ns), "n_cells": int(v1.sum()), "res_m": res}
