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


def row_spectrum(g1: np.ndarray, v1: np.ndarray, res_m: float, pitch_range=(4.0, 15.0), n_peaks: int = 3) -> list:
    """Peaks of the 2D power spectrum within the pitch band: [(row_axis_deg, pitch_m, power_share), ...]."""
    x = np.where(v1, g1 - g1[v1].mean(), 0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=res_m))
    kx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=res_m))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    band = (K > 1 / pitch_range[1]) & (K < 1 / pitch_range[0])
    ang = np.degrees(np.arctan2(KX, -KY)) % 180.0            # compass of the wave vector (image y points south)
    hist = np.zeros(180)
    np.add.at(hist, ang[band].astype(int) % 180, F[band])
    out, used = [], np.zeros(180, bool)
    for a in np.argsort(hist)[::-1]:
        if used[a] or len(out) >= n_peaks:
            continue
        sub = band & (np.abs(((ang - a) + 90) % 180 - 90) < 2)
        kpk = K[sub][np.argmax(F[sub])]
        out.append((float((a + 90) % 180), float(1 / kpk), float(hist[max(a - 2, 0):a + 3].sum() / hist.sum())))
        used[max(a - 5, 0):a + 6] = True
    return out


def row_geometry(tile_path: str, poly_crs, block: int = 5) -> dict:
    g1, v1, res = clip_grey(tile_path, poly_crs, block)
    peaks = row_spectrum(g1, v1, res)
    axis, pitch, share = peaks[0]
    ns = axis < 30.0 or axis > 150.0
    return {"row_axis_deg": axis, "pitch_m": pitch, "power_share": share, "peaks": peaks,
            "module_azimuth": None if ns else (axis + 90.0) % 360.0, "east_west": bool(ns), "n_cells": int(v1.sum()), "res_m": res}
