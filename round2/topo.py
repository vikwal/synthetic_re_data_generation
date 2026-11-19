"""WP0 — DEM-derived terrain metrics and CORINE z0 lookup.

Metrics per location (guide WP0 'Derived topo metrics'):
elevation, slope, aspect, TPI@5 km, TPI@75 km, TDI (11 km window),
elevation-std (5 km), z0 from CORINE CLC2018.
"""

import math
import os

import numpy as np
import rasterio
from pyproj import Transformer

# CLC 2018 class -> roughness length z0 [m], Silva et al. (2007)-style lookup.
# Keys are CLC grid codes 1..44 (raster values in U2018_CLC2018).
CLC_Z0 = {
    1: 1.2, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5,       # artificial
    7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
    12: 0.05, 13: 0.1, 14: 0.1, 15: 0.3, 16: 0.3, 17: 0.3,  # agricultural
    18: 0.03, 19: 0.1, 20: 0.3, 21: 0.3, 22: 0.5,
    23: 0.9, 24: 0.9, 25: 0.9,                              # forest
    26: 0.1, 27: 0.3, 28: 0.5, 29: 0.6,                     # shrub/herbaceous
    30: 0.005, 31: 0.005, 32: 0.05, 33: 0.005, 34: 0.001,   # open spaces
    35: 0.05, 36: 0.05, 37: 0.0005, 38: 0.0005, 39: 0.0005, # wetlands
    40: 0.0005, 41: 0.0005, 42: 0.0005, 43: 0.0005, 44: 0.0005,  # water
}
Z0_DEFAULT = 0.1


def circular_kernel_stats(arr: np.ndarray, cy: int, cx: int,
                          ry: int, rx: int) -> dict:
    """Stats of arr inside the ellipse of pixel radii (ry, rx) around (cy, cx)."""
    y0, y1 = max(0, cy - ry), min(arr.shape[0], cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(arr.shape[1], cx + rx + 1)
    win = arr[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
    vals = win[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    return {"mean": float(vals.mean()), "min": float(vals.min()),
            "max": float(vals.max()), "std": float(vals.std())}


def slope_aspect(arr: np.ndarray, cy: int, cx: int, px_m_y: float, px_m_x: float):
    """Horn-style slope (deg) and aspect (deg from N) from the 3x3 neighborhood."""
    if cy < 1 or cx < 1 or cy >= arr.shape[0] - 1 or cx >= arr.shape[1] - 1:
        return np.nan, np.nan
    z = arr[cy - 1:cy + 2, cx - 1:cx + 2]
    if not np.all(np.isfinite(z)):
        return np.nan, np.nan
    dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * px_m_x)
    dzdy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * px_m_y)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = math.degrees(math.atan2(dzdx, -dzdy)) % 360.0
    return slope, aspect


def location_metrics(dem_fine, dem_coarse, lat: float, lon: float) -> dict:
    """dem_fine/dem_coarse: dicts with 'arr', 'transform' (lat/lon grids)."""
    out = {}
    for name, dem, radii in (
        ("fine", dem_fine, {"tpi5": 5_000.0, "tdi11": 5_500.0, "std5": 5_000.0}),
        ("coarse", dem_coarse, {"tpi75": 75_000.0}),
    ):
        arr, transform = dem["arr"], dem["transform"]
        col, row = ~transform * (lon, lat)
        cy, cx = int(round(row)), int(round(col))
        if not (0 <= cy < arr.shape[0] and 0 <= cx < arr.shape[1]):
            continue
        px_deg_x, px_deg_y = abs(transform.a), abs(transform.e)
        px_m_y = px_deg_y * 111_320.0
        px_m_x = px_deg_x * 111_320.0 * math.cos(math.radians(lat))
        if name == "fine":
            out["elevation"] = float(arr[cy, cx])
            s, a = slope_aspect(arr, cy, cx, px_m_y, px_m_x)
            out["slope"], out["aspect"] = s, a
        for key, radius_m in radii.items():
            ry = max(1, int(round(radius_m / px_m_y)))
            rx = max(1, int(round(radius_m / px_m_x)))
            st = circular_kernel_stats(arr, cy, cx, ry, rx)
            if key.startswith("tpi"):
                out[key] = float(arr[cy, cx]) - st["mean"]
            elif key == "tdi11":
                out["tdi"] = ((st["max"] - st["min"]) / st["mean"]
                              if st["mean"] not in (0.0, np.nan) and st["mean"] > 0 else np.nan)
            elif key == "std5":
                out["elev_std"] = st["std"]
    return out


class CorineSampler:
    def __init__(self, tif_path: str):
        self.ds = rasterio.open(tif_path)
        self.tf = Transformer.from_crs("EPSG:4326", self.ds.crs, always_xy=True)

    def z0(self, lat: float, lon: float) -> float:
        x, y = self.tf.transform(lon, lat)
        try:
            val = next(self.ds.sample([(x, y)]))[0]
        except Exception:
            return Z0_DEFAULT
        return CLC_Z0.get(int(val), Z0_DEFAULT)

    def clc_class(self, lat: float, lon: float) -> int:
        x, y = self.tf.transform(lon, lat)
        try:
            return int(next(self.ds.sample([(x, y)]))[0])
        except Exception:
            return -1
