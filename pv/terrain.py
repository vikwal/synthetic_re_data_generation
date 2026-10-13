"""Terrain of a park area from the GLO-30 DEM: slope/aspect per cell, effective (tilt, azimuth)
of terrain-following rows, DEM-based horizon profile, polygon helpers (OSM or area buffer)."""

import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union

from pv import arrays as arr

GLO30_DIR = "/mnt/nvme2/synthetic/raw/round2/dem/glo30"
MOSAIC_90 = "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_90m.tif"


def tile_path(lat: float, lon: float, root: str = GLO30_DIR) -> str:
    return f"{root}/Copernicus_DSM_COG_10_N{int(math.floor(lat)):02d}_00_E{int(math.floor(lon)):03d}_00_DEM.tif"


def read_dem(poly, pad_deg: float = 0.003, root: str = GLO30_DIR) -> tuple:
    """(elevation array, transform, res_deg) covering the polygon bounds (single tile assumed)."""
    minx, miny, maxx, maxy = poly.bounds
    path = tile_path((miny + maxy) / 2, (minx + maxx) / 2, root)
    with rasterio.open(path) as d:
        win = from_bounds(minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg, d.transform)
        z = d.read(1, window=win).astype(float)
        tr = d.window_transform(win)
        res = d.res[0]
    z[z < -1000] = np.nan
    return z, tr, res


def slope_aspect(z: np.ndarray, res_deg: float, lat: float) -> tuple:
    """Horn slope (deg) and aspect (downslope direction, deg from N) on a lat/lon grid."""
    dy_m = 111_320.0 * res_deg
    dx_m = 111_320.0 * math.cos(math.radians(lat)) * res_deg
    dzdy, dzdx = np.gradient(z, dy_m, dx_m)      # rows go south (y decreasing), so flip sign of dzdy
    dzdy = -dzdy
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdy))) % 360.0     # direction of steepest descent
    return slope, aspect


def cells_in(poly, tr, shape_) -> tuple:
    """Row/col indices and lon/lat of cell centres inside the polygon."""
    rows, cols = np.indices(shape_)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    from shapely import contains_xy
    m = contains_xy(poly, xs, ys)
    return rows[m], cols[m], xs[m], ys[m]


def terrain_table(poly, mount_tilt: float, mount_azimuth: float) -> pd.DataFrame:
    """Per DEM cell inside the polygon: elevation, slope, aspect, effective tilt/azimuth."""
    z, tr, res = read_dem(poly)
    lat = poly.centroid.y
    sl, asp = slope_aspect(z, res, lat)
    r, c, x, y = cells_in(poly, tr, z.shape)
    t_eff, a_eff = arr.effective_angles(sl[r, c], asp[r, c], mount_tilt, mount_azimuth)
    cell_m2 = (111_320.0 * res) * (111_320.0 * math.cos(math.radians(lat)) * res)
    return pd.DataFrame({"lon": x, "lat": y, "elev": z[r, c], "slope": sl[r, c], "aspect": asp[r, c],
                         "tilt_eff": t_eff, "az_eff": a_eff, "area_m2": cell_m2}).dropna()


def dem_horizon(lat: float, lon: float, max_km: float = 25.0, step_m: float = 90.0, az_step: float = 7.5,
                mosaic: str = MOSAIC_90) -> pd.Series:
    """Horizon elevation (deg) by azimuth from the 90 m mosaic by ray marching (Earth curvature included)."""
    with rasterio.open(mosaic) as d:
        half = max_km / 111.0 + 0.05
        win = from_bounds(lon - half / math.cos(math.radians(lat)), lat - half, lon + half / math.cos(math.radians(lat)), lat + half, d.transform)
        z = d.read(1, window=win).astype(float)
        tr = d.window_transform(win)
    z[z < -1000] = np.nan
    r0 = int((lat - tr.f) / tr.e)
    c0 = int((lon - tr.c) / tr.a)
    z0 = np.nanmean(z[max(r0 - 1, 0):r0 + 2, max(c0 - 1, 0):c0 + 2]) + 2.0
    dists = np.arange(step_m, max_km * 1000.0, step_m)
    out = {}
    coslat = math.cos(math.radians(lat))
    for az in np.arange(0.0, 360.0, az_step):
        a = math.radians(az)
        dn = dists * math.cos(a)
        de = dists * math.sin(a)
        lats = lat + dn / 111_320.0
        lons = lon + de / (111_320.0 * coslat)
        rr = ((lats - tr.f) / tr.e).astype(int)
        cc = ((lons - tr.c) / tr.a).astype(int)
        ok = (rr >= 0) & (rr < z.shape[0]) & (cc >= 0) & (cc < z.shape[1])
        zz = np.full(len(dists), np.nan)
        zz[ok] = z[rr[ok], cc[ok]]
        drop = dists ** 2 / (2 * 6_371_000.0)                       # curvature
        ang = np.degrees(np.arctan((zz - z0 - drop) / dists))
        out[az] = float(np.nanmax(ang)) if np.isfinite(ang).any() else 0.0
    return pd.Series(out).clip(lower=0.0)


def osm_polygon(elements: list, min_area_m2: float = 2000.0):
    """Union of closed way/relation geometries from an Overpass 'out geom' answer (lon/lat)."""
    polys = []
    for e in elements:
        if e.get("type") == "way" and "geometry" in e and len(e["geometry"]) >= 4:
            pts = [(g["lon"], g["lat"]) for g in e["geometry"]]
            if pts[0] == pts[-1]:
                pg = Polygon(pts)
                if pg.is_valid and pg.area > 0:
                    polys.append(pg)
        elif e.get("type") == "relation":
            for mbr in e.get("members", []):
                if mbr.get("role") == "outer" and "geometry" in mbr and len(mbr["geometry"]) >= 4:
                    pts = [(g["lon"], g["lat"]) for g in mbr["geometry"]]
                    if pts[0] == pts[-1]:
                        pg = Polygon(pts)
                        if pg.is_valid and pg.area > 0:
                            polys.append(pg)
    if not polys:
        return None
    u = unary_union(polys)
    return u if u.area * (111_320.0 ** 2) * math.cos(math.radians(u.centroid.y)) >= min_area_m2 else None


def pick_plant_near(elements: list, lat: float, lon: float, max_km: float = 2.0):
    """Keep only polygons whose centroid lies within max_km of the park coordinate."""
    polys = []
    for e in elements:
        geom = osm_polygon([e])
        if geom is None:
            continue
        d = math.hypot((geom.centroid.y - lat) * 111.32, (geom.centroid.x - lon) * 111.32 * math.cos(math.radians(lat)))
        if d <= max_km:
            polys.append(geom)
    return unary_union(polys) if polys else None


def buffer_polygon(lat: float, lon: float, area_ha: float):
    """Disc of the given area around the coordinate (fallback when no footprint is known)."""
    r_m = math.sqrt(area_ha * 10_000.0 / math.pi)
    return Point(lon, lat).buffer(r_m / (111_320.0 * math.cos(math.radians(lat))))


def area_ha(poly) -> float:
    lat = poly.centroid.y
    return poly.area * 111_320.0 ** 2 * math.cos(math.radians(lat)) / 10_000.0
