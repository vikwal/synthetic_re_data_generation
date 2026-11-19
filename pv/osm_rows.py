"""Table-row direction from OSM power=generator polygons (Overpass JSON with geometry).

Each elongated polygon (principal-axis ratio >= min_ratio) gives one row axis (deg clockwise from north, 0-180,
PCA of its vertices in local metres); the park axis is the length-weighted circular mean over the axes (period
180). The module azimuth follows from pv.plausibility.azimuth_prior_from_rows (rows east-west -> south-facing
modules, rows north-south -> east-west pair). Same statistic as data/parks_pv_v1/osm_row_geometry.csv (2026-10-14)."""
import numpy as np

MAX_SPREAD_DEG = 10.0      # length-weighted RMS spread of the row axes above which the park axis is not a usable prior


def polygon_axis(coords: list) -> tuple:
    """(axis_deg 0-180, axis_ratio, length_m) of one polygon given [(lat, lon), ...]."""
    pts = np.array(coords, dtype=float)
    if len(pts) < 3:
        return np.nan, np.nan, np.nan
    lat0 = pts[:, 0].mean()
    xy = np.column_stack([(pts[:, 1] - pts[:, 1].mean()) * 111320.0 * np.cos(np.radians(lat0)), (pts[:, 0] - lat0) * 110540.0])
    cov = np.cov(xy.T)
    val, vec = np.linalg.eigh(cov)
    major = vec[:, np.argmax(val)]
    ratio = float(np.sqrt(max(val) / max(min(val), 1e-9)))
    axis = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)        # from north, clockwise
    length = float(np.ptp(xy @ major))
    return axis, ratio, length


def row_axis_from_osm(elements: list, min_ratio: float = 2.5, min_rows: int = 3) -> dict:
    """{'row_axis_deg', 'n_elongated', 'n_generators', 'axis_spread_deg'}; row_axis_deg is NaN with < min_rows rows."""
    axes, lens = [], []
    n_gen = 0
    for e in elements:
        tags = e.get("tags", {}) or {}
        if tags.get("power") != "generator" or not e.get("geometry"):
            continue
        n_gen += 1
        ax, ratio, length = polygon_axis([(g["lat"], g["lon"]) for g in e["geometry"]])
        if np.isfinite(ratio) and ratio >= min_ratio:
            axes.append(ax); lens.append(length)
    if len(axes) < min_rows:
        return {"row_axis_deg": np.nan, "n_elongated": len(axes), "n_generators": n_gen, "axis_spread_deg": np.nan, "prior_ok": False}
    a = np.radians(np.array(axes) * 2.0)                                    # period 180 -> 360
    w = np.array(lens)
    mean = np.degrees(np.arctan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a)))) / 2.0 % 180.0
    dev = (np.array(axes) - mean + 90.0) % 180.0 - 90.0
    spread = float(np.sqrt(np.average(dev ** 2, weights=w)))
    return {"row_axis_deg": float(mean), "n_elongated": len(axes), "n_generators": n_gen, "axis_spread_deg": spread,
            "prior_ok": bool(spread <= MAX_SPREAD_DEG)}


# ------------------------------------------------------------------ park-only selection (fix 2026-10-16)
def _utm(lat0: float, lon0: float):
    """Local equirectangular metres around (lat0, lon0): (x, y) from (lat, lon)."""
    kx, ky = 111320.0 * np.cos(np.radians(lat0)), 110540.0
    return lambda lat, lon: ((lon - lon0) * kx, (lat - lat0) * ky)


def select_park_generators(elements: list, lat: float, lon: float, dc_kwp: float, plant_dist_m: float = 300.0,
                           link_m: float = 120.0, search_m: float = 1500.0, m2_per_kwp: float = 6.0, min_ratio: float = 2.5) -> tuple:
    """The park's own power=generator polygons out of an Overpass response that covers a 3 km radius.
    Rule 1: generator polygons inside a power=plant outline (or landuse=industrial + plant:source=solar) within
    plant_dist_m of the park coordinate, outline buffered 60 m. Rule 2 (no usable outline): elongated generator
    polygons are clustered (single linkage, link_m); among clusters within search_m the one whose polygon area is
    closest (log scale) to the module area expected from dc_kwp wins, distance as tie-breaker.
    Returns (selected elements, selection name, n generator polygons in the response)."""
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    xy = _utm(lat, lon)
    def poly(e):
        return Polygon([xy(g["lat"], g["lon"]) for g in e["geometry"]]).buffer(0)
    gens = [e for e in elements if isinstance(e, dict) and e.get("type") == "way" and len(e.get("geometry") or []) >= 4
            and (e.get("tags") or {}).get("power") == "generator"]
    if not gens:
        return [], "none", 0
    plants = []
    for e in elements:
        if not isinstance(e, dict) or len(e.get("geometry") or []) < 4:
            continue
        t = e.get("tags") or {}
        if t.get("power") == "plant" or (t.get("landuse") == "industrial" and t.get("plant:source") == "solar"):
            try:
                plants.append(poly(e))
            except Exception:  # noqa: BLE001
                pass
    p0 = Point(0.0, 0.0)
    near = [pl for pl in plants if pl.distance(p0) <= plant_dist_m]
    if near:
        area = unary_union(near).buffer(60.0)
        keep = [e for e in gens if area.contains(poly(e).centroid)]
        if sum(1 for e in keep if polygon_axis([(g["lat"], g["lon"]) for g in e["geometry"]])[1] >= min_ratio) >= 3:
            return keep, "plant_outline", len(gens)
    # rule 2: cluster + area
    el = []
    for e in gens:
        ax, ratio, _ = polygon_axis([(g["lat"], g["lon"]) for g in e["geometry"]])
        if np.isfinite(ratio) and ratio >= min_ratio:
            pg = poly(e); el.append((e, np.array(pg.centroid.coords[0]), pg.area))
    if not el:
        return [], "no_elongated", len(gens)
    C = np.array([c for _, c, _ in el]); A = np.array([a for _, _, a in el])
    if len(C) > 1:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        lab = fcluster(linkage(pdist(C), "single"), link_m, "distance")
    else:
        lab = np.array([1])
    expected = max(dc_kwp, 50.0) * m2_per_kwp
    best = None
    for l in np.unique(lab):
        m = lab == l; d = float(np.hypot(C[m, 0], C[m, 1]).min()); area = float(A[m].sum())
        if d > search_m:
            continue
        score = abs(np.log(max(area, 1.0) / expected)) + d / 1000.0
        if best is None or score < best[0]:
            best = (score, l)
    if best is None:
        return [], "no_cluster_in_range", len(gens)
    return [e for (e, _, _), l in zip(el, lab) if l == best[1]], "cluster_area", len(gens)


def row_axis_park(elements: list, lat: float, lon: float, dc_kwp: float, **kw) -> dict:
    """row_axis_from_osm restricted to the park's own polygons (see select_park_generators)."""
    keep, sel, n_resp = select_park_generators(elements, lat, lon, dc_kwp, **kw)
    r = row_axis_from_osm(keep)
    r.update(selection=sel, n_generators_response=n_resp)
    return r
