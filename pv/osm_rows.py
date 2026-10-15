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
