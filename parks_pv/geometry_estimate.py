"""Angle estimation of a park: measured module geometry from open state geodata by the window method
(pv.geodata_windows, 2026-10-22). Library of scripts/pv/estimate_geometry.py and of stage `geometry` of generate_pv.py;
results are kept per park in parks_pv.geometry_store (estimated once, reused by every later release).

Per park: tiles of the state (DOP20, DGM1, surface sources: LAZ point cloud, laser DOM1, image bDOM / DOM20) ->
footprint = OSM solar plants (ways and relations) within 400 m of the register point, else a 300-m box grown while
accepted windows touch its edge (E4b) -> per surface source: height above ground -> windows -> groups -> arrays and
park acceptance (E4d) -> comparison of the main group with the MaStR orientation and tilt classes (or_agree,
tilt_agree, tilt_agree_tol) -> best accepted source (laser first, survey after commissioning or undated).

Generator arrays of a park (park_arrays): the measured arrays of the best source, else the register arrays of the
release baseline re-terrained on the state DGM1 (fallback_arrays), else the baseline arrays as they are.
Tiles are cached under <pv_shared_dir>/inputs/geodata/."""
import json
import os
import traceback

import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString, Point

from parks_pv import geometry as PG, params, paths
from pv import geodata_bayern as GB, geodata_states as GS, geodata_tiles as GT, geodata_windows as GW, site_inputs as SI
from pv.plausibility import orientation_class, tilt_class

GEO = os.path.join(paths.INPUT_DIR, "geodata")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Bayern": ("by", 32),
         "Nordrhein-Westfalen": ("nw", 32), "Hessen": ("he", 32), "Sachsen": ("sn", 33), "Thüringen": ("th", 32), "Sachsen-Anhalt": ("st", 32),
         "Baden-Württemberg": ("bw", 32), "Rheinland-Pfalz": ("rp", 32)}
SOURCE_RANK = {"laz": 0, "dom1_laser": 1, "bdom_image": 2, "dom20_image": 3}     # point cloud before DOM1 before image models (2026-10-22)
LASER = ("laz", "dom1_laser")
FENCE_MAX_M = 400.0
# review 2026-11-04 (Viktor), rules of this script; the window-method rules are pv.geodata_windows.RULES (--rules sets both)
RULES = dict(ni_bdom=True,        # R11: Niedersachsen also fetches the LGLN bDOM20 (image DSM, 20 cm) - DOM1 often predates young parks
             osm_fallback=False,  # R12 (probe): no plant near the register point -> plants near the MaStR unit points -> OSM generator areas
             gen_at_point=False,  # R19 (probe 2026-11-06): register point outside the chosen plant outline but inside a generator area lying outside it -> that area is the outline (SEL938315121090: plant polygon 130 m west of the park)
             c5_mixed=False,      # R13 (probe, not used): C5 parks = measured arrays + register arrays for the unit classes not measured
             az_known=True)       # R17 (Viktor 2026-11-05): a source with a group of undetermined azimuth is not accepted
GEN_AT_POINT_OVERLAP = 0.5        # R19: the generator area replaces the outline only if less than this share of it lies inside the outline
C5_MISSING_MAX = 0.10             # Viktor 2026-11-05: a C5 park with more capacity in unmeasured register directions leaves the fleet
EW_PAIR_TOL = 15.0                # opposite arrays within 180 +- this are one east-west construction
MIXED_AZ_TOL = 35.0               # R13: a register class counts as measured when a measured array lies within this azimuth distance
# R13: tilt of a register array = median measured tilt of the fleet run _t2g (73 measured parks, 2026-11-05) by
# (east-west, register tilt class, cohort) where >= 3 parks, else by (east-west, cohort); no random draw
MEASURED_TILT = {("EW", "< 20 Grad", "2021+"): 14.6, ("EW", "< 20 Grad", "<=2012"): 14.8,
                 ("S", "20 - 40 Grad", "<=2012"): 23.5, ("S", "20 - 40 Grad", "2013-20"): 21.3, ("S", "20 - 40 Grad", "2021+"): 19.7,
                 ("S", "< 20 Grad", "2013-20"): 15.3, ("S", "< 20 Grad", "2021+"): 17.0,
                 ("EW", "<=2012"): 12.8, ("EW", "2013-20"): 13.8, ("EW", "2021+"): 14.6, ("S", "<=2012"): 23.3, ("S", "2013-20"): 20.4, ("S", "2021+"): 18.1}
GEN_CLOSE_M = 15.0                # R12: generator areas (rows or fields) closed into outlines by a buffer of +/- this width
TILT_TOL_DEG = 2.0
CLASS_BOUNDS = {"< 20 Grad": (-90.0, 20.0), "20 - 40 Grad": (20.0, 40.0), "40 - 60 Grad": (40.0, 60.0), "> 60 Grad": (60.0, 90.0)}
DERIVED = ("ew_measured", "tilt_measured", "az_measured", "or_cls_measured", "tilt_cls_measured", "gate_after_build", "survey_date_known",
           "accept", "laser", "source_rank", "or_agree", "tilt_agree", "tilt_agree_tol", "best")


def fetch_tiles(state: str, lat: float, lon: float, year: int, landkreis: str = None, gemeinde: str = None, attempts: int = 3,
                margin_m: float = 300.0) -> dict:
    """{product: [paths]} with products dop20, dgm1 and the surface sources. NI takes the newest survey after the
    commissioning year and returns its dates; BB adds the image-based bDOM where ALS tiles are missing; Hessen needs the
    district and municipality (packages per municipality); Thüringen reads the hand-filled cache only."""
    last = None
    for k in range(attempts):
        try:
            if state == "Bayern":
                out = {"dop20": [], "dgm1": [], "dom20": []}
                for e, n in GS.tiles_for(lat, lon, 32, margin_m):        # full tile range (GB.tiles_for skips middle tiles)
                    for prod in ("dop20", "dgm1", "dom20"):
                        try:
                            out[prod].append(GB.fetch(prod, e, n, os.path.join(GEO, "bayern")))
                        except Exception as ex:
                            print(f"  {prod} {e}_{n}: {type(ex).__name__}", flush=True)
                out.update(GS.fetch_by_laser(lat, lon, GEO, margin_m))
                return out
            if state == "Niedersachsen":
                out = GS.fetch_ni(lat, lon, GEO, after=f"{year}-01-01", margin_m=margin_m)
                if RULES["ni_bdom"]:                                     # R11 (probe 2026-11-04, SEL968513427397: DOM1 2016 before the build)
                    b = GS.fetch_ni(lat, lon, GEO, products=("bdom20",), after=f"{year}-01-01", margin_m=margin_m)
                    out.update(bdom=b.get("bdom20", []), bdom_dates=b.get("bdom20_dates", {}))
                return out
            if state == "Brandenburg":
                return GS.fetch_bb(lat, lon, GEO, products=("dop20", "als", "bdom", "dgm1"), margin_m=margin_m)
            if state == "Nordrhein-Westfalen":
                return GS.fetch_nw(lat, lon, GEO, margin_m=margin_m)
            if state == "Hessen":
                return GS.fetch_he(lat, lon, GEO, landkreis, gemeinde, margin_m=margin_m)
            if state == "Sachsen":
                return GS.fetch_sn(lat, lon, GEO, margin_m=margin_m)
            if state == "Thüringen":
                return GS.fetch_th(lat, lon, GEO, margin_m=margin_m)
            if state == "Rheinland-Pfalz":
                return GS.fetch_rp(lat, lon, GEO, margin_m=margin_m)
            if state == "Sachsen-Anhalt":
                return GS.fetch_st(lat, lon, GEO, margin_m=margin_m)
            if state == "Baden-Württemberg":
                return GS.fetch_bw(lat, lon, GEO, margin_m=margin_m)
            return GS.FETCHERS[state](lat, lon, GEO, margin_m=margin_m)
        except RuntimeError as ex:                  # truncated transfers (curl 18) are retried
            last = ex
            print(f"  fetch attempt {k + 1} failed: {ex}", flush=True)
    raise last


def expected_fenced_m2(kwp: float, year: int) -> float:
    """Fenced area expected from the capacity and the era density of parks_pv (kWp per ha by commissioning year)."""
    from parks_pv.params import DENSITY_KWP_HA, DENSITY_YEARS
    return float(kwp) / float(np.interp(int(year), DENSITY_YEARS, DENSITY_KWP_HA)) * 1e4


def unit_points(r, to_m) -> list:
    """R12: points (metres) of the park's MaStR units; a Lokation point can be the mean of units far apart
    (SEL961760100570: units 0.19 and 0.63 km from it)."""
    global _UNIT_XY
    if "_UNIT_XY" not in globals():
        _UNIT_XY = pd.read_parquet(paths.mastr_file("units"),
                                   columns=["EinheitMastrNummer", "Breitengrad", "Laengengrad"]).dropna().set_index("EinheitMastrNummer")
    u = _UNIT_XY.reindex(str(r.unit_ids).split(";")).dropna()
    return [Point(*to_m(float(lo), float(la))) for la, lo in zip(u.Breitengrad, u.Laengengrad)]


def generator_fences(elements: list, points: list, to_m) -> pd.DataFrame:
    """R12: OSM solar generator areas (rows or fields) closed into outlines (buffer +/- GEN_CLOSE_M), one row per
    outline as in the fences table of osm_footprint (SEL978748582556: only generator areas, the park 270 m east)."""
    _, gen = GW.osm_footprint(elements, points, to_m, 1e9, power="generator")
    if not len(gen):
        return gen
    from shapely.ops import unary_union
    u = unary_union(list(gen.geom)).buffer(GEN_CLOSE_M).buffer(-GEN_CLOSE_M)
    parts = list(getattr(u, "geoms", [u]))
    return pd.DataFrame([dict(geom=g, name=None, typ="generator", id=None, area_ha=g.area / 1e4, dist_m=min(g.distance(p) for p in points))
                         for g in parts if g.area > 0])


def footprint(r, osm: dict, to_m, pt) -> tuple:
    """(polygon or None, fences table, n_used): OSM plants within FENCE_MAX_M of the register point, selected by
    pv.geodata_windows.select_fences (nearest plant, further ones while the union fits the capacity; 2026-10-22,
    replaces the union of all plants and the per-park exceptions of the same day). R12 (osm_fallback) only where
    this finds nothing: plants within FENCE_MAX_M of a MaStR unit point, then generator areas near the register or
    a unit point, under the same area limit. fences.attrs['kind']: osm | osm_units | osm_generator."""
    els = (osm.get(r.park_id) or {}).get("elements", [])
    exp = expected_fenced_m2(r.dc_kwp, r.year)
    _, fences = GW.osm_footprint(els, [pt], to_m, FENCE_MAX_M)
    poly, used, _ = GW.select_fences(fences, exp, FENCE_MAX_M)
    kind = "osm"
    if poly is not None and RULES["gen_at_point"] and not poly.contains(pt):      # R19
        gen = generator_fences(els, [pt], to_m)
        at = gen[[g.contains(pt) for g in gen.geom]] if len(gen) else gen
        if len(at) and at.geom.iloc[0].intersection(poly).area < GEN_AT_POINT_OVERLAP * at.geom.iloc[0].area:
            poly, used, _ = GW.select_fences(at, exp, FENCE_MAX_M)
            kind = "osm_generator_point"
    if poly is None and RULES["osm_fallback"]:
        pts = [pt] + unit_points(r, to_m)
        _, fences = GW.osm_footprint(els, pts, to_m, FENCE_MAX_M)
        poly, used, _ = GW.select_fences(fences, exp, FENCE_MAX_M)
        kind = "osm_units"
        if poly is None:
            fences = generator_fences(els, pts, to_m)
            poly, used, _ = GW.select_fences(fences, exp, FENCE_MAX_M)
            kind = "osm_generator"
    fences.attrs["kind"] = kind if poly is not None else None
    return poly, fences, used


EXTEND_ON = False   # variant c (2026-11-02, probe): --extend


def extend_ring_m(r) -> float:
    """Width of the ring around an OSM outline measured by variant c: half the side of the expected fenced square."""
    return float(max(GW.EXTEND["ring_min_m"], 0.5 * np.sqrt(expected_fenced_m2(r.dc_kwp, r.year))))


def fetch_margin(r, osm: dict, zone: int) -> float:
    """Half side (m) of the square around the register point that the tiles must cover: the OSM outline (+ 25 m, the
    analysis grid adds 20 m) or, without an outline, the largest box of measure_box; at least 300 m."""
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    x, y = to_m(float(r.lon), float(r.lat))
    poly, _, _ = footprint(r, osm, to_m, Point(x, y))
    if poly is not None:
        minx, miny, maxx, maxy = poly.buffer(25 + (GW.EXTEND["ring_max_factor"] * extend_ring_m(r) if EXTEND_ON else 0.0)).bounds
        m = max(x - minx, maxx - x, y - miny, maxy - y)
    else:
        m = GW.box_reach(float(r.dc_kwp), a_ref_m2(r)) + 60.0 + 25                 # one step (<= 60 m with R1) beyond the cap
    return float(max(m, 300.0))


def a_ref_m2(r) -> float:
    """R1: reference area of the box cap, max(fenced area by era density, MaStR area)."""
    area = float(r.area_ha) * 1e4 if pd.notna(r.area_ha) and r.area_ha > 0 else 0.0
    return max(expected_fenced_m2(r.dc_kwp, r.year), area)


def municipality(unit_ids: str) -> str:
    """Municipality of the park's MaStR units (most frequent), for the Hessen packages."""
    global _UNITS
    if "_UNITS" not in globals():
        _UNITS = pd.read_parquet(paths.mastr_file("units"), columns=["EinheitMastrNummer", "Gemeinde"]).set_index("EinheitMastrNummer")
    g = _UNITS.reindex(str(unit_ids).split(";")).Gemeinde.dropna()
    return g.mode().iloc[0] if len(g) else None


def sources_from(tiles: dict) -> list:
    src = []
    if tiles.get("dom1"):
        src.append(("dom1_laser", tiles["dom1"]))
    if tiles.get("laser") or tiles.get("als"):
        src.append(("laz", (tiles.get("laser") or []) + (tiles.get("als") or [])))
    if tiles.get("bdom"):
        src.append(("bdom_image", tiles["bdom"]))
    if tiles.get("dom20"):
        src.append(("dom20_image", tiles["dom20"]))
    return src


def rasters(tiles: dict, name: str, paths_: list, bounds: tuple, zone: int, lat: float, lon: float) -> dict:
    """Height above ground and the terrain of one surface source on the grid of bounds (0.5 m)."""
    S = GW.surface_from(name, paths_, bounds, zone, GW.RES, dates=tiles.get({"dom1_laser": "dom1_dates", "bdom_image": "bdom_dates"}.get(name, "")) or None)
    dgm = GT.mosaic(tiles["dgm1"], bounds, GW.RES, zone)[0].astype(np.float32) if tiles.get("dgm1") else None
    if dgm is None or not np.isfinite(dgm).any():                 # no DGM1: laser ground (LAZ) or a surface percentile (rasters)
        ground = S["ground"] if S["ground"] is not None else GT.ground_from_surface(S["surf"], GW.RES)
        dgm, dgm_src = np.asarray(ground, np.float32), "laser_ground" if S["ground"] is not None else "surface_percentile"
    else:
        dgm_src = "dgm1"
    hag = GW.height_above_ground(S["surf"], dgm, S["ground"])
    return dict(hag=hag, dgm=dgm, dgm_src=dgm_src, surf=S["surf"], source=name, survey_year=S["survey_year"], survey_dates=S["survey_dates"],
                conv=GT.meridian_convergence(lat, lon, zone))


def ortho_pitch(tiles: dict, bounds: tuple, inside: np.ndarray, zone: int) -> float:
    if not tiles.get("dop20"):
        return np.nan
    rgb = GT.mosaic(tiles["dop20"], bounds, GW.RES, zone, bands=[1, 2, 3])
    return GW.orthophoto_axis(rgb, inside, GW.RES)["pitch0"]


def measure(r, tiles: dict, osm: dict, zone: int, outd: str) -> tuple:
    """(rows per surface source, {source: window result}, footprint info) for one park."""
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    pt = Point(*to_m(float(r.lon), float(r.lat)))
    poly, fences, n_used = footprint(r, osm, to_m, pt)
    rows, outs = [], {}
    os.makedirs(outd, exist_ok=True)
    srcs = sources_from(tiles)
    if not srcs:
        return [dict(pv=r.park_id, source=None, status="failed", status_reason="keine Oberflächenquelle")], {}, dict(footprint="none")
    for name, paths_ in srcs:
        try:
            if poly is not None:
                g = GW.analysis_grid(poly, GW.RES)
                R = rasters(tiles, name, paths_, g["bounds"], zone, float(r.lat), float(r.lon))
                pitch0 = ortho_pitch(tiles, g["bounds"], g["inside"], zone)
                o = GW.measure_park(R["hag"], R["dgm"], g["inside"], R["conv"], name, pitch0, GW.RES, r.orientation_class,
                                    kwp=float(r.dc_kwp), centre_xy=(pt.x - g["bounds"][0], pt.y - g["bounds"][1]))
                fp = dict(footprint=fences.attrs.get("kind") or "osm", footprint_ha=round(poly.area / 1e4, 2), n_fences=int((fences.dist_m <= FENCE_MAX_M).sum()), n_fences_used=n_used)
                inside = g["inside"]
                cov0 = GW.coverage(o["park"], int(r.year))
                o0, R0, g0 = o, R, g
                if EXTEND_ON and o["park"]["status"] == "accepted" and cov0 < GW.EXTEND["cov_min"]:
                    from rasterio.features import geometry_mask
                    from shapely.affinity import translate
                    ring0 = extend_ring_m(r); ring, steps = ring0, []
                    while True:                                   # the ring grows while accepted windows touch its outer edge
                        g = GW.analysis_grid(poly.buffer(ring), GW.RES, buffer_m=20.0, inner_m=0.0)
                        core_in = ~geometry_mask([poly.buffer(-5.0)], (g["H"], g["W"]), g["transform"])
                        R = rasters(tiles, name, paths_, g["bounds"], zone, float(r.lat), float(r.lon))
                        o = GW.measure_park(R["hag"], R["dgm"], g["inside"], R["conv"], name, pitch0, GW.RES, r.orientation_class,
                                            kwp=float(r.dc_kwp), centre_xy=(pt.x - g["bounds"][0], pt.y - g["bounds"][1]), core_inside=core_in,
                                            year=int(r.year), outline_xy=translate(poly, -g["bounds"][0], -g["bounds"][1]))
                        ring_ = translate(poly.buffer(ring), -g["bounds"][0], -g["bounds"][1])
                        edge_ = ring_.exterior if ring_.geom_type == "Polygon" else MultiLineString([q.exterior for q in ring_.geoms])   # OSM outline of separate parts
                        ok_ = o["win"][o["win"].ok.astype(bool)]
                        n_edge = int(sum(edge_.distance(Point(x_, y_)) < o["win_m"] for x_, y_ in zip(ok_.x, ok_.y)))
                        cov_ = GW.coverage(o["park"], int(r.year))
                        steps.append((round(ring), n_edge, round(cov_, 3)))
                        if not n_edge or cov_ >= GW.EXTEND["cov_max"] or ring + GW.EXTEND["grow_m"] > GW.EXTEND["ring_max_factor"] * ring0:
                            break
                        ring += GW.EXTEND["grow_m"]
                    n_join = int((o["win"].ok.astype(bool) & ~o["win"].core.astype(bool)).sum()) if "core" in o["win"] else 0
                    fp.update(extend_ring_m=round(ring), coverage_outline=round(cov0, 3), extend_steps=str(steps), extend_edge_open=bool(n_edge), extend_joined=n_join)
                    if n_join and o["park"]["status"] == "accepted":
                        fp["extended"] = True
                        inside = g["inside"]
                    else:                                         # nothing joined or the extension fails: the outline result stays
                        fp["extended"] = False
                        o, R, g = o0, R0, g0
                fp["coverage"] = round(GW.coverage(o["park"], int(r.year)), 3)
            else:
                def load(bounds, name=name, paths_=paths_):
                    R_ = rasters(tiles, name, paths_, bounds, zone, float(r.lat), float(r.lon))
                    R_["pitch0"] = ortho_pitch(tiles, bounds, np.ones(R_["hag"].shape, bool), zone)
                    return R_
                o = GW.measure_box(load, (pt.x, pt.y), float(r.dc_kwp), register_orientation=r.orientation_class, a_ref_m2=a_ref_m2(r))
                R = o["rasters"]
                inside = np.ones(R["hag"].shape, bool)
                fp = dict(footprint="box", footprint_ha=round((o["park"]["box_side_m"] ** 2) / 1e4, 2), n_fences=0)
            o["dgm"], o["inside"], o["conv"] = R["dgm"], inside, R["conv"]
            outs[name] = o
            o["win"].to_csv(os.path.join(outd, f"windows_{name}.csv"), index=False)
            o["G"].to_csv(os.path.join(outd, f"groups_{name}.csv"), index=False)
            o["arrays"].to_csv(os.path.join(outd, f"arrays_{name}.csv"), index=False)
            park = dict(o["park"], survey_year=R["survey_year"], survey_dates=R["survey_dates"], dgm_src=R["dgm_src"], **fp)
            json.dump(park, open(os.path.join(outd, f"park_{name}.json"), "w"), indent=1, default=str)
            try:
                GW.plot_park(o, R["hag"], os.path.join(outd, f"windows_{name}.png"), f"{r.park_id}: {name}, Befliegung {R['survey_year']}, {park['status']}")
            except Exception as ex:                                    # a figure must never fail the measurement
                print("  fig error", r.park_id, name, ex, flush=True)
            G0 = o["G"].iloc[0] if len(o["G"]) else None
            A0 = o["arrays"][o["arrays"].group == 0] if len(o["arrays"]) else o["arrays"]
            rows.append(dict(pv=r.park_id, source=name, survey_year=R["survey_year"], survey_dates=str(R["survey_dates"])[:200], dgm_src=R["dgm_src"],
                             **fp, **{k: v for k, v in park.items() if k not in ("source", "survey_year", "survey_dates", "dgm_src", "box_steps") and k not in fp},
                             g0_tilt=G0.tilt_horiz if G0 is not None else np.nan, g0_tilt_p10=G0.tilt_p10 if G0 is not None else np.nan,
                             g0_tilt_p90=G0.tilt_p90 if G0 is not None else np.nan, g0_azimuth=float(A0.azimuth.iloc[0]) if len(A0) else np.nan,
                             g0_az_source=A0.az_source.iloc[0] if len(A0) else "", g0_ew=bool(G0.ew) if G0 is not None else False,
                             g0_pitch=G0.pitch if G0 is not None else np.nan, g0_share=G0.share if G0 is not None else np.nan,
                             g0_n_windows=int(G0.n_windows) if G0 is not None else 0, g0_n_sure=int(G0.n_sure) if G0 is not None else 0,
                             arrays=str([(round(a.tilt, 1), round(a.azimuth), round(a.share, 2), a.az_source) for a in o["arrays"].itertuples()])))
        except Exception as e:
            traceback.print_exc()
            rows.append(dict(pv=r.park_id, source=name, status="failed", status_reason=f"{type(e).__name__}: {str(e)[:100]}"))
    return rows, outs, dict(poly=poly)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Register comparison of the main group and the choice of the best source per park. A measured park needs
    status accepted and a survey after the commissioning year (unknown survey date: flagged, not rejected).
    or_agree is only defined where the azimuth was measured (an azimuth taken from the register class agrees by
    construction)."""
    df = df.copy()
    col = lambda c, d=np.nan: df[c] if c in df.columns else pd.Series(d, index=df.index)
    df["ew_measured"] = col("g0_ew", False).fillna(False).astype(bool)
    df["tilt_measured"] = col("g0_tilt").astype(float)
    df["az_measured"] = col("g0_azimuth").astype(float)
    df["or_cls_measured"] = [("Ost-West" if e else (orientation_class(a) if np.isfinite(a) else "")) for e, a in zip(df.ew_measured, df.az_measured)]
    df["tilt_cls_measured"] = [tilt_class(t) if np.isfinite(t) else "" for t in df.tilt_measured]
    sy = col("survey_year").astype(float)
    df["survey_date_known"] = sy.notna()
    df["gate_after_build"] = (sy >= col("year_built").astype(float)) | sy.isna()
    df["accept"] = (col("status", "") == "accepted") & df.gate_after_build
    if RULES["az_known"]:                                         # R17: another accepted source of the park may still be chosen
        df["accept"] &= ~col("az_undetermined", False).fillna(False).astype(bool)
    df["laser"] = col("source", "").isin(LASER)
    df["source_rank"] = col("source", "").map(SOURCE_RANK).fillna(9)
    az_measured = col("g0_az_source", "").isin(["measured", "axis"])
    df["or_agree"] = np.where(az_measured, df.or_cls_measured == col("orientation_class_reg", ""), np.nan)
    reg_t = col("tilt_class_reg", "")
    df["tilt_agree"] = df.tilt_cls_measured == reg_t
    df["tilt_agree_tol"] = [bool(np.isfinite(t) and c in CLASS_BOUNDS and CLASS_BOUNDS[c][0] - TILT_TOL_DEG <= t < CLASS_BOUNDS[c][1] + TILT_TOL_DEG)
                            for t, c in zip(df.tilt_measured, reg_t)]
    best = df[df.accept].sort_values(["pv", "source_rank"]).drop_duplicates("pv")
    df["best"] = df.index.isin(best.index)
    return df



def nominal_mounts(base_arrays: pd.DataFrame, park_id: str) -> list:
    """Nominal mounts of the register geometry (baseline_arrays.csv): one entry per nominal array (array_id prefix
    before the terrain cluster suffix), share summed, mount angles relative to the ground."""
    a = base_arrays[base_arrays.park_id == park_id]
    out = []
    for nom, g in a.groupby(a.array_id.str.rsplit("_", n=1).str[0], sort=False):
        out.append(dict(name=nom, tilt=float(g.mount_tilt.iloc[0]), azimuth=float(g.mount_azimuth.iloc[0]), share=float(g.share.sum()),
                        source=str(g.source.iloc[0])))
    return out


def fallback_arrays(r, tiles: dict, zone: int, base_arrays: pd.DataFrame, base_params: pd.DataFrame, osm: dict) -> tuple:
    """(arrays, source) of a park without an accepted measurement: the nominal register mounts of the baseline
    re-terrained on the state DGM1 over the baseline footprint (parks_pv.geometry.footprint: OSM plant within 2.5 km,
    else the MaStR area as a disc), so that baseline and fallback differ only in the terrain model (GLO-30 -> DGM1);
    without DGM1 the baseline arrays unchanged."""
    nominal = nominal_mounts(base_arrays, r.park_id)
    if nominal and tiles.get("dgm1") and zone:
        par = base_params.set_index("park_id").loc[r.park_id]
        area_fb = float(r.area_ha) if pd.notna(r.area_ha) and r.area_ha > 0 else float(par.kwp_dc) / float(par.kwp_per_ha)
        poly_ll, _ = PG.footprint(r, (osm.get(r.park_id) or {}).get("elements", []), area_fb)
        g = GW.analysis_grid(GT.to_utm(poly_ll, zone), GW.RES, inner_m=0.0)
        m_ = GT.mosaic(tiles["dgm1"], g["bounds"], GW.RES, zone)           # None when no DGM1 tile touches the footprint
        dgm = m_[0].astype(np.float32) if m_ is not None else None
        if dgm is not None and np.isfinite(dgm[g["inside"]]).mean() > 0.5:
            arr_, _ = PG.terrain_arrays_grid(nominal, dgm, g["inside"], GW.RES, GT.meridian_convergence(float(r.lat), float(r.lon), zone))
            return arr_, "register+dgm1"
    a = base_arrays[base_arrays.park_id == r.park_id]
    return [dict(name=x.array_id, tilt=x.tilt, azimuth=x.azimuth, share=x.share, mount_tilt=x.mount_tilt, mount_azimuth=x.mount_azimuth)
            for x in a.itertuples()], "register"


def _az_dist(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def register_classes(r) -> dict:
    """Orientation classes of a park by its MaStR units: {class: [kWp share, tilt class]}. Hauptausrichtung with the
    unit's kWp share, or half of it each for Haupt- and Nebenausrichtung when the unit has one (MaStR gives no split);
    north classes are skipped (register noise on ground parks)."""
    from pv.plausibility import AZ_CENTRE
    u = pd.read_parquet(paths.mastr_file("units"),
                        columns=["EinheitMastrNummer", "Bruttoleistung", "Hauptausrichtung", "HauptausrichtungNeigungswinkel", "Nebenausrichtung",
                                 "NebenausrichtungNeigungswinkel"]).set_index("EinheitMastrNummer").reindex(str(r.unit_ids).split(";")).dropna(subset=["Bruttoleistung"])
    ok = lambda c: c == "Ost-West" or c in AZ_CENTRE and not str(c).startswith("Nord")
    reg, tot = {}, float(u.Bruttoleistung.sum()) or 1.0
    for x in u.itertuples():
        parts = [(x.Hauptausrichtung, x.HauptausrichtungNeigungswinkel)] + ([(x.Nebenausrichtung, x.NebenausrichtungNeigungswinkel)] if ok(x.Nebenausrichtung) else [])
        for c, tc in parts:
            if ok(c):
                reg.setdefault(c, [0.0, tc])[0] += float(x.Bruttoleistung) / tot / len(parts)
    if "Ost" in reg and "West" in reg:                                # units registered as Ost and as West = one east-west park
        o, w = reg.pop("Ost"), reg.pop("West")
        ew = reg.setdefault("Ost-West", [0.0, o[1]])
        ew[0] += o[0] + w[0]
    return reg


def measured_classes(az_measured: list) -> set:
    """Register classes the measured arrays stand for. Two arrays facing opposite ways (180 +- EW_PAIR_TOL deg) with
    the pair closer to east/west than to north/south are one Ost-West construction, also when rotated (SEL965286735001:
    115/295 deg); every other array goes to its nearest MaStR class (within MIXED_AZ_TOL; north classes allowed)."""
    from pv.plausibility import AZ_CENTRE
    az = [float(x) % 360.0 for x in az_measured]
    out, used = set(), set()
    for i in range(len(az)):
        for k in range(i + 1, len(az)):
            if i in used or k in used:
                continue
            if abs(_az_dist(az[i], az[k]) - 180.0) <= EW_PAIR_TOL and min(_az_dist(az[i], 90.0), _az_dist(az[i], 270.0)) <= 45.0:
                out.add("Ost-West"); used |= {i, k}
    for i, x in enumerate(az):
        if i in used:
            continue
        c = min(AZ_CENTRE, key=lambda k: _az_dist(x, AZ_CENTRE[k]))
        if _az_dist(x, AZ_CENTRE[c]) <= MIXED_AZ_TOL:
            out.add(c)
    if {"Ost", "West"} <= out:
        out = (out - {"Ost", "West"}) | {"Ost-West"}
    return out


def unmeasured_classes(r, az_measured: list) -> dict:
    """Register classes of the park (register_classes) that no measured array stands for (measured_classes)."""
    m = measured_classes(az_measured)
    return {c: v for c, v in register_classes(r).items() if c not in m}


def measured_cls(arrays: pd.DataFrame) -> str:
    """Fleet class of the measured geometry (pv.selection classes): two or more measured directions C5_multi, east-west
    C3, one south class C1/C2 by the share-weighted tilt (< 20 deg C2), one off-south class C4."""
    m = measured_classes(arrays.azimuth.tolist())
    if len(m) >= 2:
        return "C5_multi"
    if m == {"Ost-West"}:
        return "C3_east_west"
    if m == {"Süd"}:
        return "C2_south_flat" if float(np.average(arrays.tilt, weights=arrays.share)) < 20.0 else "C1_south_std"
    return "C4_off_south" if m else ""


def c5_check(r, arrays: pd.DataFrame) -> dict:
    """Viktor 2026-11-05 for parks of the register class C5: usable when at most C5_MISSING_MAX of the capacity lies in
    register directions the measurement does not hold; the park then goes to the class of its measured geometry
    (measured_cls: C5 only with two or more measured directions)."""
    miss = unmeasured_classes(r, arrays.azimuth.tolist())
    share = float(sum(v[0] for v in miss.values()))
    return dict(missing=sorted(miss), missing_share=round(share, 3), usable=share <= C5_MISSING_MAX, cls_measured=measured_cls(arrays))


def mixed_register_arrays(r, measured: list) -> list:
    """R13 (c5_mixed, off; not used: Viktor 2026-11-05 takes only completely measured C5 parks): register arrays for the
    classes the measurement did not find (unmeasured_classes). Nothing is added when no register class is measured (the
    register then contradicts the measurement) or all are. Tilt: median measured tilt of the fleet (MEASURED_TILT) for
    the unit's tilt class and the cohort, azimuth the class centre, no terrain."""
    from pv.plausibility import AZ_CENTRE
    az_m = [float(a["azimuth"]) for a in measured]
    reg, miss = register_classes(r), unmeasured_classes(r, az_m)
    if not reg or len(miss) == len(reg) or not miss:
        return []
    tilt = lambda kind, tc: MEASURED_TILT.get((kind, tc, r.cohort), MEASURED_TILT[(kind, r.cohort)])
    out = []
    for c, (share, tc) in miss.items():
        if c == "Ost-West":
            t = tilt("EW", tc)
            out += [dict(name="reg_ew_e", tilt=t, azimuth=90.0, share=share / 2), dict(name="reg_ew_w", tilt=t, azimuth=270.0, share=share / 2)]
        else:
            out.append(dict(name=f"reg_{c}", tilt=tilt("S", tc), azimuth=AZ_CENTRE[c], share=share))
    return out


def measured_arrays(r, df_park: pd.DataFrame, outs: dict = None, park_dir: str = None) -> list:
    """Generator array rows of the best accepted source of one park ([] when none): from the window result in memory
    (outs) or the stored arrays_<source>.csv of the park directory."""
    best = df_park[df_park.best.astype(bool)] if "best" in df_park and len(df_park) else df_park.iloc[:0]
    if not len(best):
        return []
    src = best.source.iloc[0]
    A = outs[src]["arrays"] if outs and src in outs else pd.read_csv(os.path.join(park_dir, f"arrays_{src}.csv"))
    rows = [dict(park_id=r.park_id, array_id=f"g{int(a.group)}_{k}", tilt=round(a.tilt, 2), azimuth=round(a.azimuth, 2), share=round(a.share, 5),
                 source=f"windows:{src}", az_source=a.az_source, note=a.note if pd.notna(a.note) else "", mount_tilt=np.nan, mount_azimuth=np.nan)
            for k, a in enumerate(A.itertuples())]
    if RULES["c5_mixed"] and getattr(r, "cls", None) == "C5_multi":
        extra = mixed_register_arrays(r, rows)
        s_reg = sum(x["share"] for x in extra)
        for x in rows:
            x["share"] = round(x["share"] * (1.0 - s_reg), 5)
        rows += [dict(park_id=r.park_id, array_id=x["name"], tilt=round(x["tilt"], 2), azimuth=round(x["azimuth"], 2), share=round(x["share"], 5),
                      source="register_units", az_source="register", note="c5_mixed", mount_tilt=np.nan, mount_azimuth=np.nan) for x in extra]
    return rows


def register_rows(r, arrays: list, source: str) -> list:
    """Generator array rows of the register fallback (fallback_arrays)."""
    return [dict(park_id=r.park_id, array_id=a["name"], tilt=round(a["tilt"], 2), azimuth=round(a["azimuth"], 2), share=round(a["share"], 5),
                 source=source, az_source="register", note="", mount_tilt=round(a.get("mount_tilt", np.nan), 2),
                 mount_azimuth=round(a.get("mount_azimuth", np.nan), 2)) for a in arrays]


def park_tiles(r, osm: dict) -> tuple:
    """(tiles, zone) of one park; ({}, None) for a state without a fetcher."""
    if r.bundesland not in STATE:
        return {}, None
    zone = STATE[r.bundesland][1]
    tiles = fetch_tiles(r.bundesland, float(r.lat), float(r.lon), int(r.year), r.landkreis,
                        (municipality(r.unit_ids) if params.has_units(r) else r.gemeinde) if r.bundesland == "Hessen" else None,
                        margin_m=fetch_margin(r, osm, zone))
    return tiles, zone


def set_rules(spec: str) -> str:
    """Switch rules by name ('box_cap,-sections'); returns the active rules as text (column `rules` of the run)."""
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        on, name = not tok.startswith("-"), tok.lstrip("-+")
        d = GW.RULES if name in GW.RULES else RULES if name in RULES else None
        if d is None:
            raise SystemExit(f"unknown rule {name!r}: {sorted(GW.RULES) + sorted(RULES)}")
        d[name] = on
    return active_rules()


# rule set of the production estimates (methods t2v, w2a, v4); a new method starts from it (estimate_geometry.py --rules
# switches single rules on or off)
PRODUCTION_RULES = ("box_cap box_tab_near box_dir auto_outline ew_one_flank sections contrast_ew centreline tent_pair min2_windows axis_tol5 "
                    "ni_bdom osm_fallback gen_at_point az_known")


def method_rules(method: str, changes: str = "", extend: bool = False) -> str:
    """Set the rules of a method: an existing method keeps its stored rule set (changes must not alter it); a new one
    gets PRODUCTION_RULES with `changes` applied and is recorded in the store. Returns the active rule text."""
    from parks_pv import geometry_store as GSt
    global EXTEND_ON
    spec = GSt.method_spec(method)
    use_rules(spec["rules"] if spec else PRODUCTION_RULES)
    EXTEND_ON = bool(spec["extend"]) if spec else extend
    before = active_rules()
    set_rules(changes)
    if spec:
        if active_rules() != before or (extend and not spec["extend"]):
            raise SystemExit(f"method {method} has the rule set {spec['rules']!r} (extend {spec['extend']}); other rules need a new method")
    else:
        GSt.write_method_spec(method, active_rules(), EXTEND_ON)
    return active_rules()


def use_rules(active: str) -> None:
    """Set the rules exactly to a stored rule text (column `rules` of sources.csv): listed names on, all others off."""
    on = set(str(active).split())
    for d in (GW.RULES, RULES):
        for k in d:
            d[k] = k in on


def active_rules() -> str:
    return " ".join(k for k, v in {**GW.RULES, **RULES}.items() if v)


def summary(df: pd.DataFrame, label: str) -> None:
    pd.set_option("display.width", 250)
    print(f"\n{label}: {df.pv.nunique()} parks; accepted at {df[df.accept].pv.nunique()} (laser {df[df.accept & df.laser].pv.nunique()}); "
          f"status per source {df.status.value_counts().to_dict()}")
    b = df[df.best]
    if len(b):
        om = b[pd.notna(b.or_agree)]
        print(f"orientation class agrees {int(om.or_agree.astype(bool).sum())}/{len(om)} (measured azimuth); tilt class {int(b.tilt_agree.sum())}/{len(b)} "
              f"(+-{TILT_TOL_DEG:.0f} deg: {int(b.tilt_agree_tol.sum())}/{len(b)})")
        cols = ["pv", "source", "footprint", "n_ok", "tab_share_accepted", "orientation_class_reg", "or_cls_measured", "tilt_class_reg", "tilt_measured",
                "tilt_ends_med", "az_measured", "g0_az_source", "ew_measured", "survey_year", "year_built"]
        print(b[[c for c in cols if c in b.columns]].round(2).to_string(index=False))


def load_osm(sel: pd.DataFrame) -> dict:
    """OSM cache for the selected parks; missing entries are fetched first, never silently measured in the box."""
    d = os.path.join(paths.INPUT_DIR, "osm")
    return SI.ensure_osm(sel, os.path.join(d, "parks_osm.json"), os.path.join(d, "pvgis_horizon.json"))


def estimate_park(r, osm: dict, outd: str) -> tuple:
    """Angle estimation of one selection row: (evaluated rows per surface source, {source: window result}, tiles, zone).
    Window outputs go to outd. A state without a fetcher gives one row with status no_fetcher; errors one row failed."""
    if r.bundesland not in STATE:
        o, outs, tiles, zone = [dict(pv=r.park_id, source=None, status="no_fetcher", status_reason=f"no geodata fetcher for {r.bundesland}")], {}, {}, None
    else:
        tiles, zone = {}, STATE[r.bundesland][1]
        try:
            tiles, zone = park_tiles(r, osm)
            o, outs, _ = measure(r, tiles, osm, zone, outd)
        except Exception as e:
            traceback.print_exc()
            o, outs = [dict(pv=r.park_id, source=None, status="failed", status_reason=f"{type(e).__name__}: {str(e)[:100]}")], {}
    code = STATE.get(r.bundesland, ("",))[0]
    for x in o:
        x.update(state_code=code, orientation_class_reg=r.orientation_class, tilt_class_reg=r.tilt_class, dc_kwp=float(r.dc_kwp), year_built=int(r.year),
                 rules=active_rules())
    return evaluate(pd.DataFrame(o)), outs, tiles, zone


def estimate_parks(sel: pd.DataFrame, method: str, force: bool = False) -> list:
    """Estimate the angles of the selection rows that have no estimate of `method` in the store yet (all with force)
    and store them (parks_pv.geometry_store), with the rule set of the method (method_rules); returns the park ids."""
    from parks_pv import geometry_store as GSt
    todo = sel if force else sel[sel.park_id.isin(GSt.missing(method, sel.park_id))]
    if not len(todo):
        return []
    if GSt.method_spec(method) is None or active_rules() != GSt.method_spec(method)["rules"]:
        method_rules(method)                                      # rule set of the method (stage geometry: no changes)
    osm = load_osm(todo)
    done = []
    for r in todo.itertuples():
        print(f"== {r.park_id} {r.bundesland} {r.dc_kwp:.0f} kWp {r.year} {r.orientation_class} / {r.tilt_class}", flush=True)
        d = GSt.park_dir(method, r.park_id)
        if os.path.isdir(d):                                      # force: a new estimate replaces the park directory
            import shutil
            shutil.rmtree(d)
        df, outs, _, _ = estimate_park(r, osm, d)
        GSt.save(method, r.park_id, df, measured_arrays(r, df, outs),
                 meta=dict(rules=active_rules(), extend=EXTEND_ON, state=r.bundesland,
                           register={k: getattr(r, k, None) for k in ("lat", "lon", "dc_kwp", "year", "area_ha", "orientation_class", "tilt_class", "unit_ids")}))
        from pv import geometry_overrides as GO
        GO.apply_to_store(method, parks=[r.park_id])              # manual corrections of this park and method, if any
        print("  ", [(x.source, x.status, x.get("n_ok"), round(x.get("g0_tilt", np.nan), 1), x.get("arrays")) for _, x in df.iterrows()], flush=True)
        done.append(r.park_id)
    return done
