"""Module geometry of the Trianel parks from open state geodata (all states): row axis and pitch from the
orthophoto (2D spectrum + window map, true-north corrected), tilt from every available surface source
(laser DOM1 raster, LAZ point cloud, image-based bDOM/DOM20) via folded profile + flank plane fit.

Footprint: OSM plant polygon within 400 m of the park coordinate; else the bounds of the earlier Bavarian run
(pvX_rasters.tif); else a disc sized from the master-data capacity. Outputs (anonymous, no coordinates):
  reports/figs_pv_generator/geodata_states_trianel.csv      one row per park and surface source
  reports/figs_pv_generator/geodata_states_trianel_rows.csv row-axis peaks and blocks per park
  results/pv/geodata_states/<pv>_profile_<source>.png       folded profiles (no imagery)
Usage: synthre/bin/python scripts/pv/geometry_trianel_states.py [--parks pv1,pv2,...]"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, os.path.dirname(__file__))
from pv import geodata_tiles as GT  # noqa: E402
import trianel_common as TC  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
B = os.path.join(ROOT, "synthetic/solar/trianel_benchmark/inputs")
GEO = os.path.join(B, "geodata")
FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "FL_Contribution", "reports", "figs_pv_generator")
OUTD = os.path.join(os.path.dirname(__file__), "..", "..", "results", "pv", "geodata_states")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Sachsen-Anhalt": ("st", 32), "Bayern": ("by", 32), "Rheinland-Pfalz": ("rp", 32)}


def park_states() -> pd.DataFrame:
    mc = pd.read_csv(os.path.join(B, "mastr", "mastr_candidates.csv"))
    units = pd.read_parquet(os.path.join(ROOT, "synthetic/solar/mastr/solar_units_20250625.parquet"), columns=["EinheitMastrNummer", "Bundesland"])
    j = mc.merge(units, on="EinheitMastrNummer", how="left")
    return j.groupby("pv").agg(state=("Bundesland", lambda x: x.dropna().mode().iloc[0]), year=("Inbetriebnahmedatum", lambda x: int(str(x.min())[:4])))


def footprint(pv: str, lat: float, lon: float, kw: float, zone: int) -> tuple:
    """(polygon in CRS[zone], source)."""
    osm = json.load(open(os.path.join(B, "osm", "trianel_pv_osm.json"))).get(pv, {}).get("elements", [])
    p = Point(*GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(lon, lat))
    plants = []
    for e in osm:
        if e.get("type") == "way" and (e.get("tags") or {}).get("power") == "plant" and len(e.get("geometry", [])) >= 4:
            poly = GT.to_utm(Polygon([(g["lon"], g["lat"]) for g in e["geometry"]]), zone).buffer(0)
            if poly.distance(p) <= 400:
                plants.append(poly)
    if plants:
        from shapely.ops import unary_union
        return unary_union(plants), "osm"
    old = os.path.join(GEO, f"{pv}_rasters.tif")
    if os.path.exists(old):
        with rasterio.open(old) as r:
            bb = r.bounds
        return Polygon([(bb.left, bb.bottom), (bb.right, bb.bottom), (bb.right, bb.top), (bb.left, bb.top)]), "bayern_run"
    radius = np.sqrt(max(kw, 300.0) / 600.0 * 1e4 / np.pi) * 1.3 + 60.0
    return p.buffer(radius), "disc"


def tiles(sub: str, prod: str) -> list:
    return sorted(glob.glob(os.path.join(GEO, sub, prod, "*")))


def run_park(pv: str, p: pd.Series, st: pd.Series, args) -> list:
    sub, zone = STATE[st.state]
    poly, fp_src = footprint(pv, float(p.lat), float(p.lon), float(p.kw_master), zone)
    conv = GT.meridian_convergence(float(p.lat), float(p.lon), zone)
    minx, miny, maxx, maxy = poly.buffer(20).bounds
    bounds = (minx, miny, maxx, maxy)
    res_img, res_dsm = 0.5, 0.5
    H, W = int(round((maxy - miny) / res_img)), int(round((maxx - minx) / res_img))
    inside = ~geometry_mask([poly.buffer(-5)], (H, W), from_bounds(*bounds, W, H))
    out_rows, rows_rows = [], []
    # ---- orthophoto
    dop = tiles(sub, "dop20") + (tiles("by", "dop20") if sub == "by" else [])
    if sub == "by":
        dop = sorted(glob.glob(os.path.join(GEO, "dop20_*.tif")))
    rows = {"peaks": [], "blocks": [], "n_windows": 0}
    if args.surface_only:
        dop = []
    if dop:
        rgb = GT.mosaic(dop, bounds, res_img, zone, bands=[1, 2, 3])
        rows = GT.row_axis_from_image(rgb, inside, res_img)
        for k, (a, pch, s) in enumerate(rows["peaks"][:3]):
            rows_rows.append(dict(pv=pv, state_code=sub, footprint=fp_src, rank=k, axis_grid=round(a, 1), axis_true=round((a + conv) % 180, 1), pitch_m=round(pch, 2), share=round(s, 3)))
        for b in rows["blocks"]:
            rows_rows.append(dict(pv=pv, state_code=sub, footprint=fp_src, rank="block", axis_grid=round(b["axis_grid"], 1), axis_true=round((b["axis_grid"] + conv) % 180, 1), pitch_m=round(b["pitch_m"], 2), share=round(b["share_windows"], 3), n_windows=b["n_windows"]))
    axis_grid = rows["blocks"][0]["axis_grid"] if rows["blocks"] else (rows["peaks"][0][0] if rows["peaks"] else np.nan)
    pitch = rows["blocks"][0]["pitch_m"] if rows["blocks"] else (rows["peaks"][0][1] if rows["peaks"] else np.nan)
    if fp_src == "disc" and np.isfinite(axis_grid) and rows.get("windows") is not None and len(rows["windows"]):
        hull = GT.footprint_from_windows(rows["windows"], axis_grid, 1.0, bounds)
        if hull is not None and hull.area > 2000:
            poly = hull.intersection(poly); fp_src = "disc+windows"
            inside = ~geometry_mask([poly], (H, W), from_bounds(*bounds, W, H))
    # ---- terrain
    dgm_paths = tiles(sub, "dgm1") or (sorted(glob.glob(os.path.join(GEO, "dgm1_*.tif"))) if sub == "by" else [])
    dgm = GT.mosaic(dgm_paths, bounds, res_dsm, zone)[0] if dgm_paths else None
    # ---- surface sources
    sources = []
    if tiles(sub, "dom1"): sources.append(("dom1_laser", tiles(sub, "dom1")))
    if tiles(sub, "als") or tiles(sub, "laser"): sources.append(("laz", tiles(sub, "als") + tiles(sub, "laser")))
    if tiles(sub, "bdom"): sources.append(("bdom_image", tiles(sub, "bdom")))
    if sub == "by" and glob.glob(os.path.join(GEO, "dom20_*.tif")): sources.append(("dom20_image", sorted(glob.glob(os.path.join(GEO, "dom20_*.tif")))))
    axis_grid0, pitch0 = axis_grid, pitch
    for name, paths in sources:
        axis_grid, pitch = axis_grid0, pitch0
        try:
            if name == "laz":
                surf, ground, npts = GT.dsm_from_laz(paths, bounds, res_dsm)
                base = ground if dgm is None else np.where(np.isfinite(dgm), dgm, ground)
                hag = surf - base; dates = GT.laz_dates(paths)
            else:
                surf = GT.mosaic(paths, bounds, res_dsm, zone)[0]
                base = dgm if dgm is not None else GT.ground_from_surface(surf, res_dsm)
                hag = surf - base; dates = [os.path.basename(x) for x in paths] + (["ground: surface-percentile"] if dgm is None else [])
            axis_grid_src, pitch_src = axis_grid, pitch
            if args.surface_only or not np.isfinite(axis_grid):
                # row axis and pitch from the height image itself: spectrum of hag (clipped 0..3 m) inside a coarse block
                exp0 = float(p.kw_master) * 5.0 * 2.2; pxy0 = GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(float(p.lon), float(p.lat))
                fp0 = GT.footprint_from_hag(hag, res_dsm, bounds, exp0, pxy0, pitch_m=10.0)
                ins0 = ~geometry_mask([fp0], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0])) if fp0 is not None and fp0.area > 1500 else np.isfinite(hag)
                hclip = np.clip(np.where(np.isfinite(hag), hag, 0.0), 0.0, 3.0) * 80.0
                rs = GT.row_axis_from_image(np.stack([hclip] * 3), ins0, res_dsm)
                if rs["blocks"]:
                    axis_grid_src, pitch_src = rs["blocks"][0]["axis_grid"], rs["blocks"][0]["pitch_m"]
                elif rs["peaks"]:
                    axis_grid_src, pitch_src = rs["peaks"][0][0], rs["peaks"][0][1]
                else:
                    out_rows.append(dict(pv=pv, source=name, ok=False, reason="no row pattern in surface")); continue
                rows_rows.append(dict(pv=pv, state_code=sub, footprint="surface", rank=f"surface:{name}", axis_grid=round(axis_grid_src, 1), axis_true=round((axis_grid_src + conv) % 180, 1), pitch_m=round(pitch_src, 2), share=round(rs["peaks"][0][2], 3) if rs["peaks"] else np.nan, n_windows=rs["n_windows"]))
            axis_grid, pitch = axis_grid_src, pitch_src
            exp_m2 = float(p.kw_master) * 5.0 * 2.2        # modules ~5 m2/kWp, block area incl. gaps ~2.2x
            pxy = GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(float(p.lon), float(p.lat))
            fp_hag = GT.footprint_from_hag(hag, res_dsm, bounds, exp_m2, pxy, pitch_m=pitch if np.isfinite(pitch) else 10.0)
            if fp_hag is not None and fp_hag.area > 1500:
                blk = fp_hag.intersection(poly) if fp_src in ("osm",) and fp_hag.intersection(poly).area > 1500 else fp_hag
                ins = ~geometry_mask([blk], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0])); fp_note = f"hag-block {blk.area / 1e4:.1f} ha"
            else:
                ins = inside if hag.shape == inside.shape else ~geometry_mask([poly.buffer(-5)], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0])); fp_note = "outline"
            pitch_s, pcorr = GT.pitch_from_surface(hag, ins, res_dsm, axis_grid)
            use_pitch = pitch_s if np.isfinite(pitch_s) and pcorr >= 0.2 else pitch
            axis_ref, pitch_ref, contrast = GT.refine_fold(hag, ins, res_dsm, axis_grid, use_pitch)
            t = GT.tilt_from_surface(hag, ins, res_dsm, axis_ref, pitch_ref)
            t["pitch_surface"] = pitch_s; t["pitch_surface_corr"] = pcorr; t["pitch_used"] = pitch_ref; t["axis_surface_true"] = round((axis_ref + conv) % 180, 2); t["fold_contrast"] = round(contrast, 2); t["block"] = fp_note; t["hag_p50_block"] = float(np.nanpercentile(hag[ins], 50)); t["hag_p90_block"] = float(np.nanpercentile(hag[ins], 90))

            # --- terrain slope in the facing direction (2026-10-21): hag-based tilts are relative to the local ground
            if t.get("ok") and base is not None:
                gy_, gx_ = np.gradient(np.where(np.isfinite(base), base, np.nan), res_dsm)
                dgE, dgN = gx_, -gy_                                   # rows grow southwards -> flip the sign of d/d(row)
                dd = np.radians(t["flank_dir_grid"])                   # facing direction in the grid frame
                slope_facing = -(dgE * np.sin(dd) + dgN * np.cos(dd))  # > 0: ground falls towards the facing direction
                okc = ins & np.isfinite(slope_facing) & np.isfinite(hag) & (hag > -0.5) & (hag < 6)
                g_fac = float(np.nanmedian(slope_facing[okc])) if okc.any() else float("nan")
                t["slope_facing"] = g_fac
                t["terrain_deg"] = float(np.degrees(np.arctan(np.hypot(np.nanmedian(dgE[okc]), np.nanmedian(dgN[okc]))))) if okc.any() else float("nan")
                hz = lambda b, g: float(np.degrees(np.arctan(np.tan(np.radians(b)) + g)))
                t["tilt_profile_horiz"] = hz(t["tilt_profile"], g_fac)
                t["tilt_plane_horiz"] = hz(t["tilt_plane_median"], g_fac) if np.isfinite(t["tilt_plane_median"]) else float("nan")
                chosen_left = t["flank_left_m"] > t["flank_right_m"]   # the chosen (longer) flank faces flank_dir, the other one the opposite way
                t["tilt_left_horiz"] = hz(t["tilt_left"], g_fac if chosen_left else -g_fac)
                t["tilt_right_horiz"] = hz(t["tilt_right"], -g_fac if chosen_left else g_fac)
            try:
                import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                os.makedirs(OUTD, exist_ok=True); fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
                im = ax[0].imshow(np.clip(hag, -0.5, 4), cmap="viridis", extent=[0, hag.shape[1] * res_dsm, 0, hag.shape[0] * res_dsm]); plt.colorbar(im, ax=ax[0], label="m above ground")
                ax[0].contour(np.flipud(ins.astype(float)), levels=[0.5], colors="red", linewidths=0.8, extent=[0, hag.shape[1] * res_dsm, 0, hag.shape[0] * res_dsm]); ax[0].set_title(f"{pv} {name}: height above ground, red = block used ({fp_note})", fontsize=9)
                cen, prof = t["profile"]; ax[1].plot(cen, prof, color="#8a3b12"); ax[1].set_xlabel("across-row position (m)"); ax[1].set_ylabel("m above ground"); ax[1].set_title(f"folded profile, axis {axis_ref:.1f} pitch {pitch_ref:.2f} m: tilt {t['tilt_profile']:.1f} deg (L {t['tilt_left']:.1f} / R {t['tilt_right']:.1f}), plane {t['tilt_plane_median']:.1f} deg", fontsize=9)
                plt.tight_layout(); plt.savefig(os.path.join(OUTD, f"{pv}_hag_{name}.png"), dpi=100); plt.close()
            except Exception as e:
                print("fig error", pv, name, e)
            row = dict(pv=pv, state_code=sub, footprint=fp_src, source=name, axis_true=round((axis_grid + conv) % 180, 1), pitch_m=round(pitch, 2), convergence=round(conv, 2), year_built=int(st.year), dates=str(dates)[:120], **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items() if k != "profile"})
            if t.get("ok"):
                row["azimuth_true"] = round((t["flank_dir_grid"] + conv) % 360, 1)
                import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                cen, prof = t["profile"]; os.makedirs(OUTD, exist_ok=True)
                plt.figure(figsize=(6, 3)); plt.plot(cen, prof, color="#8a3b12"); plt.xlabel("across-row position (m)"); plt.ylabel("m above ground")
                plt.title(f"{pv} {name}: tilt profile {t['tilt_profile']:.1f} deg, plane {t['tilt_plane_median']:.1f} deg, flank {t['flank_m']:.1f} m"); plt.tight_layout(); plt.savefig(os.path.join(OUTD, f"{pv}_profile_{name}.png"), dpi=110); plt.close()
            out_rows.append(row)
        except Exception as e:
            out_rows.append(dict(pv=pv, source=name, ok=False, reason=f"{type(e).__name__}: {str(e)[:100]}"))
    if not sources:
        out_rows.append(dict(pv=pv, state_code=sub, footprint=fp_src, source=None, ok=False, reason="no surface source", axis_true=round((axis_grid + conv) % 180, 1) if np.isfinite(axis_grid) else np.nan, pitch_m=pitch))
    return out_rows, rows_rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parks", default="pv1,pv2,pv3,pv5,pv6,pv7")
    ap.add_argument("--surface-only", action="store_true", help="row axis, pitch and block from the surface model alone (no orthophoto)")
    ap.add_argument("--suffix", default=""); a = ap.parse_args()
    parks = TC.parks_table(TC.load_masterdata()); st = park_states()
    out, rows = [], []
    for pv in a.parks.split(","):
        try:
            o, r = run_park(pv, parks.loc[pv], st.loc[pv], a); out += o; rows += r
            print(pv, [(x.get("source"), x.get("ok"), x.get("tilt_profile"), x.get("tilt_plane_median"), x.get("axis_true"), x.get("pitch_m"), x.get("reason")) for x in o], flush=True)
        except Exception as e:
            print(pv, "ERROR", type(e).__name__, e, flush=True)
    os.makedirs(FIGS, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(FIGS, f"geodata_states_trianel{a.suffix}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(FIGS, f"geodata_states_trianel_rows{a.suffix}.csv"), index=False)


if __name__ == "__main__":
    main()
