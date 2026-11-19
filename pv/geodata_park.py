"""LEGACY (2026-10-22): block method, replaced by pv.geodata_windows (window method). Kept to reproduce the
results up to 2026-10-21 (geometry_release_states_legacy.py, geometry_trianel_override.py, analysis/fig_geodata.py).

One park through the geodata geometry pipeline of scripts/pv/geometry_trianel_states.py, with the inputs
(footprint, tile paths per product) passed in instead of looked up: orthophoto -> row axis, pitch and blocks;
each surface source (laser DOM1 raster, LAZ point cloud, image-based bDOM / DOM20) minus terrain -> height above
ground -> folded cross-row profile and flank plane fit (pv.geodata_tiles). Used by the release batch
(parks_pv.geometry_estimate); the Trianel driver keeps its own copy of this logic."""
import os

import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

from pv import geodata_tiles as GT


def _fig_hag(outd, label, name, hag, ins, res, fp_note, t, axis_ref, pitch_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outd, exist_ok=True)
    ext = [0, hag.shape[1] * res, 0, hag.shape[0] * res]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(np.clip(hag, -0.5, 4), cmap="viridis", extent=ext)
    plt.colorbar(im, ax=ax[0], label="m above ground")
    ax[0].contour(np.flipud(ins.astype(float)), levels=[0.5], colors="red", linewidths=0.8, extent=ext)
    ax[0].set_title(f"{label} {name}: height above ground, red = block used ({fp_note})", fontsize=9)
    cen, prof = t["profile"]
    ax[1].plot(cen, prof, color="#8a3b12")
    ax[1].set_xlabel("across-row position (m)")
    ax[1].set_ylabel("m above ground")
    ax[1].set_title(f"folded profile, axis {axis_ref:.1f} pitch {pitch_ref:.2f} m: tilt {t['tilt_profile']:.1f} deg "
                    f"(L {t['tilt_left']:.1f} / R {t['tilt_right']:.1f}), plane {t['tilt_plane_median']:.1f} deg", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outd, f"{label}_hag_{name}.png"), dpi=100)
    plt.close()


def measure_park(label: str, poly, fp_src: str, zone: int, lat: float, lon: float, kw: float, year: int,
                 dop: list, dgm_paths: list, sources: list, surface_only: bool = False, outd: str = None,
                 state_code: str = "") -> tuple:
    """(rows per surface source, row-axis rows). poly in CRS[zone]; sources = [(name, paths)] with name in
    dom1_laser | laz | bdom_image | dom20_image."""
    conv = GT.meridian_convergence(lat, lon, zone)
    minx, miny, maxx, maxy = poly.buffer(20).bounds
    bounds = (minx, miny, maxx, maxy)
    res_img, res_dsm = 0.5, 0.5
    H, W = int(round((maxy - miny) / res_img)), int(round((maxx - minx) / res_img))
    inside = ~geometry_mask([poly.buffer(-5)], (H, W), from_bounds(*bounds, W, H))
    out_rows, rows_rows = [], []
    pxy = GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(lon, lat)
    # ---- orthophoto
    rows = {"peaks": [], "blocks": [], "n_windows": 0}
    if dop and not surface_only:
        rgb = GT.mosaic(dop, bounds, res_img, zone, bands=[1, 2, 3])
        rows = GT.row_axis_from_image(rgb, inside, res_img)
        for k, (a, pch, s) in enumerate(rows["peaks"][:3]):
            rows_rows.append(dict(pv=label, state_code=state_code, footprint=fp_src, rank=k, axis_grid=round(a, 1),
                                  axis_true=round((a + conv) % 180, 1), pitch_m=round(pch, 2), share=round(s, 3)))
        for b in rows["blocks"]:
            rows_rows.append(dict(pv=label, state_code=state_code, footprint=fp_src, rank="block", axis_grid=round(b["axis_grid"], 1),
                                  axis_true=round((b["axis_grid"] + conv) % 180, 1), pitch_m=round(b["pitch_m"], 2),
                                  share=round(b["share_windows"], 3), n_windows=b["n_windows"]))
    axis_grid = rows["blocks"][0]["axis_grid"] if rows["blocks"] else (rows["peaks"][0][0] if rows["peaks"] else np.nan)
    pitch = rows["blocks"][0]["pitch_m"] if rows["blocks"] else (rows["peaks"][0][1] if rows["peaks"] else np.nan)
    if fp_src == "disc" and np.isfinite(axis_grid) and rows.get("windows") is not None and len(rows["windows"]):
        hull = GT.footprint_from_windows(rows["windows"], axis_grid, 1.0, bounds)
        if hull is not None and hull.area > 2000:
            poly = hull.intersection(poly)
            fp_src = "disc+windows"
            inside = ~geometry_mask([poly], (H, W), from_bounds(*bounds, W, H))
    # ---- terrain
    dgm = GT.mosaic(dgm_paths, bounds, res_dsm, zone)[0] if dgm_paths else None
    # ---- surface sources
    axis_grid0, pitch0 = axis_grid, pitch
    exp_m2 = float(kw) * 5.0 * 2.2        # modules ~5 m2/kWp, block area incl. gaps ~2.2x
    for name, paths in sources:
        axis_grid, pitch = axis_grid0, pitch0
        try:
            if name == "laz":
                surf, ground, npts = GT.dsm_from_laz(paths, bounds, res_dsm)
                base = ground if dgm is None else np.where(np.isfinite(dgm), dgm, ground)
                hag = surf - base
                dates = GT.laz_dates(paths)
            else:
                surf = GT.mosaic(paths, bounds, res_dsm, zone)[0]
                base = dgm if dgm is not None else GT.ground_from_surface(surf, res_dsm)
                hag = surf - base
                dates = [os.path.basename(x) for x in paths] + (["ground: surface-percentile"] if dgm is None else [])
            axis_grid_src, pitch_src = axis_grid, pitch
            if surface_only or not np.isfinite(axis_grid):
                fp0 = GT.footprint_from_hag(hag, res_dsm, bounds, exp_m2, pxy, pitch_m=10.0)
                ins0 = ~geometry_mask([fp0], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0])) if fp0 is not None and fp0.area > 1500 else np.isfinite(hag)
                hclip = np.clip(np.where(np.isfinite(hag), hag, 0.0), 0.0, 3.0) * 80.0
                rs = GT.row_axis_from_image(np.stack([hclip] * 3), ins0, res_dsm)
                if rs["blocks"]:
                    axis_grid_src, pitch_src = rs["blocks"][0]["axis_grid"], rs["blocks"][0]["pitch_m"]
                elif rs["peaks"]:
                    axis_grid_src, pitch_src = rs["peaks"][0][0], rs["peaks"][0][1]
                else:
                    out_rows.append(dict(pv=label, source=name, ok=False, reason="no row pattern in surface"))
                    continue
                rows_rows.append(dict(pv=label, state_code=state_code, footprint="surface", rank=f"surface:{name}",
                                      axis_grid=round(axis_grid_src, 1), axis_true=round((axis_grid_src + conv) % 180, 1),
                                      pitch_m=round(pitch_src, 2), share=round(rs["peaks"][0][2], 3) if rs["peaks"] else np.nan,
                                      n_windows=rs["n_windows"]))
            axis_grid, pitch = axis_grid_src, pitch_src
            fp_hag = GT.footprint_from_hag(hag, res_dsm, bounds, exp_m2, pxy, pitch_m=pitch if np.isfinite(pitch) else 10.0)
            if fp_hag is not None and fp_hag.area > 1500:
                blk = fp_hag.intersection(poly) if fp_src in ("osm",) and fp_hag.intersection(poly).area > 1500 else fp_hag
                ins = ~geometry_mask([blk], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0]))
                fp_note = f"hag-block {blk.area / 1e4:.1f} ha"
            else:
                ins = inside if hag.shape == inside.shape else ~geometry_mask([poly.buffer(-5)], hag.shape, from_bounds(*bounds, hag.shape[1], hag.shape[0]))
                fp_note = "outline"
            pitch_s, pcorr = GT.pitch_from_surface(hag, ins, res_dsm, axis_grid)
            use_pitch = pitch_s if np.isfinite(pitch_s) and pcorr >= 0.2 else pitch
            axis_ref, pitch_ref, contrast = GT.refine_fold(hag, ins, res_dsm, axis_grid, use_pitch)
            t = GT.tilt_from_surface(hag, ins, res_dsm, axis_ref, pitch_ref)
            t["pitch_surface"] = pitch_s
            t["pitch_surface_corr"] = pcorr
            t["pitch_used"] = pitch_ref
            t["axis_surface_true"] = round((axis_ref + conv) % 180, 2)
            t["fold_contrast"] = round(contrast, 2)
            t["block"] = fp_note
            t["hag_p50_block"] = float(np.nanpercentile(hag[ins], 50))
            t["hag_p90_block"] = float(np.nanpercentile(hag[ins], 90))

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
            if outd and t.get("ok"):
                try:
                    _fig_hag(outd, label, name, hag, ins, res_dsm, fp_note, t, axis_ref, pitch_ref)
                except Exception as e:                      # a figure must never fail the measurement
                    print("fig error", label, name, e)
            row = dict(pv=label, state_code=state_code, footprint=fp_src, source=name, axis_true=round((axis_grid + conv) % 180, 1),
                       pitch_m=round(pitch, 2), convergence=round(conv, 2), year_built=int(year), dates=str(dates)[:160],
                       **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items() if k != "profile"})
            if t.get("ok"):
                row["azimuth_true"] = round((t["flank_dir_grid"] + conv) % 360, 1)
            out_rows.append(row)
        except Exception as e:
            out_rows.append(dict(pv=label, source=name, ok=False, reason=f"{type(e).__name__}: {str(e)[:100]}"))
    if not sources:
        out_rows.append(dict(pv=label, state_code=state_code, footprint=fp_src, source=None, ok=False, reason="no surface source",
                             axis_true=round((axis_grid + conv) % 180, 1) if np.isfinite(axis_grid) else np.nan, pitch_m=pitch))
    return out_rows, rows_rows
