"""Window method for the module geometry of a ground-mounted PV park from open geodata (2026-10-22, replaces the
block method of pv.geodata_park / pv.geodata_tiles.tilt_from_surface, which stays as legacy).

The park footprint is cut into non-overlapping square windows (50 m, 60 m when the orthophoto pitch exceeds 12 m).
Each window is measured on its own from the height above ground `hag` (surface model minus DGM1, 0.5 m cells):
table share -> row axis from the 2-D spectrum of the clipped height map -> pitch from the cross-row
autocorrelation -> refinement of axis and pitch by the contrast of the folded profile -> folded profile: ridge,
flanks, side choice, tilt (least-squares line through the interior flank bins), azimuth -> one plane per row as a
second estimator -> terrain slope in the facing direction (tilt against the horizontal) -> gates. Accepted windows
are grouped into at most four arrays (share = table area). Origin: the prototype in
pv_sim2real/notebooks/_build_window_notebook.py (BODY, W1, W5), reviewed 2026-10-21; decisions of 2026-10-22 (Viktor):
  E1 tilt from a line fit through the interior flank bins (end points only as fallback), all sources;
  E2 side choice unchanged (no skewness rule);
  E3 pitch band of the window spectrum 4-20 m: tested and NOT adopted (pv5: the harmonic check doubled 8-9 m pitches
     to 16-18 m, two windows with a wrong pitch accepted, three right ones lost; pv7 lost one window; also with the
     doubling limited to 15 m). The band stays 4-15 m; the autocorrelation still searches 4-20 m (pv7: 15.2 m);
  E4a estimator gate 3 deg for point clouds, 5 deg for 1-m rasters;
  E4b park without OSM outline: 300-m box around the point, grown in 50-m rings while accepted windows touch the
      edge (up to 3x the area expected from the capacity), isolated accepted windows dropped;
  E4c group without >= 3 windows with a sure side choice: azimuth from the register class (flag az_source);
  E4d park accepted with >= 5 accepted windows and >= 40 % of the table cells in accepted windows; no window with
      tables = "no park at this point", tables but no acceptance = "measurement failed".

Conventions: rasters are north-up, row 0 = north; local metres X east, Y north with Y = (H - i - 0.5) * res. Grid
angles are compass angles in the UTM grid; true = grid + meridian convergence `conv`. Tilt against the horizontal,
azimuth clockwise from true north (180 = south), as pv.arrays. Every variable of a window is a column of the window
table. German reason strings are kept from the notebook (they show up in the reports)."""
import os
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.signal import find_peaks

RES = 0.5                    # analysis grid (m)
PAD = 4                      # zero padding of the window FFT
TAPER_M = 15.0               # cosine taper at the edge of the valid cells (m)
PITCH_RANGE = (4.0, 15.0)    # wavelength band of the window spectrum; 4-20 m (E3) was tested 2026-10-22 and reverted, see below
HARMONIC_MAX = 15.0          # the 2nd-harmonic check doubles the pitch only up to this value (m)
BW, BIN_W, MIN_H = 0.25, 0.5, 0.45   # autocorrelation bin, profile bin, "above ground" threshold (m)
TAB_LO, TAB_HI = 0.4, 4.5    # table cells: height above ground (m)
F_IN_MIN, F_TAB_MIN = 0.6, 0.15      # window counts with >= 60 % inside the outline; measured with >= 15 % table cells
EST_TOL = {"laz": 3.0}       # estimator gate (deg) per surface source (E4a); 1-m rasters default to EST_TOL_RASTER
EST_TOL_RASTER = 5.0
GROUP_TOL = dict(axis=10.0, az=30.0, tilt=4.0)
SYM_MIN = 0.75               # ridge symmetry of an opposite pair that is merged into "azimuth undetermined"
MIN_SURE = 3                 # windows with a sure side choice for a measured group azimuth (E4d); a smaller group needs all of its windows
MAX_GROUPS = 4
SPLINTER = dict(n=3, share=0.05)
PARK_ACCEPT = dict(n_ok=2, tab_share=0.40)   # 2026-10-22 decision 1 (B): >= 2 windows (was 5) as long as they cover >= 40 % of the table cells
PARK_ACCEPT_BOX = dict(n_ok=5, tab_share=0.40)   # box mode keeps 5: without an outline a few windows at the box edge can be another park (SEL968423559874)
BOX = dict(size_m=300.0, step_m=50.0, area_factor=3.0)
AZ_CENTRE = {"Süd": 180.0, "Süd-West": 225.0, "Süd-Ost": 135.0, "Ost": 90.0, "West": 270.0, "Nord": 0.0,
             "Nord-Ost": 45.0, "Nord-West": 315.0}   # = pv.plausibility.AZ_CENTRE (copied to keep this module light)


def ang180(d):
    """Signed difference of two axes (mod 180) in (-90, 90]."""
    return ((d + 90.0) % 180.0) - 90.0


def ang360(d):
    """Signed difference of two azimuths in (-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


def circ_mean(deg, period: float = 360.0) -> float:
    k = 360.0 / period
    r = np.radians(np.asarray(deg, float) * k)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) / k % period)


def window_size(pitch0: float) -> float:
    """50 m, 60 m when the orthophoto pitch exceeds 12 m (at least four rows per window)."""
    return 60.0 if (pitch0 is not None and np.isfinite(pitch0) and pitch0 > 12.0) else 50.0


def est_tol(source: str) -> float:
    return EST_TOL.get(source, EST_TOL_RASTER)


# ------------------------------------------------------------------------------------------------ W1 window grid
def window_grid(hag: np.ndarray, inside: np.ndarray, nw: int) -> list:
    """[(i0, j0, f_in, f_tab)] of the non-overlapping nw x nw windows with >= 60 % of their cells inside."""
    H, W = hag.shape
    grid = []
    for i0 in range(0, H - nw + 1, nw):
        for j0 in range(0, W - nw + 1, nw):
            f_in = float(inside[i0:i0 + nw, j0:j0 + nw].mean())
            if f_in < F_IN_MIN:
                continue
            hw_ = hag[i0:i0 + nw, j0:j0 + nw]
            f_tab = float((np.isfinite(hw_) & (hw_ > TAB_LO) & (hw_ < TAB_HI)).mean())
            grid.append((i0, j0, f_in, f_tab))
    return grid


def _reg_tilt(prof_: np.ndarray, cen_: np.ndarray, k_a: int, k_b: int) -> float:
    """Tilt (deg) of a least-squares line through the interior bins of one flank (end bins excluded when the flank
    has >= 5 bins); NaN with fewer than 3 usable bins (E1)."""
    kk = np.arange(min(k_a, k_b), max(k_a, k_b) + 1)
    inner = kk[1:-1] if len(kk) >= 5 else kk
    inner = inner[np.isfinite(prof_[inner])]
    if len(inner) < 3:
        return np.nan
    return float(np.degrees(np.arctan(abs(np.polyfit(cen_[inner], prof_[inner], 1)[0]))))


def _hz(b_: float, g_: float) -> float:
    """Tilt against the horizontal from the tilt against the ground b_ and the ground descent g_ (tan) towards the facing."""
    return float(np.degrees(np.arctan(np.tan(np.radians(b_)) + g_)))


# ------------------------------------------------------------------------------------------------ BODY per window
def measure_window(hag: np.ndarray, dgm: np.ndarray, cell: tuple, nw: int, conv: float, source: str, res: float = RES,
                   w: int = 0, detail: bool = False):
    """One window (cell = (i0, j0, f_in, f_tab) from window_grid). Returns the result dict (one row of the window
    table); with detail=True also a dict of the intermediates (height map, spectrum, profile, rows) for figures."""
    H = hag.shape[0]
    i0, j0, f_in, f_tab = cell
    hw_ = hag[i0:i0 + nw, j0:j0 + nw]                                          # height above ground in the window
    dw_ = dgm[i0:i0 + nw, j0:j0 + nw] if dgm is not None else np.full_like(hw_, np.nan)
    jjw, iiw = np.meshgrid(np.arange(nw), np.arange(nw))
    Xw, Yw = (j0 + jjw + 0.5) * res, (H - (i0 + iiw) - 0.5) * res              # local metres, X east, Y north
    okw = np.isfinite(hw_) & (hw_ > -0.5) & (hw_ < 6)
    res_w = dict(w=w, i0=i0, j0=j0, x=float(Xw.mean()), y=float(Yw.mean()), f_in=f_in, f_tab=f_tab, n_ok=int(okw.sum()), ok=False, reason="")
    det = dict(hw=hw_, dw=dw_, Xw=Xw, Yw=Yw, okw=okw)
    if okw.mean() < 0.5:
        res_w["reason"] = "zu wenig Oberfläche"
        return (res_w, det) if detail else res_w
    if f_tab < F_TAB_MIN:
        res_w["reason"] = "zu wenig Tischzellen"
        return (res_w, det) if detail else res_w
    # --- row axis from the 2-D spectrum of the clipped height map (tables 0-3 m)
    hc = np.clip(np.where(okw, hw_, 0.0), 0.0, 3.0)
    dtw = distance_transform_edt(np.pad(okw, 1))[1:-1, 1:-1] * res             # distance to the edge of the valid cells; the window edge counts
    xw = np.where(okw, hc - hc[okw].mean(), 0.0) * np.where(dtw < TAPER_M, 0.5 * (1 - np.cos(np.pi * np.clip(dtw, 0, TAPER_M) / TAPER_M)), 1.0)
    Fw = np.abs(np.fft.fftshift(np.fft.fft2(xw, s=(PAD * nw, PAD * nw)))) ** 2
    kw_ = np.fft.fftshift(np.fft.fftfreq(PAD * nw, d=res)); KXw, KYw = np.meshgrid(kw_, kw_); Kw = np.hypot(KXw, KYw)
    bandw = (Kw > 1 / PITCH_RANGE[1]) & (Kw < 1 / PITCH_RANGE[0])
    angw = np.degrees(np.arctan2(KXw, -KYw)) % 180.0                            # compass of the wave vector
    histw = np.zeros(180); np.add.at(histw, np.round(angw).astype(int)[bandw] % 180, Fw[bandw])
    a_w = int(np.argmax(histw)); wedgew = np.abs(((angw - a_w) + 90) % 180 - 90) < 2; selw = bandw & wedgew
    kpw = Kw[selw][np.argmax(Fw[selw])]
    if kpw / 2 >= 1 / HARMONIC_MAX and Fw[wedgew & (Kw > 0) & (np.abs(Kw - kpw / 2) < 0.05 * kpw)].sum() >= 0.5 * Fw[wedgew & (np.abs(Kw - kpw) < 0.1 * kpw)].sum():
        kpw = kpw / 2                                                           # 2nd harmonic: the fundamental at k/2 is strong enough
    share_w = histw[np.arange(a_w - 2, a_w + 3) % 180].sum() / max(histw.sum(), 1e-9)
    axis_w, pitch_fft = (a_w + 90) % 180.0, 1 / kpw                             # row axis (grid), pitch from the spectrum
    res_w.update(axis_fft=axis_w, pitch_fft=pitch_fft, share=float(share_w))
    det.update(histw=histw)
    # --- pitch from the autocorrelation of the cross-row profile
    sw0 = Xw * np.cos(np.radians(axis_w)) - Yw * np.sin(np.radians(axis_w))
    kb_ = ((sw0[okw] - sw0[okw].min()) / BW).astype(int); nbn = kb_.max() + 1
    cnb = np.bincount(kb_, minlength=nbn); prb = np.where(cnb > 5, np.bincount(kb_, weights=hw_[okw], minlength=nbn) / np.maximum(cnb, 1), np.nan)
    prb = np.where(np.isfinite(prb), prb, np.nanmean(prb)); xcb = prb - prb.mean()
    acb = np.correlate(xcb, xcb, "full")[len(xcb) - 1:]; acb /= max(acb[0], 1e-9)
    lagb = np.arange(len(acb)) * BW; mlb = (lagb >= 4.0) & (lagb <= 20.0); pkb, _ = find_peaks(acb[mlb], prominence=0.05)
    pitch_ac, corr_ac = (float(lagb[mlb][pkb[np.argmax(acb[mlb][pkb])]]), float(acb[mlb][pkb].max())) if len(pkb) else (np.nan, np.nan)
    res_w.update(pitch_ac=pitch_ac, corr_ac=corr_ac)
    # --- refine axis and pitch by the contrast (p90 - p10) of the folded profile
    Xo, Yo, ho = Xw[okw], Yw[okw], hw_[okw]
    use_ac = bool(np.isfinite(corr_ac) and corr_ac >= 0.2 and 0.75 <= pitch_ac / pitch_fft <= 1.33)
    bc, bax, bpi = -1.0, axis_w, (pitch_ac if use_ac else pitch_fft)
    res_w.update(use_ac=use_ac)
    for step in (0.5, 0.1):
        span = 4.0 if step == 0.5 else 0.6
        for ax_ in np.arange(bax - span, bax + span + 1e-9, step):
            sv = Xo * np.cos(np.radians(ax_)) - Yo * np.sin(np.radians(ax_)); nb_ = max(int(np.ceil(bpi / BW)), 4); idx_ = np.minimum((np.mod(sv, bpi) / BW).astype(int), nb_ - 1)
            cnt_ = np.bincount(idx_, minlength=nb_); pr = np.where(cnt_ > 20, np.bincount(idx_, weights=ho, minlength=nb_) / np.maximum(cnt_, 1), np.nan)
            c = float(np.nanpercentile(pr, 90) - np.nanpercentile(pr, 10)) if np.isfinite(pr).sum() >= 4 else -1.0
            if c > bc: bc, bax = c, float(ax_)
        for pm in np.arange(bpi * 0.88, bpi * 1.12, 0.05 if step == 0.5 else 0.01):
            sv = Xo * np.cos(np.radians(bax)) - Yo * np.sin(np.radians(bax)); nb_ = max(int(np.ceil(pm / BW)), 4); idx_ = np.minimum((np.mod(sv, pm) / BW).astype(int), nb_ - 1)
            cnt_ = np.bincount(idx_, minlength=nb_); pr = np.where(cnt_ > 20, np.bincount(idx_, weights=ho, minlength=nb_) / np.maximum(cnt_, 1), np.nan)
            c = float(np.nanpercentile(pr, 90) - np.nanpercentile(pr, 10)) if np.isfinite(pr).sum() >= 4 else -1.0
            if c > bc: bc, bpi = c, float(pm)
    axis_ref, pitch_ref, contrast = bax, bpi, bc
    res_w.update(axis_grid=axis_ref, axis_true=(axis_ref + conv) % 180, pitch=pitch_ref, contrast=contrast)
    # --- folded cross-row profile: ridge, foot, flank -> tilt; falling direction -> azimuth
    a_ref = np.radians(axis_ref); sw = Xw * np.cos(a_ref) - Yw * np.sin(a_ref); n_across = np.array([np.cos(a_ref), -np.sin(a_ref)])
    nb = int(np.ceil(pitch_ref / BIN_W)); cen = (np.arange(nb) + 0.5) * BIN_W
    ph = np.mod(sw[okw], pitch_ref); idx = np.minimum((ph / BIN_W).astype(int), nb - 1); h = hw_[okw]
    prof = np.array([np.median(h[idx == k_]) if (idx == k_).sum() > 30 else np.nan for k_ in range(nb)])
    if not np.isfinite(prof).any():
        res_w["reason"] = "kein Profil"
        return (res_w, det) if detail else res_w
    shift = cen[np.nanargmax(prof)]
    ph2 = np.mod(ph - shift + pitch_ref / 2, pitch_ref); idx2 = np.minimum((ph2 / BIN_W).astype(int), nb - 1)
    prof2 = np.array([np.median(h[idx2 == k_]) if (idx2 == k_).sum() > 30 else np.nan for k_ in range(nb)])
    k_r = int(np.nanargmax(prof2)); h_ridge = float(prof2[k_r]); above = prof2 > MIN_H
    lo = k_r
    while lo - 1 >= 0 and above[lo - 1]: lo -= 1
    hi = k_r
    while hi + 1 < nb and above[hi + 1]: hi += 1
    wl, wr = cen[k_r] - cen[lo], cen[hi] - cen[k_r]
    tilt_left_ends = float(np.degrees(np.arctan2(h_ridge - float(prof2[lo]), max(wl, 0.1)))); tilt_right_ends = float(np.degrees(np.arctan2(h_ridge - float(prof2[hi]), max(wr, 0.1))))
    _l = _reg_tilt(prof2, cen, lo, k_r); _r = _reg_tilt(prof2, cen, k_r, hi)     # E1: line fit, end points as fallback
    tilt_left = _l if np.isfinite(_l) else tilt_left_ends; tilt_right = _r if np.isfinite(_r) else tilt_right_ends
    # Side choice (unchanged, E2): a clear sawtooth (one side drops at once) -> the longer flank; two long flanks
    # (smoothed 1-m raster) -> the FLATTER flank (the back edge stays steeper), length only at equal slope.
    # az_sure = the choice has a basis (slope difference >= 1.5 deg or length difference >= 1 m).
    side_len = "left" if wl > wr else "right"; side_slope = "left" if tilt_left < tilt_right else "right"
    if min(wl, wr) < 1.0:
        side, az_sure = side_len, True
    else:
        side = side_slope if abs(tilt_left - tilt_right) >= 1.5 else side_len
        az_sure = bool(abs(tilt_left - tilt_right) >= 1.5 or abs(wl - wr) >= 1.0)
    k_f = lo if side == "left" else hi
    h_foot = float(prof2[k_f]); flank = float(abs(cen[k_r] - cen[k_f]))
    tilt_profile_ends = float(np.degrees(np.arctan2(h_ridge - h_foot, max(flank, 0.1))))
    _t = _reg_tilt(prof2, cen, k_r, k_f); tilt_profile = _t if np.isfinite(_t) else tilt_profile_ends
    ridge_sym = float(min(wl, wr) / max(wl, wr, 0.1)); two_flank = bool(ridge_sym > 0.6 and min(wl, wr) >= 1.5)
    dir_across = float(np.degrees(np.arctan2(n_across[0], n_across[1])) % 360)
    flank_dir_grid = dir_across if side == "right" else (dir_across + 180) % 360; azimuth_true = (flank_dir_grid + conv) % 360
    ew_pair = bool(two_flank and abs((azimuth_true % 180) - 90) <= 35)
    det.update(cen=cen, prof2=prof2, k_r=k_r, k_f=k_f, lo=lo, hi=hi, sw=sw, shift=shift)
    # --- one plane per row in the window: second estimator of the tilt
    tab = np.floor((sw[okw] - shift + pitch_ref / 2) / pitch_ref).astype(int)
    on_flank = ((ph2 >= cen[k_r] - 0.3) & (ph2 <= cen[k_f] + 0.3)) if side == "right" else ((ph2 >= cen[k_f] - 0.3) & (ph2 <= cen[k_r] + 0.3))
    Xk, Yk = Xw[okw], Yw[okw]; rows_list = []
    for t_ in np.unique(tab):
        m = (tab == t_) & on_flank & (h > MIN_H)
        if m.sum() < 100: continue
        A = np.column_stack([Xk[m], Yk[m], np.ones(m.sum())]); coef, *_ = np.linalg.lstsq(A, h[m], rcond=None)
        rows_list.append(dict(row=int(t_), cells=int(m.sum()), tilt=float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))), az_true=float((np.degrees(np.arctan2(-coef[0], -coef[1])) + conv) % 360)))
    R = pd.DataFrame(rows_list)
    tilt_rows = float(R.tilt.median()) if len(R) else np.nan
    det.update(R=R)
    # --- terrain in the window: ground descent towards the facing direction (grid frame)
    gy_, gx_ = np.gradient(np.where(np.isfinite(dw_), dw_, np.nan), res); dg_dE, dg_dN = gx_, -gy_
    az_rad = np.radians(flank_dir_grid); slope_facing = -(dg_dE * np.sin(az_rad) + dg_dN * np.cos(az_rad))
    slope_cover = float(np.isfinite(slope_facing[okw]).mean())                # share of window cells with DGM1
    g_fac = float(np.nanmedian(slope_facing[okw])) if slope_cover > 0 else 0.0
    terrain_deg = float(np.degrees(np.arctan(np.hypot(np.nanmedian(dg_dE[okw]), np.nanmedian(dg_dN[okw]))))) if slope_cover > 0 else np.nan
    chosen_left = side == "left"                                                # the chosen flank faces flank_dir_grid, the other one the opposite way
    tilt_left_h, tilt_right_h = _hz(tilt_left, g_fac if chosen_left else -g_fac), _hz(tilt_right, -g_fac if chosen_left else g_fac)
    tilt_measured = 0.5 * (tilt_left + tilt_right) if ew_pair else tilt_profile             # against the ground
    tilt_horiz = 0.5 * (tilt_left_h + tilt_right_h) if ew_pair else _hz(tilt_profile, g_fac)  # against the horizontal
    tilt_rows_h = _hz(tilt_rows, g_fac) if np.isfinite(tilt_rows) else np.nan
    tilt_horiz_ends = (0.5 * (_hz(tilt_left_ends, g_fac if chosen_left else -g_fac) + _hz(tilt_right_ends, -g_fac if chosen_left else g_fac)) if ew_pair
                       else _hz(tilt_profile_ends, g_fac))                     # end-point estimator, kept for comparison (E1)
    # --- gates per window
    est_diff = abs(tilt_left_h - tilt_right_h) if ew_pair else (abs(tilt_profile - tilt_rows) if np.isfinite(tilt_rows) else 99.0)
    gates_w = dict(g_pattern=bool(share_w >= 0.25 or (np.isfinite(corr_ac) and corr_ac >= 0.3)), g_contrast=contrast >= 0.5, g_flank=bool(1.5 <= flank <= 0.75 * pitch_ref),
                   g_ridge=0.8 <= h_ridge <= 4.5, g_rows=len(R) >= 2, g_est=est_diff <= est_tol(source))
    area_tab = float(((h > MIN_H) & (on_flank | ew_pair)).sum() * res**2)       # module area (plan view) in the window, m2; east-west: both flanks
    res_w.update(shift=shift, h_ridge=h_ridge, h_foot=h_foot, flank=flank, wl=wl, wr=wr, side=side, az_sure=az_sure, ridge_sym=ridge_sym, two_flank=two_flank, ew_pair=ew_pair,
                 tilt_profile=tilt_profile, tilt_left=tilt_left, tilt_right=tilt_right, tilt_profile_ends=tilt_profile_ends, tilt_left_ends=tilt_left_ends,
                 tilt_right_ends=tilt_right_ends, tilt_horiz_ends=tilt_horiz_ends, tilt_rows=tilt_rows, n_rows=len(R), flank_dir_grid=flank_dir_grid, azimuth_true=azimuth_true,
                 slope_facing_deg=float(np.degrees(np.arctan(g_fac))), slope_cover=slope_cover, terrain_deg=terrain_deg, tilt_measured=tilt_measured, tilt_horiz=tilt_horiz,
                 tilt_left_h=tilt_left_h, tilt_right_h=tilt_right_h, tilt_rows_h=tilt_rows_h, est_diff=est_diff, est_tol=est_tol(source), area_tab=area_tab, **gates_w,
                 ok=all(gates_w.values()), reason="" if all(gates_w.values()) else ",".join(k_ for k_, v_ in gates_w.items() if not v_))
    return (res_w, det) if detail else res_w


def measure_windows(hag: np.ndarray, dgm: np.ndarray, inside: np.ndarray, conv: float, source: str, pitch0: float = np.nan,
                    res: float = RES, win_m: float = None) -> tuple:
    """W1 + W3: (window table, nw, win_m). One row per window inside the outline, rejected ones with `reason`."""
    win_m = win_m or window_size(pitch0)
    nw = int(round(win_m / res))
    grid = window_grid(hag, inside, nw)
    rows = [measure_window(hag, dgm, cell, nw, conv, source, res, w=k) for k, cell in enumerate(grid)]
    return pd.DataFrame(rows), nw, win_m


def drop_isolated(win: pd.DataFrame, nw: int) -> pd.DataFrame:
    """Box mode (E4b): an accepted window without an accepted neighbour (8-neighbourhood) is rejected ("isoliert")."""
    win = win.copy()
    acc = {(int(r.i0) // nw, int(r.j0) // nw) for r in win[win.ok.astype(bool)].itertuples()}
    for k, r in win[win.ok.astype(bool)].iterrows():
        c = (int(r.i0) // nw, int(r.j0) // nw)
        if not any((c[0] + di, c[1] + dj) in acc for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)):
            win.at[k, "ok"] = False
            win.at[k, "reason"] = "isoliert"
    return win


def keep_central_component(win: pd.DataFrame, nw: int, centre_ij: tuple) -> pd.DataFrame:
    """Box mode (2026-10-22): the box can hold neighbouring parks. Accepted windows form 8-connected components; only
    the component nearest the register point (window centre distance) is kept, the others get reason "anderer Park"."""
    win = win.copy()
    acc = {(int(r.i0) // nw, int(r.j0) // nw): k for k, r in win[win.ok.astype(bool)].iterrows()}
    if not acc:
        return win
    seen, comps = set(), []
    for c0 in acc:
        if c0 in seen:
            continue
        stack, comp = [c0], []
        seen.add(c0)
        while stack:
            c = stack.pop(); comp.append(c)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n_ = (c[0] + di, c[1] + dj)
                    if n_ in acc and n_ not in seen:
                        seen.add(n_); stack.append(n_)
        comps.append(comp)
    ci, cj = centre_ij
    dist = lambda comp: min(((c[0] + 0.5) * nw - ci) ** 2 + ((c[1] + 0.5) * nw - cj) ** 2 for c in comp)
    keep = set(min(comps, key=dist))
    for c, k in acc.items():
        if c not in keep:
            win.at[k, "ok"] = False
            win.at[k, "reason"] = "anderer Park"
    return win


def keep_central_construction(win: pd.DataFrame, nw: int, centre_ij: tuple) -> pd.DataFrame:
    """Box mode (2026-10-22, decision 2): a neighbouring park can touch the park at the register point, so its windows
    join the central component. Groups are formed once; only the group of the accepted window nearest the register
    point and the groups of the same construction (same east-west status) are kept; the windows of the other groups
    get the reason "anderer Park (Bauweise)"."""
    G = group_windows(win)
    if len(G) <= 1:
        return win
    acc = win[win.ok.astype(bool)]
    ci, cj = centre_ij
    w0 = int(acc.loc[((acc.i0 + nw / 2 - ci) ** 2 + (acc.j0 + nw / 2 - cj) ** 2).idxmin(), "w"])
    gw = {int(g.group): {int(x) for x in str(g.windows).split()} for g in G.itertuples()}
    g0 = next((G.loc[G.group == k].iloc[0] for k, ws in gw.items() if w0 in ws), G.iloc[0])
    # only a different east-west status marks another park: sections of one park with another row direction or azimuth
    # are common (SEL910440791607: second section 13 deg off) and must stay (2026-10-22, after the first fleet run with
    # the stricter axis/azimuth rule)
    same = lambda g: bool(g.ew == g0.ew)
    drop = set().union(*[gw[int(g.group)] for g in G.itertuples() if not same(g)]) if any(not same(g) for g in G.itertuples()) else set()
    if not drop:
        return win
    win = win.copy()
    m = win.w.isin(drop)
    win.loc[m, "ok"] = False
    win.loc[m, "reason"] = "anderer Park (Bauweise)"
    return win


# ------------------------------------------------------------------------------------------------ W5 grouping
def rejected_flag(win: pd.DataFrame, nw: int, box: bool = False) -> dict:
    """Share of the table cells in rejected windows and the largest 4-connected block of rejected table windows.
    The shares assume that rejected windows look like accepted ones; a large connected rejected block can hold
    another construction -> flag (not a rejection). Box mode (no outline, 2026-10-22): only windows with a row pattern
    count as table windows, because the box also holds hedges, trees and buildings in the 0.4-4.5 m band."""
    other = win.reason.astype(str).str.startswith("anderer Park") if "reason" in win else pd.Series(False, index=win.index)
    tabw_ = win[(win.f_tab >= F_TAB_MIN) & ~other]   # windows of other parks (box mode) do not count
    if box and "g_pattern" in tabw_:
        tabw_ = tabw_[tabw_.g_pattern.fillna(False).astype(bool)]
    rej_ = tabw_[~tabw_.ok.astype(bool)]
    rej_tab_frac = float(rej_.f_tab.sum() / max(tabw_.f_tab.sum(), 1e-9))
    gi_ = {(int(r_.i0) // nw, int(r_.j0) // nw): float(r_.f_tab) for r_ in rej_.itertuples()}; seen_, clusters_ = set(), []
    for k0_ in gi_:
        if k0_ in seen_: continue
        stack_, cl_ = [k0_], []
        seen_.add(k0_)
        while stack_:
            c_ = stack_.pop(); cl_.append(c_)
            for d_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_ = (c_[0] + d_[0], c_[1] + d_[1])
                if n_ in gi_ and n_ not in seen_: seen_.add(n_); stack_.append(n_)
        clusters_.append(cl_)
    big_ = max(clusters_, key=lambda cl: sum(gi_[c] for c in cl)) if clusters_ else []
    rej_cluster_frac = float(sum(gi_[c] for c in big_) / max(sum(gi_.values()), 1e-9)) if big_ else 0.0
    warn_rejected = bool(rej_tab_frac > 0.30 and len(big_) >= 3 and rej_cluster_frac >= 0.5)
    return dict(n_tab_windows=int(len(tabw_)), rej_tab_frac=rej_tab_frac, rej_cluster_windows=len(big_), rej_cluster_frac=rej_cluster_frac,
                warn_rejected=warn_rejected)


def _group_values(win: pd.DataFrame, g_: dict) -> None:
    ws_ = win[win.w.isin(g_["w"])]; wsure = ws_[ws_.az_sure.astype(bool)]; src_ = wsure if len(wsure) else ws_
    g_["az"] = circ_mean(src_.azimuth_true); g_["n_sure"] = int(len(wsure))
    g_["axis"] = circ_mean(ws_.axis_true, 180.0); g_["tilt"] = float(ws_.tilt_horiz.median())


def group_windows(win: pd.DataFrame) -> pd.DataFrame:
    """Accepted windows -> at most four groups (W5). Greedy (largest table area first): axis <= 10 deg, azimuth
    <= 30 deg (not compared for east-west pairs and unsure side choices), tilt <= 4 deg, same east-west status; groups
    whose values agree are merged; opposite pairs with symmetric ridges become one group "azimuth undetermined";
    splinters (< 3 windows or < 5 % table area) are dropped before the cut to four. Group azimuth = circular mean of
    the windows with a sure side choice; a group with fewer than MIN_SURE of them has an undetermined azimuth (E4d)."""
    cols = ["group", "n_windows", "area_tab_ha", "axis_true", "pitch", "tilt_horiz", "tilt_p10", "tilt_p90", "azimuth", "az_undetermined",
            "n_sure", "n_opposite", "ew", "ridge_sym", "slope_deg", "share_raw", "share", "windows"]
    acc = win[win.ok.astype(bool)].sort_values("area_tab", ascending=False) if len(win) and "area_tab" in win else win.iloc[:0]
    groups = []
    for r_ in acc.itertuples():
        placed = False
        for g_ in groups:
            d_ax = abs(ang180(r_.axis_true - g_["axis"])); d_az = abs(ang360(r_.azimuth_true - g_["az"]))
            if d_ax <= GROUP_TOL["axis"] and (d_az <= GROUP_TOL["az"] or (bool(r_.ew_pair) and g_["ew"]) or not bool(r_.az_sure)) and abs(r_.tilt_horiz - g_["tilt"]) <= GROUP_TOL["tilt"] and bool(r_.ew_pair) == g_["ew"]:
                g_["w"].append(r_.w); placed = True; break
        if not placed:
            groups.append(dict(axis=r_.axis_true, az=r_.azimuth_true, tilt=r_.tilt_horiz, ew=bool(r_.ew_pair), w=[r_.w]))
    for g_ in groups:
        _group_values(win, g_)
    changed = True                                                              # merge groups whose values agree (the greedy pass compares with the seed only)
    while changed and len(groups) > 1:
        changed = False
        for ia in range(len(groups)):
            for ib in range(ia + 1, len(groups)):
                ga, gb = groups[ia], groups[ib]
                d_ax = abs(ang180(ga["axis"] - gb["axis"])); d_az = abs(ang360(ga["az"] - gb["az"]))
                if ga["ew"] == gb["ew"] and d_ax <= GROUP_TOL["axis"] and (d_az <= GROUP_TOL["az"] or ga["ew"]) and abs(ga["tilt"] - gb["tilt"]) <= GROUP_TOL["tilt"]:
                    ga["w"] = ga["w"] + gb["w"]; groups.pop(ib); changed = True
                    _group_values(win, ga)
                    break
            if changed: break
    merged = []                                                                 # opposite pairs with symmetric ridges: azimuth undetermined
    for g_ in groups:
        hit = None
        for m_ in merged:
            d_ax = abs(ang180(g_["axis"] - m_["axis"])); d_az = abs(ang360(g_["az"] - m_["az"]))
            sym_g = win.loc[win.w.isin(g_["w"]), "ridge_sym"].median(); sym_m = win.loc[win.w.isin(m_["w"]), "ridge_sym"].median()
            if d_ax <= GROUP_TOL["axis"] and abs(d_az - 180) <= GROUP_TOL["az"] and abs(g_["tilt"] - m_["tilt"]) <= GROUP_TOL["tilt"] and not g_["ew"] and not m_["ew"] and sym_g >= SYM_MIN and sym_m >= SYM_MIN:
                hit = m_; break
        if hit is None:
            merged.append(dict(g_, az_undetermined=False, w_opposite=[]))
        else:
            hit["w_opposite"] = hit["w_opposite"] + g_["w"]; hit["az_undetermined"] = True
    rows_g = []
    for k_, g_ in enumerate(sorted(merged, key=lambda g: -win.loc[win.w.isin(g["w"] + g["w_opposite"]), "area_tab"].sum())):
        sub_ = win[win.w.isin(g_["w"])]; opp_ = win[win.w.isin(g_["w_opposite"])]; all_ = pd.concat([sub_, opp_])
        sure_ = all_[all_.az_sure.astype(bool)]; src_ = sure_ if len(sure_) else sub_
        n_sure_ = int(len(sure_))
        rows_g.append(dict(group=k_, n_windows=len(all_), area_tab_ha=all_.area_tab.sum() / 1e4, axis_true=circ_mean(all_.axis_true, 180.0),
                           pitch=float(all_.pitch.median()), tilt_horiz=float(all_.tilt_horiz.median()), tilt_p10=float(all_.tilt_horiz.quantile(0.1)),
                           tilt_p90=float(all_.tilt_horiz.quantile(0.9)), azimuth=circ_mean(src_.azimuth_true),
                           az_undetermined=bool(g_["az_undetermined"] or (n_sure_ < min(MIN_SURE, len(all_)) and not g_["ew"])), n_sure=n_sure_, n_opposite=len(opp_),
                           ew=g_["ew"], ridge_sym=float(all_.ridge_sym.median()), slope_deg=float(all_.slope_facing_deg.median()),
                           windows=" ".join(str(int(x)) for x in sorted(all_.w))))
    G = pd.DataFrame(rows_g, columns=[c for c in cols if c not in ("share_raw", "share")])
    if not len(G):
        return pd.DataFrame(columns=cols)
    G["share_raw"] = G.area_tab_ha / G.area_tab_ha.sum()
    # splinters out, THEN at most four; a group below SPLINTER['n'] windows is a splinter only next to a larger group
    # (2026-10-22: a small park with 2 accepted windows in one group keeps its group)
    keep_g = ((G.n_windows >= SPLINTER["n"]) | (G.n_windows >= G.n_windows.max())) & (G.share_raw >= SPLINTER["share"])
    attrs = dict(n_splinter=int((~keep_g).sum()), splinter_share=float(G.share_raw[~keep_g].sum()), n_cut=max(int(keep_g.sum()) - MAX_GROUPS, 0))
    G = G[keep_g].head(MAX_GROUPS).reset_index(drop=True)
    G["group"] = range(len(G))
    G["share"] = G.area_tab_ha / G.area_tab_ha.sum() if len(G) else pd.Series(dtype=float)
    G = G[cols]
    G.attrs.update(attrs)
    return G


def register_azimuth(axis_true: float, register_orientation: str = None) -> tuple:
    """(azimuth, az_source, az_conflict) of a group whose azimuth is undetermined (E4c): the perpendicular of the
    measured row axis nearest the register class; class missing or more than 45 deg off both perpendiculars (e.g.
    class south on north-south rows) -> the perpendicular nearest south, flagged."""
    perp = [(axis_true + 90.0) % 360.0, (axis_true + 270.0) % 360.0]
    ref = AZ_CENTRE.get(register_orientation) if register_orientation else None
    if ref is not None:
        d = [abs(ang360(p - ref)) for p in perp]
        if min(d) <= 45.0:
            return perp[int(np.argmin(d))], "register", False
    south = perp[int(np.argmin([abs(ang360(p - 180.0)) for p in perp]))]
    return south, ("register_conflict_south" if ref is not None else "assumed_south"), ref is not None


def arrays_from_groups(G: pd.DataFrame, register_orientation: str = None) -> pd.DataFrame:
    """Generator input (tilt, azimuth, share) per group; east-west groups -> two arrays with half the share whose
    azimuths are perpendicular to the row axis (east flank first); undetermined azimuth -> register class (E4c)."""
    out = []
    for g_ in G.itertuples():
        if g_.ew:
            az_e, az_w = sorted([(g_.axis_true + 90.0) % 360, (g_.axis_true + 270.0) % 360], key=lambda a_: abs(ang360(a_ - 90.0)))
            out += [dict(group=g_.group, tilt=g_.tilt_horiz, azimuth=az_e, share=g_.share / 2, az_source="axis", az_conflict=False, note="Ost-Flanke"),
                    dict(group=g_.group, tilt=g_.tilt_horiz, azimuth=az_w, share=g_.share / 2, az_source="axis", az_conflict=False, note="West-Flanke")]
        elif g_.az_undetermined:
            az, src, conflict = register_azimuth(g_.axis_true, register_orientation)
            note = "Azimut unbestimmt (symmetrischer First)" if g_.n_opposite else f"Azimut unbestimmt ({g_.n_sure} sichere Fenster)"
            out.append(dict(group=g_.group, tilt=g_.tilt_horiz, azimuth=az, share=g_.share, az_source=src, az_conflict=conflict, note=note))
        else:
            out.append(dict(group=g_.group, tilt=g_.tilt_horiz, azimuth=g_.azimuth, share=g_.share, az_source="measured", az_conflict=False, note=""))
    return pd.DataFrame(out, columns=["group", "tilt", "azimuth", "share", "az_source", "az_conflict", "note"])


def park_status(win: pd.DataFrame, G: pd.DataFrame, flag: dict, box: bool = False) -> dict:
    """Park acceptance (E4d): accepted | failed ("Messung gescheitert") | no_park ("kein Park an dieser Koordinate")."""
    acc_ = PARK_ACCEPT_BOX if box else PARK_ACCEPT
    n_ok = int(win.ok.astype(bool).sum()) if len(win) else 0
    tab_share = 1.0 - flag["rej_tab_frac"] if flag["n_tab_windows"] else 0.0
    if flag["n_tab_windows"] == 0:
        status, why = "no_park", "kein Fenster mit Tischen"
    elif n_ok >= acc_["n_ok"] and tab_share >= acc_["tab_share"] and len(G):
        status, why = "accepted", ""
    else:
        why = "; ".join(x for x in (f"{n_ok} angenommene Fenster (< {acc_['n_ok']})" if n_ok < acc_["n_ok"] else "",
                                    f"Tischanteil angenommen {tab_share:.0%} (< {acc_['tab_share']:.0%})" if tab_share < acc_["tab_share"] else "",
                                    "keine Gruppe" if not len(G) else "") if x)
        status = "failed"
    # information only (2026-10-22): small parks reach few windows; would a relaxed minimum of 2 windows accept them?
    accept_relaxed = bool(status == "accepted" or (n_ok >= 2 and tab_share >= acc_["tab_share"] and len(G)))   # = accepted since n_ok 2
    return dict(status=status, status_reason=why, n_windows=int(len(win)), n_ok=n_ok, n_tab_windows=int(flag["n_tab_windows"]),
                tab_share_accepted=tab_share, accept_relaxed=accept_relaxed)


def measure_park(hag: np.ndarray, dgm: np.ndarray, inside: np.ndarray, conv: float, source: str, pitch0: float = np.nan,
                 res: float = RES, register_orientation: str = None, box: bool = False, win_m: float = None) -> dict:
    """Window method for one park on prepared rasters. Returns dict(win, G, arrays, park, nw, win_m):
    win = window table (every variable of every window), G = groups, arrays = generator input, park = acceptance,
    flags and summary values."""
    win, nw, win_m = measure_windows(hag, dgm, inside, conv, source, pitch0, res, win_m)
    if box and len(win):
        win = drop_isolated(win, nw)
        win = keep_central_component(win, nw, (hag.shape[0] / 2, hag.shape[1] / 2))   # the box is centred on the register point
    if box and len(win) and win.ok.astype(bool).any():
        win = keep_central_construction(win, nw, (hag.shape[0] / 2, hag.shape[1] / 2))
    flag = rejected_flag(win, nw, box) if len(win) else dict(n_tab_windows=0, rej_tab_frac=0.0, rej_cluster_windows=0, rej_cluster_frac=0.0, warn_rejected=False)
    G = group_windows(win)
    arrays = arrays_from_groups(G, register_orientation)
    park = park_status(win, G, flag, box)
    ok = win[win.ok.astype(bool)] if len(win) else win
    park.update(flag, source=source, win_m=win_m, box=box, conv=conv, pitch0=pitch0,
                tilt_med=float(ok.tilt_horiz.median()) if len(ok) else np.nan,
                tilt_p10=float(ok.tilt_horiz.quantile(0.1)) if len(ok) else np.nan,
                tilt_p90=float(ok.tilt_horiz.quantile(0.9)) if len(ok) else np.nan,
                tilt_ends_med=float(ok.tilt_horiz_ends.median()) if len(ok) else np.nan,
                tilt_rows_med=float(ok.tilt_rows_h.median()) if len(ok) else np.nan,
                n_groups=int(len(G)), az_undetermined=bool(G.az_undetermined.any()) if len(G) else False,
                az_conflict=bool(arrays.az_conflict.any()) if len(arrays) else False,
                slope_cover_min=float(ok.slope_cover.min()) if len(ok) else np.nan,
                n_splinter=int(G.attrs.get("n_splinter", 0)), n_cut=int(G.attrs.get("n_cut", 0)))
    return dict(win=win, G=G, arrays=arrays, park=park, nw=nw, win_m=win_m)


def measure_box(load, centre_xy: tuple, kw: float, source_pref=None, register_orientation: str = None, res: float = RES) -> dict:
    """Park without an OSM outline (E4b): square box around centre_xy (UTM m), grown in BOX['step_m'] rings while
    accepted windows touch the box edge, up to BOX['area_factor'] x the area expected from the capacity (kw x 5 m2
    x 2.2). load(bounds) -> dict(hag, dgm, conv, source, pitch0, ...) on a grid of `res` with these bounds."""
    cx, cy = centre_xy
    half = BOX["size_m"] / 2
    max_side = max(np.sqrt(BOX["area_factor"] * float(kw) * 5.0 * 2.2), BOX["size_m"])
    steps = []
    while True:
        bounds = (cx - half, cy - half, cx + half, cy + half)
        r = load(bounds)
        inside = np.ones(r["hag"].shape, bool)
        out = measure_park(r["hag"], r["dgm"], inside, r["conv"], r["source"], r.get("pitch0", np.nan), res, register_orientation, box=True)
        win, nw = out["win"], out["nw"]
        H, W = r["hag"].shape
        n_i, n_j = H // nw, W // nw
        edge = win[win.ok.astype(bool) & ((win.i0 // nw == 0) | (win.j0 // nw == 0) | (win.i0 // nw == n_i - 1) | (win.j0 // nw == n_j - 1))] if len(win) else win
        steps.append(dict(side_m=2 * half, n_ok=out["park"]["n_ok"], n_edge=int(len(edge))))
        if not len(edge) or 2 * (half + BOX["step_m"]) > max_side:
            break
        half += BOX["step_m"]
    out.update(rasters=r, bounds=bounds)
    out["park"].update(box_side_m=2 * half, box_steps=str(steps), box_edge_open=bool(len(edge)))
    return out


# ------------------------------------------------------------------------------------------------ raster preparation
def osm_footprint(elements: list, points: list, to_m, max_dist_m: float = 600.0) -> tuple:
    """(polygon in metres or None, fences table): solar plants (power=plant) as ways AND relations (outer members or
    members without role; ways need >= 4 nodes) within max_dist_m of any of `points` (shapely points in metres)."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform, unary_union
    fences = []
    for e in elements or []:
        tags = e.get("tags") or {}
        if tags.get("power") != "plant" or tags.get("plant:source", "solar") != "solar":
            continue
        rings = [e["geometry"]] if e["type"] == "way" else [m_["geometry"] for m_ in e.get("members", []) if m_.get("role") in ("outer", "") and len(m_.get("geometry", [])) >= 4]
        for ring in rings:
            if len(ring) >= 4:
                poly_m = shp_transform(lambda x, y, z=None: to_m(x, y), Polygon([(g["lon"], g["lat"]) for g in ring])).buffer(0)
                d = min(poly_m.distance(p) for p in points if p is not None)
                fences.append(dict(geom=poly_m, name=tags.get("name"), typ=e["type"], id=e.get("id"), area_ha=poly_m.area / 1e4, dist_m=d))
    near = [f["geom"] for f in fences if f["dist_m"] <= max_dist_m]
    return (unary_union(near) if near else None), pd.DataFrame(fences)


def analysis_grid(poly, res: float = RES, buffer_m: float = 20.0, inner_m: float = 5.0) -> dict:
    """Bounds (outline + 20 m), shape and the inside mask (outline - 5 m) of the analysis grid."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    minx, miny, maxx, maxy = poly.buffer(buffer_m).bounds
    H, W = int(round((maxy - miny) / res)), int(round((maxx - minx) / res))
    transform = from_bounds(minx, miny, maxx, maxy, W, H)
    inside = ~geometry_mask([poly.buffer(-inner_m)], (H, W), transform)
    return dict(bounds=(minx, miny, maxx, maxy), H=H, W=W, transform=transform, inside=inside)


def orthophoto_axis(rgb: np.ndarray, inside: np.ndarray, res: float = RES) -> dict:
    """Row axis and pitch of the park from the orthophoto spectrum (pv.geodata_tiles.row_axis_from_image, 4-15 m):
    first block, else first global peak; NaN without a row pattern."""
    from pv.geodata_tiles import row_axis_from_image
    rows = row_axis_from_image(rgb, inside, res)
    if rows["blocks"]:
        a, p = rows["blocks"][0]["axis_grid"], rows["blocks"][0]["pitch_m"]
    elif rows["peaks"]:
        a, p = rows["peaks"][0][0], rows["peaks"][0][1]
    else:
        a, p = np.nan, np.nan
    return dict(axis0=a, pitch0=p, rows=rows)


def laz_survey_dates(paths: list, bounds: tuple = None) -> list:
    """Acquisition dates of the LAZ tiles touching bounds, from the GPS time of the points (adjusted standard GPS
    time, median of the first chunk); the header date is the processing date (Bavaria: 2-5 months after the flight)
    and is only the fallback."""
    import datetime as _dt
    import laspy
    out = []
    for f in paths:
        if not f.lower().endswith((".laz", ".las")):
            continue
        with laspy.open(f) as lf:
            h_ = lf.header
            if bounds is not None:
                minx, miny, maxx, maxy = bounds
                if h_.maxs[0] <= minx or h_.mins[0] >= maxx or h_.maxs[1] <= miny or h_.mins[1] >= maxy:
                    continue
            ch_ = next(lf.chunk_iterator(200_000))
            gps_ = float(np.median(np.asarray(ch_.gps_time))) if "gps_time" in h_.point_format.dimension_names else np.nan
            out.append((_dt.datetime(1980, 1, 6) + _dt.timedelta(seconds=gps_ + 1e9)).date() if np.isfinite(gps_) and h_.global_encoding.gps_time_type else h_.creation_date)
    return out


def raster_tiles_touching(paths: list, bounds: tuple) -> list:
    """Raster tiles (GeoTIFF, zip with GeoTIFF, MV XYZ zip named by km and tile size) that touch bounds."""
    import rasterio
    minx, miny, maxx, maxy = bounds
    hit = []
    for f in paths:
        try:
            if f.lower().endswith(".zip") and not any(n_.lower().endswith((".tif", ".tiff")) for n_ in zipfile.ZipFile(f).namelist()):
                m_ = re.search(r"_(\d{3})_(\d{4})_(\d)_", os.path.basename(f))
                tb_ = (int(m_.group(1)) * 1000, int(m_.group(2)) * 1000, (int(m_.group(1)) + int(m_.group(3))) * 1000, (int(m_.group(2)) + int(m_.group(3))) * 1000) if m_ else bounds
            else:
                with rasterio.open(f"zip://{f}" if f.lower().endswith(".zip") else f) as r_:
                    tb_ = tuple(r_.bounds)
            if not (tb_[2] <= minx or tb_[0] >= maxx or tb_[3] <= miny or tb_[1] >= maxy):
                hit.append(f)
        except Exception:
            hit.append(f)
    return hit


def surface_from(name: str, paths: list, bounds: tuple, zone: int, res: float = RES, dates: dict = None) -> dict:
    """Surface model of one source: laz (max z per cell, ground = mean of class 2), dom1_laser, bdom_image,
    dom20_image (rasters, no ground). Returns dict(surf, ground, source, survey_year, survey_dates)."""
    from pv import geodata_tiles as GT
    if name == "laz":
        surf, ground, npts = GT.dsm_from_laz(paths, bounds, res)
        sd = laz_survey_dates(paths, bounds)
        return dict(surf=surf, ground=ground, source=name, n_points=npts, survey_dates=[str(d) for d in sd], survey_year=min(d.year for d in sd) if sd else None)
    surf = GT.mosaic(paths, bounds, res, zone)[0].astype(np.float32)
    hit = raster_tiles_touching(paths, bounds)
    if dates:                                                                   # e.g. NI STAC datetimes per tile
        ds = [str(dates[os.path.basename(f)]) for f in hit if os.path.basename(f) in dates] or [str(v) for v in dates.values()]
        yrs = [int(d[:4]) for d in ds if re.match(r"20\d\d", d)]
    else:
        ds, yrs = [], [int(m_) for f in hit for m_ in re.findall(r"_(20\d\d)", os.path.basename(f))]
    return dict(surf=surf, ground=None, source=name, survey_dates=ds or [os.path.basename(f) for f in hit], survey_year=min(yrs) if yrs else None)


def height_above_ground(surf: np.ndarray, dgm: np.ndarray, ground: np.ndarray = None) -> np.ndarray:
    """hag = surface - DGM1; where the DGM is missing, the laser ground (point clouds only)."""
    base = np.where(np.isfinite(dgm), dgm, ground) if ground is not None else dgm
    return (surf - base).astype(np.float32)


# ------------------------------------------------------------------------------------------------ figures
def plot_park(out: dict, hag: np.ndarray, path: str, title: str = "", res: float = RES) -> str:
    """Height above ground with the window grid, and the window maps of tilt, azimuth and pitch (rejected windows
    hatched). No orthophoto (safe for public parks and confidential ones alike)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    win, nw, win_m = out["win"], out["nw"], out["win_m"]
    H = hag.shape[0]
    extent = [0, hag.shape[1] * res, 0, H * res]
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].imshow(np.clip(hag, -0.5, 4), cmap="viridis", vmin=-0.5, vmax=4, extent=extent)
    for r_ in win.itertuples():
        x0, y0 = r_.j0 * res, (H - r_.i0 - nw) * res
        ax[0].add_patch(plt.Rectangle((x0, y0), win_m, win_m, fill=False, edgecolor="#B3121A" if r_.ok else "#8A8A8A", lw=0.8))
    ax[0].set_title("Höhe über Grund, Fenster (rot = angenommen)")
    for a_, col, lab, vmin, vmax, cmap in [(ax[1], "tilt_horiz", "Neigung horizontal (°)", 0, 35, "viridis"), (ax[2], "azimuth_true", "Azimut (°)", 0, 360, "hsv"),
                                           (ax[3], "pitch", "Pitch (m)", 4, 16, "plasma")]:
        a_.imshow(np.clip(hag, -0.5, 4), cmap="Greys", extent=extent, alpha=0.6)
        for r_ in win.itertuples():
            x0, y0 = r_.j0 * res, (H - r_.i0 - nw) * res
            if r_.ok:
                v_ = getattr(r_, col); cval = plt.get_cmap(cmap)((v_ - vmin) / (vmax - vmin))
                a_.add_patch(plt.Rectangle((x0, y0), win_m, win_m, facecolor=(*cval[:3], 0.75), edgecolor="k", lw=0.4))
                lab_ = f"{round(v_) % 360:.0f}" + ("" if r_.az_sure else "?") if col == "azimuth_true" else f"{v_:.0f}"
                a_.text(x0 + win_m / 2, y0 + win_m / 2, lab_, ha="center", va="center", fontsize=7, color="w" if col != "azimuth_true" else "k")
            elif r_.f_tab >= F_TAB_MIN:
                a_.add_patch(plt.Rectangle((x0, y0), win_m, win_m, facecolor=(0.5, 0.5, 0.5, 0.35), edgecolor="k", lw=0.3, hatch="//"))
        a_.set_title(lab)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)); plt.colorbar(sm, ax=a_, shrink=0.7)
    for a_ in ax:
        a_.set_xlabel("m Ost")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=90)
    plt.close(fig)
    return path
