"""Tests for pv.geodata_windows (window method) on synthetic parks with known geometry (no downloads).

Synthetic park (lengths m, angles deg): terrain z = z0 + gE*X + gN*Y; mono-facing rows with the module on phase
[0, depth) of each period (back edge at 0), hag = h0 + (tan(tilt) + G)(depth - phase) with G the terrain rise towards
the facing direction (so the absolute surface has `tilt` against the horizontal); east-west tents with the ridge at
phase `depth`. Rendered on a 0.1 m grid, then the highest point per 0.5 m cell (point cloud) or the 1 m mean
bilinearly resampled to 0.5 m (laser DOM1). Thresholds: FAIL level of the review harness (2026-10-21) -
pv_sim2real/notebooks/tests/test_window_body.py has the full scenario list with precision warnings."""
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from pv import geodata_windows as GW


def render_park(axis_grid, pitch, depth, tilt_h, face="right", h0=0.6, grad=(0.0, 0.0), size_m=200.0, mode="pc",
                noise=0.03, offset=0.0, seed=0, fine=0.1, res=0.5, z0=100.0):
    rng = np.random.default_rng(seed)
    n = int(round(size_m / fine))
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    X, Y = (jj + 0.5) * fine, (n - ii - 0.5) * fine
    a = np.radians(axis_grid)
    s = X * np.cos(a) - Y * np.sin(a)
    dir_across = float(np.degrees(np.arctan2(np.cos(a), -np.sin(a))) % 360)
    gE, gN = grad
    zt = z0 + gE * X + gN * Y
    tb = np.tan(np.radians(tilt_h))
    if face in ("right", "left"):
        az_grid = dir_across if face == "right" else (dir_across + 180) % 360
        G = gE * np.sin(np.radians(az_grid)) + gN * np.cos(np.radians(az_grid))
        u = s if face == "right" else -s
        ph = np.mod(u - offset, pitch)
        hag = np.where(ph < depth, h0 + (tb + G) * (depth - ph), 0.0)
    else:
        az_grid = None
        Ge = gE * np.sin(np.radians(dir_across)) + gN * np.cos(np.radians(dir_across))
        ph = np.mod(s - offset, pitch)
        dlt = ph - depth
        hag = np.where(ph < 2 * depth, h0 + tb * depth - tb * np.abs(dlt) - Ge * dlt, 0.0)
    hag = hag + np.where(hag > 0, 0.0, rng.uniform(0.0, 0.12, hag.shape))
    surf = zt + hag
    k = int(round(res / fine))
    if mode == "pc":
        s05 = surf.reshape(n // k, k, n // k, k).max(axis=(1, 3))
    else:
        k1 = int(round(1.0 / fine))
        s1 = surf.reshape(n // k1, k1, n // k1, k1).mean(axis=(1, 3))
        s05 = ndimage.zoom(s1, 2, order=1, grid_mode=True, mode="nearest")
    s05 = s05 + rng.normal(0, noise, s05.shape)
    m = s05.shape[0]
    j5, i5 = np.meshgrid(np.arange(m), np.arange(m))
    dgm = (z0 + gE * (j5 + 0.5) * res + gN * (m - i5 - 0.5) * res).astype(np.float32)
    return (s05 - dgm).astype(np.float32), dgm, az_grid, dir_across


def slope_grad(az_deg, g_desc_deg, along_deg=0.0):
    a = np.radians(az_deg)
    gd, ga = -np.tan(np.radians(g_desc_deg)), np.tan(np.radians(along_deg))
    v = gd * np.array([np.sin(a), np.cos(a)]) + ga * np.array([np.sin(a + np.pi / 2), np.cos(a + np.pi / 2)])
    return float(v[0]), float(v[1])


def ad(a, b, period=360.0):
    return float(abs(((a - b) + period / 2) % period - period / 2))


SCEN = [dict(name=f"axis{ax:.0f}_{f}", axis=ax, face=f) for ax in (0.0, 45.0, 90.0, 135.0, 179.0) for f in ("right", "left")]
SCEN += [dict(name="conv+1.5", axis=90.0, face="right", conv=1.5), dict(name="conv-2_axis45_left", axis=45.0, face="left", conv=-2.0),
         dict(name="tilt10", axis=90.0, face="right", tilt=10.0), dict(name="tilt30", axis=90.0, face="right", tilt=30.0, depth=3.4)]
SCEN += [dict(name=f"ew_axis{ax:.0f}", axis=ax, face="ew", tilt=10.0, pitch=14.0, depth=4.0, h0=0.5) for ax in (0.0, 12.0, 168.0)]
SCEN += [dict(name=f"slope{g:+.0f}_{f}", axis=90.0, face=f, slope=(g, 0.0)) for g in (5.0, -5.0) for f in ("right", "left")]
SCEN += [dict(name="slope_along5", axis=90.0, face="right", slope=(0.0, 5.0)), dict(name="slope_diag_axis45", axis=45.0, face="right", slope=(3.0, 4.0)),
         dict(name="ew_slope4_conv", axis=12.0, face="ew", tilt=10.0, pitch=14.0, depth=4.0, h0=0.5, slope=(4.0, 0.0), conv=1.5),
         dict(name="dense_d0.70", axis=90.0, face="right", pitch=8.0, depth=5.6, tilt=15.0)]
SCEN += [dict(name=f"offset{o}", axis=90.0, face="right", offset=o) for o in (0.25, 0.37)]
SCEN += [dict(name=f"dom1_axis{ax:.0f}_{f}", axis=ax, face=f, mode="dom1", tol=2.0) for ax in (90.0, 45.0) for f in ("right", "left")]
SCEN += [dict(name="dom1_ew", axis=0.0, face="ew", tilt=10.0, pitch=14.0, depth=4.0, h0=0.5, mode="dom1", tol=2.0),
         dict(name="dom1_slope5", axis=90.0, face="right", slope=(5.0, 0.0), mode="dom1", tol=2.0)]


@pytest.mark.parametrize("sc", SCEN, ids=[s["name"] for s in SCEN])
def test_synthetic_park(sc):
    pitch, depth, tilt, conv = sc.get("pitch", 8.5), sc.get("depth", 3.8), sc.get("tilt", 20.0), sc.get("conv", 0.0)
    a = np.radians(sc["axis"]); dir_across = float(np.degrees(np.arctan2(np.cos(a), -np.sin(a))) % 360)
    az_face = dir_across if sc["face"] in ("right", "ew") else (dir_across + 180) % 360
    grad = slope_grad(az_face, *sc["slope"]) if "slope" in sc else (0.0, 0.0)
    hag, dgm, _, _ = render_park(sc["axis"], pitch, depth, tilt, sc["face"], h0=sc.get("h0", 0.6), grad=grad, mode=sc.get("mode", "pc"),
                                 offset=sc.get("offset", 0.0))
    src = "laz" if sc.get("mode", "pc") == "pc" else "dom1_laser"
    out = GW.measure_park(hag, dgm, np.ones(hag.shape, bool), conv, src, pitch0=pitch)
    win, G, arrays, park = out["win"], out["G"], out["arrays"], out["park"]
    ok = win[win.ok]
    assert len(ok), win.reason.value_counts().to_dict()
    assert park["status"] == "accepted"
    exp_axis = (sc["axis"] + conv) % 180
    assert np.median([ad(x, exp_axis, 180) for x in ok.axis_true]) <= 1.0
    assert 100 * np.median(np.abs(ok.pitch - pitch)) / pitch <= 3.0
    assert abs(ok.tilt_horiz.median() - tilt) <= sc.get("tol", 1.0)
    if sc["face"] == "ew":
        assert ok.ew_pair.mean() >= 0.9 and len(ok) >= 0.6 * len(win)
        exp_az = sorted([(exp_axis + 90) % 360, (exp_axis + 270) % 360])
        got = sorted(arrays.azimuth.tolist())
        assert len(got) == 2 and max(ad(g, e) for g, e in zip(got, exp_az)) <= 1.5
    else:
        exp_az = (az_face + conv) % 360
        assert np.median([ad(x, exp_az) for x in ok.azimuth_true]) <= 1.0
        assert not ok.ew_pair.any()
        assert len(G) == 1 and ad(G.azimuth.iloc[0], exp_az) <= 1.5 and arrays.az_source.iloc[0] == "measured"
        if "slope" in sc:
            assert abs(ok.slope_facing_deg.median() - sc["slope"][0]) <= 0.3


# ------------------------------------------------------------------------------------------------ grouping
def _table(spec):
    rows, w = [], 0
    for n, axis, az, tilt, ew, sure, sym, area in spec:
        for k in range(n):
            a_ = az[k % len(az)] if isinstance(az, (list, tuple)) else az
            t_ = tilt[k % len(tilt)] if isinstance(tilt, (list, tuple)) else tilt
            rows.append(dict(w=w, i0=0, j0=100 * w, f_tab=0.4, ok=True, area_tab=area, axis_true=axis, azimuth_true=a_ % 360, tilt_horiz=t_,
                             ew_pair=ew, az_sure=sure, ridge_sym=sym, pitch=8.5, slope_facing_deg=0.0))
            w += 1
    return pd.DataFrame(rows)


def test_group_north_wrap_merges():
    win = _table([(6, 90.0, [350, 355, 5, 10, 352, 8], [20.0, 20.5, 21.0], False, True, 0.2, 400.0),
                  (6, 90.0, [358, 359, 2, 1], [24.5, 24.2], False, True, 0.2, 380.0)])
    G = GW.group_windows(win)
    assert len(G) == 1 and ad(G.azimuth.iloc[0], 0.0) <= 3


def test_group_axis_wrap():
    G = GW.group_windows(_table([(8, 179.0, 270.0, 20.0, False, True, 0.2, 400.0), (8, 1.0, 270.0, 20.0, False, True, 0.2, 400.0)]))
    assert len(G) == 1 and ad(G.axis_true.iloc[0], 0.0, 180) <= 2


def test_group_splinters_before_cut():
    spec = [(10, 90.0, 180.0, 10.0 + 6 * k, False, True, 0.2, 400.0) for k in range(5)] + [(2, 30.0, 120.0, 40.0, False, True, 0.2, 3000.0)]
    G = GW.group_windows(_table(spec))
    assert len(G) == 4 and G.n_windows.min() >= 3 and G.attrs["n_splinter"] == 1 and G.attrs["n_cut"] == 1


def test_group_east_west_arrays():
    G = GW.group_windows(_table([(9, 167.0, [77.0, 257.0], 9.5, True, True, 0.93, 300.0)]))
    A = GW.arrays_from_groups(G)
    assert sorted(A.azimuth.round(1).tolist()) == [77.0, 257.0] and np.allclose(A.share, 0.5) and A.note.iloc[0] == "Ost-Flanke"


def test_group_opposite_pair_undetermined_register():
    G = GW.group_windows(_table([(8, 90.0, 180.0, 12.0, False, True, 0.9, 300.0), (4, 90.0, 0.0, 12.5, False, True, 0.9, 300.0)]))
    assert len(G) == 1 and bool(G.az_undetermined.iloc[0])
    A = GW.arrays_from_groups(G, "Süd")
    assert A.az_source.iloc[0] == "register" and ad(A.azimuth.iloc[0], 180.0) < 1e-6


def test_group_unsure_do_not_vote():
    G = GW.group_windows(_table([(6, 90.0, 180.0, 15.0, False, True, 0.3, 300.0), (6, 90.0, 0.0, 15.0, False, False, 0.3, 300.0)]))
    assert len(G) == 1 and ad(G.azimuth.iloc[0], 180.0) <= 1


def test_group_few_sure_windows_undetermined():
    G = GW.group_windows(_table([(2, 90.0, 180.0, 15.0, False, True, 0.3, 300.0), (6, 90.0, 0.0, 15.0, False, False, 0.3, 300.0)]))
    assert len(G) == 1 and bool(G.az_undetermined.iloc[0]) and G.n_sure.iloc[0] == 2


def test_register_azimuth_rules():
    assert GW.register_azimuth(90.0, "Süd") == (180.0, "register", False)
    assert GW.register_azimuth(90.0, "Nord") == (0.0, "register", False)
    az, src, conflict = GW.register_azimuth(0.0, "Süd")            # south class on north-south rows
    assert src == "register_conflict_south" and conflict and az in (90.0, 270.0)
    assert GW.register_azimuth(100.0, None)[1] == "assumed_south" and ad(GW.register_azimuth(100.0, None)[0], 190.0) < 1e-6


def test_park_status_rules():
    win = _table([(6, 90.0, 180.0, 20.0, False, True, 0.2, 400.0)])
    G = GW.group_windows(win)
    flag = GW.rejected_flag(win, 100)
    assert GW.park_status(win, G, flag)["status"] == "accepted"
    win2 = win.copy(); win2.loc[2:, "ok"] = False
    assert GW.park_status(win2, GW.group_windows(win2), GW.rejected_flag(win2, 100))["status"] == "failed"
    win3 = win.copy(); win3["f_tab"] = 0.05; win3["ok"] = False
    assert GW.park_status(win3, GW.group_windows(win3), GW.rejected_flag(win3, 100))["status"] == "no_park"


def test_drop_isolated():
    win = pd.DataFrame(dict(w=range(4), i0=[0, 0, 100, 300], j0=[0, 100, 0, 300], ok=[True, True, True, True], reason=[""] * 4))
    out = GW.drop_isolated(win, 100)
    assert out.ok.tolist() == [True, True, True, False] and out.reason.iloc[3] == "isoliert"


def test_end_point_estimator_kept():
    hag, dgm, _, _ = render_park(90.0, 8.5, 3.8, 20.0, "right", offset=0.37)
    out = GW.measure_park(hag, dgm, np.ones(hag.shape, bool), 0.0, "laz", pitch0=8.5)
    ok = out["win"][out["win"].ok]
    assert abs(ok.tilt_horiz.median() - 20.0) < 0.3                  # line fit
    assert ok.tilt_horiz_ends.median() < ok.tilt_horiz.median()       # end points underestimate on this geometry


def test_keep_central_component():
    # two blocks of accepted windows; the one around the centre (window 2,2 of a 5x5 grid of nw=100) is kept. The other block
    # lies within the BRIDGE gap but is built differently (axis 90 instead of 0), so the gap rule must not join it.
    rows = [dict(w=k, i0=i * 100, j0=j * 100, ok=True, reason="", axis_true=0.0 if k < 3 else 90.0, tilt_horiz=20.0,
                 pitch=8.0, ew_pair=False) for k, (i, j) in enumerate([(2, 2), (2, 1), (1, 2), (0, 4), (1, 4)])]
    out = GW.keep_central_component(pd.DataFrame(rows), 100, (250.0, 250.0))
    assert out.ok.tolist() == [True, True, True, False, False] and set(out.reason[3:]) == {"anderer Park"}


def test_small_park_two_windows_accepted():
    # 2 accepted windows covering all table cells: one group (no splinter), measured azimuth with both sure, park accepted
    win = _table([(2, 90.0, 180.0, 20.0, False, True, 0.2, 400.0)])
    G = GW.group_windows(win)
    assert len(G) == 1 and not bool(G.az_undetermined.iloc[0])
    assert GW.park_status(win, G, GW.rejected_flag(win, 100))["status"] == "accepted"


def test_box_keeps_central_construction():
    # central east-west park (axis 0) next to a south park (axis 90): the south group is another park
    rows, w = [], 0
    for i, j, ew, axis, az in [(2, 2, True, 0.0, 90.0), (2, 1, True, 0.0, 270.0), (1, 2, True, 0.0, 90.0), (2, 3, False, 90.0, 180.0), (1, 3, False, 90.0, 180.0), (3, 3, False, 90.0, 180.0)]:
        rows.append(dict(w=w, i0=i * 100, j0=j * 100, f_tab=0.4, ok=True, reason="", area_tab=300.0, axis_true=axis, azimuth_true=az, tilt_horiz=12.0 if ew else 25.0,
                         ew_pair=ew, az_sure=True, ridge_sym=0.9 if ew else 0.2, pitch=8.0, slope_facing_deg=0.0)); w += 1
    out = GW.keep_central_construction(pd.DataFrame(rows), 100, (250.0, 250.0))
    assert out.ok.tolist() == [True, True, True, False, False, False] and set(out.reason[3:]) == {"anderer Park (Bauweise)"}


def test_select_fences_by_capacity():
    from shapely.geometry import box
    fences = pd.DataFrame(dict(geom=[box(0, 0, 100, 100), box(150, 0, 250, 100), box(400, 0, 700, 300)], dist_m=[0.0, 50.0, 300.0], area_ha=[1.0, 1.0, 9.0]))
    poly, used, n = GW.select_fences(fences, 15000.0, 400.0)       # 1.5 x 15000 = 22500 m2: the first two plants fit, the third not
    assert (used, n) == (2, 3) and abs(poly.area - 20000.0) < 1e-6


def test_trim_to_capacity():
    # two groups: a near south group and a far group of another construction; capacity fits the near one only
    rows, w = [], 0
    for j, (axis, az, tilt) in enumerate([(90.0, 180.0, 25.0)] * 4 + [(60.0, 150.0, 15.0)] * 4):
        rows.append(dict(w=w, i0=0, j0=100 * j, x=50.0 + 100 * j, y=50.0, f_tab=0.4, ok=True, reason="", area_tab=500.0, axis_true=axis, azimuth_true=az,
                         tilt_horiz=tilt, ew_pair=False, az_sure=True, ridge_sym=0.2, pitch=8.0, slope_facing_deg=0.0)); w += 1
    out, action = GW.trim_to_capacity(pd.DataFrame(rows), kwp=300.0, centre_xy=(50.0, 50.0), tab_share=1.0)   # 4000 m2 / 300 kWp = 13.3 > 12
    assert out.ok.tolist() == [True] * 4 + [False] * 4 and action
    same, action = GW.trim_to_capacity(pd.DataFrame(rows), kwp=1000.0, centre_xy=(50.0, 50.0), tab_share=1.0)  # 4 m2/kWp: nothing to do
    assert same.ok.all() and action == ""
