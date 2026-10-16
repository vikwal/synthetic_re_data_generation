"""Tests for pv.geodata_tiles / pv.geodata_states on synthetic height images (no downloads)."""
import numpy as np
import pytest

from pv import geodata_states as GS
from pv import geodata_tiles as GT


def _tables(axis_deg=85.0, pitch=10.0, tilt=20.0, width=4.0, res=0.5, size=200, ridge_h=2.0):
    """Height above ground of south-facing tables: ridge on the north edge, flank falling south."""
    n = int(size / res); jj, ii = np.meshgrid(np.arange(n), np.arange(n)); X = (jj + 0.5) * res; Y = (n - ii - 0.5) * res
    a = np.radians(axis_deg); s = X * np.cos(a) - Y * np.sin(a)                     # grows towards south-east-ish
    ph = np.mod(s, pitch)
    drop = np.tan(np.radians(tilt)) * width
    hag = np.where(ph < width, ridge_h - ph / width * drop, 0.0)
    return hag.astype(np.float32), np.ones_like(hag, bool)


def test_tiles_for_covers_full_range():
    t = GS.tiles_for(48.5883, 10.8443, 32, 800)
    es = {e for e, _ in t}; ns = {n for _, n in t}
    assert es == set(range(min(es), max(es) + 1)) and ns == set(range(min(ns), max(ns) + 1)) and len(t) >= 4


def test_meridian_convergence_sign():
    assert 1.0 < GT.meridian_convergence(48.5883, 10.8443, 32) < 2.0
    assert GT.meridian_convergence(49.3, 6.9, 32) < 0


def test_refine_fold_and_tilt_recover_geometry():
    hag, inside = _tables(axis_deg=85.0, pitch=10.0, tilt=20.0)
    axis, pitch, contrast = GT.refine_fold(hag, inside, 0.5, 87.0, 9.5)
    assert abs(axis - 85.0) < 0.6 and abs(pitch - 10.0) < 0.1 and contrast > 1.0
    t = GT.tilt_from_surface(hag, inside, 0.5, axis, pitch)
    assert t["ok"] and abs(t["tilt_plane_median"] - 20.0) < 1.5 and abs(t["tilt_profile"] - 20.0) < 4.0   # the binned profile underestimates by 1-3 deg
    assert not t["two_flank"]


def test_pitch_from_surface():
    hag, inside = _tables(axis_deg=90.0, pitch=12.0, tilt=15.0)
    p, c = GT.pitch_from_surface(hag, inside, 0.5, 90.0)
    assert abs(p - 12.0) < 0.3 and c > 0.5


def test_footprint_from_hag_picks_block():
    hag, inside = _tables(axis_deg=90.0, pitch=10.0, tilt=20.0, size=200)
    hag[:, : int(80 / 0.5)] = 0.0                                                     # tables only in the eastern 120 m
    poly = GT.footprint_from_hag(hag, 0.5, (0, 0, 200, 200), expected_m2=120 * 200, point_xy=(150, 100), pitch_m=10.0)
    assert poly is not None and 0.5 < poly.area / (120 * 200) < 1.3
    assert poly.centroid.x > 100
