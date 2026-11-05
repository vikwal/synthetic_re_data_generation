"""Tests for the dependency-light modules (wake, layout, analytics, charts). Run: pytest fastapi_app/tests"""
import math

import numpy as np
import pandas as pd

from fastapi_app import analytics, charts, layout, wake


def test_offset_roundtrip():
    lat, lon = wake.offset_to_latlon(51.0, 8.0, 500.0, -300.0)
    east, north = wake.offset_m(51.0, 8.0, lat, lon)
    assert abs(east - 500.0) < 1.0 and abs(north + 300.0) < 1.0


def test_circle_overlap():
    assert wake.circle_overlap_area(100, 10, 10) == 0
    assert math.isclose(wake.circle_overlap_area(0, 10, 5), math.pi * 25, rel_tol=1e-6)
    assert 0 < wake.circle_overlap_area(10, 10, 10) < math.pi * 100


def test_layout_count_and_spacing():
    pts = layout.grid_layout(52.0, 9.0, 2, 3, 5, 7, 100.0)
    assert len(pts) == 6
    east, north = wake.offset_m(pts[0]["lat"], pts[0]["lon"], pts[1]["lat"], pts[1]["lon"])
    assert abs(math.hypot(east, north) - 500) < 5


def test_layout_rotation_preserves_spacing():
    a = layout.grid_layout(52.0, 9.0, 1, 2, 5, 5, 100.0, 0)
    b = layout.grid_layout(52.0, 9.0, 1, 2, 5, 5, 100.0, 90)
    da = math.hypot(*wake.offset_m(a[0]["lat"], a[0]["lon"], a[1]["lat"], a[1]["lon"]))
    db = math.hypot(*wake.offset_m(b[0]["lat"], b[0]["lon"], b[1]["lat"], b[1]["lon"]))
    assert abs(da - db) < 1


def test_layout_limits():
    for bad in ((0, 3), (3, 0), (20, 20)):
        try:
            layout.grid_layout(52, 9, bad[0], bad[1], 5, 5, 100.0)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def _df(n=24 * 365 * 2, kind="wind"):
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"power": rng.uniform(0, 2e6, n)}, index=idx)
    if kind == "wind":
        d["wind_speed_hub"] = rng.uniform(0, 20, n)
        d["u_wind_100m"] = rng.normal(0, 5, n)
        d["v_wind_100m"] = rng.normal(0, 5, n)
    return d


def test_kpis_consistent():
    df = _df()
    k = analytics.kpis(df, "wind", 2e6)
    assert 40 < k["capacity_factor_pct"] < 60
    assert math.isclose(k["full_load_hours"], k["capacity_factor_pct"] / 100 * 8766, rel_tol=0.02)
    assert k["years"] > 1.9


def test_wind_rose_and_hist():
    df = _df()
    assert analytics.wind_rose(df) is not None
    assert analytics.wind_rose(df.drop(columns=["u_wind_100m"])) is None
    assert analytics.speed_histogram(df) is not None


def test_duration_curve_monotonic():
    d = analytics.duration_curve(_df(), 2e6)
    series = [v for v in d.values() if isinstance(v, list) and len(v) > 10]
    assert series
    ys = max(series, key=lambda s: max(x for x in s if x is not None))
    ys = [y for y in ys if y is not None]
    assert all(a >= b - 1e-9 for a, b in zip(ys, ys[1:])) or all(a <= b + 1e-9 for a, b in zip(ys, ys[1:]))


def test_figures_are_json_ready():
    import json
    df = _df()
    figs = [charts.seasonality_figure(analytics.seasonality(df)),
            charts.heatmap_figure(analytics.hour_month_heatmap(df, 2e6)),
            charts.duration_figure(analytics.duration_curve(df, 2e6)),
            charts.windrose_figure(analytics.wind_rose(df)),
            charts.speed_hist_figure(analytics.speed_histogram(df)),
            charts.waterfall_figure(analytics.energy_waterfall(100.0, 95.0, 90.0)),
            charts.daily_figure(df, "wind", "2020-03-01", "wind_speed_hub"),
            charts.quarterly_figure(df, "wind", 2020),
            charts.multi_year_figure(df, "wind")]
    for f in figs:
        assert "data" in f and "layout" in f
        json.dumps(f, allow_nan=False)


def test_real_csv_parsing_and_compare():
    from fastapi_app import realcompare as rc
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    syn = pd.DataFrame({"power": np.clip(rng.normal(2e6, 1e6, len(idx)), 0, 5e6)}, index=idx)
    real_p = syn["power"].values * 0.9
    txt = "timestamp;power_t1;power_t2\n" + "\n".join(
        f"{t:%Y-%m-%d %H:%M:%S};{p / 2e6:.5f};{p / 2e6:.5f}" for t, p in zip(idx, real_p))
    real = rc.parse_real_csv(txt, "MW")
    m, figs = rc.compare(real, syn, 5e6, "wind")
    assert abs(m["bias_pct"] - 11.11) < 0.5 and m["correlation"] > 0.999
    assert set(figs) == {"timeseries", "scatter", "diurnal", "monthly", "duration"}
    import json
    json.dumps(figs, allow_nan=False)


def test_real_csv_errors():
    from fastapi_app import realcompare as rc
    for bad in ("", "a;b\n1;2\n"):
        try:
            rc.parse_real_csv(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
