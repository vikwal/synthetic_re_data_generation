"""Unit tests of the cams_bc package (no data access)."""
import numpy as np
import pandas as pd
import pytest

from cams_bc import data, features as F, metrics, splits, targets
from cams_bc.config import AZIMUTHS, RADII


def _frame(n=96):
    idx = pd.date_range("2024-03-01 00:30", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    clear = np.clip(800 * np.sin(np.pi * ((idx.hour + idx.minute / 60) - 4) / 16), 0, None)
    d = pd.DataFrame(index=idx)
    d["cams_ghi_clear"] = clear
    d["cams_ghi"] = clear * rng.uniform(0.3, 1.0, n)
    d["cams_dhi"] = d["cams_ghi"] * rng.uniform(0.2, 0.9, n)
    d["cams_bhi"] = d["cams_ghi"] - d["cams_dhi"]
    d["cams_bni"] = d["cams_bhi"] * 1.2
    d["cams_dhi_clear"], d["cams_bhi_clear"] = 0.15 * clear, 0.85 * clear
    d["cams_toa"] = clear * 1.4
    d["cams_reliability"] = 1.0
    for c in data.ERA5_RENAME.values():
        d[c] = rng.uniform(0, 1, n)
    d["ghi_obs"] = d["cams_ghi"] * rng.uniform(0.8, 1.2, n)
    d["dhi_obs"] = d["cams_dhi"] * rng.uniform(0.8, 1.2, n)
    d["y_valid_ghi"] = d["y_valid_dhi"] = clear > 10
    d["day_mask"] = clear > 10
    return d


def _ring(idx):
    rng = np.random.default_rng(1)
    out = {}
    for r in RADII:
        for a in AZIMUTHS:
            g = pd.DataFrame(index=idx)
            g["ghi_clear"] = 800.0
            g["ghi"] = 800 * rng.uniform(0.3, 1.0, len(idx))
            g["dhi"] = g["ghi"] * 0.5
            out[(r, a)] = g
    return out


def test_target_roundtrip():
    d = _frame()
    for form in targets.FORMS:
        for v in ("ghi", "dhi"):
            y, w = targets.encode(form, d, v)
            back = targets.decode(form, y, d, v)
            assert np.allclose(back, d[f"{v}_obs"].clip(lower=0), atol=1e-6), form
            assert abs(w.mean() - 1.0) < 1e-9


def test_kc_weights_make_loss_in_wm2():
    d = _frame()
    y, w = targets.encode("kc", d, "ghi")
    clear = np.maximum(d["cams_ghi_clear"].to_numpy(), 1.0)
    assert np.allclose(w / w.mean() * w.mean(), (clear ** 2) / (clear ** 2).mean())


def _nb(idx):
    table = pd.DataFrame({f"{s}_{v}": np.full(len(idx), 0.5 + 0.1 * i) for i, s in enumerate(["00001", "00002", "00003", "00004", "00005"]) for v in ("kc", "kd", "kcc")}, index=idx)
    coords = pd.DataFrame({"lat": [50.2, 50.4, 50.6, 50.8, 51.0], "lon": [10.0] * 5}, index=["00001", "00002", "00003", "00004", "00005"])
    return data.neighbour_features(50.0, 10.0, idx, table, coords)


def test_assemble_columns_and_context():
    d = _frame()
    f = data.assemble(d, data.ring_features(_ring(d.index), d.index), 50.0, 10.0, 100.0, _nb(d.index))
    for c in F.columns(F.FULL):
        assert c in f, c
    assert f["kc_lag_m1"].iloc[5] == pytest.approx(f["kc_cams"].iloc[4])
    assert f["kc_lag_p1"].iloc[5] == pytest.approx(f["kc_cams"].iloc[6])
    assert (f["ring3_kc_mean"].between(0, 1)).all()
    assert f["ring_all_kc_mean"].iloc[10] == pytest.approx((f["ring3_kc_mean"].iloc[10] + f["ring8_kc_mean"].iloc[10]) / 2)
    assert (f["cos_zen"].between(0, 1)).all()


def test_splits_disjoint(tmp_path, monkeypatch):
    ids = [f"{i:05d}" for i in range(1, 31)]
    st = pd.DataFrame({"Stations_id": ids, "geoBreite": np.linspace(47.5, 54.5, 30), "geoLaenge": np.r_[np.linspace(6, 14, 15), np.linspace(6, 14, 15)],
                       "Stationshoehe": 100})
    p = tmp_path / "stations.csv"
    st.to_csv(p, index=False)
    monkeypatch.setattr(splits, "STATIONS_LIST", str(p))
    sp = splits.make_splits(ids)
    assert set(sp.role) == {"test", "train"}
    assert 4 <= (sp.role == "test").sum() <= 8
    tr, va = splits.fold_ids(sp, 0)
    assert not set(tr) & set(va)
    assert not set(splits.test_ids(sp)) & (set(tr) | set(va))
    assert sorted(set(sp.loc[sp.role == "train", "fold"])) == [0, 1, 2]


def test_metrics_skill():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.basic(y, y)["rmse"] == 0.0
    assert metrics.skill(1.0, 2.0) == 0.5
    df = pd.DataFrame({"station_id": ["a"] * 4, "ghi_obs": y, "cams_ghi": y + 1}, index=pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"))
    e = metrics.evaluate(df, {"m": y + 0.5}, "ghi", np.ones(4, bool))
    assert e["m"]["skill"] == pytest.approx(0.5)
    h = metrics.hourly(df, y + 0.5, "ghi", np.ones(4, bool))
    assert len(h) == 2


def test_neighbour_features_exclude_self():
    idx = pd.date_range("2024-03-01 00:30", periods=8, freq="30min", tz="UTC")
    table = pd.DataFrame({f"{s}_{v}": np.full(8, i + 1.0) for i, s in enumerate(["00001", "00002", "00003"]) for v in ("kc", "kd", "kcc")}, index=idx)
    coords = pd.DataFrame({"lat": [50.0, 50.1, 51.0], "lon": [10.0, 10.0, 10.0]}, index=["00001", "00002", "00003"])
    nb = data.neighbour_features(50.0, 10.0, idx, table, coords, exclude=["00001"], k=2)
    assert nb["nb0_kc"].iloc[0] == 2.0 and nb["nb1_kc"].iloc[0] == 3.0
    assert nb["nb0_dist_km"].iloc[0] == pytest.approx(11.1, abs=0.2)
    assert nb["nb0_err"].iloc[0] == 0.0
    assert np.isfinite(nb["nb_kc_wmean"]).all()
