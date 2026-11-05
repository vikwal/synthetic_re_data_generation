"""Tests for the pre-trained model pipeline (linear model path; torch models need PyTorch)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("SYNTH_RE_STORAGE_DIR", tempfile.mkdtemp())

from fastapi_app import mlmodels, mlprep, mlservice, storage  # noqa: E402


def _dataset(n=24 * 120, scale=1.0, seed=0, kind="wind"):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    ws = np.clip(6 + 3 * np.sin(np.arange(n) / 30.0) + rng.normal(0, 1, n), 0, 25)
    power = np.clip((ws / 12.0) ** 3, 0, 1) * 2e6 * scale
    return pd.DataFrame({"power": power, "wind_speed_hub": ws, "u_wind_100m": ws * 0.7, "v_wind_100m": ws * 0.7,
                         "density_hub": 1.2, "temperature_hub": 8.0}, index=idx)


def _patch_results(monkeypatch_dict):
    storage.get_result_meta = lambda rid: monkeypatch_dict[rid][0]
    storage.get_result_df = lambda rid: monkeypatch_dict[rid][1]


def test_windows_and_split():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    X[5, 0] = np.nan
    idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    Xw, yw, tw = mlprep.make_windows(X, y, idx, 4)
    assert Xw.shape[1:] == (4, 2) and len(yw) == len(tw)
    assert not np.isnan(Xw).any()
    assert not set(yw.tolist()) & {5, 6, 7, 8}     # windows ending at 5..8 contain the NaN row
    cut = mlprep.chrono_split(100, 0.2)
    assert cut == 80


def test_linear_pretrain_and_finetune_roundtrip():
    results = {
        "a" * 12: ({"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "synthetic"}, _dataset()),
    }
    _patch_results(results)
    jobs_progress = []
    mid = mlservice.run_training(
        {"architecture": "linear", "result_ids": ["a" * 12], "name": "lin",
         "hyperparams": {"lookback": 6, "ridge_alpha": 1.0}}, lambda f, m="": jobs_progress.append(f))
    meta = mlservice.describe_model(mid)
    assert meta["stage"] == "pretrained" and meta["metrics"]["val"]["rmse_pct"] < 15
    assert jobs_progress[-1] == 1.0
    for fig_name in ("predictions", "scatter", "residuals"):
        json.dumps(mlservice.model_figure(mid, fig_name), allow_nan=False)
    try:
        mlservice.model_figure(mid, "loss")          # closed-form model: no learning curve
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError")

    # real plant: same weather, systematically 25 % lower output
    syn = results["a" * 12][1]
    real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.75:.1f}" for t, p in zip(syn.index, syn["power"]))
    child = mlservice.run_finetune(
        {"model_id": mid, "feature_result_id": "a" * 12, "csv_text": real_csv, "unit": "W", "ridge_alpha": 1.0},
        lambda f, m="": None)
    cm = mlservice.describe_model(child)
    assert cm["stage"] == "finetuned" and cm["parent_id"] == mid
    assert cm["metrics"]["val"]["rmse_pct"] < cm["metrics"]["zero_shot"]["rmse_pct"]
    json.dumps(mlservice.model_figure(child, "comparison"), allow_nan=False)
    json.dumps(mlservice.model_figure(child, "residuals"), allow_nan=False)

    assert {m["id"] for m in storage.list_models()} >= {mid, child}
    storage.delete_model(child)
    assert child not in {m["id"] for m in storage.list_models()}


def test_architecture_availability_and_hp_clamp():
    ids = {a["id"]: a for a in mlmodels.availability()}
    assert ids["linear"]["available"] and set(ids) == {"linear", "gru", "tcn", "tft"}
    hp = mlmodels.clamp_hp({"hidden": 70, "lookback": 9999, "epochs": 0})
    assert hp["hidden"] % 4 == 0 and hp["lookback"] == 168 and hp["epochs"] == 1


def test_mixed_kinds_rejected():
    results = {"a" * 12: ({"kind": "wind", "summary": {"capacity_w": 2e6}}, _dataset()),
               "b" * 12: ({"kind": "solar", "summary": {"capacity_w": 1e4}}, _dataset())}
    _patch_results(results)
    try:
        mlservice.run_training({"architecture": "linear", "result_ids": ["a" * 12, "b" * 12],
                                "hyperparams": {"lookback": 6}}, lambda f, m="": None)
    except ValueError as exc:
        assert "same kind" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_feature_selection_subset_roundtrip():
    # resolve_features: defaults, a subset, and errors
    assert mlprep.resolve_features(None, "wind") == (None, mlprep.TIME_FEATURES)
    w, t = mlprep.resolve_features(["wind_speed_hub", "time_of_day"], "wind")      # time group ids are ignored
    assert w == ["wind_speed_hub"] and t == mlprep.TIME_FEATURES
    for bad in ([], ["not_a_feature"], ["Total"]):          # empty, unknown, solar column on wind
        try:
            mlprep.resolve_features(bad, "wind")
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    cat = mlprep.catalog()
    assert {f["id"] for f in cat["wind"]} == set(mlprep.WIND_FEATURES) and len(cat["time"]) == 2

    # train on a subset (wind speed only, no time features) and fine-tune: feature set must carry through
    results = {"b" * 12: ({"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "s"}, _dataset(seed=3))}
    _patch_results(results)
    mid = mlservice.run_training(
        {"architecture": "linear", "result_ids": ["b" * 12], "features": ["wind_speed_hub"],
         "hyperparams": {"lookback": 4}}, lambda f, m="": None)
    meta = mlservice.describe_model(mid)
    assert meta["features"] == ["wind_speed_hub"] + mlprep.TIME_FEATURES and meta["time_features"] == mlprep.TIME_FEATURES
    syn = results["b" * 12][1]
    real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.8:.1f}" for t, p in zip(syn.index, syn["power"]))
    child = mlservice.run_finetune({"model_id": mid, "feature_result_id": "b" * 12, "csv_text": real_csv, "unit": "W"},
                                   lambda f, m="": None)
    cm = mlservice.describe_model(child)
    assert cm["features"] == ["wind_speed_hub"] + mlprep.TIME_FEATURES and cm["time_features"] == mlprep.TIME_FEATURES


def test_nwp_weather_source_roundtrip():
    import tempfile
    from pathlib import Path
    from fastapi_app import nwp, settings

    # fake ICON-D2 folder with one grid point (CSV, so the test needs no pyarrow)
    folder = Path(tempfile.mkdtemp())
    settings.NWP_DIRS["icon_d2"] = folder
    settings.NWP_DIRS["ecmwf"] = Path(tempfile.mkdtemp())          # empty: must not depend on real downloaded data
    base = _dataset(seed=5)
    nwp_df = pd.DataFrame({"u_wind_100m": base["wind_speed_hub"] * 0.7 + 0.3, "v_wind_100m": base["wind_speed_hub"] * 0.7,
                           "u_wind_10m": 1.0, "v_wind_10m": 1.0, "temp_2m": 280.0, "dew_point_2m": 275.0,
                           "pressure": 1e5}, index=base.index)
    nwp_df.index.name = "time"
    nwp_df.to_csv(folder / "grid_51.2500_10.7500.csv")
    assert {s["id"]: s["available"] for s in nwp.sources() if s["id"] in ("era5", "icon_d2", "ecmwf")} == {"era5": True, "icon_d2": True, "ecmwf": False}

    meta = {"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "n",
            "params": {"turbines": [{"lat": 51.27, "lon": 10.77}]}}
    assert nwp.site_of(meta) == (51.27, 10.77)
    _patch_results({"c" * 12: (meta, base)})
    w, t = mlprep.resolve_features(["wind_speed_100m", "temp_2m"], "wind", "icon_d2")
    assert w == ["wind_speed_100m", "temp_2m"] and t == mlprep.TIME_FEATURES
    mid = mlservice.run_training(
        {"architecture": "linear", "result_ids": ["c" * 12], "weather_source": "icon_d2",
         "features": ["wind_speed_100m", "temp_2m", "time_of_day"], "hyperparams": {"lookback": 4}}, lambda f, m="": None)
    m = mlservice.describe_model(mid)
    assert m["weather_source"] == "icon_d2" and m["weather_points"][0]["grid_lat"] == 51.25
    assert m["features"] == ["wind_speed_100m", "temp_2m"] + mlprep.TIME_FEATURES
    assert m["metrics"]["val"]["rmse_pct"] < 20

    real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.9:.1f}" for t, p in zip(base.index, base["power"]))
    child = mlservice.run_finetune({"model_id": mid, "feature_result_id": "c" * 12, "csv_text": real_csv, "unit": "W"},
                                   lambda f, m="": None)
    assert mlservice.describe_model(child)["weather_source"] == "icon_d2"


def test_several_weather_models_in_one_run():
    from fastapi_app import nwp

    base = _dataset(seed=7)
    idx = base.index

    def fake_weather(source, meta, lead=None):
        if source == "m_a":                                                    # covers every hour
            df = pd.DataFrame({"wind_speed_100m": base["wind_speed_hub"] + 0.2, "temp_2m": 280.0}, index=idx)
        else:                                                                  # m_b starts 10 days later
            df = pd.DataFrame({"wind_speed_100m": base["wind_speed_hub"] - 0.2, "temp_2m": 281.0}, index=idx[240:])
        return df, {"source": source}

    real_for, real_lead = nwp.weather_for, nwp.has_lead
    nwp.weather_for, nwp.has_lead = fake_weather, lambda s: False
    try:
        wx, notes = nwp.weather_multi(["m_a", "m_b"], {}, None)
        assert list(wx.columns) == ["m_a:wind_speed_100m", "m_a:temp_2m", "m_b:wind_speed_100m", "m_b:temp_2m"]
        assert wx.index.min() == idx[240] and len(notes) == 2                  # only hours every model has
        w, t = mlprep.resolve_features_multi(["m_a:wind_speed_100m", "m_b:wind_speed_100m", "season"], "wind", ["m_a", "m_b"])
        assert w == ["m_a:wind_speed_100m", "m_b:wind_speed_100m"] and t == mlprep.TIME_FEATURES
        try:
            mlprep.resolve_features_multi(["m_c:temp_2m"], "wind", ["m_a", "m_b"])
            raise AssertionError("expected an unknown-feature error")
        except ValueError as e:
            assert "Unknown" in str(e)

        meta = {"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "n", "params": {"turbines": [{"lat": 51.0, "lon": 10.0}]}}
        _patch_results({"f" * 12: (meta, base)})
        spec = {"architecture": "linear", "result_ids": ["f" * 12], "weather_sources": ["m_a", "m_b"], "power_lag": 0,
                "features": ["m_a:wind_speed_100m", "m_b:wind_speed_100m", "m_b:temp_2m"], "hyperparams": {"lookback": 4}}
        mid = mlservice.run_training(spec, lambda f, m="": None)
        m = mlservice.describe_model(mid)
        assert m["weather_sources"] == ["m_a", "m_b"] and len(m["weather_points"]) == 2
        assert m["weather_features"] == ["m_a:wind_speed_100m", "m_b:wind_speed_100m", "m_b:temp_2m"]
        assert m["features"][:3] == m["weather_features"] and m["metrics"]["val"]["rmse_pct"] < 25
        real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.9:.1f}" for t, p in zip(idx, base["power"]))
        child = mlservice.run_finetune({"model_id": mid, "feature_result_id": "f" * 12, "csv_text": real_csv, "unit": "W"},
                                       lambda f, m="": None)
        assert mlservice.describe_model(child)["weather_sources"] == ["m_a", "m_b"]
        try:                                                                   # ERA5 cannot be mixed with forecast models
            mlservice.run_training({**spec, "weather_sources": ["era5", "m_a"]}, lambda f, m="": None)
            raise AssertionError("expected an error")
        except ValueError as e:
            assert "ERA5" in str(e)
    finally:
        nwp.weather_for, nwp.has_lead = real_for, real_lead


def test_recent_power_input_has_no_leakage():
    df = _dataset(n=24 * 40, seed=9)
    cap = 2e6
    L, lag = 5, 3
    X0, y0, t0, _ = mlprep.prepare(df, "wind", cap, L, weather=["wind_speed_hub"], time_cols=[])
    X, y, t, feats = mlprep.prepare(df, "wind", cap, L, weather=["wind_speed_hub"], time_cols=[], power_lag=lag)
    assert X.shape[2] == X0.shape[2] + 1                        # one extra input column: the lagged power
    power = (df["power"] / cap)
    for k in (0, 17, 100):                                      # window ending at t holds power at t-lag, t-lag-1, ...
        end = t[k]
        expected = [float(power.loc[end - pd.Timedelta(hours=lag + j)]) for j in range(L - 1, -1, -1)]
        assert np.allclose(X[k, :, -1], expected, atol=1e-5)
        assert abs(float(y[k]) - float(power.loc[end])) < 1e-6   # target is the power at t, which is not in the inputs
        assert end - pd.Timedelta(hours=lag) < end

    mid = None
    _patch_results({"d" * 12: ({"kind": "wind", "summary": {"capacity_w": cap}, "scenario_name": "p"}, df)})
    mid = mlservice.run_training({"architecture": "linear", "result_ids": ["d" * 12], "power_lag": 3, "features": ["wind_speed_hub"],
                                  "hyperparams": {"lookback": 6}}, lambda f, m="": None)
    meta = mlservice.describe_model(mid)
    assert meta["power_lag"] == 3 and meta["features"][-1] == "power_lag"
    real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.9:.1f}" for t, p in zip(df.index, df["power"]))
    child = mlservice.run_finetune({"model_id": mid, "feature_result_id": "d" * 12, "csv_text": real_csv, "unit": "W"},
                                   lambda f, m="": None)
    assert mlservice.describe_model(child)["power_lag"] == 3

    # past power switched off (power_lag_hours 0 in the form): weather-only model, no power_lag channel
    off = mlservice.run_training({"architecture": "linear", "result_ids": ["d" * 12], "power_lag": 0,
                                  "features": ["wind_speed_hub"], "hyperparams": {"lookback": 6}}, lambda f, m="": None)
    moff = mlservice.describe_model(off)
    assert not moff["power_lag"] and "power_lag" not in moff["features"]
    child_off = mlservice.run_finetune({"model_id": off, "feature_result_id": "d" * 12, "csv_text": real_csv, "unit": "W"},
                                       lambda f, m="": None)
    assert not mlservice.describe_model(child_off)["power_lag"]


def test_multistep_windows_and_scaling():
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.arange(30, dtype=float)
    times = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    Xw, yw, tw = mlprep.make_windows(X, y, times, lookback=4, horizon=3)
    assert Xw.shape == (30 - 4 - 3 + 2, 4, 2) and yw.shape == (len(Xw), 3)
    k = 5                                              # window ending at hour s holds rows s-3..s; targets are s, s+1, s+2
    s_idx = k + 3
    assert np.allclose(Xw[k], X[s_idx - 3: s_idx + 1]) and np.allclose(yw[k], [s_idx, s_idx + 1, s_idx + 2])
    assert tw[k] == times[s_idx]
    X1, y1, _ = mlprep.make_windows(X, y, times, lookback=4, horizon=1)
    assert y1.ndim == 1 and len(y1) == 27

    Xtr = np.random.default_rng(0).normal(5, 3, (200, 4, 2)).astype(np.float32)
    std = mlprep.apply_scaler(Xtr, mlprep.fit_scaler(Xtr, "standard"))
    mm = mlprep.apply_scaler(Xtr, mlprep.fit_scaler(Xtr, "minmax"))
    raw = mlprep.apply_scaler(Xtr, mlprep.fit_scaler(Xtr, "none"))
    assert abs(float(std.mean())) < 0.05 and abs(float(std.std()) - 1) < 0.05
    assert float(mm.min()) >= -1e-6 and float(mm.max()) <= 1 + 1e-6
    assert np.allclose(raw, Xtr)
    try:
        mlprep.fit_scaler(Xtr, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_multistep_linear_train_and_finetune():
    df = _dataset(n=24 * 60, seed=11)
    _patch_results({"e" * 12: ({"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "m"}, df)})
    mid = mlservice.run_training(
        {"architecture": "linear", "result_ids": ["e" * 12], "power_lag": 1,
         "hyperparams": {"lookback": 6, "horizon": 4, "scaling": "minmax"}}, lambda f, m="": None)
    meta = mlservice.describe_model(mid)
    assert meta["hyperparams"]["horizon"] == 4 and meta["scaler"]["type"] == "minmax"
    per = meta["metrics"]["val"]["per_horizon_rmse_pct"]
    assert len(per) == 4 and all(v < 25 for v in per)
    json.dumps(mlservice.model_figure(mid, "horizon"), allow_nan=False)
    for name in ("predictions", "scatter", "residuals"):
        json.dumps(mlservice.model_figure(mid, name), allow_nan=False)
    real_csv = "timestamp;power\n" + "\n".join(f"{t:%Y-%m-%d %H:%M:%S};{p * 0.85:.1f}" for t, p in zip(df.index, df["power"]))
    child = mlservice.run_finetune({"model_id": mid, "feature_result_id": "e" * 12, "csv_text": real_csv, "unit": "W"},
                                   lambda f, m="": None)
    cm = mlservice.describe_model(child)
    assert cm["hyperparams"]["horizon"] == 4 and len(cm["metrics"]["val"]["per_horizon_rmse_pct"]) == 4
    # single-step models still have no horizon figure
    one = mlservice.run_training({"architecture": "linear", "result_ids": ["e" * 12], "hyperparams": {"lookback": 4}}, lambda f, m="": None)
    try:
        mlservice.model_figure(one, "horizon")
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError")


def test_early_stopper_rules():
    S = mlmodels.EarlyStopper
    # patience: stops after 3 epochs without improvement
    st = S("val_rmse", 0.0, 3)
    out = [st.update(v, 0, 0) for v in (5.0, 4.0, 4.1, 4.2, 4.3)]
    assert [o[0] for o in out] == [True, True, False, False, False] and [o[1] for o in out] == [False, False, False, False, True]
    # min_delta: gains smaller than the threshold do not reset patience
    st = S("val_rmse", 0.5, 2)
    assert st.update(5.0, 0, 0) == (True, False)
    assert st.update(4.8, 0, 0) == (False, False)           # only 0.2 better than 5.0
    assert st.update(4.7, 0, 0) == (False, True)
    # monitor choice: watching MAE ignores RMSE changes
    st = S("val_mae", 0.0, 2)
    assert st.update(9.0, 3.0, 0)[0] and not st.update(1.0, 3.0, 0)[0]
    # disabled: never stops, but still tracks improvements
    st = S("val_rmse", 0.0, 1, enabled=False)
    assert st.update(1.0, 0, 0) == (True, False) and st.update(2.0, 0, 0) == (False, False) and st.update(3.0, 0, 0) == (False, False)
    # hyper-parameter validation
    hp = mlmodels.clamp_hp({"monitor": "bogus", "min_delta": -1, "patience": 999})
    assert hp["monitor"] == "val_rmse" and hp["min_delta"] == 0.0 and hp["patience"] == 50


def test_split_by_date():
    t = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")       # 10 days, hourly window end times
    tr, va = mlprep.split_by_date(t, "2024-01-05", horizon=1)
    assert t[tr].max() == pd.Timestamp("2024-01-05 23:00", tz="UTC") and t[va].min() == pd.Timestamp("2024-01-06 00:00", tz="UTC")
    assert not (tr & va).any() and (tr | va).all()
    # multi-step: windows whose targets run past the cut belong to neither set
    tr, va = mlprep.split_by_date(t, "2024-01-05", horizon=6)
    assert t[tr].max() == pd.Timestamp("2024-01-05 18:00", tz="UTC")
    assert int((~tr & ~va).sum()) == 5

    # end to end: a date split trains and validates on the right periods (and says so in the model record)
    df = _dataset(n=24 * 100, seed=21)                                          # 2024-01-01 .. 2024-04-09
    _patch_results({"f" * 12: ({"kind": "wind", "summary": {"capacity_w": 2e6}, "scenario_name": "d"}, df)})
    mid = mlservice.run_training({"architecture": "linear", "result_ids": ["f" * 12], "train_until": "2024-03-01",
                                  "hyperparams": {"lookback": 6}}, lambda f, m="": None)
    sp = mlservice.describe_model(mid)["split"]
    assert sp["mode"] == "date" and sp["train"]["end"] == "2024-03-01" and sp["val"]["start"] == "2024-03-02"
    for bad in ("2023-06-01", "2024-12-31"):                                    # nothing before / nothing after the cut
        try:
            mlservice.run_training({"architecture": "linear", "result_ids": ["f" * 12], "train_until": bad,
                                    "hyperparams": {"lookback": 6}}, lambda f, m="": None)
        except ValueError as e:
            assert "train until" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_split_by_date_train_from():
    import pandas as pd
    from fastapi_app import mlprep
    t = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")
    tr, va = mlprep.split_by_date(t, "2024-01-07", 1, train_from="2024-01-03")
    assert t[tr].min() == pd.Timestamp("2024-01-03", tz="UTC") and t[tr].max() < pd.Timestamp("2024-01-08", tz="UTC")
    assert t[va].min() == pd.Timestamp("2024-01-08", tz="UTC")


def test_future_windows_shape_mask_and_flag():
    n, L, H = 200, 6, 4
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    wx = np.arange(n, dtype=float)
    X = np.stack([wx, wx + 0.5, wx - 7], axis=1)              # col 2 plays the role of the lagged power
    y = np.arange(n, dtype=float) / 100.0
    Xw, yw, tw = mlprep.make_windows(X, y, idx, L, H, future=True, power_col=2, power_lag=1)
    assert Xw.shape[1:] == (L + H - 1, 4) and yw.shape[1] == H
    first = 0
    assert tw[first] == idx[L - 1]                              # first predicted hour
    # weather is known through the last predicted hour
    assert Xw[first, -1, 0] == (L + H - 2)
    # the lagged-power column is known up to the first predicted hour and NaN after it
    assert not np.isnan(Xw[first, :L, 2]).any() and np.isnan(Xw[first, L:, 2]).all()
    # flag marks the steps after the first predicted hour
    assert Xw[first, :L, 3].sum() == 0 and Xw[first, L:, 3].min() == 1
    # targets are the hours s .. s+H-1
    assert np.allclose(yw[first], y[L - 1: L - 1 + H])
    # classic (no future) windows are unchanged
    X0, y0, t0 = mlprep.make_windows(X, y, idx, L, H)
    assert X0.shape[1] == L and X0.shape[2] == 3


def test_scaler_ignores_masked_future_power_and_zero_fills():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, (50, 8, 3)).astype(np.float32)
    X[:, 5:, 1] = np.nan                                        # unknown future power
    sc = mlprep.fit_scaler(X, "standard")
    assert np.isfinite(sc["mean"]).all() and abs(sc["mean"][1] - 5) < 0.5
    out = mlprep.apply_scaler(X, sc)
    assert not np.isnan(out).any() and (out[:, 5:, 1] == 0).all()


def test_future_weather_train_and_finetune_roundtrip():
    from fastapi_app import mlservice  # noqa: F401
    df = _dataset(n=24 * 60)
    cap = 2e6
    L, H = 12, 6
    Xw, yw, tw, feats = mlprep.prepare(df, "wind", cap, L, weather=["wind_speed_hub"], power_lag=1, horizon=H,
                                       future_weather=True)
    assert Xw.shape[1] == L + H - 1 and Xw.shape[2] == len(feats) + 4 + 1 + 1    # weather + time + power_lag + flag
    cut = int(len(yw) * 0.8)
    sc = mlprep.fit_scaler(Xw[:cut])
    hp = mlmodels.clamp_hp({"ridge_alpha": 1.0, "horizon": H, "lookback": L})
    model, hist = mlmodels.train("linear", mlprep.apply_scaler(Xw[:cut], sc), yw[:cut],
                                 mlprep.apply_scaler(Xw[cut:], sc), yw[cut:], hp)
    pred = model.predict(mlprep.apply_scaler(Xw[cut:], sc))
    assert pred.shape == yw[cut:].shape
    base = float(np.sqrt(((yw[cut:] - yw[:cut].mean()) ** 2).mean()))
    assert float(np.sqrt(((pred - yw[cut:]) ** 2).mean())) < base       # beats a constant guess


def test_gaps_in_weather_break_windows_instead_of_gluing_hours():
    df = _dataset(n=24 * 30)
    wx = df[["wind_speed_hub", "u_wind_100m"]].rename(columns={"wind_speed_hub": "wind_speed_100m"}).drop(df.index[100:110])
    Xw, yw, tw, _ = mlprep.prepare(df, "wind", 2e6, 8, weather=["wind_speed_100m"], weather_df=wx, time_cols=[], horizon=1)
    gap_start, gap_end = df.index[100], df.index[109]
    # no window may end inside the gap, nor within `lookback` hours after it
    assert not any(gap_start <= t <= gap_end + pd.Timedelta(hours=7) for t in tw)
