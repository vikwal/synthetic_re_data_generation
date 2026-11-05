"""Orchestration for the pre-trained model tab: pre-train on generated data, fine-tune on real
measurements, and build figures. Runs inside background jobs (see jobs.py)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import charts, mlmodels, mlprep, nwp, realcompare, storage
from .charts import ACCENT, SLATE, _axis, _clean, _layout, fig

ProgressFn = Callable[[float, str], None]
MAX_PREVIEW = 2000


def _lead_of(model_meta: Dict[str, Any]) -> Optional[tuple]:
    """Forecast lead-time window a model was trained with (package weather sources only)."""
    wl = model_meta.get("weather_lead")
    return (int(wl["from"]), int(wl["to"])) if wl else None


def _scaled_progress(progress: ProgressFn, lo: float, hi: float) -> ProgressFn:
    return lambda f, m="": progress(lo + (hi - lo) * max(0.0, min(1.0, f)), m)


def _first_step(a) -> np.ndarray:
    a = np.asarray(a)
    return a[:, 0] if a.ndim == 2 else a


def _preview(times, y_true, preds: Dict[str, np.ndarray]) -> Dict[str, Any]:
    y_true = _first_step(y_true)                      # multi-step models: plot the first forecast hour
    preds = {k: _first_step(v) for k, v in preds.items()}
    n = len(y_true)
    step = max(1, n // MAX_PREVIEW)
    sel = slice(0, n, step)
    out = {"t": [t.strftime("%Y-%m-%d %H:%M") for t in times[sel]],
           "true": [round(float(v), 5) for v in y_true[sel]]}
    for k, v in preds.items():
        out[k] = [round(float(x), 5) for x in np.asarray(v)[sel]]
    out["step"] = step
    return out


def _split(Xw, yw, tw, train_until: Optional[str], val_frac: float, horizon: int, train_from: Optional[str] = None):
    """Train / validation windows: by calendar date when `train_until` is given, else the last `val_frac` share."""
    if train_until:
        tr, va = mlprep.split_by_date(tw, train_until, horizon, train_from)
        return (Xw[tr], yw[tr], tw[tr]), (Xw[va], yw[va], tw[va])
    cut = mlprep.chrono_split(len(yw), val_frac)
    return (Xw[:cut], yw[:cut], tw[:cut]), (Xw[cut:], yw[cut:], tw[cut:])


def _span(times_list) -> Dict[str, Optional[str]]:
    ts = [t for t in times_list if len(t)]
    if not ts:
        return {"start": None, "end": None}
    return {"start": str(min(t.min() for t in ts))[:10], "end": str(max(t.max() for t in ts))[:10]}


def _require_both(n_tr: int, n_va: int, train_until: Optional[str]) -> None:
    if n_tr == 0:
        raise ValueError(f"No training windows on or before {train_until}. Choose a later 'train until' date." if train_until
                         else "No training windows were left after preparing the data.")
    if n_va == 0:
        raise ValueError(f"No validation windows after {train_until}. Choose an earlier 'train until' date." if train_until
                         else "No validation windows were left after preparing the data.")


# ------------------------------------------------------------------ pre-training
def run_training(spec: Dict[str, Any], progress: ProgressFn) -> str:
    """Pre-train on one or more generated datasets of the same kind. Returns the model id."""
    arch = spec["architecture"]
    if arch not in mlmodels.ARCH_INFO:
        raise ValueError(f"Unknown architecture '{arch}'")
    hp = mlmodels.clamp_hp(spec.get("hyperparams"))
    result_ids: List[str] = spec["result_ids"]
    if not result_ids:
        raise ValueError("Choose at least one generated dataset.")

    progress(0.02, "Preparing data")
    kind: Optional[str] = None
    sources = list(dict.fromkeys(spec.get("weather_sources") or [spec.get("weather_source") or "era5"]))
    multi = len(sources) > 1            # inputs from several weather models in one run (columns tagged `<model>:<input>`)
    if multi and "era5" in sources:
        raise ValueError("The run's own ERA5 weather cannot be combined with forecast models.")
    source: Optional[str] = sources[0]                                 # "era5" = the run's own weather
    future = hp["horizon"] > 1          # multi-step models also see the weather forecast for the hours they predict
    lead = (int(spec.get("lead_from", 12)), int(spec.get("lead_to", 36))) if any(nwp.has_lead(s) for s in sources) else None
    time_cols: List[str] = list(mlprep.TIME_FEATURES)
    points: List[dict] = []
    power_lag: Optional[int] = int(spec["power_lag"]) if spec.get("power_lag") else None
    weather: Optional[List[str]] = None
    tr_parts, va_parts, names = [], [], []
    for rid in result_ids:
        meta = storage.get_result_meta(rid)
        if kind is None:
            kind = meta["kind"]
        elif meta["kind"] != kind:
            raise ValueError("All selected datasets must be the same kind (all wind or all solar).")
        if weather is None:                         # first dataset fixes the feature set for all of them
            if multi:
                chosen, time_cols = mlprep.resolve_features_multi(spec.get("features"), kind, sources, allow_empty=bool(power_lag))
            else:
                chosen, time_cols = mlprep.resolve_features(spec.get("features"), kind, source, allow_empty=bool(power_lag))
            weather = chosen
        wx = None
        if multi:
            wx, notes = nwp.weather_multi(sources, meta, lead)
            points.extend(notes)
        elif source != "era5":
            wx, note = nwp.weather_for(source, meta, lead)
            points.append(note)
        Xw, yw, tw, feats = mlprep.prepare(storage.get_result_df(rid), kind, meta["summary"]["capacity_w"],
                                           hp["lookback"], weather=weather, time_cols=time_cols, weather_df=wx,
                                           power_lag=power_lag, horizon=hp["horizon"], future_weather=future)
        weather = feats
        tr, va = _split(Xw, yw, tw, spec.get("train_until"), hp["val_frac"], hp["horizon"], spec.get("train_from"))
        tr_parts.append(tr)
        va_parts.append(va)
        names.append(meta.get("scenario_name") or rid)

    n_tr, n_va = sum(len(p[1]) for p in tr_parts), sum(len(p[1]) for p in va_parts)
    _require_both(n_tr, n_va, spec.get("train_until"))
    tr_parts = [p for p in tr_parts if len(p[1])]
    va_parts = [p for p in va_parts if len(p[1])]
    Xtr = np.concatenate([p[0] for p in tr_parts]); ytr = np.concatenate([p[1] for p in tr_parts])
    Xva = np.concatenate([p[0] for p in va_parts]); yva = np.concatenate([p[1] for p in va_parts])
    tva = va_parts[0][2].append([p[2] for p in va_parts[1:]]) if len(va_parts) > 1 else va_parts[0][2]
    split = {"mode": "date" if spec.get("train_until") else "share", "train_from": spec.get("train_from"), "train_until": spec.get("train_until"),
             "train": _span([p[2] for p in tr_parts]), "val": _span([p[2] for p in va_parts])}
    scaler = mlprep.fit_scaler(Xtr, hp["scaling"])
    Xtr_s, Xva_s = mlprep.apply_scaler(Xtr, scaler), mlprep.apply_scaler(Xva, scaler)

    refit = bool(spec.get("refit_full")) and arch != "linear"
    model, hist = mlmodels.train(arch, Xtr_s, ytr, Xva_s, yva, hp,
                                 progress=_scaled_progress(progress, 0.08, 0.6 if refit else 0.95))
    pred_va = model.predict(Xva_s)                           # metrics and plots always describe this held-out run
    pred_tr = model.predict(Xtr_s[:20000])

    refit_info = None
    if refit:
        # Stage 2: a fresh model on train + validation for exactly the best epoch found above (no early stopping).
        best_epoch = int(hist.get("best_epoch") or 1)
        Xall, yall = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
        scaler = mlprep.fit_scaler(Xall, hp["scaling"])
        Xall_s = mlprep.apply_scaler(Xall, scaler)
        hp2 = {**hp, "epochs": best_epoch, "early_stopping": False}
        progress(0.62, f"Retraining on all data for {best_epoch} epochs")
        model, _ = mlmodels.train(arch, Xall_s, yall, mlprep.apply_scaler(Xva, scaler), yva, hp2,
                                  progress=_scaled_progress(progress, 0.62, 0.95), restore_best=False)
        refit_info = {"best_epoch": best_epoch, "epochs": best_epoch, "n_samples": int(len(yall))}

    mid = storage.new_model_id()
    model.save(storage.model_weights_path(mid, mlmodels.weights_ext(arch)))
    meta = {
        "id": mid, "name": (spec.get("name") or f"{mlmodels.ARCH_INFO[arch][0]} on {', '.join(names)}")[:80],
        "stage": "pretrained", "parent_id": None, "architecture": arch, "kind": kind,
        "created": storage.now_iso(), "source_result_ids": result_ids, "source_names": names,
        "weather_source": source, "weather_sources": sources, "weather_points": points, "weather_features": weather,
        "time_features": time_cols, "power_lag": power_lag,
        "weather_lead": {"from": lead[0], "to": lead[1]} if lead else None,
        "features": weather + time_cols + (["power_lag"] if power_lag else []) + (["is_future"] if future else []),
        "future_weather": future,
        "lookback": hp["lookback"], "hyperparams": hp, "scaler": scaler, "history": hist,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "split": split, "refit": refit_info,
        "metrics": {"train": mlprep.regression_metrics(ytr[:20000], pred_tr),
                    "val": {**mlprep.regression_metrics(yva, pred_va), "per_horizon_rmse_pct": mlprep.per_horizon_rmse(yva, pred_va)}},
        "val_preview": _preview(tva, yva, {"pred": pred_va}),
    }
    storage.save_model_meta(meta)
    progress(1.0, "Model saved")
    return mid


# ------------------------------------------------------------------ fine-tuning on real data
def run_finetune(spec: Dict[str, Any], progress: ProgressFn) -> str:
    """Continue training a pre-trained model on real measurements paired with site weather."""
    parent = storage.get_model_meta(spec["model_id"])
    arch, kind = parent["architecture"], parent["kind"]
    src_meta = storage.get_result_meta(spec["feature_result_id"])
    if src_meta["kind"] != kind:
        raise ValueError(f"The model was pre-trained on {kind} data; the weather source is {src_meta['kind']}.")
    capacity = spec.get("capacity_w") or src_meta["summary"]["capacity_w"]

    progress(0.02, "Reading your plant data")
    real = realcompare.parse_real_csv(spec["csv_text"], spec.get("unit", "W"))
    target = real["power"].astype(float).resample("h").mean() / float(capacity)

    base = parent["hyperparams"]
    hp = mlmodels.clamp_hp({**base, "epochs": spec.get("epochs", 15), "lr": spec.get("lr", 3e-4),
                            "ridge_alpha": spec.get("ridge_alpha", 10.0), "val_frac": spec.get("val_frac", 0.2),
                            "batch_size": spec.get("batch_size", base.get("batch_size", 128))})
    for k in ("hidden", "layers", "dropout", "lookback", "horizon"):      # architecture must match the weights
        hp[k] = base.get(k, hp[k])           # older models have no `horizon`

    source = parent.get("weather_source", "era5")
    sources = parent.get("weather_sources") or [source]       # the same weather model(s) as the pre-training
    wx, points = None, []
    if len(sources) > 1:
        wx, points = nwp.weather_multi(sources, src_meta, _lead_of(parent))
    elif source != "era5":
        wx, note = nwp.weather_for(source, src_meta, _lead_of(parent))
        points.append(note)
    Xw, yw, tw, _ = mlprep.prepare(storage.get_result_df(spec["feature_result_id"]), kind, capacity,
                                   parent["lookback"], weather=parent["weather_features"], target=target,
                                   time_cols=parent.get("time_features", mlprep.TIME_FEATURES), weather_df=wx,
                                   power_lag=parent.get("power_lag"), horizon=hp["horizon"],
                                   future_weather=bool(parent.get("future_weather")))
    (Xtr, ytr, ttr), (Xva, yva, tva) = _split(Xw, yw, tw, spec.get("train_until"), hp["val_frac"], hp["horizon"], spec.get("train_from"))
    _require_both(len(ytr), len(yva), spec.get("train_until"))
    split = {"mode": "date" if spec.get("train_until") else "share", "train_from": spec.get("train_from"), "train_until": spec.get("train_until"),
             "train": _span([ttr]), "val": _span([tva])}
    scaler = parent["scaler"]                                   # keep the pre-training input scaling
    Xtr_s, Xva_s = mlprep.apply_scaler(Xtr, scaler), mlprep.apply_scaler(Xva, scaler)

    progress(0.08, "Evaluating the pre-trained model on your data")
    pre = mlmodels.load_model(arch, storage.model_weights_path(parent["id"]))
    zero_shot = pre.predict(Xva_s)

    model, hist = mlmodels.train(arch, Xtr_s, ytr, Xva_s, yva, hp, progress=_scaled_progress(progress, 0.12, 0.95),
                                 init=pre, freeze=bool(spec.get("freeze_backbone")))
    tuned = model.predict(Xva_s)

    mid = storage.new_model_id()
    model.save(storage.model_weights_path(mid, mlmodels.weights_ext(arch)))
    m_zero, m_tuned = mlprep.regression_metrics(yva, zero_shot), mlprep.regression_metrics(yva, tuned)
    meta = {
        **{k: parent[k] for k in ("architecture", "kind", "weather_features", "features", "lookback", "scaler")},
        "time_features": parent.get("time_features", mlprep.TIME_FEATURES),
        "weather_source": source, "weather_sources": sources, "weather_points": points, "power_lag": parent.get("power_lag"),
        "weather_lead": parent.get("weather_lead"),
        "id": mid, "name": (spec.get("name") or f"{parent['name']} → real")[:80],
        "stage": "finetuned", "parent_id": parent["id"], "created": storage.now_iso(),
        "source_result_ids": [spec["feature_result_id"]], "source_names": [src_meta.get("scenario_name")],
        "real_file": spec.get("filename"), "capacity_w": float(capacity), "freeze_backbone": bool(spec.get("freeze_backbone")),
        "hyperparams": hp, "history": hist, "n_train": int(len(ytr)), "n_val": int(len(yva)), "split": split,
        "metrics": {"zero_shot": m_zero, "val": {**m_tuned, "per_horizon_rmse_pct": mlprep.per_horizon_rmse(yva, tuned)},
                    "improvement_rmse_pct": round(m_zero["rmse_pct"] - m_tuned["rmse_pct"], 3)},
        "val_preview": _preview(tva, yva, {"pred": tuned, "pred_zero_shot": zero_shot}),
    }
    storage.save_model_meta(meta)
    progress(1.0, "Model saved")
    return mid


# ------------------------------------------------------------------ describe + figures
def describe_model(mid: str) -> Dict[str, Any]:
    meta = storage.get_model_meta(mid)
    return {k: v for k, v in meta.items() if k != "val_preview"}


TOP_LEGEND = {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"}


def _top(bottom: int = 52) -> dict:
    """Legend above the plot (never colliding with the axis title or range slider)."""
    return {"legend": TOP_LEGEND, "margin": {"l": 60, "r": 24, "t": 64, "b": bottom}}


def _pct(values) -> List[float]:
    """Capacity fractions (0-1) -> percent (0-100) for display."""
    return [round(float(v) * 100.0, 3) for v in values]


def _col(arch_color: str) -> dict:
    return {"color": arch_color}


def model_figure(mid: str, name: str) -> Dict[str, Any]:
    meta = storage.get_model_meta(mid)
    pv = meta.get("val_preview") or {}
    if name == "loss":
        h = meta["history"]
        tr = [100 * v ** 0.5 for v in h["train_loss"]]
        va = [100 * v ** 0.5 for v in h["val_loss"]]
        if len(tr) <= 1:
            raise LookupError("Closed-form models have no learning curve.")
        x = list(range(1, len(tr) + 1))
        return fig([{"type": "scatter", "mode": "lines", "name": "Training", "x": x, "y": _clean(tr), "line": _col(SLATE)},
                    {"type": "scatter", "mode": "lines", "name": "Validation", "x": x, "y": _clean(va), "line": _col(ACCENT)}],
                   _layout("Learning curve", xaxis=_axis("Epoch"), yaxis=_axis("RMSE (%)"), **_top()))
    if name == "predictions":
        if not pv:
            raise LookupError("No validation preview stored for this model.")
        data = [{"type": "scatter", "mode": "lines", "name": "Actual", "x": pv["t"], "y": _pct(pv["true"]), "line": {"color": SLATE, "width": 1.2}}]
        if pv.get("pred_zero_shot"):
            data.append({"type": "scatter", "mode": "lines", "name": "Pre-trained (no fine-tuning)", "x": pv["t"],
                         "y": _pct(pv["pred_zero_shot"]), "line": {"color": "#f59e0b", "width": 1.1}})
        data.append({"type": "scatter", "mode": "lines", "name": "Model", "x": pv["t"], "y": _pct(pv["pred"]),
                     "line": {"color": ACCENT, "width": 1.3}})
        return fig(data, _layout("Validation period: model vs actual", yaxis=_axis("Power (%)"),
                                 xaxis=_axis("Date", rangeslider={"visible": True}), **_top(bottom=40)))
    if name == "scatter":
        if not pv:
            raise LookupError("No validation preview stored for this model.")
        actual, predicted = _pct(pv["true"]), _pct(pv["pred"])
        top = max(max(actual), max(predicted), 1e-6)
        return fig([{"type": "scattergl", "mode": "markers", "x": actual, "y": predicted,
                     "marker": {"size": 4, "color": ACCENT, "opacity": 0.35}, "name": "Hours"},
                    {"type": "scatter", "mode": "lines", "x": [0, top], "y": [0, top], "name": "1:1",
                     "line": {"color": SLATE, "dash": "dash", "width": 1}}],
                   _layout("Predicted vs actual (validation)", hovermode="closest", showlegend=False,
                           xaxis=_axis("Actual (%)"), yaxis=_axis("Predicted (%)")))
    if name == "residuals":
        if not pv:
            raise LookupError("No validation preview stored for this model.")
        err = (np.asarray(pv["pred"]) - np.asarray(pv["true"])) * 100.0
        data = []
        if pv.get("pred_zero_shot"):
            e0 = (np.asarray(pv["pred_zero_shot"]) - np.asarray(pv["true"])) * 100.0
            data.append({"type": "histogram", "name": "Pre-trained only", "x": _clean(e0), "nbinsx": 40,
                         "marker": {"color": "#f59e0b"}, "opacity": 0.6})
        data.append({"type": "histogram", "name": "Model", "x": _clean(err), "nbinsx": 40,
                     "marker": {"color": ACCENT}, "opacity": 0.75})
        return fig(data, _layout("Error distribution (validation)", barmode="overlay", hovermode="closest",
                                 xaxis=_axis("Prediction error (%)", zeroline=True),
                                 yaxis=_axis("Hours"), **_top()))
    if name == "horizon":
        per = (meta["metrics"].get("val") or {}).get("per_horizon_rmse_pct") or []
        if len(per) < 2:
            raise LookupError("Only available for multi-step models.")
        n = len(per)
        return fig([{"type": "scatter", "mode": "lines+markers", "name": "Validation RMSE", "x": list(range(1, n + 1)),
                     "y": per, "line": {"color": ACCENT, "width": 2.5}, "marker": {"size": 6},
                     "hovertemplate": "Hour %{x}: %{y:.2f}%<extra></extra>"}],
                   _layout("Error by forecast hour", hovermode="closest", showlegend=False,
                           xaxis=_axis("Forecast hour", dtick=max(1, n // 12)),
                           yaxis=_axis("RMSE (%)", rangemode="tozero")))
    if name == "comparison":
        m = meta["metrics"]
        if "zero_shot" not in m:
            raise LookupError("Only available for fine-tuned models.")
        return fig([{"type": "bar", "name": "Pre-trained only", "x": ["MAE", "RMSE"], "y": [m["zero_shot"]["mae_pct"], m["zero_shot"]["rmse_pct"]], "marker": {"color": "#f59e0b"}},
                    {"type": "bar", "name": "Fine-tuned", "x": ["MAE", "RMSE"], "y": [m["val"]["mae_pct"], m["val"]["rmse_pct"]], "marker": {"color": ACCENT}}],
                   _layout("Effect of fine-tuning on your data", barmode="group", xaxis=_axis("Error metric"), yaxis=_axis("Error (%)"),
                           hovermode="closest", **_top()))
    raise LookupError(f"Unknown figure '{name}'")
