"""Shared pieces of the cams_bc scripts: repo path, data loading, one cross-validation run."""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from cams_bc import data, features as F, metrics, splits, targets  # noqa: E402
from cams_bc.models import MLPModel, XGBModel  # noqa: E402

VARS = ("ghi", "dhi")


def load_train(groups=F.FULL) -> tuple:
    sp = splits.load_splits()
    ids = splits.train_ids(sp)
    cols = ["station_id"] + F.TARGET_COLS + ["cams_ghi", "cams_dhi", "cams_ghi_clear"] + F.columns(groups)
    cols = list(dict.fromkeys(cols))
    df = data.load_stations(ids, cols)
    return df, sp


def fold_frames(df: pd.DataFrame, sp: pd.DataFrame, k: int) -> tuple:
    tr, va = splits.fold_ids(sp, k)
    return df[df.station_id.isin(tr)], df[df.station_id.isin(va)]


def fit_predict(model: str, params: dict, form: str, cols: list, tr: pd.DataFrame, va: pd.DataFrame, device: str, seed: int = 0) -> dict:
    """Fit on tr, predict on va for both targets. Returns {var: prediction in W/m2 (full va length)}
    plus fit info. XGB: one booster per target; MLP: one network with two outputs."""
    out, info = {}, {}
    m_tr = {v: data.valid_mask(tr, v, cols) for v in VARS}
    m_va = {v: data.valid_mask(va, v, cols) for v in VARS}
    Xtr, Xva = tr[cols].to_numpy(np.float32), va[cols].to_numpy(np.float32)
    if model == "xgb":
        for v in VARS:
            y, w = targets.encode(form, tr[m_tr[v]], v)
            yv, wv = targets.encode(form, va[m_va[v]], v)
            t0 = time.time()
            mdl = XGBModel(params, device=device, seed=seed).fit(Xtr[m_tr[v]], y, w, Xva[m_va[v]], yv, wv)
            out[v] = targets.decode(form, mdl.predict(Xva), va, v)
            info[f"best_iter_{v}"] = mdl.best_iteration
            info[f"fit_s_{v}"] = round(time.time() - t0, 1)
            info[f"model_{v}"] = mdl
    else:
        both = m_tr["ghi"] & m_tr["dhi"]
        both_v = m_va["ghi"] & m_va["dhi"]
        Y = np.column_stack([targets.encode(form, tr[both], v)[0] for v in VARS])
        W = np.column_stack([targets.encode(form, tr[both], v)[1] for v in VARS])
        Yv = np.column_stack([targets.encode(form, va[both_v], v)[0] for v in VARS])
        Wv = np.column_stack([targets.encode(form, va[both_v], v)[1] for v in VARS])
        t0 = time.time()
        mdl = MLPModel(params, device=device, seed=seed).fit(Xtr[both], Y, W, Xva[both_v], Yv, Wv)
        P = mdl.predict(Xva)
        for i, v in enumerate(VARS):
            out[v] = targets.decode(form, P[:, i], va, v)
        info.update({"epochs": mdl.epochs, "fit_s": round(time.time() - t0, 1), "model": mdl})
    out["masks"] = m_va
    out["info"] = info
    return out


def score(va: pd.DataFrame, pred: dict) -> dict:
    """Pooled metrics on the validation stations; rel_rmse = RMSE_model / RMSE_cams on identical rows."""
    res = {}
    for v in VARS:
        e = metrics.evaluate(va, {"model": pred[v]}, v, pred["masks"][v])
        res[f"rmse_{v}"] = e["model"]["rmse"]
        res[f"rmse_cams_{v}"] = e["cams"]["rmse"]
        res[f"rel_rmse_{v}"] = e["model"]["rmse"] / e["cams"]["rmse"]
        res[f"r2_{v}"] = e["model"]["r2"]
        res[f"r2_cams_{v}"] = e["cams"]["r2"]
        res[f"bias_{v}"] = e["model"]["bias"]
        res[f"n_{v}"] = e["model"]["n"]
    res["objective"] = 0.5 * (res["rel_rmse_ghi"] + res["rel_rmse_dhi"])
    return res


def cv_run(model: str, params: dict, form: str, groups: list, df: pd.DataFrame, sp: pd.DataFrame, device: str,
           folds=(0, 1, 2), report=None, seed: int = 0) -> tuple:
    """3-fold station CV. Returns (mean objective, list of per-fold score dicts)."""
    cols = F.columns(groups)
    rows = []
    for k in folds:
        tr, va = fold_frames(df, sp, k)
        pred = fit_predict(model, params, form, cols, tr, va, device, seed)
        s = score(va, pred)
        s.update({"fold": k, **{a: b for a, b in pred["info"].items() if not a.startswith("model")}})
        rows.append(s)
        if report is not None:
            report(k, s["objective"])
    return float(np.mean([r["objective"] for r in rows])), rows
