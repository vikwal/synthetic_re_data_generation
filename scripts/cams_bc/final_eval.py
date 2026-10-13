#!/usr/bin/env python
"""Final evaluation on the held-out test stations. The three cross-validation models (trained on two
station folds each, early-stopped on the third) are averaged - no further decision is taken on the
test stations. Also: a linear baseline (ridge on the CAMS + geometry groups), per-station and hourly
metrics, figures. Models are saved for the application to park points.

    python scripts/cams_bc/final_eval.py --model xgb --params-json results/cams_bc/best_xgb.json --groups ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import data, features as F, metrics, splits, targets  # noqa: E402
from cams_bc.config import RESULTS, ROOT  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
COL = {"cams": "#eb6834", "xgb": "#2a78d6", "mlp": "#1baf7a", "lin": "#52514e"}


def linear_baseline(tr, te, cols_lin, form):
    out = {}
    for v in C.VARS:
        m_tr = data.valid_mask(tr, v, cols_lin)
        X = tr.loc[m_tr, cols_lin].to_numpy(np.float64)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, med)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        y, w = targets.encode(form, tr[m_tr], v)
        mdl = Ridge(alpha=1.0).fit((X - mu) / sd, y, sample_weight=w)
        Xt = te[cols_lin].to_numpy(np.float64)
        Xt = np.where(np.isfinite(Xt), Xt, med)
        out[v] = targets.decode(form, mdl.predict((Xt - mu) / sd), te, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "mlp"], required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--groups", nargs="+", default=F.FULL)
    ap.add_argument("--form", default="kc")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    params = json.load(open(args.params_json))
    form = params.pop("form", args.form)
    cols = F.columns(args.groups)
    sp = splits.load_splits()
    df, _ = C.load_train(args.groups)
    te = data.load_stations(splits.test_ids(sp), list(dict.fromkeys(["station_id"] + F.TARGET_COLS + ["cams_ghi", "cams_dhi", "cams_ghi_clear"] + cols)))
    device = f"cuda:{args.gpu}"
    tag = args.tag or args.model
    mdir = os.path.join(ROOT, "synthetic", "cams_bc", "models", tag)
    os.makedirs(mdir, exist_ok=True)
    preds = {v: [] for v in C.VARS}
    oof = {v: pd.Series(np.nan, index=range(len(df))) for v in C.VARS}
    fold_rows = []
    for k in range(3):
        tr, va = C.fold_frames(df, sp, k)
        pred = C.fit_predict(args.model, params, form, cols, tr, va, device, seed=k)
        s = C.score(va, pred)
        fold_rows.append({"fold": k, **s})
        va_pos = np.flatnonzero(df.station_id.isin(va.station_id.unique()).to_numpy())
        for v in C.VARS:
            oof[v].iloc[va_pos] = pred[v]
        # test predictions from this fold model
        Xte = te[cols].to_numpy(np.float32)
        if args.model == "xgb":
            for v in C.VARS:
                mdl = pred["info"][f"model_{v}"]
                preds[v].append(targets.decode(form, mdl.predict(Xte), te, v))
                mdl.booster.save_model(os.path.join(mdir, f"fold{k}_{v}.json"))
        else:
            import torch
            mdl = pred["info"]["model"]
            P = mdl.predict(Xte)
            for i, v in enumerate(C.VARS):
                preds[v].append(targets.decode(form, P[:, i], te, v))
            torch.save({"state": mdl.net.state_dict(), "mu": mdl.mu, "sd": mdl.sd, "med": mdl.med, "nan_cols": mdl.nan_cols, "params": mdl.p},
                       os.path.join(mdir, f"fold{k}.pt"))
        print(f"fold {k}: objective {s['objective']:.4f} rel ghi {s['rel_rmse_ghi']:.4f} dhi {s['rel_rmse_dhi']:.4f}", flush=True)
    json.dump({"model": args.model, "params": params, "form": form, "groups": list(args.groups), "columns": cols},
              open(os.path.join(mdir, "meta.json"), "w"), indent=1)
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out, f"final_{tag}_folds.csv"), index=False)
    ens = {v: np.mean(preds[v], axis=0) for v in C.VARS}
    cols_lin = F.columns(["cams", "geom"])
    lin = linear_baseline(df, te, cols_lin, form)
    # physical constraint dhi <= ghi, applied exactly as in apply_parks so the test metric is for the shipped product
    ens["dhi"] = np.minimum(ens["dhi"], ens["ghi"])
    lin["dhi"] = np.minimum(lin["dhi"], lin["ghi"])
    rows, per, hourly_rows = [], [], []
    for v in C.VARS:
        m_all = data.valid_mask(te, v, cols)
        # primary test metric on OBSERVED rows only; rows filled by the imputation are reported separately
        m = m_all & (te[f"src_{v}"].to_numpy() == 0) if f"src_{v}" in te else m_all
        e = metrics.evaluate(te, {tag: ens[v], "lin": lin[v]}, v, m)
        for name, b in e.items():
            rows.append({"var": v, "resolution": "30min", "model": name, **b})
        if m.sum() < m_all.sum():
            e2 = metrics.evaluate(te, {tag: ens[v]}, v, m_all & ~m)
            for name, b in e2.items():
                rows.append({"var": v, "resolution": "30min_imputed_rows", "model": name, **b})
        h = metrics.hourly(te, ens[v], v, m)
        hl = metrics.hourly(te, lin[v], v, m)
        bc = metrics.basic(h[f"{v}_obs"], h[f"cams_{v}"])
        bm = metrics.basic(h[f"{v}_obs"], h["pred"])
        bl = metrics.basic(hl[f"{v}_obs"], hl["pred"])
        rows.append({"var": v, "resolution": "1h", "model": "cams", **bc})
        rows.append({"var": v, "resolution": "1h", "model": tag, **bm, "skill": metrics.skill(bm["rmse"], bc["rmse"])})
        rows.append({"var": v, "resolution": "1h", "model": "lin", **bl, "skill": metrics.skill(bl["rmse"], bc["rmse"])})
        ps = metrics.per_station(te, ens[v], v, m)
        ps["var"] = v
        per.append(ps)
        # out-of-fold on the training stations (for the record)
        mo = data.valid_mask(df, v, cols) & oof[v].notna().to_numpy()
        eo = metrics.evaluate(df, {tag: oof[v].to_numpy()}, v, mo)
        rows.append({"var": v, "resolution": "30min_oof_train", "model": tag, **eo[tag]})
        rows.append({"var": v, "resolution": "30min_oof_train", "model": "cams", **eo["cams"]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, f"final_{tag}_test.csv"), index=False)
    per = pd.concat(per)
    per.to_csv(os.path.join(args.out, f"final_{tag}_per_station.csv"), index=False)
    print(res.round(4).to_string())
    # error structure on the test stations (GHI): by cos_zen, by kc_cams, by month, diurnal bias
    m = data.valid_mask(te, "ghi", cols)
    d = te.loc[m, ["ghi_obs", "cams_ghi", "cos_zen", "kc_cams"]].copy()
    d["pred"] = ens["ghi"][m]
    d["lin"] = lin["ghi"][m]
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (key, bins, lab) in zip(axes.ravel(), [("kc_cams", np.linspace(0, 1, 11), "CAMS clear-sky index"), ("cos_zen", np.linspace(0, 0.9, 10), "cos(zenith)"),
                                                   ("month", np.arange(0.5, 13, 1), "month"), ("hour", np.arange(2.5, 21, 1), "hour UTC")]):
        g = d.groupby(pd.cut(d[key], bins), observed=True)
        x = [i.mid for i in g.size().index]
        for name, c in [("cams_ghi", COL["cams"]), ("lin", COL["lin"]), ("pred", COL[args.model])]:
            ax.plot(x, g.apply(lambda s, n=name: np.sqrt(np.mean((s[n] - s["ghi_obs"]) ** 2))), color=c, marker="o", ms=3, label={"cams_ghi": "CAMS raw", "lin": "ridge", "pred": tag}[name])
        ax.set_xlabel(lab)
        ax.set_ylabel("RMSE GHI W/m²")
    axes[0, 0].legend()
    fig.suptitle("Test stations (unseen): RMSE of raw CAMS, ridge and the corrected product", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, f"fig_final_{tag}_error_structure.png"))
    fig, ax = plt.subplots(figsize=(8, 3))
    pg = per[per["var"] == "ghi"].sort_values("skill")
    ax.bar(np.arange(len(pg)), pg["skill"] * 100, color=COL[args.model])
    ax.set_xticks(np.arange(len(pg)), pg["station_id"], rotation=90, fontsize=7)
    ax.set_ylabel("skill vs CAMS, RMSE %")
    ax.set_title("GHI skill per unseen test station", loc="left")
    fig.savefig(os.path.join(args.out, f"fig_final_{tag}_per_station.png"))


if __name__ == "__main__":
    main()
