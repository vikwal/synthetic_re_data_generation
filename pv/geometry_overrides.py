"""Manual geometry corrections (Viktor 2026-11-07): single-park corrections of the measured groups and arrays that no
fleet rule should carry (every one of them is below 1 % annual energy). They are written INTO the run (per-park groups
and arrays, state tables), once, with the original files kept as *.pre_override; reports, index and the release arrays
then show the corrected geometry without a mark (Viktor 2026-11-07). The record of what was changed and why is this
table and the review log, not the outputs.

data/parks_pv_v1/geometry_overrides.csv: park_id, run, action, params (JSON), reason, date
    merge_groups  {"groups": [0, 1]}                      groups become one (per flank: share-weighted tilt, circular azimuth)
    add_group     {"tilt", "azimuth", "share", "windows"}  one more group from accepted windows, the others scaled by 1 - share
    set_shares    {"0": 0.4, "1": 0.6}                    capacity share per group (flank rows of a group keep their ratio)

    synthre/bin/python -m pv.geometry_overrides <run suffix> [restore]   # apply all rows of that run (idempotent) / undo"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(REPO, "data", "parks_pv_v1", "geometry_overrides.csv")


def _circ(az, w=None, period=360.0) -> float:
    a = np.radians(np.asarray(az, float) * 360.0 / period); w = np.ones(len(a)) if w is None else np.asarray(w, float)
    return float(np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) % 360 * period / 360.0)


def _renumber(a: pd.DataFrame) -> pd.DataFrame:
    """Groups numbered by descending capacity share (as group_windows does), rows of a group by azimuth."""
    order = a.groupby("g").share.sum().sort_values(ascending=False, kind="stable").index
    gid = {g: k for k, g in enumerate(order)}
    a = a.assign(_o=a.g.map(gid)).sort_values(["_o", "azimuth"], kind="stable").drop(columns="_o").reset_index(drop=True)
    a["g"] = a.g.map(gid)
    a["array_id"] = [f"g{g}_{i}" for i, g in enumerate(a.g)]
    return a


def apply_arrays(arr: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    """Arrays table of one run (park_id, array_id, tilt, azimuth, share, source, az_source, note, ...) with the override
    rows `ov` applied."""
    for o in ov.itertuples():
        m = arr.park_id == o.park_id
        if not m.any():
            continue
        p = json.loads(o.params)
        a = arr[m].copy()
        a["g"] = a.array_id.str.extract(r"g(\d+)_")[0].astype(int)
        if o.action == "merge_groups":
            gs = [int(g) for g in p["groups"]]
            sub = a[a.g.isin(gs)].copy(); rest = a[~a.g.isin(gs)]
            sub["flank"] = sub.groupby("g").azimuth.rank(method="first").astype(int)     # 1 = east / single, 2 = west
            rows = [dict(s.iloc[0].to_dict(), tilt=float(np.average(s.tilt, weights=s.share)), azimuth=_circ(s.azimuth, s.share),
                         share=float(s.share.sum()), g=min(gs)) for _, s in sub.groupby("flank")]
            a = pd.concat([rest, pd.DataFrame(rows).drop(columns="flank")], ignore_index=True)
        elif o.action == "add_group":
            s = float(p["share"])
            a["share"] = a.share * (1.0 - s)
            new = dict(a.iloc[0].to_dict(), tilt=float(p["tilt"]), azimuth=float(p["azimuth"]), share=s, g=int(a.g.max()) + 1, az_source="measured", note="")
            a = pd.concat([a, pd.DataFrame([new])], ignore_index=True)
        elif o.action == "set_shares":
            tot = a.groupby("g").share.transform("sum")
            a["share"] = a.share / tot * a.g.map({int(k): float(v) for k, v in p.items()})
        else:
            raise ValueError(f"unknown override action {o.action}")
        assert abs(a.share.sum() - 1.0) < 1e-6, (o.park_id, a.share.sum())
        arr = pd.concat([arr[~m], _renumber(a).drop(columns="g")[arr.columns]], ignore_index=True)
    return arr


def apply_groups(G: pd.DataFrame, win: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    """Per-park groups table (groups_<src>.csv) with the override rows applied; `win` for a new group's values."""
    for o in ov.itertuples():
        p = json.loads(o.params)
        if o.action == "merge_groups":
            gs = [int(g) for g in p["groups"]]
            s = G[G.group.isin(gs)]; w = s.share
            row = dict(group=min(gs), n_windows=int(s.n_windows.sum()), area_tab_ha=float(s.area_tab_ha.sum()), axis_true=_circ(s.axis_true, w, 180.0),
                       pitch=float(np.average(s.pitch, weights=w)), tilt_horiz=float(np.average(s.tilt_horiz, weights=w)), tilt_p10=float(s.tilt_p10.min()),
                       tilt_p90=float(s.tilt_p90.max()), az_undetermined=bool(s.az_undetermined.astype(bool).all()),
                       n_sure=int(s.n_sure.sum()), n_opposite=int(s.n_opposite.sum()), ew=bool(s.ew.astype(bool).all()), ridge_sym=float(np.average(s.ridge_sym, weights=w)),
                       slope_deg=float(np.average(s.slope_deg, weights=w)), share_raw=float(s.share_raw.sum()), share=float(s.share.sum()),
                       windows=" ".join(str(x) for x in sorted(int(float(x)) for g_ in s.windows for x in str(g_).split())))
            # east-west groups carry one flank as azimuth (the run is not consistent about which): west flank = axis + 90
            row["azimuth"] = (row["axis_true"] + 90.0) % 360 if row["ew"] else _circ(s.azimuth, w)
            G = pd.concat([G[~G.group.isin(gs)], pd.DataFrame([row])], ignore_index=True)
        elif o.action == "add_group":
            ids = [int(x) for x in str(p["windows"]).split()]
            s = win[win.w.isin(ids)]
            share = float(p["share"]); area = float(s.area_tab.sum()) / 1e4
            G["share"] = G.share * (1.0 - share)
            row = dict(group=int(G.group.max()) + 1, n_windows=len(s), area_tab_ha=area, axis_true=_circ(s.axis_true, None, 180.0), pitch=float(s.pitch.median()),
                       tilt_horiz=float(p["tilt"]), tilt_p10=float(s.tilt_horiz.quantile(0.1)), tilt_p90=float(s.tilt_horiz.quantile(0.9)), azimuth=float(p["azimuth"]),
                       az_undetermined=False, n_sure=int(s.az_sure.astype(bool).sum()), n_opposite=0, ew=bool(s.ew_pair.astype(bool).all()) if "ew_pair" in s else False,
                       ridge_sym=float(s.ridge_sym.median()), slope_deg=float(s.slope_facing_deg.median()), share_raw=area / (G.area_tab_ha.sum() + area), share=share,
                       windows=" ".join(str(x) for x in sorted(ids)))
            G = pd.concat([G, pd.DataFrame([row])], ignore_index=True)
        elif o.action == "set_shares":
            G["share"] = G.group.map({int(k): float(v) for k, v in p.items()}).fillna(G.share)
    G = G.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    G["group"] = range(len(G))
    return G


def restore(run: str, path: str = DEFAULT, data_dir: str = None, results_dir: str = None) -> None:
    """Put the *.pre_override files of `run` back (park files of the override parks and the state tables)."""
    data_dir = data_dir or os.path.join(REPO, "data", "parks_pv_v1")
    results_dir = results_dir or os.path.join(REPO, "results", "pv", f"geodata_release{run}")
    files = glob.glob(os.path.join(data_dir, f"geodata_release_*{run}*.csv.pre_override"))
    for pid in pd.read_csv(path, dtype=str).query("run == @run").park_id.unique():
        files += glob.glob(os.path.join(results_dir, pid, "*.pre_override"))
    for f in files:
        os.replace(f, f[:-len(".pre_override")]); print("restored", f[:-len(".pre_override")])


def _arrays_str(a: pd.DataFrame) -> str:
    return str([(round(float(t), 1), int(round(float(z))), round(float(s), 2), str(src)) for t, z, s, src in a[["tilt", "azimuth", "share", "az_source"]].values])


def apply_to_run(run: str, path: str = DEFAULT, data_dir: str = None, results_dir: str = None) -> list:
    """Write the overrides of `run` into the run files of each park (best source): results/pv/geodata_release<run>/<pv>/
    {groups,arrays}_<src>.csv, data/parks_pv_v1/geodata_release_<state><run>{,_arrays}.csv. Originals are kept once as
    *.pre_override; a park whose files already carry that backup is skipped (idempotent)."""
    data_dir = data_dir or os.path.join(REPO, "data", "parks_pv_v1")
    results_dir = results_dir or os.path.join(REPO, "results", "pv", f"geodata_release{run}")
    ov = pd.read_csv(path, dtype=str); ov = ov[ov.run == run]
    done = []
    for pid, o in ov.groupby("park_id", sort=False):
        f_state = [f for f in glob.glob(os.path.join(data_dir, f"geodata_release_*{run}.csv")) if not f.endswith("_arrays.csv")
                   and pid in open(f, encoding="utf-8").read()]
        if len(f_state) != 1:
            raise FileNotFoundError(f"{pid}: state table of run {run} not found ({f_state})")
        f_state = f_state[0]; f_sarr = f_state.replace(".csv", "_arrays.csv")
        d = pd.read_csv(f_state, dtype={"pv": str}); sa = pd.read_csv(f_sarr, dtype={"park_id": str})
        best = d[(d.pv == pid) & d.best.astype(str).eq("True")]
        src = best.source.iloc[0]
        rd = os.path.join(results_dir, pid)
        fg, fa = os.path.join(rd, f"groups_{src}.csv"), os.path.join(rd, f"arrays_{src}.csv")
        if os.path.exists(fg + ".pre_override"):
            print(pid, "already applied"); continue
        for f in (fg, fa):
            shutil.copy2(f, f + ".pre_override")
        for f in (f_state, f_sarr):
            if not os.path.exists(f + ".pre_override"):
                shutil.copy2(f, f + ".pre_override")   # one backup per state table (first override of that table)
        win = pd.read_csv(os.path.join(rd, f"windows_{src}.csv"))
        G = apply_groups(pd.read_csv(fg), win, o)
        new_sa = apply_arrays(sa[sa.park_id == pid].reset_index(drop=True), o)
        pa = pd.read_csv(fa)
        pa_new = new_sa.assign(group=new_sa.array_id.str.extract(r"g(\d+)_")[0].astype(int))[["group", "tilt", "azimuth", "share", "az_source"]]
        pa_new["az_conflict"] = bool(pa.az_conflict.astype(bool).any()) if len(pa) else False
        pa_new["note"] = ["Ost-Flanke" if (G.ew.astype(bool).iloc[g] and z < 180) else "West-Flanke" if G.ew.astype(bool).iloc[g] else ""
                          for g, z in zip(pa_new.group, pa_new.azimuth)]
        G.to_csv(fg, index=False); pa_new.to_csv(fa, index=False)
        sa = pd.concat([sa[sa.park_id != pid], new_sa], ignore_index=True); sa.to_csv(f_sarr, index=False)
        g0 = G.iloc[0]; i = best.index[0]
        d.loc[i, "arrays"] = _arrays_str(pa_new); d.loc[i, "n_groups"] = len(G)
        d.loc[i, ["g0_tilt", "g0_tilt_p10", "g0_tilt_p90", "g0_azimuth", "g0_ew", "g0_pitch", "g0_share", "g0_n_windows", "g0_n_sure"]] = \
            [g0.tilt_horiz, g0.tilt_p10, g0.tilt_p90, g0.azimuth, bool(g0.ew), g0.pitch, g0.share, int(g0.n_windows), int(g0.n_sure)]
        d.loc[i, "az_measured"] = g0.azimuth; d.loc[i, "tilt_measured"] = g0.tilt_horiz
        d.to_csv(f_state, index=False)
        done.append(pid); print(pid, src, "->", _arrays_str(pa_new))
    return done


if __name__ == "__main__":
    (restore if len(sys.argv) > 2 and sys.argv[2] == "restore" else apply_to_run)(sys.argv[1])
