#!/usr/bin/env python
"""Re-snow an existing PV release (or the Trianel twins) with another snow model, without a chain run.

The release frames carry power_park_free, snow_depth_cm, temp_air and poa_global, and power_park is exactly
power_park_free * snow_factor, so a snow model change is a recomputation of snow_coverage, snow_factor and power_park
(parks_pv.generate.resnow_frame). Every other column stays bit-identical; file names, columns, dtypes and index are
those of the source. The source directory is only read.

    DATA_ROOT=/mnt/lambda1/nvme1 synthre/bin/python scripts/pv/resnow_release.py --kind release \\
        --src $DATA_ROOT/synthetic/solar/parks_pv_v1_30min --dst $DATA_ROOT/synthetic/solar/parks_pv_v1s_30min
    DATA_ROOT=/mnt/lambda1/nvme1 synthre/bin/python scripts/pv/resnow_release.py --kind twins \\
        --src $DATA_ROOT/synthetic/solar/trianel_twins_30min --dst $DATA_ROOT/synthetic/solar/trianel_twins_s_30min

release: per park synth_/manifest_ + top-level manifest.json; parks.csv, clients.csv, sites.csv, pv_arrays.csv,
         pv_masterdata_view.csv copied unchanged (parks.csv: cf_ac/cf_dc/snow_loss_share still describe the source),
         README.md = source README + a snow section.
twins:   per park synth_/manifest_ (meter_vs_twin recomputed against the source meters) + manifest.json + parks.csv with
         the recomputed meter metrics. Meters and sites_confidential.csv are NOT copied (the forecasting framework reads
         meters from data.meter_path, sites via data.sites_file or the source twin directory).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import assemble as A, drivers as DRV, generate as GEN, paths  # noqa: E402
from pv import metrics  # noqa: E402

SNOW_COLS = ["snow_coverage", "snow_factor", "power_park"]
RELEASE_TABLES = ["parks.csv", "clients.csv", "sites.csv", "pv_arrays.csv", "pv_masterdata_view.csv"]
TWIN_WINTER = ("2023-11-15", "2024-03-15")       # as scripts/pv/trianel_twins.py


def park_ids(src: str) -> list:
    return sorted(os.path.basename(f)[len("synth_"):-len(".parquet")] for f in glob.glob(os.path.join(src, "synth_*.parquet")))


def check_unchanged(old: pd.DataFrame, new: pd.DataFrame):
    """Schema identical and every non-snow column bit-identical."""
    assert list(old.columns) == list(new.columns), "column order changed"
    assert (old.dtypes == new.dtypes).all(), "dtypes changed"
    assert old.index.equals(new.index) and old.index.name == new.index.name, "index changed"
    keep = [c for c in old.columns if c not in SNOW_COLS]
    assert old[keep].equals(new[keep]), "non-snow columns changed"


def meter_metrics(rel: pd.DataFrame, meter_file: str, cap_w: float) -> dict:
    """meter_vs_twin block of scripts/pv/trianel_twins.py."""
    y = pd.read_parquet(meter_file)["power_w"].reindex(rel.index)
    day = rel["zenith"] < 90
    winter = day & (rel.index >= pd.Timestamp(TWIN_WINTER[0], tz="UTC")) & (rel.index < pd.Timestamp(TWIN_WINTER[1], tz="UTC"))
    return {"day": metrics.basic(y[day], rel.loc[day, "power_park"], cap_w), "winter": metrics.basic(y[winter], rel.loc[winter, "power_park"], cap_w),
            "n_meter_steps": int(y.notna().sum())}


def energy_stats(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Energy (sum of W over steps) of power_park old/new: whole period, per Nov-Mar winter, summer May-Sep."""
    out = {"e_free": float(old["power_park_free"].sum()), "e_old": float(old["power_park"].sum()), "e_new": float(new["power_park"].sum())}
    m = old.index.month
    for y in sorted(set(old.index.year)):
        w = ((old.index.year == y) & m.isin([11, 12])) | ((old.index.year == y + 1) & m.isin([1, 2, 3]))
        if w.sum() > 0 and (old.index.year == y + 1).any():
            out[f"winter_{y}_{(y + 1) % 100:02d}_old"] = float(old.loc[w, "power_park"].sum())
            out[f"winter_{y}_{(y + 1) % 100:02d}_new"] = float(new.loc[w, "power_park"].sum())
            out[f"winter_{y}_{(y + 1) % 100:02d}_free"] = float(old.loc[w, "power_park_free"].sum())
    s = m.isin([5, 6, 7, 8, 9])
    out["summer_max_abs_diff_w"] = float((new.loc[s, "power_park"] - old.loc[s, "power_park"]).abs().max())
    out["n_steps_changed"] = int((new["power_park"] != old["power_park"]).sum())
    out["months_changed"] = sorted({int(x) for x in old.index.month[(new["power_park"] != old["power_park"]).to_numpy()]})
    return out


def chain_text(chain: str, model: str) -> str:
    txt = A.snow_chain_text(model)
    for old in ("ERA5 snow depth_linear factor", "ERA5 snow depth_linear"):
        if old in chain:
            return chain.replace(old, txt)
    return chain + f"; snow replaced by {txt}"


def resnow_park(pid: str, a, par: dict) -> tuple:
    src_file = os.path.join(a.src, f"synth_{pid}.parquet")
    old = pd.read_parquet(src_file)
    man = json.load(open(os.path.join(a.src, f"manifest_{pid}.json")))
    tilt = GEN.park_tilt(man["arrays"])
    icon, icon_meta = DRV.icon_snow_series(pid, old.index, a.icon_dir)
    new = GEN.resnow_frame(old, a.snow_model, tilt, par, icon)
    check_unchanged(old, new)
    out_file = os.path.join(a.dst, f"synth_{pid}.parquet")
    new.to_parquet(out_file)
    m = dict(man)
    m["dataset"] = os.path.basename(a.dst.rstrip("/"))
    m["chain"] = chain_text(man["chain"], a.snow_model)
    m["snow"] = {"model": a.snow_model, "params": par, "tilt_deg_share_weighted": tilt, "driver": GEN.SNOW_DRIVER[a.snow_model],
                 "icon": icon_meta, "icon_lead_max_h": DRV.ICON_LEAD_MAX_H, "loss": "1 - pvlib dc_loss_nrel(coverage, 100)",
                 "calibration": "FL_Contribution/reports/pv_snow_fix_2026-10-19 (model e)"}
    m["source"] = {"dir": a.src, "file_sha256": DRV.sha256_file(src_file), "release_sha256": man.get("release_sha256"),
                   "created": man.get("created"), "git": man.get("git"), "chain": man.get("chain")}
    m["release_sha256"] = A.sha256_frame(new)
    if a.kind == "twins" and a.meter_dir:
        mf = os.path.join(a.meter_dir, f"meter_{pid}.parquet")
        if os.path.exists(mf):
            cap_w = float(man["config"]["kwp_dc"]) * 1000.0
            chk = meter_metrics(old, mf, cap_w)
            assert abs(chk["day"]["r2"] - man["meter_vs_twin"]["day"]["r2"]) < 1e-9, f"{pid}: meter metrics not reproduced"
            m["meter_vs_twin"] = meter_metrics(new, mf, cap_w)
            m["source"]["meter_vs_twin"] = man.get("meter_vs_twin")
    m["git"] = A.git_state()
    m["created"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    with open(os.path.join(a.dst, f"manifest_{pid}.json"), "w") as f:
        json.dump(m, f, indent=1 if a.kind == "twins" else 2, default=str)
    st = {"park_id": pid, "tilt_deg": round(tilt, 2), "icon_coverage": icon_meta["icon_coverage"], **energy_stats(old, new)}
    if "meter_vs_twin" in m and a.kind == "twins":
        st.update({"r2_day": m["meter_vs_twin"]["day"]["r2"], "nrmse_day": m["meter_vs_twin"]["day"]["nrmse_cap"],
                   "energy_ratio_day": m["meter_vs_twin"]["day"]["energy_ratio"], "r2_winter": m["meter_vs_twin"]["winter"]["r2"]})
    return st, m


def readme_section(a, par: dict) -> str:
    return (f"\n\n## Snow variant `{os.path.basename(a.dst.rstrip('/'))}`\n\n"
            f"Copy of `{os.path.basename(a.src.rstrip('/'))}` with only `snow_coverage`, `snow_factor` and `power_park` recomputed "
            f"(`scripts/pv/resnow_release.py`, snow model `{a.snow_model}`, params `{json.dumps(par)}`): coverage ramp on "
            f"max(ERA5 depth, ICON-D2 09 UTC daily h_snow) / 5 cm with tilt-dependent slide-off (0.1/h * sin(tilt) while "
            f"T_air > -POA/80); calibration FL_Contribution/reports/pv_snow_fix_2026-10-19. All other columns are bit-identical "
            f"to the source. `parks.csv` is copied unchanged: its `cf_ac`, `cf_dc` and `snow_loss_share` describe the source; the "
            f"new per-park snow loss is in `manifest.json` (`parks`).\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--kind", choices=["release", "twins"], required=True)
    ap.add_argument("--snow-model", default="depth_slide", choices=list(GEN.SNOW_MODELS))
    ap.add_argument("--icon-dir", default=paths.ICON_SL09_DIR)
    ap.add_argument("--meter-dir", default=None, help="twins: directory of meter_<pv>.parquet (default: --src)")
    ap.add_argument("--parks", default=None, help="comma-separated ids (default: all synth_*.parquet in --src)")
    a = ap.parse_args()
    a.src, a.dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    if a.src == a.dst:
        raise SystemExit("--dst must differ from --src")
    if os.path.isdir(a.dst) and os.listdir(a.dst):
        raise SystemExit(f"{a.dst} exists and is not empty")
    if a.kind == "twins" and a.meter_dir is None:
        a.meter_dir = a.src
    os.makedirs(a.dst)
    par = GEN.snow_params(a.snow_model)
    ids = a.parks.split(",") if a.parks else park_ids(a.src)
    rows = []
    for pid in ids:
        st, _ = resnow_park(pid, a, par)
        rows.append(st)
        print(pid, f"tilt {st['tilt_deg']:5.1f} icon {st['icon_coverage']:.3f} energy {st['e_new'] / st['e_old'] - 1:+.4%} "
                   f"snow loss {1 - st['e_old'] / st['e_free']:.4f} -> {1 - st['e_new'] / st['e_free']:.4f} "
                   f"summer max|d| {st['summer_max_abs_diff_w']:.3g} W", flush=True)
    summ = pd.DataFrame(rows).set_index("park_id")
    files = {}
    if a.kind == "release":
        for f in RELEASE_TABLES:
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.dst, f))
        readme = open(os.path.join(a.src, "README.md")).read() if os.path.exists(os.path.join(a.src, "README.md")) else ""
        open(os.path.join(a.dst, "README.md"), "w").write(readme + readme_section(a, par))
    else:
        pk = pd.read_csv(os.path.join(a.src, "parks.csv"), dtype={"park_id": str})
        for c in ("r2_day", "nrmse_day", "energy_ratio_day", "r2_winter"):
            if c in summ:
                pk[c] = pk.park_id.map(summ[c].round(4)).fillna(pk[c])
        pk.to_csv(os.path.join(a.dst, "parks.csv"), index=False)
    for f in sorted(os.listdir(a.dst)):
        files[f] = DRV.sha256_file(os.path.join(a.dst, f))
    parks = {pid: {"tilt_deg": r.tilt_deg, "snow_loss_share_old": 1 - r.e_old / r.e_free, "snow_loss_share_new": 1 - r.e_new / r.e_free,
                   "energy_change": r.e_new / r.e_old - 1, "icon_coverage": r.icon_coverage} for pid, r in summ.iterrows()}
    top = {"dataset": os.path.basename(a.dst), "kind": a.kind, "source": a.src, "n_parks": len(ids),
           "snow": {"model": a.snow_model, "params": par, "driver": GEN.SNOW_DRIVER[a.snow_model], "icon_dir": a.icon_dir,
                    "icon_lead_max_h": DRV.ICON_LEAD_MAX_H, "icon_ffill_days": 2, "tilt": "share-weighted mean of manifest arrays",
                    "loss": "1 - pvlib dc_loss_nrel(coverage, 100)", "calibration": "FL_Contribution/reports/pv_snow_fix_2026-10-19 (model e)"},
           "recomputed_columns": SNOW_COLS, "unchanged": "all other columns bit-identical to the source (checked per park)",
           "script": "scripts/pv/resnow_release.py", "git": A.git_state(), "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "files_sha256": files, "parks": parks}
    if a.kind == "twins":
        top["not_copied"] = ["meter_<pv>.parquet (use data.meter_path = source dir)", "sites_confidential.csv (confidential, stays in the source dir)"]
    with open(os.path.join(a.dst, "manifest.json"), "w") as f:
        json.dump(top, f, indent=1, default=str)
    print(f"{len(ids)} parks -> {a.dst}; total energy change {summ.e_new.sum() / summ.e_old.sum() - 1:+.4%}")


if __name__ == "__main__":
    main()
