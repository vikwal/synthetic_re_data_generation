"""Release frame + manifest per park.

Final column order: ERA5 base fields of the primary cell, derived 2 m fields,
per group t<i> (wind_speed_hub, wind_direction_100m, density_hub,
aging_factor, power [W, group total]), then power_park_free, wake_factor,
power_park (= free x wake) and the availability placeholder (1.0 = available;
outages/curtailment are a later, separate mask).
"""

import hashlib
import json
import os
import subprocess

import pandas as pd

from parks import era5_db, paths
from parks.synth import DERIVED_COLS

GROUP_FIELDS = ["wind_speed_hub", "wind_direction_100m", "density_hub", "aging_factor", "power"]


def release_frame(free: pd.DataFrame, w: pd.Series, group_ids: list) -> pd.DataFrame:
    w = w.reindex(free.index)
    if w.isna().any():
        raise ValueError(f"wake factor missing for {int(w.isna().sum())} hours")
    cols = era5_db.RAW_COLS + DERIVED_COLS + [f"{f}_{g}" for g in group_ids for f in GROUP_FIELDS]
    out = free[cols + ["power_park_free"]].copy()
    out["wake_factor"] = w.values
    out["power_park"] = out["power_park_free"] * out["wake_factor"]
    out["availability"] = 1.0
    return out


def frame_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def git_state(repo: str = paths.REPO) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout
    # porcelain lines are 'XY path'; no strip() on the whole output, it would eat the
    # leading status blank of the first line and cut its path
    dirty = [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    return {"commit": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty_files": dirty}


def manifest(cfg: dict, era5_meta: dict, wake_meta: dict, release: pd.DataFrame,
             free_sha: str, spec_overrides: pd.DataFrame) -> dict:
    import py_wake
    p, park = cfg["params"], cfg["park"]
    types = set(p["turbines"])
    ov = spec_overrides[spec_overrides["turbine"].isin(types)] if len(spec_overrides) else spec_overrides
    return {
        "experiment_id": cfg["round2"]["experiment_id"],
        "park_id": park["lokation"], "name": park["name"], "client": park["client"], "cls": park["cls"],
        "chain": "M4-noQM (dynamic power law, no QM, Weibull aging per group, NOJ wakes)",
        "round2": cfg["round2"],
        "groups": [{"group_id": g, "lib_name": t, "n": n, "hub_height": h, "rated_cap_kw": r,
                    "commissioning_date": d, "era5_cell": c}
                   for g, t, n, h, r, d, c in zip(p["group_ids"], p["turbines"], p["group_sizes"],
                                                  p["hub_heights"], p["rated"], p["commissioning_dates"],
                                                  p["era5_cells"])],
        "correction_branch": "C",
        "correction_note": "QM off for all parks (M4-noQM); hybrid-gate A candidates listed in parks.csv",
        "random_seed": p.get("random_seed", 42), "noise": p.get("noise", 0.0),
        "era5": era5_meta,
        "wake": {**wake_meta, "k": cfg["round2"]["wake"]["k"], "model": cfg["round2"]["wake"]["model"],
                 "py_wake": py_wake.__version__, "inflow_from_free_frame_sha256": free_sha},
        "spec_overrides": ov.to_dict("records") if len(ov) else [],
        "period": [str(release.index[0]), str(release.index[-1])], "n_hours": len(release),
        "release_sha256": frame_sha256(release),
        "git": git_state(),
        "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }


def write_release(lokation: str, release: pd.DataFrame, man: dict, out_dir: str = paths.RELEASE_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"synth_{lokation}.parquet")
    release.to_parquet(path)
    with open(os.path.join(out_dir, f"manifest_{lokation}.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    return path
