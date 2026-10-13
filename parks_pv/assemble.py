"""Release frame + manifest per PV park (counterpart of parks/assemble.py).

Release columns (hourly means, UTC, label = interval START, as the meters): drivers (corrected CAMS
ghi/dhi, raw CAMS, ERA5 ghi, temp, wind, albedo, snow depth), chain diagnostics (dni, zenith, poa,
t_cell, aging, clipping, horizon shading, snow factor) and power_park_free / power_park [W]."""
import json
import os
import subprocess

import pandas as pd

from parks_pv import paths
from parks_pv.drivers import sha256_frame

DRIVER_COLS = ["ghi", "dhi", "ghi_cams_raw", "dhi_cams_raw", "ghi_clear_cams", "ghi_era5", "temp_air", "wind_speed", "pressure",
               "albedo", "snow_depth_cm", "imputed_share", "ring_missing"]
CHAIN_COLS = ["dni", "zenith", "azimuth_sun", "poa_global", "poa_eff", "t_cell", "p_dc_raw_w", "aging_factor", "p_dc_w",
              "clipped", "horizon_shaded", "snow_coverage", "snow_factor"]
POWER_COLS = ["power_park_free", "power_park", "availability"]


def release_frame(drv: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    out = drv[[c for c in DRIVER_COLS if c in drv]].copy()
    for c in CHAIN_COLS:
        out[c] = res[c].reindex(out.index)
    out["power_park_free"] = res["power_park_free"].reindex(out.index)
    out["power_park"] = res["power_park"].reindex(out.index)
    out["availability"] = 1.0
    out.index.name = "timestamp"
    return out.astype("float64")


def git_state(repo: str = paths.REPO) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout
    dirty = [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    return {"commit": run("rev-parse", "HEAD").strip(), "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(), "dirty_files": dirty}


def manifest(sel_row, cfg: dict, par: dict, drv_meta: dict, site: dict, arrays: list, horizon_src: str, release: pd.DataFrame,
             config_sha: str, horizon_sha: str) -> dict:
    return {
        "dataset": paths.NAME, "park_id": sel_row.park_id, "client": sel_row.client, "cls": sel_row.cls, "region": int(sel_row.region),
        "chain": "PVWatts-P (corrected CAMS ghi/dhi -> closure DNI with clear-sky cap -> Hay-Davies per array, horizon + inter-row, "
                 "physical IAM, Faiman -> PVWatts DC with era gamma -> era aging -> DC losses -> PVWatts inverter (DC/AC clipping) "
                 "-> AC losses, availability 0.98 -> ERA5 snow depth_linear factor)",
        "config": cfg, "config_sha256": config_sha, "params": par, "arrays": arrays, "site": site,
        "horizon": {"source": horizon_src, "sha256": horizon_sha},
        "drivers": drv_meta,
        "period": [str(release.index[0]), str(release.index[-1])], "n_hours": int(len(release)),
        "time_label": "interval start, UTC, hourly means",
        "release_sha256": sha256_frame(release),
        "git": git_state(),
        "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }


def write_release(park_id: str, release: pd.DataFrame, man: dict, out_dir: str = paths.RELEASE_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"synth_{park_id}.parquet")
    release.to_parquet(path)
    with open(os.path.join(out_dir, f"manifest_{park_id}.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    return path
