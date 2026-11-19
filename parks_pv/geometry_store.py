"""Store of the angle estimation: one directory per park, estimated once and reused by every release.

    <geometry_dir>/<method>/<park_id>/
        sources.csv                one evaluated row per surface source (parks_pv.geometry_estimate.evaluate)
        arrays.csv                 measured generator arrays of the best accepted source (header only: no measurement)
        windows_<src>.csv, groups_<src>.csv, arrays_<src>.csv, park_<src>.json, windows_<src>.png   window outputs
        meta.json                  date, rules, git state, register fields the estimate used
    <geometry_dir>/<method>/method.json   rule set of the method (every park of a method is estimated with it)

`method` names the estimation version (rules and code state, e.g. v4); a new method is a new directory, an existing
park is never re-estimated unless asked (force). The release-specific register fallback is not stored here: it
depends on the baseline of the release (stage geometry -> data/<release>/geometry_arrays.csv)."""
import datetime
import json
import os
import subprocess

import pandas as pd

import settings

ARRAY_COLS = ["park_id", "array_id", "tilt", "azimuth", "share", "source", "az_source", "note", "mount_tilt", "mount_azimuth"]


def root(method: str) -> str:
    return os.path.join(settings.get("geometry_dir"), method)


def park_dir(method: str, park_id: str) -> str:
    return os.path.join(root(method), str(park_id))


def has(method: str, park_id: str) -> bool:
    """True when the park has a completed estimate of this method (sources.csv and arrays.csv written)."""
    d = park_dir(method, park_id)
    return os.path.exists(os.path.join(d, "sources.csv")) and os.path.exists(os.path.join(d, "arrays.csv"))


def missing(method: str, park_ids) -> list:
    return [p for p in park_ids if not has(method, p)]


def sources(method: str, park_id: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(park_dir(method, park_id), "sources.csv"), dtype={"pv": str})


def arrays(method: str, park_id: str) -> pd.DataFrame:
    """Measured generator arrays of the park (empty frame: no accepted measurement)."""
    return pd.read_csv(os.path.join(park_dir(method, park_id), "arrays.csv"), dtype={"park_id": str})


def all_sources(method: str, park_ids=None) -> pd.DataFrame:
    ids = park_ids if park_ids is not None else sorted(os.listdir(root(method))) if os.path.isdir(root(method)) else []
    out = [sources(method, p) for p in ids if has(method, p)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _git() -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=settings.REPO, capture_output=True, text=True).stdout.strip()
        return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except OSError:
        return {}


def save(method: str, park_id: str, src_rows: pd.DataFrame, array_rows: list, meta: dict = None):
    """Write sources.csv, arrays.csv and meta.json of one park (the window outputs are already in park_dir)."""
    d = park_dir(method, park_id)
    os.makedirs(d, exist_ok=True)
    src_rows.to_csv(os.path.join(d, "sources.csv"), index=False)
    m = dict(park_id=park_id, method=method, date=datetime.datetime.now().isoformat(timespec="seconds"), git=_git(), **(meta or {}))
    json.dump(m, open(os.path.join(d, "meta.json"), "w"), indent=1, default=str)
    pd.DataFrame(array_rows, columns=ARRAY_COLS).to_csv(os.path.join(d, "arrays.csv"), index=False)   # last: marks the park done


def method_spec(method: str) -> dict:
    """Rule set of a method ({rules, extend}) or None for a new method."""
    f = os.path.join(root(method), "method.json")
    return json.load(open(f)) if os.path.exists(f) else None


def write_method_spec(method: str, rules: str, extend: bool):
    os.makedirs(root(method), exist_ok=True)
    json.dump(dict(method=method, rules=rules, extend=bool(extend), created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(root(method), "method.json"), "w"), indent=1)
