"""Master-data overrides from the fleet dashboard (pv_sim2real/dashboard, 2026-10-23).

The dashboard never edits source tables; it appends entries to overrides_pv.json / overrides_wind.json:
    {"park_id" | "turbine_id": ..., ["park_id": ... for wind,] "field": ..., "old": ..., "new": ..., "timestamp": ..., "comment": ...}
The latest entry per (record, field) wins. The pipelines apply them only when asked to, so a release stays reproducible:

    PV   PV_OVERRIDES=<path to overrides_pv.json>     -> scripts/pv/run_parks_pv_v1.load_selection (also used by v2)
    Wind WIND_OVERRIDES=<path to overrides_wind.json> -> parks.layout.build_turbine_table (park_layouts columns)

Field names are those of the tables the hooks see (selection columns for PV, park_layouts columns for wind), which are
the names the dashboard shows. `old` is checked against the table: a mismatch is reported (the source changed since the
edit), the new value is applied anyway. Every applied override is logged, and the list is returned for the manifest.
"""
import json
import os

import pandas as pd

ID_KEY = {"pv": "park_id", "wind": "turbine_id"}


def load(path: str) -> list:
    return json.load(open(path)) if path and os.path.exists(path) else []


def latest(entries: list, id_key: str) -> list:
    """Latest entry per (record, field), in file order."""
    last = {}
    for e in entries:
        last[(str(e[id_key]), e["field"])] = e
    return list(last.values())


def apply(df: pd.DataFrame, path: str, kind: str, id_col: str = None, log=print) -> tuple:
    """(copy of df with the overrides applied, list of applied entries). id_col defaults to the kind's id key."""
    key = ID_KEY[kind]
    id_col = id_col or key
    entries = latest(load(path), key)
    if not entries:
        return df, []
    out = df.copy()
    applied = []
    for e in entries:
        m = out[id_col].astype(str) == str(e[key])
        if not m.any():
            log(f"override skipped: {key} {e[key]} not in the table")
            continue
        if e["field"] not in out:
            log(f"override skipped: field {e['field']} not in the table")
            continue
        cur = out.loc[m, e["field"]].iloc[0]
        if pd.notna(cur) and e.get("old") is not None and str(cur) != str(e["old"]):
            try:
                same = abs(float(cur) - float(e["old"])) < 1e-9
            except (TypeError, ValueError):
                same = False
            if not same:
                log(f"override {e[key]}.{e['field']}: table has {cur!r}, entry expected old {e['old']!r} (source changed since the edit)")
        if pd.api.types.is_numeric_dtype(out[e["field"]]) and e["new"] is not None:
            out.loc[m, e["field"]] = float(e["new"])
        else:
            out[e["field"]] = out[e["field"]].astype(object)
            out.loc[m, e["field"]] = e["new"]
        applied.append(e)
        log(f"override {e[key]}.{e['field']}: {cur!r} -> {e['new']!r} ({e.get('comment', '')})")
    return out, applied


def from_env(df: pd.DataFrame, kind: str, id_col: str = None) -> pd.DataFrame:
    """Apply the overrides named by PV_OVERRIDES / WIND_OVERRIDES; unchanged without the variable."""
    path = os.environ.get({"pv": "PV_OVERRIDES", "wind": "WIND_OVERRIDES"}[kind])
    return apply(df, path, kind, id_col)[0] if path else df
