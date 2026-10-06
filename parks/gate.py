"""Hybrid correction gate (round-2 final rule, wp2f) evaluated for the real parks.

INFORMATION ONLY: parks_v1 runs without QM (M4-noQM). The table documents
which parks the round-2 park gate would have corrected (branch A: terrain-
similar to the nearest QM station AND <= 20 km; coastal -> C; parks never B),
so the decision is traceable. Sector features and dominant sectors come from
the round-2 similarity-gate implementation (scripts/round2/wp2e_*.py),
evaluated at the park centroid against the nearest station that has a QM table.
"""

import importlib.util
import os

import numpy as np
import pandas as pd

from parks import paths

DIST_MAX_KM = 20.0


def _wp2e():
    path = os.path.join(paths.REPO, "scripts", "round2", "wp2e_similarity_gate.py")
    spec = importlib.util.spec_from_file_location("wp2e_similarity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def hybrid_gate(centroids: pd.DataFrame) -> pd.DataFrame:
    """centroids: park_id, latitude, longitude -> one gate row per park."""
    sg = _wp2e()
    pred = pd.read_csv(os.path.join(paths.REPO, "data", "round2", "predicted_classes.csv"),
                       dtype={"location_id": str})
    table = pd.read_parquet(os.path.join(paths.REPO, "data", "round2", "station_correction_table.parquet"))
    st = pred[(pred.kind == "station") & pred.location_id.isin(table.station_id.astype(str))] \
        .set_index("location_id")
    sampler = sg.SectorSampler(sg.CORINE_TIF)
    rows = []
    for r in centroids.itertuples(index=False):
        d = haversine_km(r.latitude, r.longitude, st.latitude.values, st.longitude.values)
        i = int(np.argmin(d))
        sid, dkm = st.index[i], float(d[i])
        dom = sg.dominant_sectors(sid) or list(range(sg.N_SECTORS))
        fl = sampler.features(r.latitude, r.longitude)
        fs = sampler.features(st.latitude.iloc[i], st.longitude.iloc[i])
        row = {"lokation": r.park_id, "station_id": sid, "dist_km": round(dkm, 2)}
        if fl is None or fs is None:
            rows.append({**row, "branch": "C", "reason": "no CORINE coverage"})
            continue
        union = lambda f: int(sum(f[s]["macro_counts"] for s in dom)[1:].argmax() + 1)  # noqa: E731
        macro_ok = union(fl) == union(fs)
        z0r = max(max(fl[s]["z0"], fs[s]["z0"]) / min(fl[s]["z0"], fs[s]["z0"]) for s in dom)
        marine = max(max(f[s]["marine_share_20km"] for s in dom) for f in (fl, fs))
        lake = max(max(f[s]["lake_share_20km"] for s in dom) for f in (fl, fs))
        similar = macro_ok and z0r <= sg.Z0_RATIO_MAX
        if marine >= sg.MARINE_SHARE_MIN or lake >= sg.LAKE_SHARE_MIN:
            branch, reason = "C", "coastal guard"
        elif similar and dkm <= DIST_MAX_KM:
            branch, reason = "A", f"similar & {dkm:.1f} km"
        elif similar:
            branch, reason = "C", "similar but too far"
        else:
            branch, reason = "C", "macro mismatch" if not macro_ok else "z0 ratio > 2"
        rows.append({**row, "macro_ok": macro_ok, "z0_ratio_max": round(z0r, 3),
                     "marine_max": round(marine, 4), "lake_max": round(lake, 4),
                     "branch": branch, "reason": reason})
    return pd.DataFrame(rows)
