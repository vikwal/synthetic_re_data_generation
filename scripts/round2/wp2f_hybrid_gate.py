#!/usr/bin/env python3
"""Hybrid correction gate (2nd, POST-HOC gate iteration — not pre-registered).

Rule: branch A iff terrain-similar (ask-21a similarity gate: same macro
class, sector z0 ratio <= 2, not coastal) AND nearest-station distance
<= 20 km; coastal locations -> C regardless of the classifier (guards B);
else B iff predicted class >= 2; else C.

3rd iteration (user decision 2026-07-08): **parks never get branch B** —
the LGBM correction is LSO-validated at the 204 stations but went 0/3 on
parks (00282, 02483, 05347-under-distance-gate all degraded vs their
uncorrected twins). Parks therefore get A or C only; the B rule remains
for station/site locations.

The 20-km bound is INFORMED BY the observed distance decay of the QM
transfer in ask 21a (18 km works, 26-30 km noise, 37-40 km breaks) and is
therefore a data-guided choice, to be reported as such.

Inputs: data/round2_simgate/branch_assignment.csv (similarity verdicts),
data/round2/branch_assignment.csv (distances). Output: data/round2_hybgate/
(branch_assignment.csv, correction/*.json, topo_features.csv) — nothing
outside this directory is touched.
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.correction import QUANTILES  # noqa: E402

OUT_BASE = os.path.join(REPO, "data", "round2_hybgate")
OUT_DIR = os.path.join(OUT_BASE, "correction")
DIST_MAX_KM = 20.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = pd.read_csv(os.path.join(REPO, "data", "round2_simgate",
                                   "branch_assignment.csv"),
                      dtype={"location_id": str, "station_id": str})
    dist = pd.read_csv(os.path.join(REPO, "data", "round2",
                                    "branch_assignment.csv"),
                       dtype={"location_id": str}) \
        .set_index("location_id")["distance_km"].astype(float)
    table = pd.read_parquet(os.path.join(
        REPO, "data", "round2", "station_correction_table.parquet")) \
        .set_index("station_id")
    model_q = pd.read_csv(os.path.join(REPO, "data", "round2",
                                       "model_quantiles.csv"),
                          dtype={"location_id": str}).set_index("location_id")
    q_cols_era5 = [f"q_era5_{q:.3f}" for q in QUANTILES]
    q_cols_dwd = [f"q_dwd_{q:.3f}" for q in QUANTILES]

    rows = []
    for _, r in sim.iterrows():
        loc_id, sid = r["location_id"], r["station_id"]
        d_km = float(dist.get(loc_id, np.nan))
        coastal = r["gate_reason"] == "coastal guard"
        similar = (r["gate_reason"] == "similar") or (
            r["branch"] == "A" and not coastal)
        if coastal:
            branch, reason = "C", "coastal guard"
        elif similar and d_km <= DIST_MAX_KM:
            branch, reason = "A", f"similar & {d_km:.1f} km"
        elif r["kind"] == "park":
            # parks never get B (0/3 on parks; LGBM only station-validated)
            branch, reason = "C", ("similar but too far"
                                   if similar else r["gate_reason"])
        elif int(r["predicted_class"]) >= 2:
            branch, reason = "B", ("similar but too far"
                                   if similar else r["gate_reason"])
        else:
            branch, reason = "C", ("similar but too far"
                                   if similar else r["gate_reason"])
        rows.append({**r.to_dict(), "distance_km": d_km,
                     "branch": branch, "gate_reason": reason})

        if r["kind"] in ("park", "station") and sid in table.index:
            q_model = None
            if loc_id in model_q.index:
                q_model = model_q.loc[
                    loc_id, [f"q_model_{q:.3f}" for q in QUANTILES]] \
                    .values.astype(float).tolist()
            payload = {"station_id": sid, "location_id": loc_id,
                       "branch": branch,
                       "q_era5": table.loc[sid, q_cols_era5]
                       .values.astype(float).tolist(),
                       "q_target_station": table.loc[sid, q_cols_dwd]
                       .values.astype(float).tolist(),
                       "q_target_model": q_model}
            name = sid if r["kind"] == "station" else loc_id.split("park_")[1]
            with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
                json.dump(payload, f)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_BASE, "branch_assignment.csv"), index=False)
    shutil.copy(os.path.join(REPO, "data", "round2", "topo_features.csv"),
                os.path.join(OUT_BASE, "topo_features.csv"))
    print(df.groupby(["kind", "branch"]).size().to_string(), "\n")
    parks = df[df["kind"] == "park"]
    print(parks[["location_id", "station_id", "distance_km", "macro_match",
                 "z0_ratio_max", "branch", "gate_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
