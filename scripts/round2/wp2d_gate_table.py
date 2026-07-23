#!/usr/bin/env python3
"""WP2-D — gated application: branch table + per-location correction JSONs.

Gate (guide):
  A  nearest station <= 15 km AND |delta elev_std| <= 20 m -> nearest-station
     empirical QM
  B  predicted class >= 2 -> model-predicted local quantiles (WP2-C)
  C  else no correction

Note: the chain runs on the DWD station driving each park/site, so for
'A' at a station location the station IS its own nearest station -> the
empirical QM equals the own-station table (distance 0). For parks the
distance is park-centroid <-> station.

Outputs: data/round2/branch_assignment.csv, data/round2/correction/<sid>.json,
data/round2/lab_notebook.md (falsifiable predictions, written BEFORE the run).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.correction import QUANTILES  # noqa: E402

TABLE = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")
PRED = os.path.join(REPO, "data", "round2", "predicted_classes.csv")
TOPO = os.path.join(REPO, "data", "round2", "topo_features.csv")
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dist-max-km", type=float, default=15.0)
_ap.add_argument("--out-base", default=os.path.join(REPO, "data", "round2"))
_args, _ = _ap.parse_known_args()

OUT_BASE = _args.out_base
OUT_DIR = os.path.join(OUT_BASE, "correction")

DIST_MAX_KM = _args.dist_max_km
ELEV_STD_MAX = 20.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table = pd.read_parquet(TABLE).set_index("station_id")
    pred = pd.read_csv(PRED, dtype={"location_id": str})
    topo = pd.read_csv(TOPO, dtype={"location_id": str}).set_index("location_id")
    model_q_path = os.path.join(REPO, "data", "round2", "model_quantiles.csv")
    model_q = pd.read_csv(model_q_path, dtype={"location_id": str}) \
        .set_index("location_id") if os.path.exists(model_q_path) else None

    q_cols_era5 = [f"q_era5_{q:.3f}" for q in QUANTILES]
    q_cols_dwd = [f"q_dwd_{q:.3f}" for q in QUANTILES]

    st = pred[pred["kind"] == "station"].set_index("location_id")
    rows = []
    # one correction JSON per DWD station that drives a park or site
    for loc_id, row in pred.set_index("location_id").iterrows():
        kind = row["kind"]
        if kind == "station":
            sid, dist = loc_id, 0.0
            d_elev_std = 0.0
        elif kind == "site":
            sid = loc_id.split("_")[1]
            dist, d_elev_std = 0.0, 0.0   # sites ARE stations
        else:  # park
            sid = loc_id.split("_")[1][:5]
            if sid not in st.index:
                continue
            dist = haversine_km(row["latitude"], row["longitude"],
                                st.loc[sid, "latitude"], st.loc[sid, "longitude"])
            d_elev_std = abs(row["elev_std"] - st.loc[sid, "elev_std"])
        if sid not in table.index:
            continue
        pred_class = int(row["predicted_class"])
        obs_class = table.loc[sid, "era5_class"]
        # gate: for parks use the park's own predicted class; branch A needs
        # a *representative* nearby station
        if dist <= DIST_MAX_KM and d_elev_std <= ELEV_STD_MAX:
            branch = "A"
        elif pred_class >= 2:
            branch = "B"
        else:
            branch = "C"
        q_era5 = table.loc[sid, q_cols_era5].values.astype(float).tolist()
        q_station = table.loc[sid, q_cols_dwd].values.astype(float).tolist()
        q_model = None
        if model_q is not None and loc_id in model_q.index:
            q_model = model_q.loc[loc_id, [f"q_model_{q:.3f}" for q in QUANTILES]] \
                .values.astype(float).tolist()
        rows.append({"location_id": loc_id, "kind": kind, "station_id": sid,
                     "distance_km": round(float(dist), 2),
                     "delta_elev_std": round(float(d_elev_std), 2),
                     "predicted_class": pred_class,
                     "observed_class": obs_class, "branch": branch})
        if kind in ("park", "station"):
            payload = {"station_id": sid, "location_id": loc_id, "branch": branch,
                       "q_era5": q_era5, "q_target_station": q_station,
                       "q_target_model": q_model}
            # stations -> <sid>.json; parks -> <park_id>.json (the chain looks
            # up park_id first, then falls back to the station id)
            name = sid if kind == "station" else loc_id.split("park_")[1]
            with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
                json.dump(payload, f)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_BASE, "branch_assignment.csv"),
              index=False)
    print(df.groupby(["kind", "branch"]).size().to_string())

    write_lab_notebook(df)


def write_lab_notebook(df: pd.DataFrame):
    parks = df[df["kind"] == "park"].copy()
    nb = os.path.join(OUT_BASE, "lab_notebook.md")
    lines = [
        "# WP2-D Lab Notebook — falsifiable predictions (written BEFORE the park re-validation)",
        "",
        f"Gate: A if dist<= {DIST_MAX_KM} km AND |d elev_std| <= {ELEV_STD_MAX} m; "
        "B if predicted class >= 2; else C.",
        "",
        "Guide predictions: Lower Saxony improves via A; Bavaria/Palatinate/"
        "Hesse-South improve via B; Schleswig-Holstein West & Mecklenburg-WP "
        "unchanged via C.",
        "",
        "## Park branch assignment (before running the chain)",
        "",
        parks[["location_id", "station_id", "distance_km", "delta_elev_std",
               "predicted_class", "branch"]].to_markdown(index=False),
        "",
        "Expected: branch A/B parks improve the WP1 metrics (energy ratio "
        "closer to 1, higher R2); branch C parks stay within noise of M1.",
    ]
    with open(nb, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("lab notebook:", nb)


if __name__ == "__main__":
    main()
