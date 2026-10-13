#!/usr/bin/env python
"""Select the 90 real PV parks (8 clients x 10 + holdout 10) for parks_pv_v1 from the parsed MaStR pool.
Writes pv_park_selection_v1.csv and pv_park_pool_v1.csv next to the wind selection (public register data).

    synthre/bin/python scripts/pv/select_parks.py [--seed 42] [--out ~/Work/FL_Contribution/data/mastr]
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "pv"))
from pv import selection as S  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
MASTR = f"{ROOT}/synthetic/solar/mastr"


def validation_parks() -> pd.DataFrame:
    """Coordinates of the confidential meter-validated parks (excluded from the pool); empty if unavailable."""
    try:
        import trianel_common as T
        m = T.load_masterdata()
        return m.rename(columns=str.lower)[["lat", "lon"]]
    except Exception as e:  # noqa: BLE001
        print("validation parks not excluded:", str(e)[:80])
        return pd.DataFrame(columns=["lat", "lon"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", default=f"{MASTR}/solar_parks_20250625.parquet")
    ap.add_argument("--units", default=f"{MASTR}/solar_units_20250625.parquet")
    ap.add_argument("--eeg", default=f"{MASTR}/solar_eeg_20250625.parquet")
    ap.add_argument("--terrain", default=f"{MASTR}/pool_terrain_20250625.parquet")
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/data/mastr"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--min-kwp", type=float, default=750.0)
    a = ap.parse_args()
    pool = S.build_pool(pd.read_parquet(a.parks), pd.read_parquet(a.units), pd.read_parquet(a.terrain), a.min_kwp,
                        validation_parks(), eeg=pd.read_parquet(a.eeg))
    pool = S.add_regions(pool, a.regions, a.seed)
    ok = pool[pool.reject.isna()]
    print(f"pool {len(pool)} parks ({(pool.n_lokationen > 1).sum()} merged from several Lokationen), {len(ok)} eligible; rejects:",
          pool.reject.value_counts().to_dict())
    print("foreign neighbour within 500 m (eligible):", ok.foreign_within_km.notna().sum())
    print("classes:", ok.cls.value_counts().reindex(S.CLASSES).to_dict())
    print("class x region:\n", pd.crosstab(ok.cls, ok.region).to_string())
    print("terrain:", ok.terrain_cls.value_counts().to_dict(), "| cohort:", ok.cohort.value_counts().to_dict(), "| size:", ok.size_bin.value_counts().to_dict())
    sel = S.select(pool, a.regions, a.seed)
    os.makedirs(a.out, exist_ok=True)
    sel.to_csv(os.path.join(a.out, "pv_park_selection_v1.csv"), index=False)
    pool.drop(columns=["unit_ids"]).to_csv(os.path.join(a.out, "pv_park_pool_v1.csv"), index=False)
    f = sel.dropna(subset=["park_id"])
    print(f"\nselected {len(f)}/{len(sel)} slots | {f.dc_kwp.sum() / 1e3:.0f} MWp | units {int(f.n_units.sum())} | merged parks {(f.n_lokationen > 1).sum()}")
    for k in ("cohort", "size_bin", "terrain_cls", "orientation_class", "tilt_class", "bundesland"):
        print(f"{k}:", f[k].value_counts().to_dict())
    print(f.groupby("client").agg(parks=("park_id", "size"), mwp=("dc_kwp", lambda s: s.sum() / 1e3), regions=("region", "nunique"),
                                  cohorts=("cohort", "nunique"), terrain=("terrain_cls", "nunique")).round(1).to_string())


if __name__ == "__main__":
    main()
