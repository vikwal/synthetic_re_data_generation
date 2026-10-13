#!/usr/bin/env python
"""Parse the MaStR solar export once and cache units, EEG plants and parks as parquet under
${DATA_ROOT}/synthetic/solar/mastr/. Public register data, no Trianel content.

    synthre/bin/python scripts/pv/mastr_parse.py [--min-kw 100] [--workers 12]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pv import mastr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mastr-dir", default=os.path.join(REPO, "data", "Marktstammdatenregister_20250625_25"))
    ap.add_argument("--out-dir", default=f"{ROOT}/synthetic/solar/mastr")
    ap.add_argument("--min-kw", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    units, eeg = mastr.parse_register(a.mastr_dir, a.min_kw, a.workers)
    print(f"parsed {len(units)} units, {len(eeg)} EEG plants in {time.time() - t0:.0f} s", flush=True)
    stamp = os.path.basename(a.mastr_dir.rstrip("/")).split("_")[1]
    units.to_parquet(os.path.join(a.out_dir, f"solar_units_{stamp}.parquet"), index=False)
    eeg.to_parquet(os.path.join(a.out_dir, f"solar_eeg_{stamp}.parquet"), index=False)
    parks = mastr.aggregate_parks(units, eeg)
    parks.to_parquet(os.path.join(a.out_dir, f"solar_parks_{stamp}.parquet"), index=False)
    ff = parks[parks["mounting"].str.contains("Freifl", na=False) & (parks["status"] == "In Betrieb")]
    print(f"parks {len(parks)}; open-ground in operation {len(ff)}; of which dc >= 750 kWp: {(ff.dc_kwp >= 750).sum()}, "
          f">= 1 MWp: {(ff.dc_kwp >= 1000).sum()}, >= 5 MWp: {(ff.dc_kwp >= 5000).sum()}")
    print("with coordinates:", ff[ff.dc_kwp >= 750].lat.notna().mean().round(3), "secondary orientation share:",
          ff[ff.dc_kwp >= 750].has_secondary.mean().round(3), "mixed classes:", ff[ff.dc_kwp >= 750].mixed_orientation.mean().round(3))
    print(f"done in {time.time() - t0:.0f} s -> {a.out_dir}")


if __name__ == "__main__":
    main()
