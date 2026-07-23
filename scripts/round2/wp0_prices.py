#!/usr/bin/env python3
"""WP0.8 — DE-LU day-ahead prices (curtailment proxy, WP1.3).

Source: Fraunhofer energy-charts API (no auth). Fallback: SMARD.de bulk API.
Output: data/round2/prices_delu.csv (UTC hourly, EUR/MWh) and a copy under
/mnt/nvme2/synthetic/raw/round2/prices/.
"""

import json
import os
import urllib.request

import pandas as pd

START, END = "2023-06-01", "2024-06-30"
OUT_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "data", "round2", "prices_delu.csv")
OUT_RAW = "/mnt/nvme2/synthetic/raw/round2/prices/prices_delu.csv"


def from_energy_charts() -> pd.DataFrame:
    url = f"https://api.energy-charts.info/price?bzn=DE-LU&start={START}&end={END}"
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.load(r)
    ts = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({"timestamp": ts, "price_eur_mwh": payload["price"]})
    assert payload.get("unit", "EUR/MWh").lower().startswith("eur"), payload.get("unit")
    return df.set_index("timestamp").sort_index()


def from_smard() -> pd.DataFrame:
    # SMARD chart_data API: filter 4169 = day-ahead price DE-LU, weekly hour files
    base = "https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{ts}.json"
    idx_url = "https://www.smard.de/app/chart_data/4169/DE/index_hour.json"
    with urllib.request.urlopen(idx_url, timeout=60) as r:
        stamps = json.load(r)["timestamps"]
    lo = pd.Timestamp(START, tz="UTC").value // 10**6
    hi = pd.Timestamp(END, tz="UTC").value // 10**6 + 7 * 86400_000
    rows = []
    for ts in [s for s in stamps if lo - 7 * 86400_000 <= s <= hi]:
        with urllib.request.urlopen(base.format(ts=ts), timeout=60) as r:
            rows += json.load(r)["series"]
    df = pd.DataFrame(rows, columns=["ms", "price_eur_mwh"]).dropna()
    df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    return df.set_index("timestamp")[["price_eur_mwh"]].sort_index()


def main():
    try:
        df = from_energy_charts()
        src = "energy-charts"
    except Exception as e:
        print(f"energy-charts failed ({e}), falling back to SMARD")
        df = from_smard()
        src = "smard"
    df = df.loc[START:END]
    df = df[~df.index.duplicated(keep="first")]
    n_neg = int((df["price_eur_mwh"] <= 0).sum())
    print(f"source={src} rows={len(df)} {df.index.min()} .. {df.index.max()} hours<=0: {n_neg}")
    assert len(df) > 8000, "suspiciously few hours"
    for out in (OUT_REPO, OUT_RAW):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        df.to_csv(out)
    print("written:", OUT_REPO, "and", OUT_RAW)


if __name__ == "__main__":
    main()
