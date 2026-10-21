#!/usr/bin/env python
"""Validation of the parks_pv_v1 release: capacity factors by region, era and class against literature
(Germany open-ground PV ≈ 1000–1150 kWh/kWp DC/a), monthly CF profile, clipping and snow shares, specific
yield vs latitude, distributions against the Trianel meters (pv1..pv7, anonymised) and a reproducibility
check (one park regenerated, release SHA256 compared). Figures -> reports/figs_pv_park_synthesis_v1/.

    synthre/bin/python scripts/pv/validate_parks_pv_v1.py [--repro-park ID]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "pv"))
from parks_pv import paths  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
FIGS = os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_park_synthesis_v1")
COL = {"flat": "#2a78d6", "rolling": "#1baf7a", "hilly": "#eb6834"}


def load_release():
    parks = pd.read_csv(os.path.join(paths.RELEASE_DIR, "parks.csv"))
    series = {}
    for pid in parks.park_id:
        d = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{pid}.parquet"), columns=["power_park", "power_park_free", "ghi", "clipped", "snow_factor"])
        series[pid] = d
    return parks, series


def trianel_reference(start, end) -> pd.DataFrame:
    """Specific yield of the metered Trianel parks over the same window (anonymised pv1..pv7); empty if unavailable."""
    try:
        import trianel_common as T
        m = T.load_masterdata()
        parks = T.parks_table(m)
        meter = T.load_meter_hourly(parks)
        meter = meter[(meter.index >= start) & (meter.index < end)]
        rows = []
        for pv in meter.columns:
            kw = float(parks.loc[pv, "kw_master"])
            s = meter[pv].dropna()
            if len(s) < 24 * 200:
                continue
            rows.append({"pv": pv, "kwh_per_kw_a": float(s.mean() / 1000.0 / kw * 8760), "hours": int(len(s))})
        return pd.DataFrame(rows)
    except Exception as e:  # noqa: BLE001
        print("no Trianel reference:", str(e)[:80])
        return pd.DataFrame(columns=["pv", "kwh_per_kw_a", "hours"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-park", default=None)
    ap.add_argument("--out", default=FIGS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parks, series = load_release()
    rows = []
    for r in parks.itertuples():
        d = series[r.park_id]
        full_years = d[(d.index >= "2023-08-01") & (d.index < "2026-08-01")]      # 3 full years
        dt_h = pd.Series(full_years.index).diff().median() / pd.Timedelta("1h")   # step length in h, 0.5 for the 30-min release
        e_kwh = full_years["power_park"].sum() * dt_h / 1000.0
        rows.append({"park_id": r.park_id, "client": r.client_id, "region": r.region, "cohort": r.cohort, "cls": r.archetype,
                     "terrain_cls": r.terrain_cls, "lat": r.latitude, "kwp_dc": r.capacity_kwp_dc, "ac_kw": r.capacity_kw, "dc_ac": r.dc_ac_ratio,
                     "kwh_per_kwp_a": e_kwh / r.capacity_kwp_dc / 3.0, "cf_ac": full_years["power_park"].mean() / (r.capacity_kw * 1000.0),
                     "clipped_share_day": float(full_years.loc[full_years.ghi > 10, "clipped"].mean()),
                     "snow_loss": float(1 - full_years["power_park"].sum() / full_years["power_park_free"].sum()),
                     "ghi_mean": float(full_years["ghi"].mean())})
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(a.out, "parks_validation.csv"), index=False)
    summary = {
        "yield_kwh_per_kwp_a": v.kwh_per_kwp_a.describe().round(1).to_dict(),
        "cf_ac": v.cf_ac.describe().round(4).to_dict(),
        "by_region": v.groupby("region").kwh_per_kwp_a.agg(["mean", "min", "max"]).round(0).to_dict("index"),
        "by_cohort": v.groupby("cohort").kwh_per_kwp_a.agg(["mean", "count"]).round(0).to_dict("index"),
        "by_class": v.groupby("cls").kwh_per_kwp_a.agg(["mean", "count"]).round(0).to_dict("index"),
        "by_terrain": v.groupby("terrain_cls").kwh_per_kwp_a.agg(["mean", "count"]).round(0).to_dict("index"),
        "clipped_share_day": v.clipped_share_day.describe().round(4).to_dict(),
        "snow_loss": v.snow_loss.describe().round(4).to_dict(),
        "yield_vs_lat_corr": float(np.corrcoef(v.lat, v.kwh_per_kwp_a)[0, 1]),
        "yield_vs_ghi_corr": float(np.corrcoef(v.ghi_mean, v.kwh_per_kwp_a)[0, 1]),
    }
    ref = trianel_reference("2023-08-01", "2024-06-01")
    if len(ref):
        summary["trianel_reference_kwh_per_kw_master_a"] = ref.set_index("pv").kwh_per_kw_a.round(0).to_dict()
        # same window for the synthetic parks
        w = []
        for r in parks.itertuples():
            d = series[r.park_id]; d = d[(d.index >= "2023-08-01") & (d.index < "2024-06-01")]
            w.append(d.power_park.mean() / 1000.0 / r.capacity_kwp_dc * 8760)
        summary["synthetic_same_window_kwh_per_kwp_a"] = pd.Series(w).describe().round(0).to_dict()
    # monthly profile
    mon = pd.concat({pid: s["power_park"].resample("MS").mean() / (parks.set_index("park_id").loc[pid, "capacity_kw"] * 1000) for pid, s in series.items()}, axis=1)
    summary["monthly_cf_ac_mean"] = mon.mean(axis=1).groupby(mon.index.month).mean().round(3).to_dict()
    if a.repro_park:
        from parks_pv.drivers import sha256_frame
        man = json.load(open(os.path.join(paths.RELEASE_DIR, f"manifest_{a.repro_park}.json")))
        rel = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{a.repro_park}.parquet"))
        summary["repro_sha_matches_manifest"] = sha256_frame(rel) == man["release_sha256"]
    json.dump(summary, open(os.path.join(a.out, "validation_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for t, g in v.groupby("terrain_cls"):
        axes[0].scatter(g.lat, g.kwh_per_kwp_a, s=14, color=COL.get(t, "#888"), label=t)
    axes[0].set_xlabel("latitude"); axes[0].set_ylabel("kWh/kWp/a (DC)"); axes[0].legend(fontsize=7); axes[0].set_title("specific yield vs latitude", loc="left")
    axes[1].hist(v.kwh_per_kwp_a, bins=20, color="#2a78d6"); axes[1].set_xlabel("kWh/kWp/a"); axes[1].set_title("distribution (90 parks)", loc="left")
    if len(ref):
        for x in ref.kwh_per_kw_a:
            axes[1].axvline(x, color="#eb6834", lw=1)
    axes[2].plot(mon.mean(axis=1).index, mon.mean(axis=1).values, color="#2a78d6"); axes[2].fill_between(mon.index, mon.quantile(0.1, axis=1), mon.quantile(0.9, axis=1), color="#2a78d6", alpha=0.2)
    axes[2].set_ylabel("CF (AC)"); axes[2].set_title("monthly CF, mean and 10–90 %", loc="left")
    fig.savefig(os.path.join(a.out, "fig_validation.png"))


if __name__ == "__main__":
    main()
