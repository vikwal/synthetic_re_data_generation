#!/usr/bin/env python
"""Calibrate the orientation plausibility check on the synthetic parks (truth known): estimate tilt/azimuth from
power_park with RAW CAMS as the estimator's driver (realistic driver mismatch), for several history lengths, and
score the error, the confidence-region coverage and the detection of the deliberate register deviations.

    synthre/bin/python scripts/pv/plausibility_calibration.py [--windows 14 30 90 365] [--workers 8]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import paths  # noqa: E402
from pv import angle_diagnostics, angles, arrays as arr, plausibility as PL  # noqa: E402

WINDOW_STARTS = {"summer": "2024-06-01", "winter": "2023-12-01", "spring": "2024-03-01"}


def one(task):
    pid, windows, reg_or, reg_tilt, true_tilt, true_az, deviation, bank_kind, row_axis = task
    d = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{pid}.parquet"))
    man = json.load(open(os.path.join(paths.RELEASE_DIR, f"manifest_{pid}.json")))
    cfg = man["config"]
    lat, lon, alt = cfg["latitude"], cfg["longitude"], cfg["altitude"]
    drv = pd.DataFrame({"ghi": d["ghi_cams_raw"], "dhi": np.minimum(d["dhi_cams_raw"], d["ghi_cams_raw"]),
                        "temp_air": d["temp_air"], "wind_speed": d["wind_speed"], "albedo": 0.2})
    exclude = (d["clipped"] > 0) | (d["snow_depth_cm"] > 0.5) | (d["imputed_share"] > 0)
    prior = PL.azimuth_prior_from_rows(row_axis) if row_axis is not None and np.isfinite(row_axis) else None
    az_prior, prior_ew = (None, False) if prior is None else ((prior["azimuths"][0], True) if prior["east_west"] else (prior["azimuth"], False))
    azs = list(PL.AZIMUTHS) if az_prior is None else ([az_prior, (az_prior + 180.0) % 360.0] if prior_ew else [az_prior])
    # the bank once on the whole span (the estimator slices it per window); raw CAMS as the driver, no horizon
    bank = PL.chain_bank(cfg, drv, PL.TILTS, azs) if bank_kind == "chain" else angles.poa_bank(drv, lat, lon, alt, PL.TILTS, azs)
    rows = []
    for season, start in WINDOW_STARTS.items():
        for w in windows:
            s = pd.Timestamp(start, tz="UTC")
            sl = slice(s, s + pd.Timedelta(days=w))
            try:
                res = PL.assess(d["power_park"].loc[sl], drv.loc[sl], lat, lon, alt, exclude.loc[sl], reg_or, reg_tilt, n_boot=30,
                                bank={k: v.loc[sl] for k, v in bank.items()}, azimuth_prior=az_prior, prior_east_west=prior_ew)
            except Exception as e:  # noqa: BLE001
                rows.append({"park_id": pid, "season": season, "window_days": w, "error": str(e)[:80]})
                continue
            g, m, p, h = res["grid"], res["mixture"], res["plausibility"], res["east_west_model"]
            tg = float(round(g["tilt"] / 5) * 5) if np.isfinite(g["tilt"]) else 25.0
            diag = angle_diagnostics.run(d["power_park"].loc[sl], drv.loc[sl], lat, lon, alt, {k: v.loc[sl] for k, v in bank.items()}, res["mask"], tilt_guess=tg)
            c, n = diag["centroid_azimuth"], diag["noon_tilt"]
            daz = ((g["azimuth"] - true_az + 180) % 360) - 180
            rows.append({"park_id": pid, "season": season, "window_days": w, "n_clear_hours": res["n_clear_hours"], "n_days": g.get("n_days"),
                         "n_suspect": res["n_suspect"], "noon_zenith_spread": res["zenith_spread"], "tilt_identifiable": bool(res["tilt_identifiable"]),
                         "true_tilt": true_tilt, "true_az": true_az, "register_orientation": reg_or, "register_tilt_class": reg_tilt, "deviation": deviation,
                         "tilt_grid": g["tilt"], "az_grid": g["azimuth"], "r2_grid": g["r2"], "tilt_lo": g["tilt_range"][0], "tilt_hi": g["tilt_range"][1],
                         "az_lo": g["azimuth_range"][0], "az_hi": g["azimuth_range"][1],
                         "tilt_boot_lo": g.get("tilt_boot", (np.nan,) * 3)[0], "tilt_boot_hi": g.get("tilt_boot", (np.nan,) * 3)[2],
                         "az_boot_lo": g.get("azimuth_boot", (np.nan,) * 3)[0], "az_boot_hi": g.get("azimuth_boot", (np.nan,) * 3)[2],
                         "az_centroid": c["azimuth"], "az_centroid_iqr": c["iqr"], "tilt_noon": n.get("tilt"), "zenith_spread": n.get("zenith_spread"),
                         "mix_components": len(m["components"]), "mix_east_west": m["east_west"], "mix_east_west_nnls": m.get("east_west_nnls"), "mix_flat": m["flat"], "w_flat": m["w_flat"],
                         "ew_model_preferred": h.get("preferred"), "ew_model_tilt": h.get("tilt"), "ew_model_share_east": h.get("share_east"), "ew_model_delta_bic": h.get("delta_bic"),
                         "bank": bank_kind, "azimuth_prior": az_prior, "prior_east_west": prior_ew, "w_east": m["w_east"], "w_west": m["w_west"], "w_south": m["w_south"], "mix_tilt_eff": m["tilt_eff"], "mix_az_eff": m["azimuth_eff"],
                         "err_tilt_grid": g["tilt"] - true_tilt, "err_az_grid": daz, "err_az_centroid": ((c["azimuth"] - true_az + 180) % 360) - 180 if np.isfinite(c["azimuth"]) else np.nan,
                         "err_tilt_noon": (n.get("tilt") - true_tilt) if n.get("tilt") is not None and np.isfinite(n.get("tilt", np.nan)) else np.nan,
                         "tilt_in_region": g["tilt_range"][0] - 2.5 <= true_tilt <= g["tilt_range"][1] + 2.5,
                         "az_in_region": g["azimuth_range"][0] - 5 <= true_az <= g["azimuth_range"][1] + 5,
                         "register_az_ok": p["azimuth_ok"], "register_tilt_ok": p["tilt_ok"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[14, 30, 90, 365])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_park_synthesis_v1"))
    ap.add_argument("--summary-only", action="store_true", help="recompute the summary from the existing per-row CSV")
    ap.add_argument("--bank", choices=["poa", "chain"], default="poa", help="basis series: plain POA or the production chain (IAM, gamma, inter-row)")
    ap.add_argument("--azimuth-prior", choices=["none", "osm"], default="none", help="osm: fix the azimuth from the mapped row axis where present, estimate only the tilt")
    a = ap.parse_args()
    suffix = "" if (a.bank == "poa" and a.azimuth_prior == "none") else f"_{a.bank}_{a.azimuth_prior}"
    if a.summary_only:
        summarise(pd.read_csv(os.path.join(a.out, f"plausibility_calibration{suffix}.csv")), a.out, suffix)
        return
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv")).set_index("site_id")
    parks = pd.read_csv(os.path.join(paths.RELEASE_DIR, "parks.csv"))
    arrays = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    view = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_masterdata_view.csv")).set_index("park_id")
    tasks = []
    for r in parks.itertuples():
        a_ = arrays[arrays.park_id == r.park_id]
        t, az = arr.mean_angles(a_.tilt.values, a_.azimuth.values, a_.share.values)
        dev = r.register_deviation if isinstance(r.register_deviation, str) else ""
        ax = float(sites.loc[r.park_id, "row_axis_osm_deg"]) if a.azimuth_prior == "osm" and r.park_id in sites.index else None
        tasks.append((r.park_id, a.windows, view.loc[r.park_id, "orientation_class"], view.loc[r.park_id, "tilt_class"], float(t), float(az), dev, a.bank, ax))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(one, tasks):
            rows += out
            print(out[0]["park_id"], "done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"plausibility_calibration{suffix}.csv"), index=False)
    summarise(df, a.out, suffix)


def _rate(flag: pd.Series) -> float:
    """Share of False among the decided (non-null) flags."""
    f = flag.dropna()
    return float((~f.astype(bool)).mean()) if len(f) else np.nan


def summarise(df: pd.DataFrame, out: str, suffix: str = "") -> pd.DataFrame:
    ok = (df[df["error"].isna()] if "error" in df else df).copy()
    ok["deviation"] = ok["deviation"].fillna("")
    for c in ("register_az_ok", "register_tilt_ok"):            # object columns (None where undecidable) -> nullable bool
        ok[c] = ok[c].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    ok["tilt_identifiable"] = ok["tilt_identifiable"].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    g = ok.groupby(["season", "window_days"])
    summ = pd.DataFrame({
        "n": g.size(), "clear_hours_med": g.n_clear_hours.median(),
        "az_grid_mae": g.err_az_grid.apply(lambda s: s.abs().median()), "az_grid_p90": g.err_az_grid.apply(lambda s: s.abs().quantile(0.9)),
        "az_centroid_mae": g.err_az_centroid.apply(lambda s: s.abs().median()),
        "tilt_grid_mae": g.err_tilt_grid.apply(lambda s: s.abs().median()), "tilt_grid_p90": g.err_tilt_grid.apply(lambda s: s.abs().quantile(0.9)),
        "tilt_noon_mae": g.err_tilt_noon.apply(lambda s: s.abs().median()),
        "tilt_region_cov": g.tilt_in_region.mean(), "az_region_cov": g.az_in_region.mean(),
        "az_assessed_share": g.apply(lambda x: x.register_az_ok.notna().mean()),
        "tilt_assessed_share": g.tilt_identifiable.mean(),
        "tilt_grid_mae_assessed": g.apply(lambda x: x.err_tilt_grid[x.tilt_identifiable].abs().median()),
        "false_alarm_az": g.apply(lambda x: _rate(x.register_az_ok[x.deviation == ""])),
        "false_alarm_tilt": g.apply(lambda x: _rate(x.register_tilt_ok[(x.deviation == "") & x.tilt_identifiable])),
        "detect_deviation": g.apply(lambda x: _rate(x.register_az_ok[x.deviation != ""]) if (x.deviation != "").any() else np.nan),
        "ew_flag_rate_south": g.apply(lambda x: x.mix_east_west[(x.register_orientation == "Süd") & (x.deviation == "")].mean()),
        "ew_flag_rate_ew": g.apply(lambda x: x.mix_east_west[(x.register_orientation == "Ost-West") & (x.deviation == "")].mean()),
        "ew_model_rate_ew": g.apply(lambda x: x.ew_model_preferred[(x.register_orientation == "Ost-West") & (x.deviation == "")].astype(float).mean()) if "ew_model_preferred" in df.columns else np.nan,
    })
    summ = summ.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    summ.to_csv(os.path.join(out, f"plausibility_calibration_summary{suffix}.csv"))
    print(summ.round(3).to_string())
    return summ


if __name__ == "__main__":
    main()
