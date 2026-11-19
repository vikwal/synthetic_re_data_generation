#!/usr/bin/env python
"""Efficiency / technology heterogeneity between parks: what the module (CEC single diode),
inverter (Sandia), DC/AC ratio, aging era and loss factors change in (a) annual energy and
(b) the hourly SHAPE of normalised power (R2 against a reference park on the same driver).
Driver: CAMS of one Trianel park point (anonymised); outputs tables + one figure."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv import aging, chain, config as pvconfig, drivers, inverter, modules, timebase  # noqa: E402
from parks import era5_db  # noqa: E402
import trianel_common as tc  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})


def r2(a, b):
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--pv", default="pv2")
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    args = ap.parse_args()
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    p = parks.loc[args.pv]
    conn = era5_db.connect()
    drvs, _ = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, args.pv)
    drv = drvs["cams"]
    base = {"park_id": "x", "latitude": p["lat"], "longitude": p["lon"], "altitude": p["alt"], "kwp_dc": 10000.0, "dc_ac": 1.15,
            "arrays": [{"tilt": 25.0, "azimuth": 180.0, "share": 1.0}], "transposition": {"albedo": "era5", "iam": "physical"},
            "losses": {"dc": 0.02, "ac": 0.01, "availability": 1.0}, "shading": {"horizon": False}}

    # 1. module library spread: temperature coefficient and low-light efficiency per era proxy
    rows = []
    for y0, y1, label, names, g in modules.ERA_TABLE:
        for n in names:
            try:
                mod = modules.module_params(n)
            except KeyError:
                continue
            ll = modules.low_light_curve(mod)
            rows.append({"era": label, "module": n, "stc_w": mod["STC"], "eff_stc": mod["STC"] / float(mod["A_c"]) / 1000.0 if pd.notna(mod.get("A_c")) else np.nan,
                         "gamma_1perK": modules.temperature_coefficient(mod), "rel_eff_200": ll.loc[200], "rel_eff_100": ll.loc[100], "library": mod.get("library", "")})
    lib = pd.DataFrame(rows)
    lib.to_csv(os.path.join(args.out, "module_library_spread.csv"), index=False)

    # 2. chain variants on the same driver: energy (CF) and shape (R2 vs reference)
    variants = {"ref: PVWatts γ −0.35 %/K, dc_ac 1.15": dict(module={"model": "pvwatts", "gamma_pdc": -0.0035, "year": 2016}, inverter={"model": "pvwatts", "eta_inv_nom": 0.97})}
    for y0, y1, label, names, g in modules.ERA_TABLE:
        for n in names[:2]:
            try:
                modules.module_params(n)
            except KeyError:
                continue
            variants[f"CEC {label}: {n.split(' ')[0]} {int(modules.module_params(n)['STC'])} W"] = dict(module={"model": "cec", "name": n, "year": (y0 + y1) // 2}, inverter={"model": "sandia", "unit_kw": None})
    for r_ in (1.0, 1.1, 1.25, 1.4):
        variants[f"PVWatts dc_ac {r_:.2f}"] = dict(module={"model": "pvwatts", "gamma_pdc": -0.0035, "year": 2016}, inverter={"model": "pvwatts", "eta_inv_nom": 0.97}, dc_ac=r_)
    for yr, era in ((2008, "multi_si_early"), (2012, "multi_si"), (2019, "mono_perc"), (2023, "topcon_bifacial")):
        variants[f"aging {era} since {yr}"] = dict(module={"model": "pvwatts", "gamma_pdc": -0.0035, "year": yr}, inverter={"model": "pvwatts", "eta_inv_nom": 0.97},
                                                  aging={"enabled": True, "commissioning_date": f"{yr}-07-01", "era": era})
    for u0, u1, lab in ((25.0, 6.84, "Faiman open rack"), (20.0, 4.0, "Faiman ventilated less"), (30.9, 0.0, "Faiman no wind term")):
        variants[f"temp model {lab}"] = dict(module={"model": "pvwatts", "gamma_pdc": -0.0035, "year": 2016}, inverter={"model": "pvwatts", "eta_inv_nom": 0.97}, temperature={"model": "faiman", "u0": u0, "u1": u1})
    variants["losses dc 8 %, ac 2 %, avail 0.97"] = dict(module={"model": "pvwatts", "gamma_pdc": -0.0035, "year": 2016}, inverter={"model": "pvwatts", "eta_inv_nom": 0.97}, losses={"dc": 0.08, "ac": 0.02, "availability": 0.97})
    out, ref = [], None
    series = {}
    for name, over in variants.items():
        cfg = pvconfig.normalise({**base, **over})
        r = chain.run_park(cfg, drv)
        s = r["p_ac_w"] / (cfg["kwp_dc"] * 1000.0)
        series[name] = s
        if ref is None:
            ref = s
        norm = s / s.quantile(0.999)
        refn = ref / ref.quantile(0.999)
        day = r["zenith"] < 90
        out.append({"variant": name, "cf": s.mean(), "cf_rel_ref": s.mean() / ref.mean(), "r2_vs_ref_day": r2(ref[day], s[day]),
                    "r2_vs_ref_peaknorm": r2(refn[day], norm[day]), "clipped_hours": int(r["clipped"].sum()),
                    "summer_noon_rel": (s["2023-08"].groupby(s["2023-08"].index.hour).mean().max()) / (ref["2023-08"].groupby(ref["2023-08"].index.hour).mean().max()),
                    "winter_rel": s["2023-12":"2024-01"].mean() / ref["2023-12":"2024-01"].mean()})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(args.out, "chain_variant_spread.csv"), index=False)
    print(lib.round(3).to_string())
    print(df.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df["cf_rel_ref"], 1 - df["r2_vs_ref_peaknorm"], color="#2a78d6", s=25)
    for _, q in df.iterrows():
        ax.annotate(q["variant"], (q["cf_rel_ref"], 1 - q["r2_vs_ref_peaknorm"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Jahresenergie relativ zur Referenz (Niveau-Effekt)")
    ax.set_ylabel("1 − R² gegen Referenz nach Peak-Normierung (Form-Effekt)")
    ax.set_yscale("log")
    ax.set_title("Was Technik-Parameter am Niveau und an der Form der Leistungskurve ändern (gleicher Antrieb)", loc="left", fontsize=8.5)
    fig.savefig(os.path.join(args.out, "fig_chain_variant_spread.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
