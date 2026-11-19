#!/usr/bin/env python3
"""Round 3 (reviewer 4a-2): split the gain over Renewables.ninja into R->M1 (reanalysis + simple chain) and M1->M4 (methodology). Median per rung, median paired difference and win count, clean variant, from results/quantitative/scores.csv.
Run on l2 from the repo root. Output: results/round3/attribution_rn_m1_m4.csv"""
import pandas as pd, numpy as np
sc = pd.read_csv("results/quantitative/scores.csv", dtype={"park_id": str})
sc = sc[sc.variant == "clean"].copy()
sc["abs_er_dev"] = (sc.energy_ratio - 1).abs()
rn = ["00198_1","00198_2","00282","01303_2","02483","04745","05347","05426","07374"]
rows = []
for parks, lab in [(rn, "9 RN-Parks"), (sorted(sc.park_id.unique()), "13 Parks")]:
    s = sc[sc.park_id.isin(parks)]
    for m in ["r2", "abs_er_dev"]:
        p = s.pivot_table(index="park_id", columns="experiment", values=m)
        r = {"Parks": lab, "Metrik": m}
        for e, k in [("M0","R"),("M1noage","M1"),("M2noage","M2"),("M2all","M3"),("M5all_PL","M4")]:
            if e in p: r[k] = round(p[e].median(), 3)
        if "M0" in p:
            r["R->M1 Median Diff"] = round((p["M1noage"]-p["M0"]).median(), 3)
            r["R->M1 Wins"] = int(((p["M1noage"]-p["M0"]) * (1 if m=="r2" else -1) > 0).sum())
        r["M1->M4 Median Diff"] = round((p["M5all_PL"]-p["M1noage"]).median(), 3)
        r["M1->M4 Wins"] = int(((p["M5all_PL"]-p["M1noage"]) * (1 if m=="r2" else -1) > 0).sum())
        if "M0" in p:
            r["R->M4 Median Diff"] = round((p["M5all_PL"]-p["M0"]).median(), 3)
        rows.append(r)
pd.set_option("display.width", 250)
out = pd.DataFrame(rows); print(out.to_string(index=False))
out.to_csv("results/round3/attribution_rn_m1_m4.csv", index=False)
