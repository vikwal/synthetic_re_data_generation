"""Fetch open state geodata (orthophoto, laser surface model / point cloud, terrain model) for the Trianel parks
outside Bavaria, plus the Bavarian laser point clouds for the Bavarian parks, into
${DATA_ROOT}/synthetic/solar/trianel_benchmark/inputs/geodata/<state>/. Coordinates stay private (trianel_common)."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from pv import geodata_states as GS
import trianel_common as TC

OUT = os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic/solar/trianel_benchmark/inputs/geodata")
parks = TC.parks_table(TC.load_masterdata())
mc = pd.read_csv(os.path.join(os.path.dirname(OUT), "mastr", "mastr_candidates.csv"))
units = pd.read_parquet(os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic/solar/mastr/solar_units_20250625.parquet"), columns=["EinheitMastrNummer", "Bundesland"])
j = mc.merge(units, on="EinheitMastrNummer", how="left")
info = j.groupby("pv").agg(state=("Bundesland", lambda x: x.dropna().mode().iloc[0]), lat=("Breitengrad", "median"), lon=("Laengengrad", "median"), year=("Inbetriebnahmedatum", lambda x: str(x.min())[:4]))
if parks is not None:   # prefer the park coordinates of the benchmark config over register medians
    for pv, p in parks.iterrows():
        if pv in info.index and "lat" in p.index: info.loc[pv, ["lat", "lon"]] = [float(p["lat"]), float(p["lon"])]
summary = {}
for pv, r in info.iterrows():
    t0 = time.time(); res = {}
    try:
        if r.state == "Niedersachsen": res = GS.fetch_ni(r.lat, r.lon, OUT, products=("dop20", "dom1", "dgm1"), margin_m=400, after=f"{int(r.year) + 1}-01-01")
        elif r.state == "Brandenburg": res = GS.fetch_bb(r.lat, r.lon, OUT, products=("dop20", "als", "dgm1", "bdom"), margin_m=400)
        elif r.state == "Mecklenburg-Vorpommern": res = GS.fetch_mv(r.lat, r.lon, OUT, products=("dop20", "dom1", "dgm1"), margin_m=400)
        elif r.state == "Bayern": res = GS.fetch_by_laser(r.lat, r.lon, OUT, margin_m=400)
        else: res = {"skipped": r.state}
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}"}
    summary[pv] = {"state": r.state, "year": r.year, "seconds": round(time.time() - t0), **{k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}}
    print(pv, summary[pv], flush=True)
json.dump(summary, open(os.path.join(OUT, "fetch_states_summary.json"), "w"), indent=1, default=str)
