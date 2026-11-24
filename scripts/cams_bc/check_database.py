"""Exercise the production database path and optionally compare model outputs between deployments."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cams_bc import weather_db,data
from pv import drivers as pv_drivers
from cams_bc.neighbour_db import load_neighbours
from apply_parks import load_models,predict,era5_solar_30min

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models-dir',required=True);p.add_argument('--fixture',required=True)
p.add_argument('--output',required=True);p.add_argument('--reference')
a=p.parse_args();conn=weather_db.connect()
points=weather_db.load_grid_points(conn);ids,_=weather_db.nearest_cells(points,[50.],[10.])
met=weather_db.fetch_cell(conn,int(ids[0]),'2026-08-01 00:00','2026-08-01 23:00')
solar=era5_solar_30min(conn,50.,10.,met.index)
pv_solar,_=pv_drivers.era5_solar(conn,50.,10.,'2026-08-01','2026-08-02')
pv_met,_=pv_drivers.era5_met(conn,50.,10.,'2026-08-01','2026-08-02')
if len(met)!=24 or solar.empty or len(pv_solar)!=24 or len(pv_met)!=24:raise RuntimeError('Weather/PV DB sample failed')
f=pd.read_parquet(a.fixture);nb,coords=load_neighbours(conn,f.index)
sid='00183';station=coords.loc[sid]
features=data.neighbour_features(station.lat,station.lon,f.index,nb,coords,exclude=[sid])
meta,models=load_models(a.models_dir,'cpu')
cols=[col for col in meta['columns'] if col.startswith('nb')]
f[cols]=features[cols];pred=predict(meta,models,f)
np.savez(a.output,ghi=pred['ghi'],dhi=pred['dhi'],neighbour_features=f[cols].to_numpy(np.float32))
maxdiff=0.
if a.reference:
    ref=np.load(a.reference)
    np.testing.assert_allclose(f[cols].to_numpy(np.float32),ref['neighbour_features'],rtol=1e-6,atol=2e-6,equal_nan=True)
    for v in ['ghi','dhi']:
        np.testing.assert_allclose(pred[v],ref[v],rtol=1e-6,atol=1e-5)
        maxdiff=max(maxdiff,float(np.max(np.abs(pred[v]-ref[v]))))
print(json.dumps(dict(status='PASS',stations=len(coords),met_points=len(points),met_sample_rows=len(met),solar_rows=len(solar),prediction_rows=len(f),reference_checked=bool(a.reference),max_prediction_difference=maxdiff)))
