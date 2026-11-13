"""Compare production and sensitivity callers on real prepared inputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()))
import generate_wind as gw
from round2 import chain,aging,correction,meterdata
from round2.paper_style import PSEUDONYM
from utils.tools import knn_imputer
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
curves=pd.read_csv('power_curves/turbine_power.csv',sep=';',decimal=',',index_col=0)
curves=curves.loc[:,~curves.columns.duplicated()]*1000
cases=[{},dict(shear='most'),dict(correction='off',aging_model='off',wake_enabled=False),
       dict(wind_level_factor=.95,power_curve_scale=1.03,wake_k=.0625,aging_kappa=1.5,aging_lambda=65.),
       dict(shear='most',z0_scale=2.,wind_level_factor=1.05,power_curve_scale=.97,wake_k=.0875),
       dict(density='static_1225',aging_model='off'),dict(density='v1_mixed'),dict(aging_model='const'),
       dict(correction='wind10_only',shear='most',z0_scale=.5,aging_model='weibull_step')]
rows=[]
for pid in sorted(meterdata.load_mapping().park_id):
    pc=chain.ParkCache(pid);pc.force_all_aging();params=chain._load_park_config(pid)['params'];assert params['noise']==0
    path=Path(chain.ERA5_DIR_V2)/f'Station_{pid[:5]}.csv'
    raw=pd.read_csv(path);raw.index=pd.to_datetime(raw.pop('timestamp'),utc=True)
    prepared=knn_imputer(raw,n_neighbors=5).loc[pc.index]
    specs={t:dict(diameter=s['diameter'],cut_in=s['cut_in'],cut_out=s['cut_out'],rated=s['rated_ws']) for t,s in pc.specs.items()}
    for case,extra in enumerate(cases):
        th={**chain.DEFAULT_THETA,**extra};fast=pc.power_series(th)
        model=th['aging_model'];df=np.ones(len(pc.index)) if model in ['off','none'] else aging.get_degradation_vector(pc.index,model=model,commissioning_date=pc.commissioning_date,lam=th['aging_lambda'],kappa=th['aging_kappa'])[0]
        if model=='const':
            df=aging.get_degradation_vector(pc.full_index,model=model,commissioning_date=pc.commissioning_date)[0][pc.full_index.get_indexer(pc.index)]
        r2={**gw.R2_DEFAULTS,**th,'density':'iec' if th['density']=='dynamic' else th['density'],'wake':dict(enabled=th['wake_enabled'],k=th['wake_k'])}
        context=dict(q_era5=pc.q_era5,q_target=pc.q_target,z0=pc.z0*th['z0_scale'])
        data=prepared.copy();names=[]
        for j,(t,h) in enumerate(zip(pc.turbines,pc.hub_heights),1):
            data=gw.gen_full_dataframe(curves,t,params,data,h,specs,pc.rated_powers[j-1],df,f'_t{j}',r2,context)
            names.append(f'power_t{j}')
        full=data[names].sum(axis=1).values
        if th['wake_enabled']:full*=gw.load_wake_factor(r2,pid,pc.index).values
        normalized=float(np.max(abs(full-fast))/meterdata.rated_power_w(pid))
        rows.append(dict(park=PSEUDONYM[pid],case=case,n_hours=len(full),max_normalized_difference=normalized))
        np.testing.assert_allclose(full,fast,rtol=1e-12,atol=1e-5,err_msg=f'{pid}, case {case}')
    print(PSEUDONYM[pid],'PASS',flush=True)
pd.DataFrame(rows).to_csv(out/'equivalence.csv',index=False)
(out/'summary.json').write_text(json.dumps(dict(n_comparisons=len(rows),max_normalized_difference=max(r['max_normalized_difference'] for r in rows),cases=cases),indent=2))
print('ALL PASS',len(rows),flush=True)
