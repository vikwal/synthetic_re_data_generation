"""Score regenerated series on frozen masks, including the existing price screen."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/round2')]
from round2 import evaluation,meterdata
from error_decomposition import decompose
BASE=Path('/mnt/nvme2/synthetic/physics_20261113')
DATA=BASE/'wind/round2';OUT=BASE/'analyses/synthesis';OUT.mkdir(parents=True,exist_ok=False)
MASK=Path('/mnt/nvme2/synthetic/raw/round2/masks')
rows=[];dec=[];hist=[];curves=[];density=[];hashes={}
prices=pd.read_csv(ROOT/'data/round2/prices_delu.csv',index_col=0,parse_dates=True).price_eur_mwh
for directory in sorted(DATA.iterdir()):
    exp=directory.name
    if exp=='SITE_v2':continue
    if not (directory/'COMPLETE.json').exists():raise RuntimeError('Incomplete generation '+exp)
    for path in sorted(directory.glob('synth_*.csv')):
        pid=path.stem[len('synth_'):];rated=meterdata.rated_power_w(pid)
        df=pd.read_csv(path,sep=';',index_col=0,parse_dates=True)
        if exp=='M5all_PL':
            rho=df.filter(regex=r'^density_t\d+$').mean(axis=1)
            density.append(dict(park_id=pid,mean=rho.mean(),p05=rho.quantile(.05),p95=rho.quantile(.95),DJF=rho[rho.index.month.isin([12,1,2])].mean(),JJA=rho[rho.index.month.isin([6,7,8])].mean()))
        df=df.loc['2023-06-01':'2024-06-01'];syn=df.power_park
        meas=meterdata.load_park_power(pid,('2023-06-01','2024-06-01'))
        maskpath=MASK/exp/f'mask_{pid}.parquet';keep=pd.read_parquet(maskpath)['keep']
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(maskpath)]=hashlib.sha256(maskpath.read_bytes()).hexdigest()
        clean=meas[keep.reindex(meas.index).fillna(False)]
        for variant,m,pr in [('raw',meas,None),('clean',clean,None),('clean_exkl_curt',clean,prices)]:
            res=evaluation.evaluate(m,syn,p_rated=rated,prices=pr)
            res.pop('acf_meas',None);res.pop('acf_synth',None)
            if variant=='clean_exkl_curt':
                res={k.replace('_excl_curt',''):v for k,v in res.items() if (k.endswith('_excl_curt') and not k.startswith('acf')) or k in ('n_hours','window','curtailed_share')}
            rows.append(dict(experiment=exp,park_id=pid,variant=variant,outlier_share=float(1-keep.mean()),**res))
        pair=pd.concat([meas.rename('meas'),syn.rename('synth')],axis=1).dropna();pair['keep']=keep.reindex(pair.index).fillna(False).astype(bool)
        cl=pair[pair.keep];terms=decompose(cl.meas.values/rated,cl.synth.values/rated);assert abs(terms['closure_resid'])<1e-12
        dec.append(dict(experiment=exp,park_id=pid,n_hours=len(cl),**terms))
        if exp=='M5all_PL':
            bins=np.linspace(-1,1,101);counts,_=np.histogram((cl.synth-cl.meas)/rated,bins)
            hist.extend(dict(experiment=exp,park_id=pid,left=bins[i],right=bins[i+1],count=int(n)) for i,n in enumerate(counts))
            if pid in ['00198_1','00282']:
                pair['wind']=df.filter(regex=r'^wind_speed_t\d+$').mean(axis=1).reindex(pair.index)
                pair[['meas','synth']]/=rated;pair['experiment']=exp;pair['park_id']=pid;curves.append(pair.reset_index(drop=True))
        print(exp,pid,'scored',flush=True)
for name,data in [('scores',rows),('decomposition',dec),('error_hist',hist),('density_parks',density)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
pd.concat(curves).to_csv(OUT/'power_curves.csv',index=False)
(OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
print('COMPLETE',len(rows),'scores',flush=True)
