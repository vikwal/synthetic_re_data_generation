"""Repeat published sensitivity designs with the shared corrected physics."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/round2')]
from round2 import chain,aging,meterdata,parkinfo
BASE=Path('/mnt/nvme2/synthetic/physics_20261113/analyses')

def screen():
    import wp6_morris as m,wp6_sobol as s
    # Only redirect output locations. Chain still reads its unchanged inputs.
    dest=BASE/'sensitivity';(dest/'results/round2').mkdir(parents=True,exist_ok=True);(dest/'figs/round2').mkdir(parents=True,exist_ok=True)
    m.REPO=str(dest);m.R3=True;sys.argv=['morris','--r3','--factors','9'];m.main()
    s.REPO=str(dest);sys.argv=['sobol','--r3'];s.main()

def placebo():
    dest=BASE/'placebo';dest.mkdir(parents=True,exist_ok=False)
    parks=sorted(meterdata.load_mapping().park_id);pcs={pid:chain.ParkCache(pid) for pid in parks}
    dates={pid:parkinfo.commissioning_date(pid) for pid in parks}
    for pc in pcs.values():pc.force_all_aging()
    models={'M3':('M2all',dict(shear='power_law',wake_enabled=False)),
            'M4':('M5all_PL',dict(shear='power_law',wake_enabled=True)),
            'S3':('M5all',dict(shear='most',wake_enabled=True))}
    rng=np.random.default_rng(25);perms=[]
    for i in range(1000):
        perm=rng.permutation(len(parks))
        while np.array_equal(perm,np.arange(len(parks))):perm=rng.permutation(len(parks))
        perms.append(perm)
    nullrows=[];prows=[];summ=[]
    for model,(maskexp,extra) in models.items():
        er0=[];gain=np.empty((13,13));energies=[];noage_energy=0
        th=dict(density='iec',aging_model='weibull',aging_lambda=54.5,aging_kappa=2.,**extra)
        for j,pid in enumerate(parks):
            pc=pcs[pid];original=pc.ages.copy()
            mask=pd.read_parquet(Path('/mnt/nvme2/synthetic/raw/round2/masks')/maskexp/f'mask_{pid}.parquet')['keep'].reindex(pc.index).fillna(False).to_numpy(bool)
            ok=mask&np.isfinite(pc.meas);den=pc.meas[ok].sum()
            p0=pc.power_series({**th,'aging_model':'off'});e0=p0[ok].sum()/den;er0.append(e0);noage_energy+=p0[ok].sum()
            for k,other in enumerate(parks):
                pc.ages=np.maximum((pc.index-pd.to_datetime(dates[other],utc=True)).days/365.25,0)
                power=pc.power_series(th);e=power[ok].sum()/den;gain[j,k]=abs(e0-1)-abs(e-1)
                if j==k:energies.append(e*den)
            pc.ages=original
        scale=sum(energies)/noage_energy;true=gain.diagonal();obs=true.mean()
        for i,perm in enumerate(perms):
            vals=gain[np.arange(13),perm];nullrows.append(dict(model=model,draw=i,T_mean=vals.mean(),T_median=np.median(vals)))
        nullrows.append(dict(model=model,draw='true',T_mean=obs,T_median=np.median(true)))
        ns=np.array([gain[np.arange(13),p].mean() for p in perms]);pval=(1+np.count_nonzero(ns>=obs))/1001
        for j,pid in enumerate(parks):prows.append(dict(model=model,park_id=pid,er_noage=er0[j],gain_true=true[j],gain_uniform=abs(er0[j]-1)-abs(er0[j]*scale-1),uniform_scale=scale))
        summ.append(dict(model=model,T_mean=obs,p_perm=pval,uniform_scale=scale))
        np.savez_compressed(dest/f'permutation_matrix_{model}.npz',gain=gain,permutations=perms)
        print('Permutation',model,'p',pval,flush=True)
    for name,rows in [('placebo_null_iec',nullrows),('placebo_parks_iec',prows),('placebo_summary_iec',summ)]:pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['screen','placebo','anchor']);a=p.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    if a.action=='screen':screen()
    elif a.action=='placebo':placebo()
    else:
        import fixed_anchor_sensitivity as f
        sys.argv=['anchor','--out',str(BASE/'fixed_anchor')];f.main()
    print('COMPLETE',a.action,flush=True)
