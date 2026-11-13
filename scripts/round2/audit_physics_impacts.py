"""Quantify corrected synthesis against frozen production series and masks.

Writes a new audit directory only; never overwrites data, models, or paper scores.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
sys.path.insert(0,str(Path.cwd()))
from round2 import chain,meterdata
from round2.paper_style import PSEUDONYM

BASE=Path('/mnt/nvme2/synthetic/wind/round2')
MASK=Path('/mnt/nvme2/synthetic/raw/round2/masks')
CASES={
 'M1noage':dict(correction='off',aging_model='off',wake_enabled=False),
 'M2noage':dict(aging_model='off',wake_enabled=False),
 'M2all':dict(wake_enabled=False),
 'M5all_PL':{},
 'M3noage':dict(shear='most',aging_model='off',wake_enabled=False),
 'M4all':dict(shear='most',wake_enabled=False),
 'M5all':dict(shear='most'),
 'M5all_PL_noage':dict(aging_model='off'),
 'M5all_noage':dict(shear='most',aging_model='off'),
 'M5all_PL_noWake':dict(wake_enabled=False),
}
def metrics(y,p,rated):
    e=p-y
    return dict(n_hours=len(y),r2=1-np.sum(e*e)/np.sum((y-y.mean())**2),
                rmse_n=np.sqrt(np.mean(e*e))/rated,mae_n=np.mean(abs(e))/rated,
                energy_ratio=p.sum()/y.sum(),wasserstein=wasserstein_distance(y/rated,p/rated))

out=Path(sys.argv[1]);out.mkdir(exist_ok=False);rows=[];training=[];hashes={}
for pid in sorted(meterdata.load_mapping().park_id):
    pc=chain.ParkCache(pid);pc.force_all_aging();rated=meterdata.rated_power_w(pid)
    for exp,extra in CASES.items():
        source=BASE/exp/f'synth_{pid}.csv';maskfile=MASK/exp/f'mask_{pid}.parquet'
        if not source.exists() or not maskfile.exists():raise FileNotFoundError(f'{exp}/{pid}')
        hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        hashes[str(maskfile)]=hashlib.sha256(maskfile.read_bytes()).hexdigest()
        old=pd.read_csv(source,sep=';',usecols=['timestamp','power_park']);old.index=pd.to_datetime(old.pop('timestamp'),utc=True)
        old=old.power_park.reindex(pc.index).to_numpy()
        new=pc.power_series(dict(density='iec',**extra))
        keep=pd.read_parquet(maskfile)['keep'].reindex(pc.index).fillna(False).to_numpy(dtype=bool)
        observed=np.isfinite(pc.meas)&np.isfinite(old)&np.isfinite(new)
        for target,ok in [('raw',observed),('clean',observed&keep)]:
            for version,p in [('before',old),('after',new)]:
                rows.append(dict(experiment=exp,park_id=pid,park=PSEUDONYM[pid],variant=target,version=version,**metrics(pc.meas[ok],p[ok],rated)))
        if exp in ['M1noage','M2noage','M5all_PL']:
            ok=(pc.index<pd.Timestamp('2024-02-01',tz='UTC'))&np.isfinite(old)&np.isfinite(new)
            delta=abs(new[ok]-old[ok])/rated
            training.append(dict(experiment=exp,park=PSEUDONYM[pid],n_hours=int(ok.sum()),changed_hours=int((delta>1e-12).sum()),max_abs_change_pp=100*delta.max(),mean_abs_change_pp=100*delta.mean()))
        np.savez_compressed(out/f'{exp}_{pid}.npz',timestamp_ns=pc.index.asi8,new_power_w=new,old_power_w=old,measured_power_w=pc.meas,keep=keep,rated_w=rated)
    print(PSEUDONYM[pid],'scored',flush=True)
pd.DataFrame(rows).to_csv(out/'scores_before_after.csv',index=False)
pd.DataFrame(training).to_csv(out/'pretraining_input_changes.csv',index=False)
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
print('COMPLETE',len(rows),'scores',flush=True)
