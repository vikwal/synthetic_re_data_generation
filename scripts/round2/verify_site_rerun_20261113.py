"""Verify the corrected release preserves all inputs, site choices and schema."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OLD=Path('/mnt/nvme2/synthetic/wind/site_v5_20261102')
NEW=Path('/mnt/nvme2/synthetic/physics_20261113/site_release')
OUT=NEW.parent/'analyses/site_audit';OUT.mkdir(parents=True,exist_ok=True)
oldmeta=json.loads((OLD/'manifests.json').read_text());newmeta=json.loads((NEW/'manifests.json').read_text())
files=sorted(NEW.glob('synth_*.parquet'));assert len(files)==200 and {p.name for p in files}=={p.name for p in OLD.glob('synth_*.parquet')}
parts={f'density_t{i}':[] for i in range(1,7)};rows=[]
for p in files:
    sid=p.stem[6:];a=pd.read_parquet(OLD/p.name);b=pd.read_parquet(p)
    assert a.shape==b.shape==(24288,40) and list(a.columns)==list(b.columns) and a.index.equals(b.index)
    for field in ['round2','branch','z0','random_seed','era5_dir']:assert oldmeta[sid][field]==newmeta[sid][field],(sid,field)
    unchanged=[c for c in a if not c.startswith(('power_','density'))]
    np.testing.assert_allclose(a[unchanged],b[unchanged],rtol=0,atol=0,equal_nan=True,err_msg=sid)
    for c in parts:parts[c].append(b[c].to_numpy())
    rows.append(dict(site_id=sid,rows=len(b),data_columns=len(b.columns),unchanged_columns=len(unchanged),old_sha256=hashlib.sha256((OLD/p.name).read_bytes()).hexdigest(),new_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for name in ['wind_parameter.csv','turbine_parameter.csv','turbine_parameter_per_site.csv']:
    a=pd.read_csv(OLD/name,sep=';');b=pd.read_csv(NEW/name,sep=';');pd.testing.assert_frame_equal(a,b)
stats=[]
for c,values in parts.items():
    a=np.concatenate(values);stats.append(dict(turbine=c,mean=a.mean(),p05=np.percentile(a,5),p95=np.percentile(a,95),n=len(a)))
pd.DataFrame(stats).to_csv(OUT/'density_v5.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'release_file_audit.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(dict(sites=200,rows_per_site=24288,data_columns=40,serialized_columns_including_timestamp=41,unchanged='all weather, wind, measured data, metadata, branch choices, seeds and schema',changed='density and synthetic power only',complete=True),indent=2))
print('ALL 200 SITES VERIFIED');print(pd.DataFrame(stats).to_string(index=False))
