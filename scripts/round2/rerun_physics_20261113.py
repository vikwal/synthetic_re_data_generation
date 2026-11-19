"""Rebuild existing park corpora into a separate, immutable campaign root."""
from pathlib import Path
import concurrent.futures as cf
import hashlib,json,os,subprocess,sys,time
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/mnt/nvme2/synthetic/physics_20261113')
RUNGS=['M1','M2','M4','M3','S1','S2','S3','M4_noage','M4_noWake','S3_noage']
sys.path.insert(0,str(ROOT))
from round2 import meterdata

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def rung(name):
    block=yaml.safe_load((ROOT/f'configs/round2/{name}.yaml').read_text())['round2']
    eid=block['experiment_id'];dest=OUT/'wind/round2'/eid
    if (dest/'COMPLETE.json').exists():
        complete=json.loads((dest/'COMPLETE.json').read_text())
        expected=sorted(meterdata.load_mapping().park_id) if name!='M4_noQM' else ['04745','05347']
        records=complete['records']
        if complete['rung']!=name or complete['experiment']!=eid or len(records)!=len(expected) or sorted(r['park_id'] for r in records)!=expected:
            raise ValueError('Incomplete generator manifest: '+name)
        for row in records:
            pid=row['park_id']
            source=ROOT/f'archiv/legacy_configs/real_wind_parks_era5/config_{pid}.yaml'
            generated=ROOT/f'configs/round2/_generated/physics_20261113/{name}/config_{pid}.yaml'
            if sha(source)!=row['source_config_sha256'] or sha(generated)!=row['generated_config_sha256'] or sha(dest/f'synth_{pid}.csv')!=row['output_sha256']:
                raise ValueError('Changed completed generator input or output: '+name)
            config=yaml.safe_load(generated.read_text())
            if config['round2']!=block or config['data']['synth_dir']!=str(OUT):
                raise ValueError('Changed generator protocol: '+name)
        return eid
    gen=ROOT/'configs/round2/_generated/physics_20261113'/name
    gen.mkdir(parents=True,exist_ok=True)
    records=[]
    ids=sorted(meterdata.load_mapping().park_id)
    if name=='M4_noQM':ids=['04745','05347']
    for pid in ids:
        source=ROOT/f'archiv/legacy_configs/real_wind_parks_era5/config_{pid}.yaml'
        cfg=yaml.safe_load(source.read_text());cfg['round2']=block
        cfg['data']['synth_dir']=str(OUT)
        path=gen/f'config_{pid}.yaml';path.write_text(yaml.safe_dump(cfg))
        log=OUT/'logs'/f'{name}_{pid}.log'
        with log.open('w') as f:
            ret=subprocess.run([str(ROOT/'synthre/bin/python'),'-c',"import sys; sys.argv=['rerun']; import generate_wind as gw; gw.main("+repr(str(path.relative_to(ROOT/'configs')))+")"],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        target=dest/f'synth_{pid}.csv'
        if ret.returncode or not target.exists():raise RuntimeError(f'Generation failed {name}/{pid}: {log}')
        records.append(dict(park_id=pid,source_config_sha256=sha(source),generated_config_sha256=sha(path),output_sha256=sha(target)))
        print(name,pid,'complete',flush=True)
    (dest/'COMPLETE.json').write_text(json.dumps(dict(rung=name,experiment=eid,records=records,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2))
    return eid

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(rung,RUNGS):print('RUNG COMPLETE',result,flush=True)
    # The two no-QM diagnostics use the same M4 settings with correction off.
    p=ROOT/'configs/round2/M4_noQM.yaml'
    if not p.exists():
        c=yaml.safe_load((ROOT/'configs/round2/M4.yaml').read_text())
        c['round2'].update(experiment_id='M5all_PL_noQM',correction='off')
        p.write_text(yaml.safe_dump(c))
    rung('M4_noQM')
    (OUT/'GENERATION_COMPLETE.json').write_text(json.dumps(dict(rungs=RUNGS+['M4_noQM'],completed=time.time()),indent=2))
