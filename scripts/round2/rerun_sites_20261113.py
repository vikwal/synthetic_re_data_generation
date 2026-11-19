"""Repeat the 200-site release with its saved configs into a new directory."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/round2')]
import ask33_site_dataset as site
BASE=Path('/mnt/nvme2/synthetic/physics_20261113')
site.SYNTH_BASE=str(BASE);site.RUN_DIR=str(BASE/'wind/round2/SITE_v2')
site.GEN_DIR=str(ROOT/'configs/round2/_generated/physics_20261113/SITE_v2')
RELEASE=BASE/'site_release'

def main():
    Path(site.GEN_DIR).mkdir(parents=True,exist_ok=True)
    out=Path(site.RUN_DIR);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for sid in site.site_ids():
        old=ROOT/f'configs/round2/_generated/site_v2/config_{sid}.yaml'
        cfg=yaml.safe_load(old.read_text());cfg['data']['synth_dir']=str(BASE)
        target=Path(site.GEN_DIR)/old.name;target.write_text(yaml.safe_dump(cfg))
        path=out/f'synth_{sid}.csv'
        if not path.exists():
            code="import sys; sys.argv=['sites']; import generate_wind as g; g.main("+repr(str(target.relative_to(ROOT/'configs')))+")"
            with (BASE/'logs'/f'site_{sid}.log').open('w') as log:
                result=subprocess.run([str(ROOT/'synthre/bin/python'),'-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not path.exists():raise RuntimeError('Failed site '+sid)
        records.append(dict(site=sid,original_config_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('Site',sid,'complete',flush=True)
    site.postprocess(str(RELEASE));site.tables(str(RELEASE));site.verify(str(RELEASE))
    (RELEASE/'physics_rerun_manifest.json').write_text(json.dumps(dict(records=records,source='saved SITE_v2 configs; unchanged commissioning and input data',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2))
    print('SITE RELEASE COMPLETE',flush=True)
if __name__=='__main__':main()
