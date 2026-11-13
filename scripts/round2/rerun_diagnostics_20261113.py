"""Repeat existing supplementary numerical diagnostics in isolated outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/round3')]
import alpha_clip_sensitivity as alpha,aging_forms as forms,filter_classification as filters
BASE=Path('/mnt/nvme2/synthetic/physics_20261113');OUT=BASE/'analyses/diagnostics';OUT.mkdir(parents=True,exist_ok=True)
alpha.OUT=str(OUT);alpha.main()
forms.OUT=str(OUT);forms.main()
filters.OUT=str(OUT);filters.SYNTH_BASE=str(BASE/'wind/round2')
original=filters.pd.read_csv
def read_csv(path,*args,**kwargs):
    if str(path)==str(ROOT/'results/quantitative/scores.csv'):path=BASE/'analyses/synthesis/scores.csv'
    return original(path,*args,**kwargs)
filters.pd.read_csv=read_csv;sys.argv=['filter','--exp','M5all_PL'];filters.main()
print('DIAGNOSTICS COMPLETE',flush=True)
