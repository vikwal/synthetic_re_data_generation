"""R3 extension audit. Run on l2 from generator repo; only writes --out.

No fitted input distribution is inferred. Fixed D20 enforces the exact aging
constraint. Independent driving coordinates: wind, curve, kappa, wake.
Profiles are audited for available inputs and Gryning's stated domain only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from round2 import chain as ch, correction, meterdata
from round2.paper_style import PSEUDONYM
from SALib.sample import sobol as sample_sobol
from SALib.analyze import sobol

def scale_from_anchor(d20, kappa):
    return 20 / (-np.log1p(-d20)) ** (1 / kappa)

class CachedObjective:
    """Evaluate the shared production physics on already loaded park inputs."""
    def __init__(self, parks):
        self.parks = parks

    def __call__(self, x, d20=.126):
        wind, curve, kappa, wake_k = x
        theta = dict(wind_level_factor=wind, power_curve_scale=curve,
                     aging_model="weibull", aging_kappa=kappa,
                     aging_lambda=scale_from_anchor(d20, kappa),
                     wake_k=wake_k, density="iec", shear="power_law",
                     correction="height_consistent", wake_enabled=True)
        ratios = np.asarray([pc.energy_ratio(theta) for pc in self.parks])
        return np.nanmean(np.abs(ratios - 1))


def audit_profiles(parks, out):
    topo = pd.read_csv('data/round2/topo_features.csv', dtype={'location_id':str}).set_index('location_id')
    rows = []
    for pc in parks:
        root = Path(ch.ERA5_DIR_V2)
        path = root/f'Station_{pc.park_id[:5]}.csv'
        if not path.exists():
            path = Path(ch.ERA5_DIR_DEFAULT)/path.name
        raw = pd.read_csv(path)
        raw.index = pd.to_datetime(raw.timestamp, utc=True)
        raw = raw.reindex(pc.index)
        zi = raw.blh.to_numpy()
        lat = topo.loc[pc.park_id[:5], 'latitude']
        f = 2*7.292115e-5*np.sin(np.deg2rad(lat))
        good = np.isfinite(zi) & (zi>0) & np.isfinite(pc.ustar) & (pc.ustar>0) & ~np.isnan(pc.L)
        strong = pc.v10 > 3.  # Gryning 2007 analysis, uncorrected ERA5 proxy
        domain = good & strong & (zi > max(100., max(pc.hub_heights)))
        rows.append(dict(park_id=pc.park_id, park=PSEUDONYM.get(pc.park_id,pc.park_id),
                         n_hours=len(zi), min_hub=min(pc.hub_heights), max_hub=max(pc.hub_heights),
                         available_inputs_pct=100*good.mean(), wind10_gt3_pct=100*strong.mean(),
                         anchor_and_hubs_below_blh_pct=100*(zi>max(100., max(pc.hub_heights))).mean(),
                         gryning_joint_domain_pct=100*domain.mean(), coriolis=f))
    pd.DataFrame(rows).to_csv(out/'profile_feasibility.csv', index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ids = sorted(meterdata.load_mapping().park_id)
    chain = ch.Chain(ids)
    for pc in chain.parks:
        pc.force_all_aging()
    audit_profiles(chain.parks, args.out)
    objective = CachedObjective(chain.parks)
    rng = np.random.default_rng(20261109)
    bounds = np.array([[.95,1.05],[.97,1.03],[1.5,2.5],[.05,.10]])
    checks = []
    for i in range(12):
        x = rng.uniform(bounds[:,0], bounds[:,1])
        d20 = rng.uniform(.08,.18)
        th = dict(wind_level_factor=x[0],power_curve_scale=x[1],aging_kappa=x[2],
                  aging_lambda=scale_from_anchor(d20,x[2]),wake_k=x[3],density='dynamic')
        reference = chain.run_chain(th)
        cached = objective(x,d20)
        checks.append(dict(reference=reference,cached=cached,abs_diff=abs(reference-cached)))
    pd.DataFrame(checks).to_csv(args.out/'implementation_check.csv',index=False)
    assert max(row['abs_diff'] for row in checks)<1e-10, checks
    print('Reference agreement passed; domain audit saved.', flush=True)
    problem = dict(num_vars=4,names=['wind_level_factor','power_curve_scale','aging_kappa','wake_k'],bounds=bounds.tolist())
    rows = []
    for seed in [91,193]:
        x = sample_sobol.sample(problem,args.n,calc_second_order=False,scramble=True,seed=seed)
        start = time.time()
        y = np.array([objective(v) for v in x])
        np.savez_compressed(args.out/f'fixed_anchor_draws_seed{seed}.npz',X=x,Y=y)
        res = sobol.analyze(problem,y,calc_second_order=False,seed=seed,num_resamples=1000)
        for i, factor in enumerate(problem['names']):
            rows.append(dict(seed=seed,N=args.n,D20=.126,parameter=factor,
                             **{k:float(res[k][i]) for k in ['S1','S1_conf','ST','ST_conf']}))
        print(f'Seed {seed}, {len(x)} runs, {time.time()-start:.1f} s: '+str(dict(zip(problem['names'],res['ST']))),flush=True)
    pd.DataFrame(rows).to_csv(args.out/'fixed_anchor_sobol.csv',index=False)
    # Deterministic coupling check: no sampled curve violates the anchor.
    kappa = np.linspace(1.5,2.5,101)
    lam = scale_from_anchor(.126,kappa)
    loss = 1-np.exp(-(20/lam)**kappa)
    assert np.max(np.abs(loss-.126))<1e-14
    pd.DataFrame(dict(kappa=kappa,lambda_years=lam,loss20=loss)).to_csv(args.out/'fixed_anchor_curve_parameters.csv',index=False)
    meta = dict(output='mean absolute energy-ratio error across 13 farms, raw available hours',
                fixed_D20=.126, N=args.n, seeds=[91,193],
                interpretation='constrained sensitivity in independent driving coordinates; not a fitted copula or Shapley analysis',
                chain_source_sha256=hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                max_reference_difference=max(row['abs_diff'] for row in checks))
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE '+str(args.out),flush=True)

if __name__=='__main__':
    main()
