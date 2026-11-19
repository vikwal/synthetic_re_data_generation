"""Read-only generator extraction for restored R3 figures; writes only --out.

Run from generator checkout with its synthre Python. Current IEC full-chain
power is reconstructed and checked before rescoring two no-QM references.
All existing masks are retained. No training, mask fitting or source mutation.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'scripts/round2')]
from round2 import chain,aging,parkinfo,meterdata,stability,evaluation
from regen_paper_figs import _park_ideal_curve
from rerun_gryning_20261113 import full_power

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    base=Path('/mnt/nvme2/synthetic/physics_20261113/wind/round2')
    maskbase=Path('/mnt/nvme2/synthetic/raw/round2/masks')
    specs=pd.read_csv('power_curves/turbine_specs.csv',sep=';').drop_duplicates('Turbine',keep='first').set_index('Turbine')
    hashes={}; rows=[]; checks=[]; full=[]; curves=[]
    def tracked(p):
        hashes[str(p)]=sha(p);return p
    for pid in sorted(meterdata.load_mapping().park_id):
        a=pd.read_csv(tracked(base/'M5all_PL'/f'synth_{pid}.csv'),sep=';',index_col=0,parse_dates=True)
        b=pd.read_csv(tracked(base/'M5all'/f'synth_{pid}.csv'),sep=';',index_col=0,parse_dates=True)
        ix=a.index.intersection(b.index);a=a.loc[ix];b=b.loc[ix]
        L=stability.obukhov_length(a.friction_wind.values,a.density.values,a.temp_2m.values,a.sshf.values)
        cl=stability.stability_class(L)
        valid=np.isfinite(a.power_park.values)&np.isfinite(b.power_park.values)
        for group in ['stable','neutral','unstable','all']:
            ok=valid & ((cl==group) if group!='all' else True)
            full.append(dict(park_id=pid,stability=group,n_hours=int(ok.sum()),
                             pl_energy_wh=float(a.power_park.values[ok].sum()),most_energy_wh=float(b.power_park.values[ok].sum()),
                             start=str(ix[ok].min()),end=str(ix[ok].max())))
        if pid in ['00198_1','00282']:
            grid,power=_park_ideal_curve(pid)
            curves.extend(dict(park_id=pid,wind=float(v),power_mw=float(p)) for v,p in zip(grid,power))
        if pid not in ['04745','05347']:continue
        a=a.loc['2023-07-24':'2024-06-01']
        n=pd.read_csv(tracked(base/'M5all_PL_noQM'/f'synth_{pid}.csv'),sep=';',index_col=0,parse_dates=True).reindex(a.index)
        old_manifest=json.loads(tracked(base/'M5all_PL_noQM'/f'manifest_{pid}.json').read_text())
        assert old_manifest['round2']['correction']=='off' and old_manifest['round2']['shear']=='power_law'
        pars=chain._load_park_config(pid)['params'];ts=pars['turbines']; hs=pars['hub_heights']
        pcs=chain._power_curves(ts,[float(specs.loc[t,'Abschaltgeschwindigkeit']) for t in ts])
        df=aging.get_degradation_vector(a.index,model='weibull',commissioning_date=parkinfo.commissioning_date(pid))[0]
        recon=np.zeros(len(a));pred=np.zeros(len(a))
        for j,(t,h) in enumerate(zip(ts,hs),1):
            sr=specs.loc[t];sp=dict(cut_in=float(sr['Einschaltgeschwindigkeit']),cut_out=float(sr['Abschaltgeschwindigkeit']))
            diam=float(sr['Rotordurchmesser']) if 'Rotordurchmesser' in specs else float(sr['Rotordurchmesser [m]'])
            ticks,watts=pcs[t];rated=float(pars['rated'][j-1])*1000 if 'rated' in pars else watts.max()
            rho=a[f'density_t{j}'].values
            np.testing.assert_allclose(rho,n[f'density_t{j}'].values,rtol=0,atol=1e-12)
            recon+=full_power(a[f'wind_speed_t{j}'].values,rho,df,ticks,watts,sp,diam,rated)
            pred+=full_power(n[f'wind_speed_t{j}'].values,rho,df,ticks,watts,sp,diam,rated)
        np.testing.assert_allclose(a.wake_factor,n.wake_factor,rtol=0,atol=1e-12)
        recon*=a.wake_factor.values;pred*=a.wake_factor.values
        rated=meterdata.rated_power_w(pid);error=float(np.max(abs(recon-a.power_park.values))/rated)
        assert error<1e-7,(pid,error)
        y=meterdata.load_park_power(pid,('2023-07-24','2024-06-01')).reindex(a.index)
        keep=pd.read_parquet(tracked(maskbase/'M5all_PL_noQM'/f'mask_{pid}.parquet'))['keep'].reindex(a.index).fillna(False).astype(bool)
        p=pd.Series(pred,index=a.index)
        for variant in ['raw','clean']:
            measured=y if variant=='raw' else y[keep]
            res=evaluation.evaluate(measured,p,rated)
            res.pop('acf_meas',None);res.pop('acf_synth',None)
            rows.append(dict(experiment='M5all_PL_noQM',park_id=pid,variant=variant,**res))
        np.savez_compressed(out/f'noQM_{pid}.npz',timestamp_ns=a.index.asi8,y_w=y.to_numpy(),pred_w=pred,keep=keep.to_numpy(),rated_w=rated)
        checks.append(dict(park_id=pid,max_reconstruction_error_normalized=error,unchanged_density=True,unchanged_wake=True))
    pd.DataFrame(rows).to_csv(out/'noQM_scores.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'noQM_verification.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'ideal_curves.csv',index=False)
    f=pd.DataFrame(full);f.to_csv(out/'stability_energy_per_farm.csv',index=False)
    g=f.groupby('stability')[['n_hours','pl_energy_wh','most_energy_wh']].sum()
    g['share_hours']=g.n_hours/g.loc['all','n_hours'];g['delta_energy_pct']=(g.pl_energy_wh/g.most_energy_wh-1)*100
    g.to_csv(out/'stability_energy.csv')

    # Same fast-chain sensitivity engine as the current Morris/Sobol analysis.
    assert chain.DENSITY_METHOD=='iec'
    c=chain.Chain()
    for pc in c.parks:pc.force_all_aging()
    d20=.126;kap=2.
    lam=lambda d,k:20/(-np.log1p(-d))**(1/k)
    baseline=dict(wind_level_factor=1.,correction='height_consistent',shear='most',z0_scale=1.,aging_model='weibull',
                  aging_lambda=lam(d20,kap),aging_kappa=kap,power_curve_scale=1.,density='dynamic',wake_k=.075,wake_enabled=True)
    def energy(th):
        ers=np.array([pc.energy_ratio(th) for pc in c.parks]);weights=np.array([np.nansum(pc.meas) for pc in c.parks])
        assert np.isfinite(ers).all();return float(ers@weights),float(np.mean(abs(ers-1)))
    e0,a0=energy(baseline)
    sweeps={
       'wind_level_factor':[(str(v),dict(wind_level_factor=v)) for v in [.95,.975,1.025,1.05]],
       'correction_mode':[('off',dict(correction='off')),('10 m only',dict(correction='wind10_only'))],
       'aging_model':[('off',dict(aging_model='off'))],
       'shear_method':[('power law',dict(shear='power_law'))],
       'density_mode':[('reference',dict(density='static_1225'))],
       'wake_k':[(str(v),dict(wake_k=v)) for v in [.05,.10]],
       'z0_scale':[(str(v),dict(z0_scale=v)) for v in [.5,.71,1.41,2.]],
       'aging_d20':[(str(v),dict(aging_lambda=lam(v,kap))) for v in [.08,.10,.15,.18]],
       'aging_kappa':[(str(v),dict(aging_kappa=v,aging_lambda=lam(d20,v))) for v in [1.5,2.,2.5]],
       'power_curve_scale':[(str(v),dict(power_curve_scale=v)) for v in [.97,1.03]]}
    effect=[dict(parameter='baseline',level='S3',d_energy_pct=0.,pooled_abs_er_dev=a0)]
    for param,levels in sweeps.items():
        for label,over in levels:
            en,ae=energy({**baseline,**over});effect.append(dict(parameter=param,level=label,d_energy_pct=(en/e0-1)*100,pooled_abs_er_dev=ae))
    ages=[pc.ages.copy() for pc in c.parks]
    for factor in [1.1,1.2,1.3]:
        for pc,a in zip(c.parks,ages):pc.ages=a*factor
        en,ae=energy(baseline);effect.append(dict(parameter='age_scaling',level=str(factor),d_energy_pct=(en/e0-1)*100,pooled_abs_er_dev=ae))
    for pc,a in zip(c.parks,ages):pc.ages=a
    np.testing.assert_allclose(energy(baseline),[e0,a0],rtol=1e-14)
    pd.DataFrame(effect).to_csv(out/'main_effects.csv',index=False)
    for p,h in hashes.items():assert sha(p)==h,p
    (out/'manifest.json').write_text(json.dumps(dict(source_hashes=hashes,baseline=baseline,
        noQM='Current full-power conversion on archived uncorrected winds; checked reconstruction of current M4; original noQM masks',
        stability='Full available synthetic PL/MOST overlap, no operational mask; not forecast evaluation',
        sensitivity='Current IEC fast chain on original measured overlap; S3 baseline; D20 and kappa replace independent lambda/kappa',
        code_sha256=sha(__file__),full_power_sha256=sha(Path(__file__).parent/'rerun_gryning_20261113.py'),
        chain_sha256=sha(chain.__file__)),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False));print(g.to_string());print('COMPLETE',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
