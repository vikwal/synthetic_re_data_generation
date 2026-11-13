"""Controlled full-generator profile substitution and operational plateau audit.
Run from generator repo on l2. Reads frozen M4/S3, writes only --out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
sys.path.insert(0,str(Path.cwd()))
from round2 import chain, stability, aging, parkinfo, meterdata
from round2.paper_style import PSEUDONYM

def gryning_shape(z,z0,L,ustar,blh,latitude,b=5.):
    """Gryning 2007 Eqs 8/15/20/21/31. No above-BLH extrapolation."""
    L,ustar,blh=np.broadcast_arrays(np.asarray(L,float),np.asarray(ustar,float),np.asarray(blh,float))
    f=2*7.292115e-5*np.sin(np.deg2rad(latitude))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        invL=np.where(np.isinf(L),0.,1/L)
        invm=(f/ustar)*(55-2*np.log(ustar/(f*z0)))*np.exp(-(ustar*invL/f)**2/400)
        shape=np.log(z/z0)+z*invm*(1-z/(2*blh))
        stable=invL>0
        shape=np.where(stable,shape+b*z*invL*(1-z/(2*blh)),shape)
        x=np.cbrt(1-12*z*np.minimum(invL,0))
        psi=1.5*np.log((1+x+x*x)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+np.pi/np.sqrt(3)
        shape=np.where(invL<0,shape-psi,shape)
    valid=(blh>z)&(ustar>0)&(f>0)&(z>z0)&np.isfinite(shape)&(shape>0)&(invm>=0)&~np.isnan(L)&(L!=0)
    return np.where(valid,shape,np.nan)

def gryning_wind(v100,z,z0,L,ustar,blh,latitude):
    return np.asarray(v100)*gryning_shape(z,z0,L,ustar,blh,latitude)/gryning_shape(100.,z0,L,ustar,blh,latitude)

def profile_tests():
    L=np.array([np.inf,100.,-100.]); u=np.full(3,.4); zi=np.full(3,1000.)
    np.testing.assert_allclose(gryning_wind(np.full(3,8.),100.,.1,L,u,zi,52.),8.,rtol=1e-14)
    # Neutral analytic shape and infinite-MBL limit independently checked.
    f=2*7.292115e-5*np.sin(np.deg2rad(52.)); im=f/.4*(55-2*np.log(.4/(f*.1)))
    expected=np.log(150/.1)+150*im*(1-150/2000)
    np.testing.assert_allclose(gryning_shape(150,.1,[np.inf],[.4],[1000.],52.),expected)
    for sign in [-1,1]:
        np.testing.assert_allclose(gryning_shape(150,.1,[sign*1e12],[.4],[1000.],52.),expected,rtol=1e-8)
    assert np.isnan(gryning_shape(150,.1,[100.],[.4],[120.],52.)[0])
    # Stable derivative equals Eq 14; numerical check within the BL.
    h=.001; z=150.; L0=100.; zi0=1000.
    im=f/.4*(55-2*np.log(.4/(f*.1)))*np.exp(-(.4/(f*L0))**2/400)
    der=(gryning_shape(z+h,.1,[L0],[.4],[zi0],52.)-gryning_shape(z-h,.1,[L0],[.4],[zi0],52.))/(2*h)
    expected_der=(1-z/zi0)*((1+5*z/L0)/z+im+1/(zi0-z))
    np.testing.assert_allclose(der,expected_der,rtol=1e-7)

def full_power(v,rho,df,ticks,watts,spec,diameter,rated):
    from round2.wind_physics import turbine_power
    return turbine_power(v,rho,ticks,watts,diameter,rated,spec['cut_in'],spec['cut_out'],df,'iec')

def metrics(y,p,rated):
    return dict(n_hours=len(y),r2=1-np.sum((p-y)**2)/np.sum((y-y.mean())**2),
                rmse_n=np.sqrt(np.mean((p-y)**2))/rated,energy_ratio=p.sum()/y.sum(),
                abs_er_dev=abs(p.sum()/y.sum()-1),wasserstein=wasserstein_distance(y/rated,p/rated))

def plateau_mask(index,p,v,available,tolerance=.01,minwindspan=1.,duration=3):
    """Contiguous low-variance subrated plateaus; no cleaning mask is an input.
    Uses observed level + varying modeled wind + modeled available-power margin.
    All thresholds are fixed before inspecting results and swept for robustness.
    """
    n=len(p); mask=np.zeros(n,dtype=bool)
    for i in range(n-duration+1):
        sl=slice(i,i+duration)
        if (index[i+duration-1]-index[i])!=pd.Timedelta(hours=duration-1): continue
        a=p[sl]; w=v[sl]; av=available[sl]
        if not (np.isfinite(a).all() and np.isfinite(w).all() and np.isfinite(av).all()): continue
        level=np.median(a)
        if (.05<=level<=.90 and np.ptp(a)<=tolerance and np.ptp(w)>=minwindspan
            and np.min(av-a)>=.10): mask[sl]=True
    return mask

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False); profile_tests()
    topo=pd.read_csv('data/round2/topo_features.csv',dtype={'location_id':str}).set_index('location_id')
    specs=pd.read_csv('power_curves/turbine_specs.csv',sep=';').drop_duplicates('Turbine',keep='first').set_index('Turbine')
    stats=[]; checks=[]; cover=[]; plateau=[]; events=[]; sensitivity=[]; examples=[]
    for pid in sorted(meterdata.load_mapping().park_id):
        path=Path('/mnt/nvme2/synthetic/physics_20261113/wind/round2')
        a=pd.read_csv(path/'M5all_PL'/f'synth_{pid}.csv',sep=';',index_col=0,parse_dates=True)
        a=a[(a.index>=pd.Timestamp('2023-07-24',tz='UTC'))&(a.index<pd.Timestamp('2024-06-01',tz='UTC'))]
        m=pd.read_csv(path/'M5all'/f'synth_{pid}.csv',sep=';',index_col=0,parse_dates=True).reindex(a.index)
        params=chain._load_park_config(pid)['params']
        ts=params['turbines']; hs=params['hub_heights']
        curves=chain._power_curves(ts,[float(specs.loc[t,'Abschaltgeschwindigkeit']) for t in ts])
        df=aging.get_degradation_vector(a.index,model='weibull',commissioning_date=parkinfo.commissioning_date(pid))[0]
        z0=float(topo.loc[pid[:5],'z0']); latitude=float(topo.loc[pid[:5],'latitude'])
        L=stability.obukhov_length(a.friction_wind.values,a.density.values,a.temp_2m.values,a.sshf.values)
        gyro=[]; recon=np.zeros(len(a)); recon_m=np.zeros(len(a)); domain=np.ones(len(a),bool)
        for j,(t,h) in enumerate(zip(ts,hs),1):
            sr=specs.loc[t]; spec=dict(cut_in=float(sr['Einschaltgeschwindigkeit']),cut_out=float(sr['Abschaltgeschwindigkeit']))
            diam=float(sr['Rotordurchmesser']) if 'Rotordurchmesser' in specs else float(sr['Rotordurchmesser [m]'])
            ticks,watts=curves[t]; rated=float(params['rated'][j-1])*1000 if 'rated' in params else watts.max()
            rho=a[f'density_t{j}'].values
            def power(v):return full_power(v,rho,df,ticks,watts,spec,diam,rated)
            recon+=power(a[f'wind_speed_t{j}'].values)
            recon_m+=power(m[f'wind_speed_t{j}'].values)
            v=np.round(gryning_wind(a.wind_speed_100m.values,h,z0,L,a.friction_wind.values,a.blh.values,latitude),2)
            domain &= np.isfinite(v)
            gyro.append(power(v))
        rated=meterdata.rated_power_w(pid)
        recon*=a.wake_factor.values; recon_m*=m.wake_factor.values
        assert np.isfinite(recon[np.isfinite(a.power_park.values)]).all(),pid
        assert np.isfinite(recon_m[np.isfinite(m.power_park.values)]).all(),pid
        epl=np.nanmax(abs(recon-a.power_park.values))/rated
        emo=np.nanmax(abs(recon_m-m.power_park.values))/rated
        checks.append(dict(park_id=pid,max_pl_normalized_difference=epl,max_most_normalized_difference=emo))
        assert max(epl,emo)<1e-7,(pid,epl,emo)
        rawv10=np.hypot(a.u_wind_10m.values,a.v_wind_10m.values)
        domain &= rawv10>3.
        g=np.sum(gyro,axis=0)*a.wake_factor.values
        domain &= np.isfinite(g)
        y=meterdata.load_park_power(pid,('2023-07-24','2024-06-01')).reindex(a.index).values
        observed=np.isfinite(y)&np.isfinite(a.power_park.values)&np.isfinite(m.power_park.values)
        keep=pd.read_parquet(Path('/mnt/nvme2/synthetic/raw/round2/masks/M5all_PL')/f'mask_{pid}.parquet')['keep'].reindex(a.index).fillna(False).values.astype(bool)
        common=observed&domain; cl=stability.stability_class(L)
        cover.append(dict(park_id=pid,park=PSEUDONYM[pid],n_observed=int(observed.sum()),n_common=int(common.sum()),common_share=float(common.sum()/observed.sum()),
                          max_hub=max(hs),terrain_elev_std=float(topo.loc['park_'+pid,'elev_std']),z0=z0))
        for support,mask in [('common_raw',common),('common_screened',common&keep),('full_raw',observed),('full_screened',observed&keep)]:
            variants={'PL':a.power_park.values,'MOST':m.power_park.values}
            variants['Gryning' if support.startswith('common') else 'Gryning_PL_fallback']=g if support.startswith('common') else np.where(domain,g,a.power_park.values)
            for group in ['all','stable','neutral','unstable']:
                ok=mask & ((cl==group) if group!='all' else True)
                if ok.sum()<100:continue
                for model,p in variants.items():
                    stats.append(dict(park_id=pid,park=PSEUDONYM[pid],support=support,stability=group,model=model,**metrics(y[ok],p[ok],rated)))
        idx=a.index[observed]; py=y[observed]/rated; v=a.filter(regex=r'^wind_speed_t\d+$').mean(axis=1).values[observed]; av=a.power_park.values[observed]/rated
        removed=~keep[observed]; detected=plateau_mask(idx,py,v,av)
        for tol,span,duration in [(.005,1.,3),(.01,1.,3),(.02,1.,3),(.01,.5,3),(.01,1.5,3),(.01,1.,4)]:
            flags=plateau_mask(idx,py,v,av,tol,span,duration)
            sensitivity.append(dict(park_id=pid,tolerance=tol,wind_span=span,duration=duration,n_candidates=int(flags.sum()),n_removed_candidates=int((flags&removed).sum()),n_removed=int(removed.sum())))
        plateau.append(dict(park_id=pid,park=PSEUDONYM[pid],n_hours=len(py),n_removed=int(removed.sum()),n_candidates=int(detected.sum()),n_removed_candidates=int((detected&removed).sum()),n_kept_candidates=int((detected&~removed).sum()),n_removed_zero=int((removed&(py==0)).sum())))
        labels=np.cumsum(np.r_[True,(np.diff(idx.asi8)!=3600*10**9)|(detected[1:]!=detected[:-1])])
        for label in np.unique(labels[detected]):
            sel=(labels==label)&detected
            events.append(dict(park_id=pid,start=str(idx[sel][0]),end=str(idx[sel][-1]),hours=int(sel.sum()),median_power=float(np.median(py[sel])),power_span=float(np.ptp(py[sel])),wind_span=float(np.ptp(v[sel])),removed_hours=int((sel&removed).sum())))
        examples.append(pd.DataFrame(dict(park_id=pid,timestamp=idx,wind=v,power=py,available=av,removed=removed,plateau=detected)))
        print(pid,'profile reconstruction',round(epl,12),round(emo,12),'common',int(common.sum()),'plateau',int(detected.sum()),flush=True)
    for name,rows in [('profile_metrics',stats),('profile_reconstruction',checks),('profile_coverage',cover),('plateau_summary',plateau),('plateau_events',events),('plateau_sensitivity',sensitivity)]:
        pd.DataFrame(rows).to_csv(args.out/(name+'.csv'),index=False)
    pd.concat(examples).to_parquet(args.out/'plateau_hourly.parquet',index=False)
    pd.concat(examples).to_csv(args.out/'plateau_hourly.csv.gz',index=False)
    meta=dict(gryning='Gryning2007 Eqs8,15,20,21,31; stable b=5 as in deployed Dyer; 100m ratio anchor; ERA5 BLH',
              mask='v10_raw>3, 100m and all hubs below BLH, finite positive shapes; horizontal homogeneity not established',
              plateau='diagnostic candidates, not confirmed curtailment; 3 contiguous hours, range<=1% capacity, level5-90%, wind range>=1m/s, modeled available margin>=10% capacity',
              reference='frozen full-generator PL and MOST; common M4 screening mask',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE '+str(args.out),flush=True)

if __name__=='__main__':main()
