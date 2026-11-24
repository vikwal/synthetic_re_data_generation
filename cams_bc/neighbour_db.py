"""Database-backed CAMS neighbour inputs, preserving xgb_v5 preprocessing.
Raw data: DWD J/cm2 per 10min and CAMS Wh/m2 per 15min, both UTC interval END.
QC constants mirror NWP/ERA5/config.yaml and solar_bc/data.py (2026-11-24).
No imputation; partially observed DWD half-hours retain mean over available samples.
"""
import numpy as np
import pandas as pd
QC_VERSION='xgb_v5_bsrn_30min_v1'
STATION_TABLE='public.cams_neighbour_stations'
OBS_TABLE='public.dwd_solar_observations_10min'
CAMS_TABLE='public.cams_station_radiation_15min'
READY_TABLE='public.cams_neighbour_inputs_30min'
COLUMNS=['ghi_raw','dhi_raw','cams_ghi','cams_ghi_clear','cams_toa','n_obs_ghi','n_obs_dhi','day_mask','y_valid_ghi','y_valid_dhi','ghi_rare','dhi_rare','dhi_gt_ghi']

def prepare_station(obs,cams,idx,record_start=None,record_end=None):
    """Pure transformation, independently comparable against existing processed Parquets."""
    kw=dict(rule='30min',label='right',closed='right')
    radiation=obs[['ghi_j_cm2','dhi_j_cm2']].astype(float)*(10000.0/600.0)
    r=radiation.resample(**kw)
    values=r.mean().reindex(idx);counts=r.count().reindex(idx).fillna(0).astype('int16')
    # Keep the original float32 conversion BEFORE CAMS averaging.
    rad=(cams[['ghi_wh_m2','ghi_clear_wh_m2','toa_wh_m2']].astype(float)*4).astype('float32')
    cr=rad.resample(**kw);cm=cr.mean().where(cr.count()==2).reindex(idx)
    ghi,dhi=values['ghi_j_cm2'],values['dhi_j_cm2']
    clear=cm['ghi_clear_wh_m2'];toa=cm['toa_wh_m2']
    day=clear>10.0
    e0n=1361.0*(1.0+0.033*np.cos(2*np.pi*idx.dayofyear.to_numpy()/365.0))
    mu=np.clip(toa.to_numpy(float)/e0n,0.0,1.0)
    rare_g=ghi.notna() & (ghi>1.2*e0n*mu**1.2+50.0)
    rare_d=dhi.notna() & (dhi>0.75*e0n*mu**1.2+30.0)
    dhi_gt=dhi.notna() & ghi.notna() & (dhi>ghi)
    window=np.ones(len(idx),dtype=bool)
    if record_start is not None:
        window=(idx>=pd.Timestamp(record_start)) & (idx<=pd.Timestamp(record_end))
    valid_g=ghi.notna() & day & window & ~rare_g
    valid_d=dhi.notna() & day & window & ~rare_d & ~dhi_gt
    return pd.DataFrame(dict(ghi_raw=ghi.where(valid_g).astype('float32'),dhi_raw=dhi.where(valid_d).astype('float32'),cams_ghi=cm['ghi_wh_m2'],cams_ghi_clear=clear,cams_toa=toa,n_obs_ghi=counts['ghi_j_cm2'],n_obs_dhi=counts['dhi_j_cm2'],day_mask=day,y_valid_ghi=valid_g,y_valid_dhi=valid_d,ghi_rare=rare_g,dhi_rare=rare_d,dhi_gt_ghi=dhi_gt),index=idx)[COLUMNS]

def ratios(frame):
    """Same float32 arithmetic, masks and clipping as data.neighbour_table(source='raw')."""
    ghi=frame.ghi_raw.astype('float32').where(frame.y_valid_ghi)
    dhi=frame.dhi_raw.astype('float32').where(frame.y_valid_dhi & frame.y_valid_ghi)
    clear=np.maximum(frame.cams_ghi_clear.astype('float32'),1.0)
    return pd.DataFrame({'kc':(ghi/clear).clip(0,2).where(frame.day_mask).astype('float32'),'kd':(dhi/np.maximum(ghi,1.0)).clip(0,1).where(frame.day_mask).astype('float32'),'kcc':(frame.cams_ghi.astype('float32')/clear).where(frame.day_mask).astype('float32')})

def frame_from_rows(rows,columns):
    df=pd.DataFrame(rows,columns=['ts']+columns)
    df['ts']=pd.to_datetime(df.ts,utc=True)
    return df.set_index('ts')

def load_neighbours(conn,idx):
    """Read each station once per requested period, then reuse for every park.
    Explicit DB mode: errors do not silently fall back to the obsolete Parquet cache.
    The caller should use a read-only REPEATABLE READ transaction for a stable version.
    """
    if not isinstance(idx,pd.DatetimeIndex) or idx.tz is None or idx.empty or not idx.is_monotonic_increasing:
        raise ValueError('A nonempty sorted timezone-aware index is required')
    with conn.cursor() as c:
        c.execute("SELECT qc_version,stations FROM cams_neighbour_dataset WHERE name='xgb_v5'")
        marker=c.fetchone()
        if not marker or marker[0]!=QC_VERSION:raise RuntimeError('Neighbour database is not ready for this QC version')
        c.execute(f'SELECT station_id,latitude,longitude,elevation FROM {STATION_TABLE} ORDER BY station_id')
        coords=pd.DataFrame(c.fetchall(),columns=['station_id','lat','lon','elev']).set_index('station_id').astype(float)
        if len(coords)!=marker[1]:raise RuntimeError('Station registry changed; refresh neighbour inputs first')
        parts=[]
        for sid in coords.index:
            c.execute(f'SELECT ts,ghi_raw,dhi_raw,cams_ghi,cams_ghi_clear,day_mask,y_valid_ghi,y_valid_dhi FROM {READY_TABLE} WHERE station_id=%s AND ts>=%s AND ts<=%s ORDER BY ts',(sid,idx.min().to_pydatetime(),idx.max().to_pydatetime()))
            df=frame_from_rows(c.fetchall(),['ghi_raw','dhi_raw','cams_ghi','cams_ghi_clear','day_mask','y_valid_ghi','y_valid_dhi']).reindex(idx)
            for k in ['day_mask','y_valid_ghi','y_valid_dhi']:df[k]=df[k].fillna(False).astype(bool)
            parts.append(ratios(df).add_prefix(sid+'_'))
    return pd.concat(parts,axis=1),coords
