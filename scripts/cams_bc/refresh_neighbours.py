"""Build database neighbour inputs from mirrored raw tables; no Parquet dependency.
Run after setup_database.py using the configured database writer connection.
All stations and completion marker commit atomically. A failure leaves previous data.
"""
import io,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cams_bc.neighbour_db import *
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--start',default='2023-07-24')
args=parser.parse_args()
conn=psycopg2.connect(os.environ.get('WEATHER_DB_WRITE_URL','service=synthetic_weather_writer'))
conn.set_session(isolation_level='REPEATABLE READ');c=conn.cursor();c.execute("SET TIME ZONE 'UTC'; SET statement_timeout='15min'")
c.execute('SELECT pg_try_advisory_xact_lock(20261124,513)')
if not c.fetchone()[0]:raise RuntimeError('Refresh already running')
# Synchronizers must not modify the mirrored inputs during this snapshot's build.
c.execute('SELECT pg_try_advisory_xact_lock(20261124,512)')
if not c.fetchone()[0]:raise RuntimeError('Source mirror still running')
c.execute('CREATE TEMP TABLE new_inputs(LIKE cams_neighbour_inputs_30min INCLUDING DEFAULTS)')
c.execute(f'SELECT station_id FROM {STATION_TABLE} ORDER BY station_id');stations=[r[0] for r in c.fetchall()]
if not stations:raise RuntimeError('No model stations')
c.execute(f'SELECT min(ts),max(ts) FROM {CAMS_TABLE}');cams_bounds=c.fetchone()
c.execute(f'SELECT min(ts),max(ts) FROM {OBS_TABLE}');obs_bounds=c.fetchone()
if None in cams_bounds+obs_bounds:raise RuntimeError('Empty mirror')
start=max(pd.to_datetime(args.start,utc=True).ceil('30min'),pd.to_datetime(obs_bounds[0],utc=True).floor('30min'))
end=min(pd.to_datetime(cams_bounds[1],utc=True).floor('30min'),pd.to_datetime(obs_bounds[1],utc=True).ceil('30min'))
if end<start:raise RuntimeError('Mirror has no usable half-hour range')
idx=pd.date_range(start,end,freq='30min');total=0
for i,sid in enumerate(stations):
    c.execute(f'SELECT ts,ghi_j_cm2,dhi_j_cm2 FROM {OBS_TABLE} WHERE station_id=%s AND ts>%s AND ts<=%s ORDER BY ts',(sid,(start-pd.Timedelta('30min')).to_pydatetime(),end.to_pydatetime()))
    obs=frame_from_rows(c.fetchall(),['ghi_j_cm2','dhi_j_cm2'])
    c.execute(f'SELECT ts,ghi_wh_m2,ghi_clear_wh_m2,toa_wh_m2 FROM {CAMS_TABLE} WHERE station_id=%s AND ts>%s AND ts<=%s ORDER BY ts',(sid,(start-pd.Timedelta('30min')).to_pydatetime(),end.to_pydatetime()))
    cams=frame_from_rows(c.fetchall(),['ghi_wh_m2','ghi_clear_wh_m2','toa_wh_m2'])
    # Original record window uses nonmissing aggregated GHI within the configured period.
    nonmissing=(obs.ghi_j_cm2.astype(float)*(10000/600)).resample('30min',label='right',closed='right').mean().reindex(idx).dropna()
    first,last=(nonmissing.index.min(),nonmissing.index.max()) if len(nonmissing) else (None,None)
    df=prepare_station(obs,cams,idx,first,last)
    df.insert(0,'ts',idx);df.insert(0,'station_id',sid)
    buf=io.StringIO();df.to_csv(buf,index=False,header=False,na_rep='');buf.seek(0)
    c.copy_expert('COPY new_inputs FROM STDIN WITH (FORMAT csv)',buf);total+=len(df)
    print(json.dumps(dict(station=sid,done=i+1,stations=len(stations),rows=len(df),valid_ghi=int(df.y_valid_ghi.sum()),valid_dhi=int(df.y_valid_dhi.sum()))),flush=True)
# Replace the derived dataset atomically; raw source mirrors stay untouched.
c.execute('DELETE FROM cams_neighbour_inputs_30min')
c.execute('INSERT INTO cams_neighbour_inputs_30min SELECT * FROM new_inputs')
c.execute("INSERT INTO cams_neighbour_dataset VALUES('xgb_v5',%s,%s,%s,%s,now()) ON CONFLICT(name) DO UPDATE SET qc_version=excluded.qc_version,stations=excluded.stations,first_ts=excluded.first_ts,last_ts=excluded.last_ts,built_at=excluded.built_at",(QC_VERSION,len(stations),start.to_pydatetime(),end.to_pydatetime()))
c.execute("COMMENT ON TABLE cams_neighbour_inputs_30min IS 'UTC interval END; radiation W/m2, BSRN-derived validity, no imputed values. n_obs counts available original 10min values (0..3). QC version in cams_neighbour_dataset.'")
conn.commit();conn.autocommit=True;c.execute('ANALYZE cams_neighbour_inputs_30min')
print(json.dumps(dict(status='REFRESH_COMPLETE',rows=total,stations=len(stations),start=str(start),end=str(end),qc_version=QC_VERSION)),flush=True)
