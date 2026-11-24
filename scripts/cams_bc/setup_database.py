"""Emit transactional PostgreSQL setup SQL; apply with psql as database administrator.
One inference interface supports spatial source tables or normalized point-ID tables.
With --source-inputs, expose source station observations/CAMS via views (no duplicate raw copy).
Existing mirrored raw tables are used otherwise. No credentials or data enter Git.
"""
import argparse,csv,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from parks.era5_db import RAW_COLS


def setup_sql(layout,stations=None,source_inputs=False,reader_role=None,start='2023-07-24'):
    import datetime
    start=datetime.date.fromisoformat(start).isoformat()
    out=['BEGIN;','CREATE SCHEMA IF NOT EXISTS cams_weather;',
    '''CREATE TABLE IF NOT EXISTS public.cams_neighbour_stations(station_id text PRIMARY KEY,latitude double precision NOT NULL,longitude double precision NOT NULL,elevation double precision,source_latitude real,source_longitude real,source_elevation real);''']
    if stations:
        for row in stations:
            sid=row['station_id']
            if not re.fullmatch(r'\d{5}',sid):raise ValueError('Expected five-digit station ID')
            import math
            values=[float(row[k]) for k in ['lat','lon','elev']]
            if not all(map(math.isfinite,values)):raise ValueError('Nonfinite station coordinates')
            if not (-90<=values[0]<=90 and -180<=values[1]<=180):raise ValueError('Invalid coordinates')
            out.append(f"INSERT INTO public.cams_neighbour_stations(station_id,latitude,longitude,elevation) VALUES('{sid}',{','.join(map(repr,values))}) ON CONFLICT(station_id) DO UPDATE SET latitude=excluded.latitude,longitude=excluded.longitude,elevation=excluded.elevation;")
    if layout=='spatial':
        met_points='SELECT cell_id::integer,lat::double precision,lon::double precision FROM public.era5_wind_grid_points'
        met='SELECT p.cell_id::integer,g.timestamp,'+','.join('g.'+c for c in RAW_COLS)+' FROM public.era5_wind_grid g JOIN public.era5_wind_grid_points p ON p.geom=g.geom'
        solar_points='SELECT cell_id::integer,lat::double precision,lon::double precision FROM public.era5_solar_grid_points'
        solar_from='public.era5_solar s JOIN public.era5_solar_grid_points p ON p.geom=s.geom'
        solar_id='p.cell_id::integer'
    elif layout=='point_id':
        met_points='SELECT id AS cell_id,latitude AS lat,longitude AS lon FROM public.grid_points'
        met='SELECT grid_id AS cell_id,ts AS timestamp,'+','.join(RAW_COLS)+' FROM public.era5'
        solar_points='SELECT grid_id AS cell_id,latitude AS lat,longitude AS lon FROM public.era5_solar_grid_points'
        solar_from='public.era5_solar s';solar_id='s.grid_id AS cell_id'
    else:raise ValueError('Unknown weather layout')
    solar_columns=['ssrd','fdir','ssrdc','tisr','fal','tcc','lcc','mcc','hcc','tcwv','strd','strdc','cdir']
    views={'met_points':met_points,'met':met,'solar_points':solar_points,'solar':'SELECT '+solar_id+',s.valid_time,'+','.join('s.'+c for c in solar_columns)+' FROM '+solar_from}
    for name,query in views.items():out.append(f'CREATE OR REPLACE VIEW cams_weather.{name} AS {query};')
    if source_inputs:
        # CREATE OR REPLACE VIEW is repeatable and refuses to replace a physical raw mirror.
        out.extend([f'''CREATE OR REPLACE VIEW public.dwd_solar_observations_10min AS SELECT s.station_id::text,s.timestamp AS ts,s.ghi AS ghi_j_cm2,s.dhi AS dhi_j_cm2 FROM public.stations s JOIN public.cams_neighbour_stations p ON p.station_id=s.station_id WHERE s.timestamp>=TIMESTAMPTZ '{start} 00:00:00+00';''',
        f'''CREATE OR REPLACE VIEW public.cams_station_radiation_15min AS SELECT s.station_id,s.valid_time AS ts,s.ghi AS ghi_wh_m2,s.ghi_clear AS ghi_clear_wh_m2,s.toa AS toa_wh_m2,s.reliability FROM public.cams_radiation s JOIN public.cams_neighbour_stations p USING(station_id) WHERE s.valid_time>=TIMESTAMPTZ '{start} 00:00:00+00'-interval '30 minutes';'''])
    out.extend(['''CREATE TABLE IF NOT EXISTS public.cams_neighbour_inputs_30min(station_id text REFERENCES public.cams_neighbour_stations,ts timestamptz,ghi_raw real,dhi_raw real,cams_ghi real,cams_ghi_clear real,cams_toa real,n_obs_ghi smallint,n_obs_dhi smallint,day_mask boolean,y_valid_ghi boolean,y_valid_dhi boolean,ghi_rare boolean,dhi_rare boolean,dhi_gt_ghi boolean,PRIMARY KEY(station_id,ts));''',
    '''CREATE TABLE IF NOT EXISTS public.cams_neighbour_dataset(name text PRIMARY KEY,qc_version text NOT NULL,stations integer NOT NULL,first_ts timestamptz,last_ts timestamptz,built_at timestamptz NOT NULL);'''])
    if reader_role:
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*',reader_role):raise ValueError('Invalid role name')
        out.extend([f'GRANT USAGE ON SCHEMA cams_weather TO "{reader_role}";',f'GRANT SELECT ON ALL TABLES IN SCHEMA cams_weather TO "{reader_role}";',f'GRANT SELECT ON public.cams_neighbour_stations,public.cams_neighbour_inputs_30min,public.cams_neighbour_dataset TO "{reader_role}";'])
    return '\n'.join(out+['COMMIT;'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',required=True,choices=['spatial','point_id'])
    p.add_argument('--stations',help='Existing model splits.csv; omitted to preserve installed registry')
    p.add_argument('--source-inputs',action='store_true')
    p.add_argument('--reader-role')
    p.add_argument('--start',default='2023-07-24')
    a=p.parse_args();stations=None
    if a.stations:
        with open(a.stations) as f:stations=list(csv.DictReader(f))
        if not stations or len({r['station_id'] for r in stations})!=len(stations):raise ValueError('Empty or duplicate station registry')
    print(setup_sql(a.layout,stations,a.source_inputs,a.reader_role,a.start))

if __name__=='__main__':main()
