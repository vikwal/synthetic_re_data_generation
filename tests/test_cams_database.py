"""Regression for units, interval boundaries and missingness in DB neighbour preprocessing."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cams_bc.neighbour_db import prepare_station, ratios


def test_partial_observations_and_incomplete_cams_interval():
    oi=pd.date_range('2026-06-01 12:10',periods=6,freq='10min',tz='UTC')
    obs=pd.DataFrame({'ghi_j_cm2':[6.,12.,np.nan,6.,6.,6.],'dhi_j_cm2':[3.]*6},index=oi)
    ci=pd.date_range('2026-06-01 12:15',periods=4,freq='15min',tz='UTC')
    cams=pd.DataFrame({'ghi_wh_m2':[50.]*4,'ghi_clear_wh_m2':[100.,100.,np.nan,100.],'toa_wh_m2':[200.]*4},index=ci)
    idx=pd.date_range('2026-06-01 12:30',periods=2,freq='30min',tz='UTC')
    d=prepare_station(obs,cams,idx)
    assert d.n_obs_ghi.tolist()==[2,3]
    assert d.ghi_raw.iloc[0]==pytest.approx(150.)
    assert d.cams_ghi.iloc[0]==200.
    assert d.cams_ghi_clear.iloc[0]==400.
    assert d.day_mask.tolist()==[True,False]
    assert ratios(d).kc.iloc[0]==pytest.approx(.375)
    assert ratios(d).iloc[1].isna().all()


def test_qc_rejects_extreme_global_and_diffuse_exceeding_global():
    oi=pd.date_range('2026-06-01 12:10',periods=6,freq='10min',tz='UTC')
    obs=pd.DataFrame({'ghi_j_cm2':[180.]*3+[6.]*3,'dhi_j_cm2':[3.]*3+[12.]*3},index=oi)
    ci=pd.date_range('2026-06-01 12:15',periods=4,freq='15min',tz='UTC')
    cams=pd.DataFrame({'ghi_wh_m2':50.,'ghi_clear_wh_m2':100.,'toa_wh_m2':200.},index=ci)
    idx=pd.date_range('2026-06-01 12:30',periods=2,freq='30min',tz='UTC')
    d=prepare_station(obs,cams,idx);r=ratios(d)
    assert d.ghi_rare.iloc[0]
    assert np.isnan(r.kc.iloc[0]) and np.isnan(r.kd.iloc[0])
    assert d.dhi_gt_ghi.iloc[1]
    assert np.isfinite(r.kc.iloc[1]) and np.isnan(r.kd.iloc[1])


def test_database_setup_rejects_unsafe_registry_and_role():
    path=Path(__file__).resolve().parents[1]/'scripts/cams_bc/setup_database.py'
    spec=importlib.util.spec_from_file_location('database_setup',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with pytest.raises(ValueError):mod.setup_sql('spatial',reader_role='reader;DROP SCHEMA public')
    with pytest.raises(ValueError):mod.setup_sql('spatial',[dict(station_id='1',lat=50,lon=10,elev=0)])
    with pytest.raises(ValueError):mod.setup_sql('point_id',[dict(station_id='00001',lat=float('nan'),lon=10,elev=0)])


def test_solar_query_keeps_hourly_brackets_for_half_hour_interpolation(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/cams_bc'))
    import apply_parks

    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,params):self.sql,self.params=sql,params
        def fetchall(self):
            return [(ts,*([float(value)]*12)) for ts,value in zip(
                pd.date_range('2026-06-01 12:00',periods=3,freq='h',tz='UTC'),[100,200,300])]

    class Connection:
        def __init__(self):self.cur=Cursor()
        def cursor(self):return self.cur

    conn=Connection()
    monkeypatch.setattr(apply_parks.era5_db,'load_solar_points',lambda _:pd.DataFrame(
        {'cell_id':[7],'lat':[50.],'lon':[10.]}))
    idx=pd.date_range('2026-06-01 12:30',periods=3,freq='30min',tz='UTC')
    h=apply_parks.era5_solar_30min(conn,50.,10.,idx)
    result=apply_parks.onto_30min(h,idx)
    assert 's.valid_time >= %s AND s.valid_time <= %s' in conn.cur.sql
    assert conn.cur.params==(7,idx.min().floor('h').to_pydatetime(),idx.max().ceil('h').to_pydatetime())
    np.testing.assert_allclose(result.era5_albedo,[150,200,250])
    np.testing.assert_allclose(result.era5_ghi,np.array([200,200,300])/3600)


@pytest.mark.parametrize('tz',[None,'UTC'])
def test_shared_weather_coverage_accepts_naive_and_aware_database_times(tz):
    from parks import era5_db
    bounds=pd.date_range('2026-06-01',periods=2,freq='h',tz=tz)

    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,params):pass
        def fetchone(self):return tuple(bounds.to_pydatetime())

    class Connection:
        def cursor(self):return Cursor()

    lo,hi=era5_db.coverage(Connection(),7)
    assert lo==pd.Timestamp('2026-06-01 00:00',tz='UTC')
    assert hi==pd.Timestamp('2026-06-01 01:00',tz='UTC')
