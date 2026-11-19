import numpy as np
import pandas as pd
import pytest
from round2 import wind_physics as wp

def test_supersaturation_does_not_change_other_hours_units():
    # A fractional humidity of 1.0001 is not evidence that every value is percent.
    pressure=np.array([101325.,101325.]);temperature=np.array([288.15,288.15])
    saturation=wp.vapor_pressure(temperature-273.15)
    fractions=np.array([.8,1.0001])
    rho=wp.moist_air_density(pressure,temperature,fractions*saturation)
    expected=(pressure[0]-.8*saturation[0])/(287.05*temperature[0])+.8*saturation[0]/(461.5*temperature[0])
    assert rho[0]==pytest.approx(expected,abs=1e-14)
    assert rho[1]<rho[0]  # More water vapor lowers density at constant p,T.

@pytest.mark.parametrize('rho',[1.1,1.225,1.3])
def test_curve_shift_persists_above_nominal_rated_speed(rho):
    ticks=np.arange(0.,31.,.01);watts=np.minimum(2e6,(ticks/12.)**3*2e6)
    v=np.array([0.,2.,8.,12.,13.,20.,25.,25.01]);df=.75
    expected=np.interp(v*df**(1/3)*(rho/1.225)**(1/3),ticks,watts)
    expected=np.minimum(expected,.5*rho*np.pi*45**2*v**3)
    expected=np.where((v<3)|(v>25),0,expected)
    expected=np.minimum(expected,1.8e6)
    actual=wp.turbine_power(v,rho,ticks,watts,90,1.8e6,3,25,df)
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=1e-8)
    assert actual[-2]>0 and actual[-1]==0

def test_fixed_nameplate_does_not_scale_with_curve():
    v=np.array([20.]);ticks=np.array([0.,10.,30.]);watts=np.array([0.,2e6,2e6])*1.03
    assert wp.turbine_power(v,1.225,ticks,watts,100,2e6,3,25)[0]==pytest.approx(2e6)

def test_cp_cap_is_applied():
    v=np.array([4.]);rho=1.2;diameter=10.
    actual=wp.turbine_power(v,rho,[0,3,25],[0,2e6,2e6],diameter,2e6,3,25)
    expected=.5*rho*np.pi*(diameter/2)**2*v**3
    np.testing.assert_allclose(actual,expected)

def test_wake_interpolation_preserves_grid():
    factors={.05:np.array([.7,1]),.075:np.array([.8,1]),.1:np.array([.9,1])}
    for k,w in factors.items():np.testing.assert_array_equal(wp.interpolate_wake_factor(k,factors),w)
    np.testing.assert_allclose(wp.interpolate_wake_factor(.0625,factors),[.75,1])

def test_generator_density_consumes_fraction_without_inference():
    import generate_wind as gw
    data=pd.DataFrame({'pressure':[101325.,101325.],'temp_2m':[288.15,288.15],
                       'sat_vap_pressure':[1705.,1705.],'relhum':[.8,1.0001]})
    one=gw.get_rho(data.iloc[:1]).iloc[0]
    together=gw.get_rho(data).iloc[0]
    assert together==pytest.approx(one,abs=1e-14)
    assert together==pytest.approx((101325-.8*1705)/(287.05*288.15)+(.8*1705)/(461.5*288.15))

def test_generator_and_shared_power_agree():
    import generate_wind as gw
    index=pd.date_range('2024-01-01',periods=100,freq='h')
    v=np.linspace(0,26,100);rho=np.linspace(1.05,1.30,100);df=np.linspace(.70,.95,100)
    curve=pd.Series(np.minimum(2e6,(np.arange(0,26,.1)/12)**3*2e6),index=np.arange(0,26,.1))
    data=pd.DataFrame({'wind_speed_hub':v,'density_hub':rho},index=index)
    cp=gw.get_cp_from_power_curve_hardcoded(data,curve,'wind_speed_hub','density_hub',90,df,'iec')
    actual=gw.generate_wind_power(data,{'noise':0.},1.8e6,cp,90,3,25,12,df,density_mode='iec')
    expected=wp.turbine_power(v,rho,curve.index,curve.values,90,1.8e6,3,25,df)
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=1e-8)
