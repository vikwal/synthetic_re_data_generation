import pvlib
from renewable_energy_forecast.src import openmeteo
from renewable_energy_forecast.src.utils import check_inputs
from renewable_energy_forecast.src.utils import make_like


PARAM_WIND = 'wind_speed_10m'
PARAM_TEMP = 'temperature_2m'
PARAM_PRESSURE = 'surface_pressure'
PARAM_ELEVATION = 'elevation'
ELEVATION_DEFAULT = 10
GAMMA_PDC = -0.0035


def get_power_pvwatts(weather, latitude, longitude, elevation, azimuth, tilt, installed_power):
    # Set missing values
    if elevation is None:
        try:
            elevation = weather[PARAM_ELEVATION].dropna().values[0]
        except [KeyError, IndexError]:
            elevation = ELEVATION_DEFAULT
    try:
        pressure = weather[PARAM_PRESSURE]
    except KeyError:
        pressure = pvlib.atmosphere.alt2pres(elevation)

    # Get solar position
    solpos = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=latitude,
        longitude=longitude,
        altitude=elevation,
        temperature=weather[PARAM_TEMP],
        pressure=pressure,
    )
    # Get total irradiance
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        tilt,
        azimuth,
        solpos['apparent_zenith'],
        solpos['azimuth'],
        weather['dni'],
        weather['ghi'],
        weather['dhi'],
        dni_extra=pvlib.irradiance.get_extra_radiation(weather.index),
        model='haydavies',
    )
    # Calculate solar power
    cell_temperature = pvlib.temperature.faiman(total_irradiance['poa_global'], weather[PARAM_TEMP],
                                                weather[PARAM_WIND], u0=25.0, u1=6.84)
    power_dc = pvlib.pvsystem.pvwatts_dc(total_irradiance['poa_global'], cell_temperature, installed_power,
                                         gamma_pdc=GAMMA_PDC, temp_ref=25.0)
    return pvlib.inverter.pvwatts(power_dc, installed_power, eta_inv_nom=0.96, eta_inv_ref=0.9637)


def get_power_hist(latitude, longitude, elevation, azimuth, tilt, installed_power, start, end,
                   weather=None, weather_provider='open-meteo'):
    # Check mandatory input
    try:
        check_inputs(latitude, longitude, azimuth, tilt, installed_power)
    except ValueError as e:
        print(e)
        return None
    # Make these lists if necessary
    elevation = make_like(elevation, latitude)
    start = make_like(start, latitude)
    end = make_like(end, latitude)
    weather = make_like(weather, latitude)

    # Calculate weather/radiation if not provided as input
    if (weather is None) or all(v is None for v in weather):
        if weather_provider == 'open-meteo':
            weather = openmeteo.get_weather_hist(latitude, longitude, elevation, start, end)
        else:
            print(f'Invalid weather provider {weather_provider}')
            return None

    # Calculate powers
    if isinstance(latitude, list):
        power = [get_power_pvwatts(*items) for items in
                 zip(*(weather, latitude, longitude, elevation, azimuth, tilt, installed_power))]
    else:
        power = get_power_pvwatts(weather, latitude, longitude, elevation, azimuth, tilt, installed_power)
    return power