"""Shared deterministic wind-to-power physics for generation and sensitivity.

Units are explicit: pressure Pa, temperature K, density kg/m3, speed m/s,
power W. Relative humidity is a fraction, never inferred from sample maxima.
"""
import numpy as np

RHO_REFERENCE = 1.225

def vapor_pressure(temperature_c, model="huang"):
    t = np.asarray(temperature_c, dtype=float)
    if model == "huang":
        return np.where(t > 0,
                        np.exp(34.494 - 4924.99 / (t + 237.1)) / (t + 105) ** 1.57,
                        np.exp(43.494 - 6545.8 / (t + 278)) / (t + 868) ** 2)
    if model == "improved_magnus":
        return np.where(t > 0, 610.94 * np.exp(17.625 * t / (t + 243.04)),
                        611.21 * np.exp(22.587 * t / (t + 273.86)))
    raise ValueError(f"Unknown saturation-vapor-pressure model: {model}")

def moist_air_density(pressure, temperature, water_vapor_pressure):
    p, t, e = np.broadcast_arrays(np.asarray(pressure, float),
                                  np.asarray(temperature, float),
                                  np.asarray(water_vapor_pressure, float))
    return (p - e) / (287.05 * t) + e / (461.5 * t)

def density_at_height(rho2m, temperature2m, height, lapse_rate=0.00649):
    return np.asarray(rho2m) * (1 - lapse_rate * (height - 2) / np.asarray(temperature2m)) ** (
        (0.028949 * 9.81) / (lapse_rate * 8.31451) - 1)

def curve_coefficient(speed, density, curve_speed, curve_power, rotor_diameter,
                      degradation=1.0, density_mode="iec"):
    """Reference-curve lookup with persistent age shift and bounded Cp.

Legacy modes retain their documented density treatment. In particular,
`dynamic` in the full generator is a historical unnormalized lookup;
the sensitivity caller maps its dynamic-density switch to `iec` explicitly.
"""
    v, rho, df = np.broadcast_arrays(np.asarray(speed, float),
                                     np.asarray(density, float),
                                     np.asarray(degradation, float))
    if density_mode == "iec":
        reference_density = rho
        normalized = v * (rho / RHO_REFERENCE) ** (1 / 3)
    elif density_mode == "dynamic":
        reference_density, normalized = rho, v
    elif density_mode in ("v1_mixed", "static_1225"):
        reference_density, normalized = RHO_REFERENCE, v
    else:
        raise ValueError(f"Unknown density mode: {density_mode}")
    power = np.interp(normalized * df ** (1 / 3), curve_speed, curve_power)
    denominator = 0.5 * reference_density * np.pi * (rotor_diameter / 2) ** 2 * v ** 3
    cp = np.divide(power, denominator, out=np.zeros_like(power), where=v != 0)
    return np.clip(cp, 0, 1)

def power_from_coefficient(speed, density, coefficient, rotor_diameter,
                           rated_power, cut_in, cut_out, density_mode="iec"):
    v = np.asarray(speed, float)
    rho = RHO_REFERENCE if density_mode == "static_1225" else np.asarray(density, float)
    power = 0.5 * rho * np.pi * (rotor_diameter / 2) ** 2 * coefficient * v ** 3
    return np.minimum(np.where((v < cut_in) | (v > cut_out), 0., power), rated_power)

def turbine_power(speed, density, curve_speed, curve_power, rotor_diameter,
                  rated_power, cut_in, cut_out, degradation=1.0, density_mode="iec"):
    cp = curve_coefficient(speed, density, curve_speed, curve_power, rotor_diameter,
                           degradation, density_mode)
    return power_from_coefficient(speed, density, cp, rotor_diameter, rated_power,
                                  cut_in, cut_out, density_mode)

def interpolate_wake_factor(k, factors):
    """Interpolate hourly factors without changing endpoint values."""
    keys = sorted(factors)
    if not keys:
        raise ValueError("No wake factors supplied")
    if len(keys) == 1:
        return factors[keys[0]]
    value = np.clip(k, keys[0], keys[-1])
    upper = min(np.searchsorted(keys, value, side="right"), len(keys) - 1)
    lower = upper - 1
    return factors[keys[lower]] + (value - keys[lower]) / (keys[upper] - keys[lower]) * (
        factors[keys[upper]] - factors[keys[lower]])
