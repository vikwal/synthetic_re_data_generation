"""Synthetic PV power: modular chain (driver -> transposition -> module -> inverter -> losses).

Replaces the monolithic generate_pv.py. Modules:
  timebase       timestamp conventions, hourly means, solar position at interval centre
  drivers        irradiance/met drivers: imputed DWD station, CAMS, ERA5 (DB)
  transposition  DNI closure, POA per (tilt, azimuth), horizon shading, IAM
  arrays         sub-array mixtures, terrain-derived (tilt, azimuth) distributions
  modules        cell temperature, DC power (PVWatts or CEC single diode), library lookup
  inverter       AC power (PVWatts or Sandia), inverter selection
  aging          PV degradation (LID + linear + late acceleration) by technology era
  soiling        PM-deposition soiling model (fixed sign)
  chain          run one park config end to end
  angles         estimate effective tilt/azimuth from power measurements
  metrics        benchmark metrics against meter data
"""
