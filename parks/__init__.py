"""Real MaStR wind parks (park = MaStR Lokation) for the FL benchmark, v1.

A park is a list of turbine groups (power-curve type, hub height,
commissioning year, rating cap, ERA5 cell) with real per-turbine coordinates.
Each group runs through the unchanged round-2 chain functions of
generate_wind.py (M4-noQM: dynamic power law, no QM, Weibull aging per group);
the park's free-stream sum is multiplied by a PyWake NOJ wake factor computed
from the same run's hub wind. ERA5 is read directly from Postgres
public.era5_wind_grid. Driver: scripts/parks/run_parks_v1.py.
"""
