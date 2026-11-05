"""Wind-farm layout generator: a rotated rectangular grid of turbines."""
from __future__ import annotations

import math
from typing import Dict, List

from .wake import offset_to_latlon


def grid_layout(center_lat: float, center_lon: float, rows: int, cols: int,
                spacing_along_d: float, spacing_across_d: float,
                rotor_diameter_m: float, rotation_deg: float = 0.0) -> List[Dict[str, float]]:
    """
    Turbine positions on a rows x cols grid centred on (center_lat, center_lon).

    spacing_along_d  : spacing between turbines within a row, in rotor diameters
    spacing_across_d : spacing between rows, in rotor diameters
    rotation_deg     : clockwise rotation of the grid from north (rows run east-west at 0)
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if rows * cols > 200:
        raise ValueError("A layout may contain at most 200 turbines.")
    if rotor_diameter_m <= 0 or spacing_along_d <= 0 or spacing_across_d <= 0:
        raise ValueError("Rotor diameter and spacings must be positive.")

    dx = spacing_along_d * rotor_diameter_m
    dy = spacing_across_d * rotor_diameter_m
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    out = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * dx          # east before rotation
            y = (r - (rows - 1) / 2.0) * dy          # north before rotation
            # clockwise rotation from north
            east = x * cos_t + y * sin_t
            north = -x * sin_t + y * cos_t
            lat, lon = offset_to_latlon(center_lat, center_lon, east, north)
            out.append({"lat": round(float(lat), 6), "lon": round(float(lon), 6)})
    return out
