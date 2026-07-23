"""Shared paper figure style: pseudonyms, ages, fonts (HANDOFF asks 9-11).

Park pseudonyms are the authoritative Table-4.1 names (source: round-1
results_summarize.ipynb cell-5 mapping, spelling corrected to the manuscript:
'North Rhine-Westphalia', 'Mecklenburg-Western Pomerania'). No station IDs or
real park/site names may appear in any figure (confidentiality).
"""

import os

import matplotlib
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# station_id -> paper pseudonym (Table 4.1)
PSEUDONYM = {
    "00198_1": "Saxony-Anhalt",
    "00198_2": "Thuringia",
    "00282": "Bavaria",
    "00298": "Mecklenburg-Western Pomerania",
    "01200_1": "Schleswig-Holstein West",
    "01200_2": "Schleswig-Holstein East",
    "01303_1": "North Rhine-Westphalia East",
    "01303_2": "North Rhine-Westphalia West",
    "02483": "Hesse South",
    "04745": "Lower Saxony",
    "05347": "Hesse North",
    "05426": "Palatinate",
    "07374": "North Rhine-Westphalia North",
}

# abbreviated pseudonyms for space-constrained axis labels (e.g. the
# dumbbell figure y-axis) -- same identities as PSEUDONYM, shorter strings
PSEUDONYM_SHORT = {
    "00198_1": "Saxony-Anhalt",
    "00198_2": "Thuringia",
    "00282": "Bavaria",
    "00298": "MWP",
    "01200_1": "SH-West",
    "01200_2": "SH-East",
    "01303_1": "NRW-East",
    "01303_2": "NRW-West",
    "02483": "Hesse South",
    "04745": "Lower Saxony",
    "05347": "Hesse North",
    "05426": "Palatinate",
    "07374": "NRW-North",
}

# SA parameter names -> paper Table 4.3
SA_NAME = {
    "wind_level_factor": "Wind-speed factor",
    "correction_mode": "Correction mode",
    "shear_method": "Shear method",
    "z0_scale": "Roughness scale $z_0$",
    "z0_scale_log": "Roughness scale $z_0$",
    "aging_lambda": r"Aging scale $\lambda_a$",
    "aging_kappa": r"Aging shape $\kappa_a$",
    "power_curve_scale": "Power-curve scale",
    "density_mode": "Density mode",
    "wake_k": "Wake decay $k_w$",
}

# paper rung names (renumbered: M1-M4 is the deployed dynamic-exponent
# path, M2all=M3 (was unlabeled "M2+aging"), M5all_PL=M4 (was M6, deployed).
# S1-S4 is the validated-but-not-deployed stability-profile path (was
# M3/M4/M5/M7): S1=+stability, S2=+aging, S3=+wakes (full alt. chain),
# S4=height-gated hybrid (no single config id, not in this dict).
PAPER_RUNG = {"M1noage": "M1", "M2noage": "M2", "M2all": "M3",
              "M5all_PL": "M4",
              "M3noage": "S1", "M4all": "S2", "M5all": "S3", "MK": "MK",
              "M5all_noQM": "S3-noQM",
              "M5all_noage": "S3-noage",
              "M5all_PL_noage": "M4-noage", "M5all_PL_noQM": "M4-noQM",
              "M5all_PL_noWake": "M4-noWake"}


def rung(r: str) -> str:
    return PAPER_RUNG.get(r, r)


def park_age(park_id: str, at: str = "2023-12-01") -> float:
    # Ask 12: single authoritative source (config-derived table)
    from round2 import parkinfo
    cd = parkinfo.commissioning_date(park_id)
    return (pd.Timestamp(at) - pd.Timestamp(cd)).days / 365.25


def parks_by_age():
    """[(park_id, pseudonym, age_years), ...] sorted by age ascending."""
    rows = [(pid, name, park_age(pid)) for pid, name in PSEUDONYM.items()]
    return sorted(rows, key=lambda r: r[2])


def apply_print_style():
    """Ask 13a: match the round-1 figure ratio (fig5: ~22 pt at 10x6.5 in).

    Single-column figures use figsize ~(10, 6.5) with these sizes; multi-panel
    figure*-wide exports scale figsize up so the per-panel ratio stays similar.
    """
    matplotlib.rcParams.update({
        "font.size": 20, "axes.labelsize": 22, "axes.titlesize": 20,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 20, "figure.dpi": 100,
        "axes.titlepad": 10,
    })
