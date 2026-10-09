"""Metrics of the validation report (FL_Contribution/reports/curtailment_synthesis_v1.md).

Pure functions on frames/arrays; I/O and figures live in parks/curtail_validate.py.
Reference values (rows of calibration_numbers.md) are given where a metric is
compared against one.
"""

import numpy as np
import pandas as pd

from curtailment import areas as areas_mod

# [C7] KIT 2015-17 and [C15] EWE: curtailed hours per affected unit and year, p10 / p50 / p90
REF_HOURS = {"A1_SH": ("SH Netz (KIT, C7)", 11, 481, 1881), "A2_NI_NW": ("EWE 2024 (C15)", 12, 55, 367),
             "A3_NI_O_ST": ("E.DIS (KIT, C7)", 2, 33, 620), "A4_NO": ("E.DIS (KIT, C7)", 2, 33, 620),
             "A5_MITTE_W": ("Avacon (KIT, C7)", 1.0, 11, 72)}
# [C19] depth-weighted share of (ever affected) units by CF_DA bin
REF_COUPLING = {"A1_SH": (0.8, 9.5, 23.5, 31.9), "A2_NI_NW": (0.2, 1.0, 5.4, 8.6),
                "A3_NI_O_ST": (0.0, 1.5, 5.9, 12.5), "A4_NO": (0.0, 1.5, 5.9, 12.5),
                "A5_MITTE_W": (0.0, 0.1, 1.3, 3.9)}
COUPLING_BINS = ((0.0, 0.05), (0.2, 0.25), (0.4, 0.45), (0.6, 0.65))
# [C28] area event start rate [1/h] by CF bin 0-0.1 / 0.3-0.4 / 0.5-0.6
REF_START = {"A1_SH": (0.34, 3.66, 3.61), "A2_NI_NW": (0.14, 1.33, 2.25), "A3_NI_O_ST": (0.037, 0.97, 1.98),
             "A4_NO": (0.037, 0.97, 1.98), "A5_MITTE_W": (0.002, 0.10, 0.32)}
START_BINS = ((0.0, 0.1), (0.3, 0.4), (0.5, 0.6))
# [C40]-[C43] Netzampel SH 2023-25 (commune flags)
REF_RUNS = {"p10": 1, "p25": 3, "p50": 6, "p75": 12, "p90": 22, "p99": 64, "mean": 11.1}
REF_GAPS = {"p25": 8, "p50": 23, "p75": 79, "p90": 168}
REF_PHI = {"<10": 0.88, "10-20": 0.52, "20-40": 0.52, "40-80": 0.39, "80-150": 0.31}
DIST_BINS = ((0, 10), (10, 20), (20, 40), (40, 80), (80, 150))
# [C3] KIT / [C12] EWE: share of curtailed hours at setpoint 0 / 30 / 60 %
REF_SETPOINT_HOURS = {"SH Netz (C3)": (90.7, 5.6, 3.6), "E.DIS (C3)": (79.6, 12.7, 6.4),
                      "Avacon (C3)": (88.9, 6.1, 4.8), "EWE 2024 (C12)": (77.5, 10.0, 12.6),
                      "EWE 2025 (C12)": (70.1, 12.9, 17.0)}
# [C25] monthly depth-weighted share Dec / Jun; [C45] Netzampel flagged-hour share Nov-Jan vs summer
REF_SEASON = {"SH Netz (C25)": (15.2, 5.5), "E.DIS (C25)": (4.2, 1.0), "Avacon (C25)": (0.9, 0.2),
              "EWE (C25)": (2.6, 0.8)}
# [D7b] monthly onshore market value (Hirth 2026, scaled to the official annual value)
REF_RM = {2023: (85, 101, 82, 86, 78, 88, 59, 65, 84, 67, 73, 44),
          2024: (64, 53, 55, 49, 59, 58, 53, 65, 64, 68, 89, 72),
          2025: (84, 112, 77, 75, 63, 55, 80, 68, 64, 57, 86, 81)}
# [D4] exposure / response / curtailment of wind in negative hours [%]
REF_D4 = {2023: (56, 32, 18), 2024: (57, 26, 15), 2025: (64, 27, 17), 2026: (70, 32, 22)}
# [D5] curtailment by episode length 1-3 h / 4-5 h / >= 6 h [% of potential in negative hours]
REF_D5 = {2023: (10.3, 10.8, 20.5), 2024: (-2.1, 3.6, 22.0), 2025: (-0.2, 6.8, 20.3), 2026: (6.0, 5.5, 32.8)}


def quantiles(x, qs=(0.1, 0.5, 0.9)) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {f"p{int(100 * q)}": np.nan for q in qs}
    return {f"p{int(100 * q)}": float(np.quantile(x, q)) for q in qs}


def runs(flag) -> tuple:
    """Lengths of the runs of 1 and of the gaps between runs (in samples)."""
    f = np.concatenate([[0], np.asarray(flag, np.int8), [0]])
    d = np.diff(f)
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return b - a, (a[1:] - b[:-1]) if len(a) > 1 else np.array([], int)


def phi_by_distance(flags: pd.DataFrame, lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    """Pairwise phi (Pearson on 0/1) of hourly flags by distance class."""
    ids = list(flags.columns)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = flags[ids[i]].to_numpy(float), flags[ids[j]].to_numpy(float)
            if a.std() == 0 or b.std() == 0:
                continue
            d = float(areas_mod.haversine_km(lat[ids[i]], lon[ids[i]], lat[ids[j]], lon[ids[j]]))
            rows.append({"a": ids[i], "b": ids[j], "dist_km": d, "phi": float(np.corrcoef(a, b)[0, 1])})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["bin"] = pd.cut(df["dist_km"], [b[0] for b in DIST_BINS] + [DIST_BINS[-1][1]],
                       labels=list(REF_PHI), right=False)
    return df


def setpoint_hour_shares(s_q: np.ndarray) -> tuple:
    """Shares [%] of curtailed quarter-hours at setpoint 0 / 0.3 / 0.6."""
    s = np.asarray(s_q)
    c = s < 1
    n = c.sum()
    if not n:
        return (np.nan, np.nan, np.nan)
    return tuple(float(100 * (np.isclose(s[c], v)).sum() / n) for v in (0.0, 0.3, 0.6))


def detector(power: pd.Series, wind: pd.Series, cap_w: float, q: float = 0.9, ratio: float = 0.7,
             min_q: float = 0.2, bin_ms: float = 0.5) -> pd.Series:
    """UniWind-style rule (report 3(b)): q-quantile power curve by wind bin; an hour is
    flagged as curtailed if p / q_curve <= ratio while q_curve >= min_q * P_cap."""
    b = np.floor(wind / bin_ms)
    curve = power.groupby(b).quantile(q)
    qc = b.map(curve).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        flag = (qc >= min_q * cap_w) & (power.to_numpy() / qc <= ratio)
    return pd.Series(flag, index=power.index)


def detection_scores(flag: pd.Series, true_flag: pd.Series, loss: pd.Series) -> dict:
    tp = (flag & true_flag).sum()
    return {"hours_flagged": int(flag.sum()), "hours_true": int(true_flag.sum()),
            "precision": float(tp / flag.sum()) if flag.sum() else np.nan,
            "recall_hours": float(tp / true_flag.sum()) if true_flag.sum() else np.nan,
            "recall_energy": float(loss[flag].sum() / loss.sum()) if loss.sum() > 0 else np.nan}
