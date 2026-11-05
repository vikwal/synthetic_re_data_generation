"""Real vs synthetic comparison: parse an uploaded plant file, align it with a stored
synthetic result, and compute error metrics plus Plotly figure dicts."""
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import charts
from .charts import ACCENT, SLATE, _axis, _clean, _layout, fig

UNIT_FACTORS = {"W": 1.0, "kW": 1e3, "MW": 1e6}
MAX_CSV_CHARS = 30_000_000
REAL_COLOR = SLATE
_TS_NAMES = {"timestamp", "time", "date", "datetime", "index", "utc", "ts"}


# ------------------------------------------------------------------ parsing
def _read_table(text: str) -> pd.DataFrame:
    """Read a CSV with ';' or ',' separators; fall back to decimal commas."""
    head = text[:4096]
    sep = ";" if head.count(";") >= head.count(",") and ";" in head else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    non_ts = df.iloc[:, 1:]
    if len(non_ts.columns) and all(non_ts[c].dtype == object for c in non_ts.columns):
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=",")
    return df


def parse_real_csv(text: str, unit: str = "W") -> pd.DataFrame:
    """
    Real plant file -> DataFrame indexed by UTC timestamp with `power` (W) and, if present,
    `wind_speed`. Accepts `power` or `power_t1..tN` (summed), `wind_speed`, `wind_speed_mean`
    or `wind_speed_t1..tN` (averaged). The timestamp is the column named like a time column,
    otherwise the first column.
    """
    if unit not in UNIT_FACTORS:
        raise ValueError(f"Unit must be one of {', '.join(UNIT_FACTORS)}.")
    if not text or not text.strip():
        raise ValueError("The uploaded file is empty.")
    if len(text) > MAX_CSV_CHARS:
        raise ValueError("The uploaded file is too large (limit about 30 MB).")
    try:
        df = _read_table(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the file as CSV: {exc}")
    if df.shape[1] < 2:
        raise ValueError("The file needs a timestamp column and at least one data column.")

    ts_col = next((c for c in df.columns if str(c).strip().lower() in _TS_NAMES), df.columns[0])
    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if idx.isna().mean() > 0.5:
        raise ValueError(f"Could not read timestamps from column '{ts_col}'.")
    df = df.drop(columns=[ts_col])
    df.index = idx
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce")

    low = {str(c).strip().lower(): c for c in df.columns}
    power_t = [low[c] for c in low if re.fullmatch(r"power_t\d+", c)]
    if power_t:
        power = df[power_t].sum(axis=1, min_count=1)
    elif "power" in low:
        power = df[low["power"]]
    else:
        raise ValueError("No power column found. Expected `power` or `power_t1`, `power_t2`, …")

    speed_t = [low[c] for c in low if re.fullmatch(r"wind_speed_t\d+", c)]
    out = pd.DataFrame({"power": power * UNIT_FACTORS[unit]})
    if speed_t:
        out["wind_speed"] = df[speed_t].mean(axis=1)
    else:
        for name in ("wind_speed", "wind_speed_mean"):
            if name in low:
                out["wind_speed"] = df[low[name]]
                break
    return out


# ------------------------------------------------------------------ alignment + metrics
def align(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """Hourly means of both series over their overlap: columns real, synthetic (W)."""
    r = real["power"].astype(float).resample("h").mean()
    s = synth["power"].astype(float).resample("h").mean()
    if r.index.tz is None:
        r.index = r.index.tz_localize("UTC")
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    joined = pd.concat({"real": r, "synthetic": s}, axis=1).dropna()
    if joined.empty:
        raise ValueError("The real file and the synthetic result share no overlapping hours. "
                         "Check that the date ranges and timestamps (UTC) match.")
    return joined


def metrics(joined: pd.DataFrame, capacity_w: float, kind: str) -> Dict[str, Optional[float]]:
    div, unit = charts.power_unit(kind)
    real, syn = joined["real"].values, joined["synthetic"].values
    err = syn - real
    mean_r, mean_s = float(real.mean()), float(syn.mean())
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(real, syn)[0, 1]) if real.std() > 0 and syn.std() > 0 else None
    hours = len(joined)
    e_real, e_syn = float(real.sum() / 1e6), float(syn.sum() / 1e6)      # MWh (hourly steps)
    cap = capacity_w if capacity_w and capacity_w > 0 else None

    def r(x, d=3):
        return None if x is None or not np.isfinite(x) else round(float(x), d)

    return {
        "unit": unit, "hours": hours,
        "start": joined.index.min().strftime("%Y-%m-%d"), "end": joined.index.max().strftime("%Y-%m-%d"),
        "mean_real": r(mean_r / div), "mean_synthetic": r(mean_s / div),
        "bias": r((mean_s - mean_r) / div), "bias_pct": r(100 * (mean_s - mean_r) / mean_r, 2) if mean_r else None,
        "mae": r(mae / div), "rmse": r(rmse / div),
        "nmae_pct": r(100 * mae / cap, 2) if cap else None,
        "nrmse_pct": r(100 * rmse / cap, 2) if cap else None,
        "correlation": r(corr, 3),
        "energy_real_mwh": r(e_real, 1), "energy_synthetic_mwh": r(e_syn, 1),
        "energy_diff_pct": r(100 * (e_syn - e_real) / e_real, 2) if e_real else None,
        "cf_real_pct": r(100 * mean_r / cap, 2) if cap else None,
        "cf_synthetic_pct": r(100 * mean_s / cap, 2) if cap else None,
    }


# ------------------------------------------------------------------ figures
def _lines(joined: pd.DataFrame, x, kind: str, title: str, ytitle: str, **layout) -> dict:
    div, unit = charts.power_unit(kind)
    data = [
        {"type": "scatter", "mode": "lines", "name": "Real", "x": x, "y": _clean(joined["real"] / div),
         "line": {"color": REAL_COLOR, "width": 1.4}},
        {"type": "scatter", "mode": "lines", "name": "Synthetic", "x": x, "y": _clean(joined["synthetic"] / div),
         "line": {"color": ACCENT, "width": 1.4}},
    ]
    return fig(data, _layout(title, yaxis=_axis(ytitle), **layout))


def figures(joined: pd.DataFrame, kind: str) -> Dict[str, dict]:
    div, unit = charts.power_unit(kind)
    out: Dict[str, dict] = {}

    # 1. time series (daily means for long periods keeps the page fast)
    ts = joined if len(joined) <= 24 * 45 else joined.resample("D").mean().dropna()
    label = "hourly" if ts is joined else "daily mean"
    x = [d.strftime("%Y-%m-%d %H:%M") for d in ts.index]
    out["timeseries"] = _lines(ts, x, kind, f"Real vs synthetic power ({label})", f"Power ({unit})",
                               xaxis=_axis("Date", rangeslider={"visible": len(ts) > 200}))

    # 2. scatter with 1:1 line
    samp = joined if len(joined) <= 4000 else joined.sample(4000, random_state=0)
    top = float(max(joined["real"].max(), joined["synthetic"].max()) / div)
    out["scatter"] = fig([
        {"type": "scattergl", "mode": "markers", "name": "Hours", "x": _clean(samp["real"] / div),
         "y": _clean(samp["synthetic"] / div), "marker": {"size": 4, "color": ACCENT, "opacity": 0.35}},
        {"type": "scatter", "mode": "lines", "name": "1:1", "x": [0, top], "y": [0, top],
         "line": {"color": SLATE, "dash": "dash", "width": 1}},
    ], _layout("Hourly agreement", hovermode="closest", showlegend=False,
               xaxis=_axis(f"Real ({unit})"), yaxis=_axis(f"Synthetic ({unit})")))

    # 3. diurnal profile (UTC hours)
    byh = joined.groupby(joined.index.hour).mean()
    out["diurnal"] = _lines(byh, [int(h) for h in byh.index], kind, "Average daily profile (UTC)",
                            f"Power ({unit})", xaxis=_axis("Hour of day (UTC)", dtick=3))

    # 4. monthly energy
    m = (joined / 1e6).groupby([joined.index.year, joined.index.month]).sum()
    labels = [f"{y}-{mo:02d}" for y, mo in m.index]
    out["monthly"] = fig([
        {"type": "bar", "name": "Real", "x": labels, "y": _clean(m["real"]), "marker": {"color": REAL_COLOR}},
        {"type": "bar", "name": "Synthetic", "x": labels, "y": _clean(m["synthetic"]), "marker": {"color": ACCENT}},
    ], _layout("Monthly energy", barmode="group", xaxis=_axis("Month"), yaxis=_axis("Energy (MWh)")))

    # 5. duration curves
    n = min(len(joined), 200)
    pos = np.linspace(0, 100, n)
    take = np.linspace(0, len(joined) - 1, n).astype(int)
    d_real = np.sort(joined["real"].values)[::-1][take] / div
    d_syn = np.sort(joined["synthetic"].values)[::-1][take] / div
    out["duration"] = fig([
        {"type": "scatter", "mode": "lines", "name": "Real", "x": _clean(pos), "y": _clean(d_real),
         "line": {"color": REAL_COLOR, "width": 2}},
        {"type": "scatter", "mode": "lines", "name": "Synthetic", "x": _clean(pos), "y": _clean(d_syn),
         "line": {"color": ACCENT, "width": 2}},
    ], _layout("Duration curves", xaxis=_axis("Share of time (%)"), yaxis=_axis(f"Power ({unit})")))
    return out


def compare(real: pd.DataFrame, synth: pd.DataFrame, capacity_w: float, kind: str) -> Tuple[dict, Dict[str, dict]]:
    joined = align(real, synth)
    return metrics(joined, capacity_w, kind), figures(joined, kind)
