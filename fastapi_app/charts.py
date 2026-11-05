"""
Plotly figure builders. They return plain JSON-able dicts for Plotly.js, so the Python
plotly package is not needed. Visual language: slate text, light grid, emerald accent.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import analytics

ACCENT = "#059669"
SLATE = "#0f172a"
GRID = "#e2e8f0"
SERIES = ["#059669", "#334155", "#f59e0b", "#0d9488", "#84cc16", "#94a3b8"]   # emerald first, slate/amber as contrast
QUARTER_COLORS = ["#a7f3d0", "#6ee7b7", "#34d399", "#059669"]
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"


def _clean(values) -> List[Optional[float]]:
    arr = np.asarray(values, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]


def _axis(title: str = "", **extra) -> dict:
    ax = {"title": {"text": title}, "gridcolor": GRID, "zerolinecolor": GRID, "linecolor": "#94a3b8",
          "showline": True, "linewidth": 1, "tickangle": 0,
          "ticks": "outside", "tickcolor": "#cbd5e1", "automargin": True}
    ax.update(extra)
    return ax


def _layout(title: str = "", **extra) -> dict:
    layout = {
        "title": {"text": f"<b>{title}</b>" if title else "", "x": 0.5, "xref": "container", "xanchor": "center", "y": 0.97, "yanchor": "top", "font": {"size": 15, "color": SLATE}},
        "margin": {"l": 56, "r": 24, "t": 48, "b": 48},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 12, "color": "#334155"},
        # Legend above the plot, right-aligned, so it never collides with the x-axis title.
        "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"},
        "xaxis": _axis(), "yaxis": _axis(), "hovermode": "x unified",
    }
    if extra.get("showlegend") is not False:
        layout["margin"] = {"l": 64, "r": 24, "t": 72, "b": 56}
    layout.update(extra)
    return layout


def fig(data: list, layout: dict) -> dict:
    return {"data": data, "layout": layout}


def power_unit(kind: str):
    """(divisor from W, label) — wind in MW, solar in kW."""
    return (1e6, "MW") if kind == "wind" else (1e3, "kW")


FEATURE_META = {
    "wind_speed_hub": ("Wind speed", "m/s"),
    "density_hub": ("Air density", "kg/m³"),
    "temperature_hub": ("Temperature", "°C"),
    "Total": ("Total irradiance", "W/m²"),
    "Direct": ("Direct irradiance", "W/m²"),
    "Diffuse": ("Diffuse irradiance", "W/m²"),
}


def available_features(df: pd.DataFrame, kind: str) -> List[str]:
    cands = ["wind_speed_hub", "density_hub", "temperature_hub"] if kind == "wind" else ["Total", "Direct", "Diffuse"]
    return [c for c in cands if c in df.columns]


# ---------------------------------------------------------------- overview
def seasonality_figure(data: dict, unit: str = "MWh") -> dict:
    return fig([{"type": "bar", "x": data["months"], "y": _clean(data["values"]),
                 "marker": {"color": ACCENT}, "name": "Mean monthly energy",
                 "hovertemplate": "%{x}: %{y:,.1f} " + unit + "<extra></extra>"}],
               _layout("Seasonality — mean monthly energy", xaxis=_axis("Month"), yaxis=_axis(f"Energy ({unit})"), showlegend=False,
                       hovermode="closest"))


def heatmap_figure(data: dict) -> dict:
    return fig([{"type": "heatmap", "x": data["x"], "y": data["y"], "z": data["z"],
                 "colorscale": [[0, "#f0fdf4"], [0.5, "#34d399"], [1, "#065f46"]],
                 "colorbar": {"title": {"text": "% of capacity"}, "thickness": 12},
                 "hovertemplate": "%{x}, %{y}:00 — %{z:.1f}%<extra></extra>"}],
               _layout("Typical day — output by hour and month", xaxis=_axis("Month"), yaxis=_axis("Hour of day", autorange="reversed",
                                                                          dtick=3), hovermode="closest"))


def duration_figure(data: dict) -> dict:
    return fig([{"type": "scatter", "mode": "lines", "x": data["x"], "y": data["y"], "fill": "tozeroy",
                 "line": {"color": ACCENT, "width": 2}, "fillcolor": "rgba(5,150,105,0.12)", "name": "Output",
                 "hovertemplate": "%{y:.1f}% of capacity exceeded %{x:.1f}% of the time<extra></extra>"}],
               _layout("Power duration curve", xaxis=_axis("Share of time (%)", range=[0, 100]),
                       yaxis=_axis("Output (% of capacity)", rangemode="tozero"), showlegend=False,
                       hovermode="closest"))


def windrose_figure(data: dict) -> dict:
    colors = ["#d1fae5", "#6ee7b7", "#10b981", "#047857", "#064e3b"]
    traces = [{"type": "barpolar", "r": s["values"], "theta": data["sectors"], "name": s["label"],
               "marker": {"color": c}, "hovertemplate": "%{theta}: %{r:.1f}%<extra>" + s["label"] + "</extra>"}
              for s, c in zip(data["series"], colors)]
    return fig(traces, _layout("Wind rose (direction the wind comes from)",
                               polar={"angularaxis": {"direction": "clockwise", "rotation": 90, "gridcolor": GRID},
                                      "radialaxis": {"ticksuffix": "%", "gridcolor": GRID}},
                               hovermode="closest", margin={"l": 40, "r": 110, "t": 64, "b": 40},
                               # Vertical legend in the right margin: a horizontal one wraps into the title on narrow cards.
                               legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 0.5, "yanchor": "middle"}))


def speed_hist_figure(data: dict) -> dict:
    return fig([{"type": "bar", "x": data["x"], "y": data["y"], "marker": {"color": ACCENT},
                 "hovertemplate": "%{x:.1f} m/s: %{y:.1f}%<extra></extra>"}],
               _layout("Wind speed distribution (hub height)", xaxis=_axis("Wind speed (m/s)"),
                       yaxis=_axis("Share of time (%)"), showlegend=False, hovermode="closest"))


def irradiance_monthly_figure(df: pd.DataFrame) -> Optional[dict]:
    if "Total" not in df.columns:
        return None
    dt = analytics.step_hours(df)
    per = (df["Total"].astype(float) * dt / 1000.0).groupby([df.index.year, df.index.month]).sum()
    mean = per.groupby(level=1).mean()
    return fig([{"type": "bar", "x": analytics.MONTHS, "y": _clean([mean.get(m, np.nan) for m in range(1, 13)]),
                 "marker": {"color": ACCENT}, "hovertemplate": "%{x}: %{y:.0f} kWh/m²<extra></extra>"}],
               _layout("Plane-of-array irradiation per month", xaxis=_axis("Month"), yaxis=_axis("kWh/m² per month"), showlegend=False,
                       hovermode="closest"))


# ---------------------------------------------------------------- performance (wind parks)
def waterfall_figure(e: dict) -> dict:
    """Ideal -> ageing -> wake -> net. Built from stacked bars (invisible base + coloured bar) so every
    step gets the same colour it has in the 'Loss per turbine' chart: ageing amber, wake light green."""
    unit, div = ("GWh", 1000.0) if e["ideal_mwh"] >= 5000 else ("MWh", 1.0)
    ideal, ageing, wake, net = (e["ideal_mwh"] / div, e["ageing_loss_mwh"] / div, e["wake_loss_mwh"] / div, e["net_mwh"] / div)
    labels = ["Ideal", "Ageing", "Wake", "Net"]
    heights = [ideal, ageing, wake, net]
    bases = [0.0, ideal - ageing, ideal - ageing - wake, 0.0]
    return fig([
        {"type": "bar", "x": labels, "y": [round(v, 4) for v in bases], "marker": {"color": "rgba(0,0,0,0)"},
         "hoverinfo": "skip", "showlegend": False},
        {"type": "bar", "x": labels, "y": [round(v, 4) for v in heights],
         "marker": {"color": [ACCENT, "#f59e0b", "#6ee7b7", "#334155"]},
         "text": [f"{v:,.1f}" if i in (0, 3) else f"−{v:,.2f}" for i, v in enumerate(heights)], "textposition": "outside",
         "cliponaxis": False, "hovertemplate": "%{x}: %{y:,.2f} " + unit + "<extra></extra>", "showlegend": False},
    ], _layout("From ideal to net energy", barmode="stack", xaxis=_axis("Loss stage"), yaxis=_axis(f"Energy ({unit})"),
               showlegend=False, hovermode="closest", margin={"l": 64, "r": 24, "t": 64, "b": 56}))


def degradation_summary_figure(sites: List[dict]) -> dict:
    labels = [f"T{s['slot']}" for s in sites]
    wake = [s.get("wake_loss_pct") or 0.0 for s in sites]
    ageing = [s.get("ageing_loss_pct") or 0.0 for s in sites]
    combined = [100 * (1 - (1 - w / 100) * (1 - a / 100)) for w, a in zip(wake, ageing)]
    traces = [{"type": "bar", "name": n, "x": labels, "y": _clean(v), "marker": {"color": c},
               "hovertemplate": "%{x}: %{y:.2f}%<extra>" + n + "</extra>"}
              for n, v, c in zip(("Wake loss", "Ageing loss", "Combined"), (wake, ageing, combined),
                                 ("#6ee7b7", "#f59e0b", "#334155"))]
    return fig(traces, _layout("Loss per turbine", barmode="group", yaxis=_axis("Energy loss (%)"),
                               xaxis=_axis("Turbine", type="category")))


def ageing_timeline_figure(sites: List[dict]) -> Optional[dict]:
    traces = []
    for k, s in enumerate([s for s in sites if s.get("ageing_curve")]):
        c = s["ageing_curve"]
        traces.append({"type": "scatter", "mode": "lines", "name": f"T{s['slot']}", "x": c["x"],
                       "y": _clean(np.asarray(c["y"]) * 100), "line": {"color": SERIES[k % len(SERIES)], "width": 2}})
    if not traces:
        return None
    return fig(traces, _layout("Ageing efficiency over time", xaxis=_axis("Date"), yaxis=_axis("Efficiency (%)")))


# ---------------------------------------------------------------- explorer
def daily_figure(df: pd.DataFrame, kind: str, date: str, feature: Optional[str]) -> dict:
    div, unit = power_unit(kind)
    day = df.loc[date]
    x = [t.isoformat() for t in day.index]
    traces = [{"type": "scatter", "mode": "lines", "name": f"Power ({unit})", "x": x,
               "y": _clean(day["power"] / div), "line": {"color": ACCENT, "width": 2.5}}]
    layout = _layout(f"Power on {date}", xaxis=_axis("Time of day"), yaxis=_axis(f"Power ({unit})"))
    if feature and feature in day.columns:
        label, funit = FEATURE_META.get(feature, (feature, ""))
        traces.append({"type": "scatter", "mode": "lines", "name": f"{label} ({funit})", "x": x,
                       "y": _clean(day[feature]), "yaxis": "y2", "line": {"color": "#334155", "width": 2, "dash": "dot"}})
        layout["yaxis2"] = _axis(f"{label} ({funit})", overlaying="y", side="right", showgrid=False)
    return fig(traces, layout)


def quarterly_figure(df: pd.DataFrame, kind: str, year: int) -> dict:
    div, unit = power_unit(kind)
    sub = df[df.index.year == year]
    quarters = [("Jan–Mar", (1, 2, 3)), ("Apr–Jun", (4, 5, 6)), ("Jul–Sep", (7, 8, 9)), ("Oct–Dec", (10, 11, 12))]
    traces = []
    for (name, months), colour in zip(quarters, QUARTER_COLORS):
        vals = sub.loc[sub.index.month.isin(months), "power"] / div
        # One accent colour, light fill + thin outline; the dashed line marks the mean (the solid one is the median).
        traces.append({"type": "box", "name": name, "y": _clean(vals), "boxpoints": False, "boxmean": True,
                       "width": 0.45, "marker": {"color": ACCENT}, "line": {"color": ACCENT, "width": 1.5},
                       "fillcolor": "rgba(5,150,105,0.18)"})
    return fig(traces, _layout(f"Quarterly power distribution — {year}", xaxis=_axis("Quarter"), yaxis=_axis(f"Power ({unit})"),
                               showlegend=False, hovermode="closest"))


def hour_profile_figure(df: pd.DataFrame, kind: str, year: int, hour: int) -> dict:
    """Power at one hour of the day, every day of the chosen year, plus its annual average."""
    div, unit = power_unit(kind)
    sub = df[(df.index.year == year) & (df.index.hour == hour)]
    if sub.empty:
        raise LookupError(f"No data for {hour:02d}:00 in {year}.")
    daily = (sub["power"] / div).groupby(sub.index.normalize()).mean()
    mean = float(daily.mean())
    x = [d.strftime("%Y-%m-%d") for d in daily.index]
    roll = daily.rolling(30, center=True, min_periods=7).mean()
    traces = [
        {"type": "scatter", "mode": "lines", "name": "Daily value", "x": x, "y": _clean(daily.values),
         "line": {"color": "rgba(5,150,105,0.35)", "width": 1}},
        {"type": "scatter", "mode": "lines", "name": "30-day average", "x": x, "y": _clean(roll.values),
         "line": {"color": ACCENT, "width": 2.5}},
        {"type": "scatter", "mode": "lines", "name": f"Annual average ({mean:,.3g} {unit})", "x": [x[0], x[-1]], "y": [mean, mean],
         "line": {"color": "#334155", "width": 1.6, "dash": "dash"}},
    ]
    return fig(traces, _layout(f"Power at {hour:02d}:00 through {year}", xaxis=_axis("Date"),
                               yaxis=_axis(f"Power ({unit})", rangemode="tozero")))


def hour_box_figure(df: pd.DataFrame, kind: str, year: int) -> dict:
    """Distribution of power for each of the 24 hours of the day (one box per hour) over the chosen year."""
    div, unit = power_unit(kind)
    sub = df[df.index.year == year]
    if sub.empty:
        raise LookupError(f"No data for {year}.")
    power = sub["power"] / div
    traces = []
    for hr in range(24):
        vals = power[sub.index.hour == hr]
        traces.append({"type": "box", "name": f"{hr:02d}", "y": _clean(vals), "boxpoints": False, "boxmean": True,
                       "width": 0.6, "marker": {"color": ACCENT}, "line": {"color": ACCENT, "width": 1.2},
                       "fillcolor": "rgba(5,150,105,0.18)", "showlegend": False})
    # Legend: the box plot draws the median as a solid line and the mean as a dashed line. Two empty traces carry
    # those line styles so the legend (top right, like the other plots) explains them.
    for name, dash in (("Median", "solid"), ("Mean", "dash")):
        traces.append({"type": "scatter", "mode": "lines", "name": name, "x": [None], "y": [None], "hoverinfo": "skip",
                       "line": {"color": ACCENT, "width": 1.6, "dash": dash}})
    return fig(traces, _layout(f"Power distribution by hour of day — {year}", xaxis=_axis("Hour of day", type="category"),
                               yaxis=_axis(f"Power ({unit})", rangemode="tozero"), hovermode="closest",
                               margin={"l": 64, "r": 24, "t": 72, "b": 56}))


def multi_year_figure(df: pd.DataFrame, kind: str) -> dict:
    div, unit = power_unit(kind)
    yearly = (df["power"] / div).groupby(df.index.year).sum() * analytics.step_hours(df)
    return fig([{"type": "bar", "x": [str(y) for y in yearly.index], "y": _clean(yearly.values),
                 "marker": {"color": ACCENT}, "hovertemplate": "%{x}: %{y:,.0f} " + unit + "h<extra></extra>"}],
               _layout("Energy per year", yaxis=_axis(f"Energy ({unit}h)"), xaxis=_axis("Year", type="category"),
                       showlegend=False, hovermode="closest"))


def turbine_curve_figure(points: dict, turbine: str) -> dict:
    # One style for every dot: manufacturer and interpolated points are merged.
    x = list(points["original"]["x"]) + list(points["interpolated"]["x"])
    y = list(points["original"]["y"]) + list(points["interpolated"]["y"])
    return fig([
        {"type": "scatter", "mode": "markers", "x": x, "y": y,
         "marker": {"color": ACCENT, "size": 5, "opacity": 0.7}},
    ], _layout(f"Power curve — {turbine}", xaxis=_axis("Wind speed (m/s)"), yaxis=_axis("Power (MW)"),
               showlegend=False, hovermode="closest"))


def park_curve_figure(speed, power_mw) -> dict:
    """Whole-park power curve: hourly park output against mean hub-height wind speed."""
    return fig([
        {"type": "scatter", "mode": "markers", "x": _clean(speed), "y": _clean(power_mw),
         "marker": {"color": ACCENT, "size": 4, "opacity": 0.45}},
    ], _layout("Power curve — whole park", xaxis=_axis("Mean hub-height wind speed (m/s)"),
               yaxis=_axis("Park power (MW)"), showlegend=False, hovermode="closest"))


# ---------------------------------------------------------------- compare
def compare_figures(items: List[dict]) -> Dict[str, dict]:
    """items: [{name, seasonality, monthly, duration}] → overlay figures."""
    seas, dur, series = [], [], []
    for k, it in enumerate(items):
        color = SERIES[k % len(SERIES)]
        seas.append({"type": "bar", "name": it["name"], "x": it["seasonality"]["months"],
                     "y": _clean(it["seasonality"]["values"]), "marker": {"color": color}})
        dur.append({"type": "scatter", "mode": "lines", "name": it["name"], "x": it["duration"]["x"],
                    "y": it["duration"]["y"], "line": {"color": color, "width": 2.5}})
        series.append({"type": "scatter", "mode": "lines", "name": it["name"], "x": it["monthly"]["labels"],
                       "y": _clean(it["monthly"]["values"]), "line": {"color": color, "width": 2}})
    # Legend above the plot (right-aligned) so it never collides with the x-axis title.
    top = {"legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"},
           "margin": {"l": 64, "r": 24, "t": 72, "b": 56}}
    return {
        "seasonality": fig(seas, _layout("Mean monthly energy", barmode="group", xaxis=_axis("Month"), yaxis=_axis("Energy (MWh)"), **top)),
        "duration": fig(dur, _layout("Power duration curves", xaxis=_axis("Share of time (%)", range=[0, 100]),
                                     yaxis=_axis("Output (% of capacity)"), hovermode="closest", **top)),
        "monthly": fig(series, _layout("Monthly energy over time", xaxis=_axis("Month"), yaxis=_axis("Energy (MWh)"), **top)),
    }
