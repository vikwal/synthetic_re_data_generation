"""Application services: run a scenario end to end and describe stored results."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from . import analytics, charts, core, storage
from .schemas import SolarParams, WindParams

ProgressFn = Callable[[float, str], None]


def _public_sites(sites: List[dict]) -> List[dict]:
    """JSON-safe site list: numpy arrays replaced by a downsampled ageing curve."""
    keep = ("slot", "lat", "lon", "turbine", "hub_height", "commissioning_date", "grid_lat", "grid_lon",
            "distance_km", "wake_loss_pct", "ageing_loss_pct", "rated_mw", "energy_mwh",
            "capacity_factor_pct", "mean_wind_speed")
    out = []
    for s in sites:
        item = {k: s.get(k) for k in keep}
        if s.get("ageing_efficiency") is not None:
            item["ageing_curve"] = storage.downsample_curve(s["ageing_index"], s["ageing_efficiency"])
        out.append(item)
    return out


def run_scenario(scenario_id: str, progress: ProgressFn) -> str:
    """Execute a stored scenario and persist the result. Returns the result id."""
    sc = storage.get_scenario(scenario_id)
    kind = sc["kind"]

    if kind == "wind":
        params = WindParams(**sc["params"])
        if not params.turbines:
            raise ValueError("Add at least one turbine before running.")
        unknown = sorted({t.turbine for t in params.turbines} - set(core.turbine_types()))
        if unknown:
            raise ValueError(f"Unknown turbine type(s): {', '.join(unknown)}")
        df, sites, summary = core.generate_wind_park(
            [t.model_dump() for t in params.turbines], apply_ageing=params.apply_ageing,
            apply_wake=params.apply_wake, wake_decay=params.wake_decay,
            annual_degradation=params.annual_degradation,
            default_commissioning_date=params.default_commissioning_date, progress=progress)
        public_sites = _public_sites(sites)
    else:
        params = SolarParams(**sc["params"])
        if params.lat is None or params.lon is None:
            raise ValueError("Pick a location before running.")
        overrides = params.model_dump(exclude={"lat", "lon"})
        df, summary = core.generate_solar(params.lat, params.lon, overrides, progress=progress)
        public_sites = None

    kp = analytics.kpis(df, kind, summary["capacity_w"])
    meta: Dict[str, Any] = {
        "scenario_id": scenario_id, "scenario_name": sc["name"], "kind": kind,
        "params": params.model_dump(), "summary": summary, "kpis": kp, "sites": public_sites,
    }
    rid = storage.save_result(meta, df)
    storage.update_scenario(scenario_id, {"last_result_id": rid, "last_run_at": storage.now_iso()})
    return rid


def describe_result(rid: str) -> Dict[str, Any]:
    """Result metadata plus everything the results page needs to build its selectors."""
    meta = storage.get_result_meta(rid)
    df = storage.get_result_df(rid)
    kind = meta["kind"]
    sites = meta.get("sites") or []
    features = charts.available_features(df, kind)
    summary = meta["summary"]
    return {
        **meta,
        "dates": sorted({d.strftime("%Y-%m-%d") for d in df.index}),
        "years": sorted({int(y) for y in df.index.year.unique()}),
        "features": features,
        "feature_labels": {f: charts.FEATURE_META.get(f, (f, ""))[0] for f in features},
        "turbines": sorted({s["turbine"] for s in sites}),
        "has_performance": bool(kind == "wind" and (summary.get("wake_applied") or summary.get("ageing_applied"))),
        "has_rose": bool(kind == "wind" and {"u_wind_100m", "v_wind_100m"}.issubset(df.columns)),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }


def build_figure(rid: str, name: str, **q) -> Dict[str, Any]:
    """Dispatch a named figure for a stored result. Raises LookupError if unavailable."""
    meta = storage.get_result_meta(rid)
    df = storage.get_result_df(rid)
    kind = meta["kind"]
    capacity_w = meta["summary"]["capacity_w"]
    unit = "MWh"

    if name == "seasonality":
        return charts.seasonality_figure(analytics.seasonality(df), unit)
    if name == "heatmap":
        return charts.heatmap_figure(analytics.hour_month_heatmap(df, capacity_w))
    if name == "duration":
        return charts.duration_figure(analytics.duration_curve(df, capacity_w))
    if name == "windrose":
        rose = analytics.wind_rose(df)
        if rose is None:
            raise LookupError("Wind rose needs 100 m wind components, which this dataset lacks.")
        return charts.windrose_figure(rose)
    if name == "speed-hist":
        hist = analytics.speed_histogram(df)
        if hist is None:
            raise LookupError("No wind-speed data.")
        return charts.speed_hist_figure(hist)
    if name == "irradiance":
        f = charts.irradiance_monthly_figure(df)
        if f is None:
            raise LookupError("No irradiance data.")
        return f
    if name == "waterfall":
        energies = (meta["summary"] or {}).get("energies")
        if not energies:
            raise LookupError("No wake/ageing breakdown for this run.")
        return charts.waterfall_figure(energies)
    if name == "loss-per-turbine":
        if not meta.get("sites"):
            raise LookupError("Only available for wind parks.")
        return charts.degradation_summary_figure(meta["sites"])
    if name == "ageing-timeline":
        f = charts.ageing_timeline_figure(meta.get("sites") or [])
        if f is None:
            raise LookupError("Ageing was not enabled for this run.")
        return f
    if name == "daily":
        date = q.get("date")
        if not date:
            raise ValueError("date is required")
        return charts.daily_figure(df, kind, date, q.get("feature"))
    if name == "quarterly":
        year = q.get("year")
        if year is None:
            raise ValueError("year is required")
        return charts.quarterly_figure(df, kind, int(year))
    if name == "hour-profile":
        year, hour = q.get("year"), q.get("hour")
        if year is None or hour is None:
            raise ValueError("year and hour are required")
        if not 0 <= int(hour) <= 23:
            raise ValueError("hour must be 0-23")
        return charts.hour_profile_figure(df, kind, int(year), int(hour))
    if name == "hour-box":
        year = q.get("year")
        if year is None:
            raise ValueError("year is required")
        return charts.hour_box_figure(df, kind, int(year))
    if name == "multi-year":
        return charts.multi_year_figure(df, kind)
    if name == "turbine-curve":
        turbine = q.get("turbine")
        if turbine == "__park__":
            if "wind_speed_hub" not in df.columns:
                raise LookupError("This result has no wind-speed data for a park power curve.")
            d = df[["wind_speed_hub", "power"]].dropna()
            if len(d) > 6000:
                d = d.sample(6000, random_state=0)
            return charts.park_curve_figure(d["wind_speed_hub"].astype(float).tolist(),
                                            (d["power"].astype(float) / 1e6).tolist())
        if turbine not in set(core.turbine_types()):
            raise ValueError("Unknown turbine type")
        return charts.turbine_curve_figure(core.turbine_power_curve_points(turbine), turbine)
    raise LookupError(f"Unknown figure '{name}'")


def compare_results(result_ids: List[str]) -> Dict[str, Any]:
    items, fig_items = [], []
    for rid in result_ids:
        meta = storage.get_result_meta(rid)
        df = storage.get_result_df(rid)
        cap = meta["summary"]["capacity_w"]
        name = meta.get("scenario_name") or rid
        s = meta["summary"]
        # Imported package parks carry no per-turbine ageing loss: average only the sites that have one.
        ageing = [x["ageing_loss_pct"] for x in (meta.get("sites") or []) if x.get("ageing_loss_pct") is not None]
        items.append({
            "result_id": rid, "name": name, "kind": meta["kind"], "created": meta.get("created"),
            "kpis": meta["kpis"],
            "wake_loss_pct": s.get("park_wake_loss_pct"),
            "ageing_loss_pct": round(sum(ageing) / len(ageing), 3) if ageing and s.get("ageing_applied") else None,
            "n_turbines": s.get("n_turbines"),
        })
        fig_items.append({"name": name, "seasonality": analytics.seasonality(df),
                          "monthly": analytics.monthly_energy(df), "duration": analytics.duration_curve(df, cap)})
    return {"items": items, "figures": charts.compare_figures(fig_items)}
