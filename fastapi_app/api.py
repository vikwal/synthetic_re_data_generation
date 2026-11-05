"""HTTP API (JSON). Mounted under /api by main.py."""
from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import core, jobs, layout, mlmodels, mlprep, mlservice, nwp, parkdata, realcompare, service, storage
from .schemas import CompareRequest, FinetuneRequest, RealCompareRequest, TrainRequest, LayoutRequest, ScenarioCreate, ScenarioUpdate, SolarParams, WindParams

router = APIRouter(prefix="/api")


def _wind_defaults() -> dict:
    d = WindParams().model_dump()
    cfg = core.load_config().get("wind_params", {})          # config.yaml is the source of truth
    if "annual_degradation" in cfg:
        d["annual_degradation"] = float(cfg["annual_degradation"])
    return d


def _defaults(kind: str) -> dict:
    return _wind_defaults() if kind == "wind" else SolarParams().model_dump()


# ---------------------------------------------------------------- reference data
@router.get("/bootstrap")
def bootstrap():
    stations = core.solar_stations()
    return {
        "turbine_types": core.turbine_types(),
        "solar_defaults": SolarParams().model_dump(),
        "wind_defaults": _wind_defaults(),
        "solar_stations": [{"id": r.station_id, "lat": float(r.latitude), "lon": float(r.longitude)}
                           for r in stations.itertuples()],
    }


@router.get("/search")
def search(q: str = Query(min_length=1, max_length=120)):
    return {"results": core.search_location(q)}


@router.get("/nearest-grid")
def nearest_grid(lat: float, lon: float):
    near = core.find_nearest_grid_point(lat, lon)
    if near is None:
        raise HTTPException(404, "No NWP grid points found.")
    return near


@router.get("/nearest-station")
def nearest_station(lat: float, lon: float):
    return core.find_nearest_station(lat, lon)


@router.get("/grid-points")
def grid_points(south: float, west: float, north: float, east: float):
    return {"points": core.grid_points_in_bounds(south, west, north, east)}


@router.get("/turbine-specs")
def turbine_specs(name: str):
    if name not in set(core.turbine_types()):
        raise HTTPException(404, "Unknown turbine type")
    return core.turbine_specs(name)


@router.post("/layout")
def make_layout(req: LayoutRequest):
    if req.turbine not in set(core.turbine_types()):
        raise HTTPException(400, "Unknown turbine type")
    diameter = core.turbine_specs(req.turbine)["diameter"]
    if not diameter:
        raise HTTPException(400, f"The rotor diameter of {req.turbine} is not in the turbine table, so no layout can be spaced.")
    try:
        points = layout.grid_layout(req.center_lat, req.center_lon, req.rows, req.cols,
                                    req.spacing_along_d, req.spacing_across_d, diameter, req.rotation_deg)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"rotor_diameter_m": diameter, "points": points}


# ---------------------------------------------------------------- scenarios
@router.get("/scenarios")
def list_scenarios():
    parkdata.ensure_scenarios()                                       # imported package parks appear as scenarios too
    return {"scenarios": storage.list_scenarios()}


@router.post("/scenarios", status_code=201)
def create_scenario(req: ScenarioCreate):
    return storage.create_scenario(req.name, req.kind, _defaults(req.kind))


@router.get("/scenarios/{sid}")
def get_scenario(sid: str):
    return storage.get_scenario(sid)


@router.put("/scenarios/{sid}")
def update_scenario(sid: str, req: ScenarioUpdate):
    patch = {}
    if req.name is not None:
        patch["name"] = req.name.strip()
    if req.params is not None:
        current = storage.get_scenario(sid)
        model = WindParams if current["kind"] == "wind" else SolarParams
        try:
            patch["params"] = model(**req.params).model_dump()      # validate + normalise
        except Exception as exc:
            raise HTTPException(422, str(exc))
    return storage.update_scenario(sid, patch)


@router.delete("/scenarios/{sid}", status_code=204)
def delete_scenario(sid: str):
    storage.delete_scenario(sid)


@router.post("/scenarios/{sid}/duplicate", status_code=201)
def duplicate_scenario(sid: str):
    return storage.duplicate_scenario(sid)


@router.post("/scenarios/{sid}/run", status_code=202)
def run_scenario(sid: str):
    storage.get_scenario(sid)                                         # 404 early if missing
    return jobs.manager.submit(lambda progress: service.run_scenario(sid, progress), scenario_id=sid)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ---------------------------------------------------------------- results
@router.get("/results")
def list_results():
    from . import settings
    out = []
    for p in sorted((settings.STORAGE_DIR / "results").glob("*.json")) if (settings.STORAGE_DIR / "results").exists() else []:
        try:
            m = storage.get_result_meta(p.stem)
            out.append({**{k: m.get(k) for k in ("id", "scenario_id", "scenario_name", "kind", "created", "kpis")},
                        "start": (m.get("summary") or {}).get("start"), "end": (m.get("summary") or {}).get("end")})
        except storage.NotFound:
            continue
    return {"results": sorted(out, key=lambda r: r.get("created") or "", reverse=True)}


@router.get("/results/{rid}")
def get_result(rid: str):
    return service.describe_result(rid)


@router.get("/results/{rid}/figure/{name}")
def get_figure(rid: str, name: str, date: Optional[str] = None, feature: Optional[str] = None,
               year: Optional[int] = None, turbine: Optional[str] = None, hour: Optional[int] = None):
    try:
        return service.build_figure(rid, name, date=date, feature=feature, year=year, turbine=turbine, hour=hour)
    except LookupError as exc:
        raise HTTPException(404, str(exc.args[0]))
    except (ValueError, KeyError) as exc:
        if isinstance(exc, storage.NotFound):
            raise
        raise HTTPException(400, str(exc.args[0]) if exc.args else "Bad request")


@router.get("/results/{rid}/table")
def result_table(rid: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500)):
    df = storage.get_result_df(rid)
    total = len(df)
    pages = max(1, -(-total // page_size))
    page = min(page, pages)
    chunk = df.iloc[(page - 1) * page_size: page * page_size].round(4)
    rows = [[idx.strftime("%Y-%m-%d %H:%M")] + [None if v != v else v for v in r]
            for idx, r in zip(chunk.index, chunk.itertuples(index=False, name=None))]
    return {"columns": ["timestamp"] + [str(c) for c in chunk.columns], "rows": rows,
            "total": total, "page": page, "pages": pages}


@router.get("/results/{rid}/csv")
def result_csv(rid: str):
    meta = storage.get_result_meta(rid)
    buf = io.StringIO()
    storage.get_result_df(rid).to_csv(buf, index_label="timestamp")
    name = f"{meta['kind']}_{rid}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/results/{rid}/parquet")
def result_parquet(rid: str):
    meta = storage.get_result_meta(rid)
    return FileResponse(storage.result_parquet_path(rid), media_type="application/octet-stream",
                        filename=f"{meta['kind']}_{rid}.parquet")


@router.get("/results/{rid}/summary.json")
def result_summary(rid: str):
    meta = storage.get_result_meta(rid)
    return JSONResponse(meta, headers={"Content-Disposition": f'attachment; filename="{meta["kind"]}_{rid}_summary.json"'})


@router.delete("/results/{rid}", status_code=204)
def delete_result(rid: str):
    storage.delete_result(rid)


@router.post("/compare")
def compare(req: CompareRequest):
    return service.compare_results(req.result_ids)


@router.post("/compare/real")
def compare_real(req: RealCompareRequest):
    """Compare an uploaded real plant file against a stored synthetic result."""
    meta = storage.get_result_meta(req.result_id)
    synth = storage.get_result_df(req.result_id)
    try:
        real = realcompare.parse_real_csv(req.csv_text, req.unit)
        m, figs = realcompare.compare(real, synth, meta["summary"]["capacity_w"], meta["kind"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"result_id": req.result_id, "name": meta.get("scenario_name"), "kind": meta["kind"],
            "real_rows": int(len(real)), "metrics": m, "figures": figs}


# ---------------------------------------------------------------- pre-trained models
@router.get("/ml/architectures")
def ml_architectures():
    return {"architectures": mlmodels.availability(), "torch": mlmodels.torch_available(),
            "features": mlprep.catalog(), "weather_sources": nwp.sources(), "package": parkdata.status()}


@router.post("/package/import")
def package_import():
    """Store the sample package's wind parks as generated results (replaces earlier imports)."""
    if parkdata.package_dir() is None:
        raise HTTPException(404, "Sample-data package not found. Put it in data/ or set SYNTH_RE_PACKAGE_DIR.")
    try:
        ids = parkdata.import_wind_parks()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"imported": len(ids), "result_ids": ids}


@router.get("/ml/models")
def ml_models():
    return {"models": storage.list_models()}


@router.post("/ml/train", status_code=202)
def ml_train(req: TrainRequest):
    opts = next(a for a in mlmodels.availability() if a["id"] == req.architecture)
    if not opts["available"]:
        raise HTTPException(400, opts["reason"])
    for rid in req.result_ids:
        storage.get_result_meta(rid)                                  # 404 early if missing
    wanted = list(dict.fromkeys(req.weather_sources or [req.weather_source]))        # one or several weather models
    available = {x["id"]: x for x in nwp.sources()}
    if len(wanted) > 1 and "era5" in wanted:
        raise HTTPException(400, "The run's own ERA5 weather cannot be combined with forecast models.")
    for sid in wanted:
        src = available.get(sid)
        if src is None or not src["available"]:
            raise HTTPException(400, (src or {}).get("reason") or f"Unknown weather source '{sid}'")
        if nwp.has_lead(sid):
            if req.lead_from > req.lead_to:
                raise HTTPException(400, "The lead-time window must start before it ends.")
            if req.lead_to > src["lead"]["max"]:
                raise HTTPException(400, f"{src['label']} forecasts reach {src['lead']['max']} h ahead; lower the lead time.")
        if nwp.is_package(sid):
            if not all(storage.get_result_meta(rid).get("package_park_id") for rid in req.result_ids):
                raise HTTPException(400, f"{src['label']} forecasts only exist for the imported wind parks. Choose one of those parks.")
    spec = {"name": req.name, "architecture": req.architecture, "result_ids": req.result_ids,
            "features": req.features, "weather_source": wanted[0], "weather_sources": wanted, "power_lag": req.power_lag_hours,
            "train_from": req.train_from, "train_until": req.train_until, "refit_full": req.refit_full, "lead_from": req.lead_from, "lead_to": req.lead_to,
            "hyperparams": req.model_dump(exclude={"name", "architecture", "result_ids", "features", "weather_source", "weather_sources",
                                                   "power_lag_hours", "train_from", "train_until", "refit_full", "lead_from", "lead_to"})}
    return jobs.manager.submit(lambda progress: mlservice.run_training(spec, progress))


@router.get("/ml/models/{mid}")
def ml_model(mid: str):
    return mlservice.describe_model(mid)


@router.delete("/ml/models/{mid}", status_code=204)
def ml_delete(mid: str):
    storage.delete_model(mid)


@router.get("/ml/models/{mid}/figure/{name}")
def ml_figure(mid: str, name: str):
    try:
        return mlservice.model_figure(mid, name)
    except LookupError as exc:
        raise HTTPException(404, str(exc.args[0]))


@router.get("/ml/models/{mid}/download")
def ml_download(mid: str):
    meta = storage.get_model_meta(mid)
    path = storage.model_weights_path(mid)
    return FileResponse(path, media_type="application/octet-stream", filename=f"{meta['architecture']}_{mid}{path.suffix}")


@router.post("/ml/models/{mid}/finetune", status_code=202)
def ml_finetune(mid: str, req: FinetuneRequest):
    parent = storage.get_model_meta(mid)
    storage.get_result_meta(req.feature_result_id)
    opts = next(a for a in mlmodels.availability() if a["id"] == parent["architecture"])
    if not opts["available"]:
        raise HTTPException(400, opts["reason"])
    spec = req.model_dump()
    spec["model_id"] = mid
    return jobs.manager.submit(lambda progress: mlservice.run_finetune(spec, progress))
