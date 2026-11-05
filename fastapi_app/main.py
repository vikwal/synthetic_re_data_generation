"""
Synthetic Renewables — FastAPI application.

Run from the project root:
    uvicorn fastapi_app.main:app --reload
then open http://127.0.0.1:8000
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import api, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Synthetic Renewables", version="2.0.0",
              description="Scenario workspace for synthetic wind and solar generation.")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.include_router(api.router)


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Local dev app: never let the browser serve stale JS/CSS."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(storage.NotFound)
async def _not_found(_: Request, exc: storage.NotFound):
    return JSONResponse(status_code=404, content={"detail": "Not found (it may have been deleted)."})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "app.html", {})
