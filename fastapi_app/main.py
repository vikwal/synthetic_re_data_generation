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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import api, auth, authapi, settings, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

BASE_DIR = Path(__file__).resolve().parent
_docs = {} if not settings.AUTH_ENABLED else {"docs_url": None, "redoc_url": None, "openapi_url": None}   # no public API docs when logins are on
app = FastAPI(title="Synthetic Renewables", version="2.0.0",
              description="Scenario workspace for synthetic wind and solar generation.", **_docs)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.include_router(authapi.router)
app.include_router(api.router)


_PUBLIC = ("/login", "/healthz", "/api/auth/login", "/api/auth/logout", "/api/auth/me")


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    """With SYNTH_RE_AUTH=1: nothing but the login page, its static files and the health check is reachable without a
    valid session. The signed-in user is set for the rest of the request (and for background jobs it starts), which is
    what keeps every user's data apart (see storage._root)."""
    path = request.url.path
    token = None
    if settings.AUTH_ENABLED:
        user = auth.user_from_token(request.cookies.get(auth.COOKIE_NAME))
        if user is None and not (path.startswith("/static/") or path in _PUBLIC):
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Please sign in."})
            return RedirectResponse(f"{settings.BASE_PATH}/login", status_code=303)
        if user is not None:
            token = auth.set_current(user)
    try:
        response = await call_next(request)
    finally:
        if token is not None:
            auth.reset_current(token)
    if path.startswith("/static/") or path in ("/", "/login") or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"       # never serve stale JS/CSS, never cache a user's data
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.exception_handler(auth.AuthError)
async def _auth_error(_: Request, exc: auth.AuthError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.exception_handler(storage.NotFound)
async def _not_found(_: Request, exc: storage.NotFound):
    return JSONResponse(status_code=404, content={"detail": "Not found (it may have been deleted)."})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if settings.AUTH_ENABLED and auth.user_from_token(request.cookies.get(auth.COOKIE_NAME)):
        return RedirectResponse(f"{settings.BASE_PATH}/", status_code=303)
    if not settings.AUTH_ENABLED:
        return RedirectResponse(f"{settings.BASE_PATH}/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"base": settings.BASE_PATH})


@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "app.html", {"base": settings.BASE_PATH})
