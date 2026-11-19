"""Login / logout / who-am-I routes (mounted by main.py). The gate that protects everything else is in main.py."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import auth, settings

router = APIRouter(prefix="/api/auth")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


def _https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


@router.post("/login")
def login(req: LoginRequest, request: Request, response: Response):
    if not settings.AUTH_ENABLED:
        raise HTTPException(400, "Login is not enabled on this server.")
    ip = request.client.host if request.client else ""
    try:
        token = auth.login(req.username, req.password, ip)
    except auth.AuthError as exc:
        raise HTTPException(exc.status, str(exc))
    response.set_cookie(auth.COOKIE_NAME, token, max_age=int(settings.SESSION_HOURS * 3600), httponly=True,
                        samesite="lax", secure=_https(request), path=settings.BASE_PATH or "/")
    return {"user": req.username.strip().lower()}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path=settings.BASE_PATH or "/")
    return {"ok": True}


@router.get("/me")
def me():
    """`auth` tells the page whether this server uses logins at all; `user` is the signed-in name."""
    return {"auth": settings.AUTH_ENABLED, "user": auth.current()}
