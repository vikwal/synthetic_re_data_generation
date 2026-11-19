"""Basic login for the dashboard: accounts created by an admin, one private data space per user.

Only active when SYNTH_RE_AUTH=1 (the server). Without it the app behaves as before: no login, one local storage folder.

* Accounts live in  <storage>/auth/users.json  (password stored as a salted scrypt hash, never in clear text).
* After login the browser gets a signed, HttpOnly cookie. The signing key is SYNTH_RE_SECRET, or a random key kept in
  <storage>/auth/secret.key. Changing a user's password invalidates that user's open sessions.
* The signed-in user is held in a context variable; storage.py uses it to pick the user's own folder, so a request can only
  ever read and write that user's scenarios, results and models.

Standard library only, so it can be tested without the web framework.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from . import settings

COOKIE_NAME = "sr_session"
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
MIN_PASSWORD_LEN = 10
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}

_current: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("synth_re_user", default=None)
_lock = threading.RLock()
_failures: Dict[str, list] = {}                     # login throttle: key -> timestamps of recent failures
MAX_FAILURES, FAILURE_WINDOW_S = 5, 300


class AuthError(Exception):
    """Wrong credentials, too many attempts, bad user data ..."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ current user (per request / per job)
def current() -> Optional[str]:
    return _current.get()


def set_current(username: Optional[str]) -> "contextvars.Token":
    return _current.set(username)


def reset_current(token: "contextvars.Token") -> None:
    _current.reset(token)


def enabled() -> bool:
    return settings.AUTH_ENABLED


# ------------------------------------------------------------------ user store
def _auth_dir() -> Path:
    d = settings.STORAGE_DIR / "auth"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _users_path() -> Path:
    return _auth_dir() / "users.json"


def _read_users() -> Dict[str, dict]:
    p = _users_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        raise AuthError("The user file is damaged.", 500)


def _write_users(users: Dict[str, dict]) -> None:
    p = _users_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, p)


def valid_username(name: str) -> bool:
    return bool(_USERNAME_RE.match(name or ""))


_PBKDF2_ROUNDS = 600_000


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """scrypt where available; PBKDF2-SHA256 where Python was built without scrypt (e.g. macOS system Python)."""
    salt = salt or os.urandom(16)
    pw = password.encode("utf-8")
    if hasattr(hashlib, "scrypt"):
        digest = hashlib.scrypt(pw, salt=salt, dklen=32, **_SCRYPT)
        return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"
    digest = hashlib.pbkdf2_hmac("sha256", pw, salt, _PBKDF2_ROUNDS, dklen=32)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Understands both formats, so accounts made on one machine work on the other."""
    try:
        parts = stored.split("$")
        pw = password.encode("utf-8")
        if parts[0] == "scrypt":
            _, n, r, p, salt_hex, digest_hex = parts
            digest = hashlib.scrypt(pw, salt=bytes.fromhex(salt_hex), dklen=len(digest_hex) // 2, n=int(n), r=int(r), p=int(p))
        elif parts[0] == "pbkdf2":
            _, rounds, salt_hex, digest_hex = parts
            digest = hashlib.pbkdf2_hmac("sha256", pw, bytes.fromhex(salt_hex), int(rounds), dklen=len(digest_hex) // 2)
        else:
            return False
    except (ValueError, TypeError, AttributeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


_DUMMY_HASH = None


def add_user(username: str, password: str) -> None:
    username = (username or "").strip().lower()
    if not valid_username(username):
        raise AuthError("User names are 3 to 32 characters: lower-case letters, digits, '.', '_' or '-'.", 400)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"The password needs at least {MIN_PASSWORD_LEN} characters.", 400)
    with _lock:
        users = _read_users()
        if username in users:
            raise AuthError(f"User '{username}' already exists.", 409)
        users[username] = {"hash": hash_password(password), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_users(users)


def set_password(username: str, password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"The password needs at least {MIN_PASSWORD_LEN} characters.", 400)
    with _lock:
        users = _read_users()
        if username not in users:
            raise AuthError(f"User '{username}' does not exist.", 404)
        users[username]["hash"] = hash_password(password)
        _write_users(users)


def delete_user(username: str) -> None:
    with _lock:
        users = _read_users()
        if username not in users:
            raise AuthError(f"User '{username}' does not exist.", 404)
        del users[username]
        _write_users(users)


def list_users() -> Dict[str, dict]:
    return {k: {"created": v.get("created")} for k, v in sorted(_read_users().items())}


# ------------------------------------------------------------------ login + session token
def _throttle_key(username: str, ip: str) -> str:
    return f"{username}|{ip}"


def _recent_failures(key: str) -> list:
    now = time.time()
    fails = [t for t in _failures.get(key, []) if now - t < FAILURE_WINDOW_S]
    _failures[key] = fails
    return fails


def login(username: str, password: str, ip: str = "") -> str:
    """Check the credentials and return a session token. Raises AuthError."""
    global _DUMMY_HASH
    username = (username or "").strip().lower()
    key = _throttle_key(username, ip)
    with _lock:
        if len(_recent_failures(key)) >= MAX_FAILURES:
            raise AuthError("Too many failed attempts. Please wait a few minutes and try again.", 429)
        user = _read_users().get(username)
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("not-a-real-password")
    ok = verify_password(password or "", user["hash"] if user else _DUMMY_HASH)     # same work for unknown users
    if not (user and ok):
        with _lock:
            _failures.setdefault(key, []).append(time.time())
        raise AuthError("Wrong user name or password.", 401)
    with _lock:
        _failures.pop(key, None)
    return make_token(username, user["hash"])


def _secret() -> bytes:
    env = os.environ.get("SYNTH_RE_SECRET")
    if env:
        return env.encode("utf-8")
    p = _auth_dir() / "secret.key"
    with _lock:
        if not p.exists():
            p.write_text(secrets.token_hex(32), encoding="utf-8")
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
        return p.read_text(encoding="utf-8").strip().encode("utf-8")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _version(stored_hash: str) -> str:
    return hashlib.sha256(stored_hash.encode("utf-8")).hexdigest()[:10]          # changes with the password


def make_token(username: str, stored_hash: str, now: Optional[float] = None) -> str:
    exp = int((now if now is not None else time.time()) + settings.SESSION_HOURS * 3600)
    payload = _b64(json.dumps({"u": username, "exp": exp, "v": _version(stored_hash)}, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def user_from_token(token: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """The user a valid, unexpired token belongs to, else None."""
    if not token or "." not in token:
        return None
    payload, _, sig = token.partition(".")
    expected = hmac.new(_secret(), payload.encode("ascii", "ignore"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
    except (ValueError, TypeError):
        return None
    if data.get("exp", 0) < (now if now is not None else time.time()):
        return None
    user = _read_users().get(data.get("u", ""))
    if not user or data.get("v") != _version(user["hash"]):
        return None                                   # deleted user, or password changed since this session began
    return data["u"]
