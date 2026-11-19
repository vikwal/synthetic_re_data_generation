"""Admin tool for the dashboard's logins (run on the server, from the project folder):

    python -m fastapi_app.manage_users add alice              # asks for the password
    python -m fastapi_app.manage_users add alice --generate   # makes a random password and prints it once
    python -m fastapi_app.manage_users passwd alice
    python -m fastapi_app.manage_users delete alice [--data]  # --data also deletes that user's scenarios, results, models
    python -m fastapi_app.manage_users list
    python -m fastapi_app.manage_users adopt-legacy alice     # give the pre-login storage/ data to alice (one time)

Uses the same SYNTH_RE_STORAGE_DIR as the app. Admins cannot see users' data through the dashboard; this tool only manages accounts.
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import shutil
import sys

from . import auth, settings

LEGACY = ("scenarios", "results", "models")


def _ask_password() -> str:
    a = getpass.getpass("Password: ")
    if a != getpass.getpass("Repeat password: "):
        raise auth.AuthError("The passwords differ.", 400)
    return a


def _password(args) -> str:
    if args.generate:
        pw = secrets.token_urlsafe(12)
        print(f"Password: {pw}   (shown only now)")
        return pw
    return _ask_password()


def adopt_legacy(name: str) -> list:
    """Move storage/{scenarios,results,models} into storage/users/<name>/. Refuses if the target already has data."""
    if name not in auth.list_users():
        raise auth.AuthError(f"User '{name}' does not exist. Create it first.", 404)
    dest = settings.STORAGE_DIR / "users" / name
    moved = []
    for sub in LEGACY:
        src, dst = settings.STORAGE_DIR / sub, dest / sub
        if not src.exists():
            continue
        if dst.exists() and any(dst.iterdir()):
            raise auth.AuthError(f"{dst} already has data; nothing was moved.", 409)
    dest.mkdir(parents=True, exist_ok=True)
    for sub in LEGACY:
        src, dst = settings.STORAGE_DIR / sub, dest / sub
        if src.exists():
            if dst.exists():
                dst.rmdir()
            shutil.move(str(src), str(dst))
            moved.append(sub)
    return moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="manage_users", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("add", "passwd"):
        p = sp.add_parser(c)
        p.add_argument("name")
        p.add_argument("--generate", action="store_true")
    p = sp.add_parser("delete")
    p.add_argument("name")
    p.add_argument("--data", action="store_true")
    sp.add_parser("list")
    sp.add_parser("adopt-legacy").add_argument("name")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "add":
            auth.add_user(args.name, _password(args)); print(f"User '{args.name.lower()}' created.")
        elif args.cmd == "passwd":
            auth.set_password(args.name.strip().lower(), _password(args)); print("Password changed; open sessions of this user are signed out.")
        elif args.cmd == "delete":
            name = args.name.strip().lower()
            auth.delete_user(name)
            if args.data:
                shutil.rmtree(settings.STORAGE_DIR / "users" / name, ignore_errors=True)
            print(f"User '{name}' deleted" + (" with all data." if args.data else "; data folder kept."))
        elif args.cmd == "list":
            for k, v in auth.list_users().items():
                print(f"{k}\tcreated {v.get('created')}")
        elif args.cmd == "adopt-legacy":
            moved = adopt_legacy(args.name.strip().lower())
            print("Moved: " + ", ".join(moved) if moved else "No old data found.")
    except auth.AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
