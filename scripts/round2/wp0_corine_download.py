#!/usr/bin/env python3
"""WP0.5 — CORINE Land Cover 2018 (100 m raster) via CLMS API.

Auth: JWT service token from ~/.config/clms_token.json (user-provided).
Flow: token -> @datarequest_post -> poll @datarequest_search -> download zip.
Output: /mnt/nvme2/synthetic/raw/round2/corine/
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import jwt

CFG_PATH = os.path.expanduser("~/.config/clms_token.json")
BASE = "https://land.copernicus.eu/api"
DATASET_UID = "0407d497d3c44bcd93ce8fd5bf78596a"          # CORINE Land Cover 2018
# Prepackaged full-Europe 100 m Geotiff (125 MB) — full datasets must go through
# the prepackaged route, the DatasetDownloadInformationID route returns 400.
FILE_ID = "3d9e2413-46ca-4f6e-abf8-b6eb429a4e48"
OUT_DIR = "/mnt/nvme2/synthetic/raw/round2/corine"
POLL_S = 60
TIMEOUT_S = 4 * 3600


def get_token() -> str:
    cfg = json.load(open(CFG_PATH))
    claims = {"iss": cfg["client_id"], "sub": cfg["user_id"], "aud": cfg["token_uri"],
              "iat": int(time.time()), "exp": int(time.time()) + 3600}
    assertion = jwt.encode(claims, cfg["private_key"], algorithm="RS256")
    data = urllib.parse.urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion}).encode()
    req = urllib.request.Request(cfg["token_uri"], data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    return json.load(urllib.request.urlopen(req, timeout=60))["access_token"]


def api(path: str, token: str, payload=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Accept": "application/json", "Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST" if payload is not None else "GET")
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    token = get_token()
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
    else:
        resp = api("/@datarequest_post", token, {
            "Datasets": [{"DatasetID": DATASET_UID, "FileID": FILE_ID}]})
        print("datarequest response:", resp, flush=True)
        task_ids = list(resp.get("TaskIds", [])) or [resp]
        task_id = str(task_ids[0].get("TaskID") if isinstance(task_ids[0], dict) else task_ids[0])
    print("task:", task_id, flush=True)

    t0 = time.time()
    url = None
    while time.time() - t0 < TIMEOUT_S:
        time.sleep(POLL_S)
        token = get_token()
        status = api("/@datarequest_search?status=All", token)
        task = status.get(task_id) or {}
        st = task.get("Status")
        print(f"[{int(time.time()-t0)}s] status={st}", flush=True)
        if st in ("Finished_ok", "Completed"):
            url = task.get("DownloadURL")
            break
        if st in ("Rejected", "Finished_nok", "Cancelled"):
            print("FAILED task:", json.dumps(task)[:2000], flush=True)
            sys.exit(2)
    if not url:
        print("TIMEOUT waiting for CLMS extraction", flush=True)
        sys.exit(3)

    dest = os.path.join(OUT_DIR, "u2018_clc2018_v2020_20u1_raster100m.zip")
    print("downloading", url, "->", dest, flush=True)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {get_token()}"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dest + ".part", dest)
    print("done:", dest, os.path.getsize(dest), "bytes", flush=True)


if __name__ == "__main__":
    main()
