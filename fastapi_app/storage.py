"""
Tiny file-based persistence for a single local user.

    storage/scenarios/<id>.json     scenario definitions (inputs)
    storage/results/<id>.json       run metadata (KPIs, sites, summary)
    storage/results/<id>.parquet    the generated time series

Writes are atomic (temp file + rename) and guarded by a lock. Swapping this for a real
database later only requires re-implementing this module's functions.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import settings

_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_lock = threading.RLock()
_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


class NotFound(KeyError):
    pass


def _dirs():
    sc = settings.STORAGE_DIR / "scenarios"
    rs = settings.STORAGE_DIR / "results"
    sc.mkdir(parents=True, exist_ok=True)
    rs.mkdir(parents=True, exist_ok=True)
    return sc, rs


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _check(item_id: str) -> str:
    if not _ID_RE.match(item_id or ""):
        raise NotFound(item_id)
    return item_id


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars and arrays to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2))
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise NotFound(path.stem)
    return json.loads(path.read_text())


def downsample_curve(index, values, points: int = 400) -> Dict[str, List]:
    """Keep ~`points` samples of a smooth series (e.g. an ageing curve)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    vals = np.asarray(values, float)
    step = max(1, len(idx) // points)
    return {"x": [t.isoformat() for t in idx[::step]], "y": [float(v) for v in vals[::step]]}


# ------------------------------------------------------------------ scenarios
def list_scenarios() -> List[Dict[str, Any]]:
    sc, _ = _dirs()
    with _lock:
        items = [json.loads(p.read_text()) for p in sc.glob("*.json")]
    return sorted(items, key=lambda s: s.get("updated", ""), reverse=True)


def get_scenario(sid: str) -> Dict[str, Any]:
    sc, _ = _dirs()
    with _lock:
        return _read_json(sc / f"{_check(sid)}.json")


def create_scenario(name: str, kind: str, params: Dict[str, Any], sid: Optional[str] = None,
                    last_result_id: Optional[str] = None, when: Optional[str] = None) -> Dict[str, Any]:
    """New scenario. A fixed `sid` (12 hex chars) and `last_result_id` are used for scenarios built around an existing result."""
    sc, _ = _dirs()
    sid = _check(sid) if sid else new_id()
    now = when or now_iso()
    data = {"id": sid, "name": name, "kind": kind, "params": params,
            "created": now, "updated": now, "last_result_id": last_result_id, "last_run_at": now if last_result_id else None}
    with _lock:
        _write_json(sc / f"{sid}.json", data)
    return data


def set_result_scenario(rid: str, sid: str) -> None:
    """Point a stored result at the scenario it belongs to (deleting the scenario then deletes the result)."""
    _, rs = _dirs()
    with _lock:
        path = rs / f"{_check(rid)}.json"
        meta = _read_json(path)
        meta["scenario_id"] = sid
        _write_json(path, meta)


def update_result_meta(rid: str, patch: Dict[str, Any]) -> None:
    """Change fields of a stored result's metadata (the data itself stays as it is)."""
    _, rs = _dirs()
    with _lock:
        path = rs / f"{_check(rid)}.json"
        meta = _read_json(path)
        meta.update(patch)
        _write_json(path, meta)


def result_metas() -> List[Dict[str, Any]]:
    """Metadata of every stored result."""
    _, rs = _dirs()
    out = []
    with _lock:
        for p in rs.glob("*.json"):
            try:
                out.append(json.loads(p.read_text()))
            except (ValueError, OSError):
                continue
    return out


def update_scenario(sid: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    sc, _ = _dirs()
    with _lock:
        data = _read_json(sc / f"{_check(sid)}.json")
        for key in ("name", "params", "last_result_id", "last_run_at"):
            if key in patch:
                data[key] = patch[key]
        data["updated"] = now_iso()
        _write_json(sc / f"{sid}.json", data)
    return data


def delete_scenario(sid: str) -> None:
    sc, rs = _dirs()
    with _lock:
        path = sc / f"{_check(sid)}.json"
        if not path.exists():
            raise NotFound(sid)
        path.unlink()
        for meta_path in rs.glob("*.json"):
            try:
                if json.loads(meta_path.read_text()).get("scenario_id") == sid:
                    delete_result(meta_path.stem)
            except (ValueError, OSError):
                continue


def duplicate_scenario(sid: str) -> Dict[str, Any]:
    src = get_scenario(sid)
    return create_scenario(f"{src['name']} (copy)", src["kind"], json.loads(json.dumps(src["params"])))


# ------------------------------------------------------------------ results
def save_result(meta: Dict[str, Any], df: pd.DataFrame, rid: Optional[str] = None) -> str:
    """Store a result. A fixed `rid` (12 hex chars) replaces an existing result with that id."""
    _, rs = _dirs()
    rid = _check(rid) if rid else new_id()
    meta = dict(meta, id=rid, created=now_iso())
    with _lock:
        df.to_parquet(rs / f"{rid}.parquet")
        _write_json(rs / f"{rid}.json", meta)
        _remember(rid, df)
    return rid


def get_result_meta(rid: str) -> Dict[str, Any]:
    _, rs = _dirs()
    with _lock:
        return _read_json(rs / f"{_check(rid)}.json")


def get_result_df(rid: str) -> pd.DataFrame:
    _, rs = _dirs()
    with _lock:
        if rid in _frames:
            _frames.move_to_end(rid)
            return _frames[rid]
        path = rs / f"{_check(rid)}.parquet"
        if not path.exists():
            raise NotFound(rid)
        df = pd.read_parquet(path)
        _remember(rid, df)
        return df


def result_parquet_path(rid: str) -> Path:
    _, rs = _dirs()
    path = rs / f"{_check(rid)}.parquet"
    if not path.exists():
        raise NotFound(rid)
    return path


def delete_result(rid: str) -> None:
    _, rs = _dirs()
    with _lock:
        for suffix in (".json", ".parquet"):
            p = rs / f"{_check(rid)}{suffix}"
            if p.exists():
                p.unlink()
        _frames.pop(rid, None)


def _remember(rid: str, df: pd.DataFrame) -> None:
    _frames[rid] = df
    _frames.move_to_end(rid)
    while len(_frames) > settings.MAX_RESULTS_IN_MEMORY:
        _frames.popitem(last=False)


# ------------------------------------------------------------------ trained models
def _models_dir() -> Path:
    d = settings.STORAGE_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_weights_path(mid: str, ext: Optional[str] = None) -> Path:
    """Path of a model's weights file (`.npz` for linear, `.pt` for torch models)."""
    d = _models_dir()
    _check(mid)
    if ext:
        return d / f"{mid}{ext}"
    for e in (".npz", ".pt"):
        if (d / f"{mid}{e}").exists():
            return d / f"{mid}{e}"
    raise NotFound(mid)


def new_model_id() -> str:
    return new_id()


def save_model_meta(meta: Dict[str, Any]) -> None:
    with _lock:
        _write_json(_models_dir() / f"{_check(meta['id'])}.json", meta)


def get_model_meta(mid: str) -> Dict[str, Any]:
    with _lock:
        return _read_json(_models_dir() / f"{_check(mid)}.json")


def list_models() -> List[Dict[str, Any]]:
    with _lock:
        items = []
        for p in _models_dir().glob("*.json"):
            try:
                m = json.loads(p.read_text())
            except ValueError:
                continue
            # list view: omit heavy arrays
            items.append({k: v for k, v in m.items() if k not in ("val_preview", "history", "scaler")})
    return sorted(items, key=lambda m: m.get("created", ""), reverse=True)


def delete_model(mid: str) -> None:
    d = _models_dir()
    with _lock:
        if not (d / f"{_check(mid)}.json").exists():
            raise NotFound(mid)
        for suffix in (".json", ".npz", ".pt"):
            p = d / f"{mid}{suffix}"
            if p.exists():
                p.unlink()
