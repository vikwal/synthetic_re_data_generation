"""Read a user's turbine file (CSV or Excel, one row per turbine) and group it into wind parks.

The park is found from the columns whose names contain "park" together with "name" and/or "id". Every park then
offers its turbines with the details the generator needs: position, turbine type, hub height, commissioning date.
Turbine types that are not in the turbine table are matched to the closest known name.
"""
from __future__ import annotations

import difflib
import io
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import settings

MAX_BYTES = 25_000_000
DEFAULT_TYPE_CUTOFF = 0.0          # always suggest the closest known turbine; the page flags suggested matches


# ------------------------------------------------------------------ column detection (pure, no pandas)
# Column names may be English or German. Umlauts are folded (ä -> ae, ß -> ss), so 'Nabenhöhe' and 'Nabenhoehe' both work.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def _prep(col) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col)).lower().translate(_FOLD)      # split camelCase, then fold


def _compact(col) -> str:
    """'Park ID', 'park_id', 'ParkID' -> 'parkid'."""
    return re.sub(r"[^a-z0-9]+", "", _prep(col))


def _tokens(col) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _prep(col)) if t]


def _is_park_id(col) -> bool:
    c, tk = _compact(col), _tokens(col)
    return "park" in c and ("id" in tk or c.endswith("id") or "nr" in tk or "number" in c or "nummer" in c or "kennung" in c)


def _is_park_name(col) -> bool:
    c = _compact(col)
    if _is_park_id(col):
        return False
    return ("park" in c and ("name" in c or "bezeichnung" in c)) or c in ("park", "windpark", "parkname", "windparkname")


def _is_lat(c: str) -> bool:
    return c.startswith("lat") or "breite" in c or c in ("nord", "y")


def _is_lon(c: str) -> bool:
    return c.startswith("lon") or c.startswith("lng") or (("laenge" in c) and "rotor" not in c and "blatt" not in c) or c in ("ost", "x")


def _is_type(c: str) -> bool:
    kind = "typ" in c or "model" in c                              # typ(e), modell, model
    return (kind and any(w in c for w in ("turbin", "anlage", "wea", "wka"))) or c in ("type", "typ", "model", "modell", "bezeichnung")


def find_columns(columns: Sequence) -> Dict[str, Optional[object]]:
    """Which column holds what. Values are the original column labels (or None when not found)."""
    found: Dict[str, Optional[object]] = {k: None for k in ("park_name", "park_id", "lat", "lon", "type", "maker", "hub", "date")}
    for col in columns:
        c = _compact(col)
        if found["park_id"] is None and _is_park_id(col):
            found["park_id"] = col
        elif found["park_name"] is None and _is_park_name(col):
            found["park_name"] = col
        elif found["lat"] is None and _is_lat(c):
            found["lat"] = col
        elif found["lon"] is None and _is_lon(c):
            found["lon"] = col
        elif found["maker"] is None and ("hersteller" in c or "manufacturer" in c or "fabrikat" in c or c in ("make", "maker", "oem")):
            found["maker"] = col
        elif found["hub"] is None and ("hub" in c or "nabenh" in c):
            found["hub"] = col
        elif found["date"] is None and any(w in c for w in ("commission", "inbetrieb", "installation", "baujahr", "errichtung", "betriebsbeginn")):
            found["date"] = col
    taken = list(found.values())
    for col in columns:                                          # turbine type last: it must not steal park/hub columns
        if col not in taken and _is_type(_compact(col)):
            found["type"] = col
            break
    if found["type"] is None:                                    # looser fallback
        for col in columns:
            c = _compact(col)
            if col not in taken and any(w in c for w in ("typ", "model", "turbin", "anlage")):
                found["type"] = col
                break
    return found


# ------------------------------------------------------------------ turbine type matching (pure)
def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def match_turbine(name: str, known: Sequence[str]) -> Tuple[str, str]:
    """(turbine from the table, 'exact' | 'suggested'). Suggested = closest name, to be confirmed by the user."""
    keys = {_key(k): k for k in known}
    k = _key(name)
    if k in keys:
        return keys[k], "exact"
    inside = [orig for kk, orig in keys.items() if k and (k in kk or kk in k)]     # 'V112-3.0' inside 'Vestas V112-3.0 Offshore'
    if inside:
        return min(inside, key=len), "suggested"
    best = max(keys.items(), key=lambda kv: difflib.SequenceMatcher(None, k, kv[0]).ratio())
    return best[1], "suggested"


# ------------------------------------------------------------------ reading
def _read_frames(filename: str, data: bytes):
    import pandas as pd
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except ImportError:
            raise ValueError("Reading Excel files needs the 'openpyxl' package (pip install openpyxl).")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read the Excel file: {exc}")
        return list(sheets.values())
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    head = text[:4096]
    sep = ";" if head.count(";") >= head.count(",") and ";" in head else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
        non_key = df.iloc[:, 1:]
        if len(non_key.columns) and sum(non_key[c].dtype == object for c in non_key.columns) == len(non_key.columns):
            df = pd.read_csv(io.StringIO(text), sep=sep, decimal=",")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the file as CSV: {exc}")
    return [df]


def _num(series):
    import pandas as pd
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _date(series):
    import pandas as pd
    years = pd.to_numeric(series, errors="coerce")
    if years.notna().mean() > 0.5 and years.dropna().between(1980, 2100).all():          # a build year, e.g. 2013
        return pd.to_datetime(years.round().astype("Int64").astype(str) + "-01-01", errors="coerce")
    as_text = series.astype(str)
    dayfirst = bool(as_text.str.contains(r"^\d{1,2}\.\d{1,2}\.\d{4}", regex=True).any())
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def parse_parks(filename: str, data: bytes, known_types: Sequence[str]) -> dict:
    """Group the turbines of an uploaded file into parks."""
    import pandas as pd
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise ValueError("The uploaded file is too large (limit about 25 MB).")
    frames = _read_frames(filename, data)

    chosen, cols = None, None
    for df in frames:                                             # first sheet that has everything needed
        c = find_columns(list(df.columns))
        if (c["park_name"] is not None or c["park_id"] is not None) and all(c[k] is not None for k in ("lat", "lon", "type", "hub")):
            chosen, cols = df, c
            break
    if chosen is None:
        c = find_columns(list(frames[0].columns))
        missing = []
        if c["park_name"] is None and c["park_id"] is None:
            missing.append("a park column (its name must contain 'park' and 'name' or 'id')")
        for k, label in (("lat", "latitude"), ("lon", "longitude"), ("type", "turbine type"), ("hub", "hub height")):
            if c[k] is None:
                missing.append(f"a {label} column")
        raise ValueError("Could not find " + ", ".join(missing) + ". Columns in the file: "
                         + ", ".join(str(x) for x in frames[0].columns[:30]) + ".")

    df = chosen.copy()
    n_rows = len(df)
    df["_lat"], df["_lon"] = _num(df[cols["lat"]]), _num(df[cols["lon"]])
    df["_hub"] = _num(df[cols["hub"]])
    df["_date"] = _date(df[cols["date"]]) if cols["date"] is not None else pd.NaT
    df = df[df["_lat"].between(-90, 90) & df["_lon"].between(-180, 180)]
    warnings: List[str] = []
    if len(df) < n_rows:
        warnings.append(f"{n_rows - len(df)} rows without a valid latitude/longitude were skipped.")

    has_id, has_name = cols["park_id"] is not None, cols["park_name"] is not None
    id_s = df[cols["park_id"]].astype(str).str.strip() if has_id else None
    name_s = df[cols["park_name"]].astype(str).str.strip() if has_name else None
    df["_key"] = id_s if has_id else name_s
    df = df[df["_key"].notna() & (df["_key"] != "") & (df["_key"].str.lower() != "nan")]

    parks: List[dict] = []
    for key, grp in df.groupby("_key", sort=False):
        name = str(name_s.loc[grp.index[0]]) if has_name else str(key)
        pid = str(key) if has_id else None
        label = f"{name} ({pid})" if has_name and has_id else name
        turbines = []
        for _, r in grp.iterrows():
            src_type = str(r[cols["type"]]).strip()
            if cols["maker"] is not None:                         # 'Enercon' + 'E-82 E2' -> 'Enercon E-82 E2'
                maker = str(r[cols["maker"]]).strip()
                if maker and maker.lower() != "nan" and maker.lower() not in src_type.lower():
                    src_type = f"{maker} {src_type}".strip()
            tname, how = match_turbine(src_type, known_types) if src_type and src_type.lower() != "nan" else (known_types[0], "suggested")
            hub = r["_hub"]
            turbines.append({
                "lat": round(float(r["_lat"]), 6), "lon": round(float(r["_lon"]), 6), "turbine": tname,
                "hub_height": round(float(hub), 2) if pd.notna(hub) and hub > 0 else None,
                "commissioning_date": r["_date"].strftime("%Y-%m-%d") if pd.notna(r["_date"]) else None,
                "match": how, "source_type": src_type,
            })
        if len(turbines) > settings.MAX_TURBINES:
            warnings.append(f"{label}: only the first {settings.MAX_TURBINES} of {len(turbines)} turbines are used.")
            turbines = turbines[:settings.MAX_TURBINES]
        parks.append({"key": str(key), "label": label, "name": name, "id": pid, "n": len(turbines), "turbines": turbines})
    if not parks:
        raise ValueError("No parks with turbines were found in the file.")
    parks.sort(key=lambda p: p["label"].lower())
    return {"filename": filename, "columns": {k: (str(v) if v is not None else None) for k, v in cols.items()},
            "parks": parks, "warnings": warnings}
