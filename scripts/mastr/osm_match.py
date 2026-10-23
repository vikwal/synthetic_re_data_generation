#!/usr/bin/env python3
"""MaStR turbine coordinates vs OpenStreetMap wind generators.

OSM masts are mostly digitised from aerial imagery and are independent of the
operator self-declarations in MaStR, so a large MaStR-OSM offset almost always
means a MaStR coordinate error (duplicated park coordinate, typo).

Per park (Lokation) the MaStR units are matched one-to-one to the OSM masts
within SEARCH_M (Hungarian assignment on UTM32 distance). A unit is
  ok          match <= OK_M
  corrected   match > OK_M, the OSM node is tag-compatible and unambiguous
              (no other compatible free node within d + AMBIGUITY_M)
  unresolved  otherwise (no compatible node, ambiguous, or nothing in range)
A park is OSM-consistent iff every unit is 'ok' and no pair is closer than one
rotor diameter.

Usage:
  osm_match.py fetch <out.json>          # Overpass snapshot of all DE wind masts
  osm_match.py check <matched.csv> <osm.json> <lokation> [...]
  osm_match.py corrections <matched.csv> <osm.json> <selection.csv> <out.csv> [--fixes coordinate_fixes.csv]

MaStR coordinates on whole degrees (e.g. 54.0/10.0) are placeholders; such units
get status 'placeholder' (never 'ok'), the automatic search around them is
meaningless. Manually researched positions (coordinate_fixes.csv: unit, lat,
lon, osm_id, evidence) override everything with status 'corrected_manual'.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = ('[out:json][timeout:240];area["ISO3166-1"="DE"][admin_level=2]->.de;'
                  '(node["generator:source"="wind"](area.de);way["generator:source"="wind"](area.de););'
                  'out center tags;')
SEARCH_M = 600.0
OK_M = 100.0
AMBIGUITY_M = 100.0
HUB_TOL_M = 5.0
RATED_TOL = 0.15
OFFSET_MIN_UNITS = 3
OFFSET_TOL_M = 15.0
OFFSET_HIT_M = 30.0
MANUFACTURER_ALIAS = {"repower": "senvion", "an": "bonus", "general": "ge", "neg": "neg-micon",
                      "micon": "neg-micon", "neg-micon": "neg-micon", "siemens-gamesa": "siemens"}

_TF = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def to_utm32(lon, lat):
    return _TF.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def fetch_osm(out_path: str) -> None:
    import requests
    r = requests.post(OVERPASS_URL, data={"data": OVERPASS_QUERY}, timeout=300,
                      headers={"User-Agent": "fl-contribution-park-check/1.0 (research)",
                               "Accept": "application/json"})
    r.raise_for_status()
    with open(out_path, "w") as f:
        f.write(r.text)


def _parse_float(s):
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(s))
    return float(m.group(0).replace(",", ".")) if m else np.nan


def parse_rated_kw(s) -> float:
    """'2.0 MW' / '850 kW' / '2500 KW' / 'yes' -> kW (NaN if unparseable)."""
    v = _parse_float(s)
    if np.isnan(v):
        return v
    return v * 1000.0 if "mw" in str(s).lower() else v


def norm_manufacturer(s) -> str:
    tok = re.split(r"[\s/_]+", str(s).strip().lower())[0] if pd.notna(s) and str(s).strip() else ""
    return MANUFACTURER_ALIAS.get(tok, tok)


def load_osm(path: str) -> pd.DataFrame:
    with open(path) as f:
        els = json.load(f)["elements"]
    rows = []
    for e in els:
        tags = e.get("tags", {})
        if tags.get("power") != "generator":
            continue
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None:
            continue
        rows.append({"osm_id": f"{e['type']}/{e['id']}", "osm_lat": lat, "osm_lon": lon,
                     "osm_manufacturer": tags.get("manufacturer"), "osm_model": tags.get("model"),
                     "osm_hub_m": _parse_float(tags.get("height:hub", np.nan)),
                     "osm_rated_kw": parse_rated_kw(tags.get("generator:output:electricity", np.nan))})
    o = pd.DataFrame(rows)
    o["osm_x"], o["osm_y"] = to_utm32(o.osm_lon, o.osm_lat)
    return o.reset_index(drop=True)


def tag_compatible(unit: pd.Series, node: pd.Series) -> bool:
    """False only if an OSM tag that is present contradicts the MaStR unit."""
    if pd.notna(node.osm_manufacturer):
        if norm_manufacturer(node.osm_manufacturer) != norm_manufacturer(unit.lib_name):
            return False
    if pd.notna(node.osm_hub_m) and abs(node.osm_hub_m - unit.Nabenhoehe) > HUB_TOL_M:
        return False
    if pd.notna(node.osm_rated_kw) and node.osm_rated_kw > 0:
        if abs(node.osm_rated_kw - unit.Nettonennleistung) / unit.Nettonennleistung > RATED_TOL:
            return False
    return True


class OsmIndex:
    def __init__(self, osm: pd.DataFrame):
        self.osm = osm
        self.tree = cKDTree(osm[["osm_x", "osm_y"]].values)

    def match_park(self, units: pd.DataFrame) -> pd.DataFrame:
        """One row per MaStR unit with status ok/corrected/unresolved and the
        coordinate to use (x_utm32/y_utm32, coord_source)."""
        units = units.reset_index(drop=True)
        ux, uy = to_utm32(units.Laengengrad, units.Breitengrad)
        P = np.c_[ux, uy]
        cand = sorted({j for p in P for j in self.tree.query_ball_point(p, SEARCH_M)})
        out = units[["EinheitMastrNummer", "lokation"]].copy()
        out["x_mastr"], out["y_mastr"] = ux, uy
        out["osm_id"], out["d_osm_m"], out["status"] = None, np.nan, "unresolved"
        out["x_utm32"], out["y_utm32"], out["coord_source"] = ux, uy, "mastr"
        out["lat"], out["lon"] = units.Breitengrad.values, units.Laengengrad.values
        if not cand:
            out["reason"] = "no OSM mast within 600 m"
            return self._mark_placeholders(out, units)
        Q = self.osm.iloc[cand]
        D = np.hypot(P[:, None, 0] - Q.osm_x.values[None, :], P[:, None, 1] - Q.osm_y.values[None, :])
        # pairs beyond the search radius are forbidden, and every unit may stay
        # unassigned at cost SEARCH_M (dummy columns): otherwise the assignment
        # pulls a far-off unit onto a mast that belongs to a correctly placed
        # neighbour and displaces that neighbour instead
        n = len(units)
        cost = np.hstack([np.where(D <= SEARCH_M, D, 1e9), np.full((n, n), SEARCH_M)])
        ri, ci = linear_sum_assignment(cost)
        assigned = {i: j for i, j in zip(ri, ci) if j < D.shape[1] and D[i, j] <= SEARCH_M}
        out["reason"] = ""
        taken_ok = {assigned[i] for i in assigned if D[i, assigned[i]] <= OK_M}
        for i in range(len(units)):
            j = assigned.get(i)
            if j is None:
                out.loc[i, "reason"] = "no free OSM mast within 600 m"
                continue
            out.loc[i, ["osm_id", "d_osm_m"]] = [Q.osm_id.iloc[j], D[i, j]]
            if D[i, j] <= OK_M:
                out.loc[i, "status"] = "ok"
                continue
            node = Q.iloc[j]
            if not tag_compatible(units.iloc[i], node):
                out.loc[i, "reason"] = "assigned OSM mast contradicts type/hub/rating tags"
                continue
            rivals = [k for k in range(len(cand)) if k != j and k not in taken_ok
                      and D[i, k] <= D[i, j] + AMBIGUITY_M and tag_compatible(units.iloc[i], Q.iloc[k])]
            if rivals:
                out.loc[i, "reason"] = f"ambiguous: {len(rivals)} other compatible masts within d+{AMBIGUITY_M:.0f} m"
                continue
            out.loc[i, ["status", "x_utm32", "y_utm32", "lat", "lon", "coord_source"]] = \
                ["corrected", node.osm_x, node.osm_y, node.osm_lat, node.osm_lon, f"osm:{node.osm_id}"]
            out.loc[i, "reason"] = f"MaStR {D[i, j]:.0f} m off the OSM mast"
        return self._mark_placeholders(self._apply_park_offset(out, Q), units)

    @staticmethod
    def _mark_placeholders(out: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
        """Whole-degree MaStR coordinates are register placeholders, not positions."""
        ph = is_placeholder(units["Breitengrad"].values, units["Laengengrad"].values)
        for i in np.flatnonzero(ph):
            if out.loc[i, "status"] != "corrected":
                out.loc[i, ["status", "osm_id", "d_osm_m"]] = ["placeholder", None, np.nan]
                out.loc[i, "reason"] = "MaStR placeholder coordinate (whole degrees)"
        return out

    def _apply_park_offset(self, out: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
        """Systematic offset: if >= OFFSET_MIN_UNITS corrected units share one
        shift vector (each within OFFSET_TOL_M of the mean), an ambiguous unit
        whose assigned mast lies within OFFSET_HIT_M of MaStR + mean shift is
        corrected to that mast (operator registered the whole park shifted)."""
        cor = out[out.status == "corrected"]
        if len(cor) < OFFSET_MIN_UNITS:
            return out
        dx, dy = cor.x_utm32 - cor.x_mastr, cor.y_utm32 - cor.y_mastr
        mx, my = dx.mean(), dy.mean()
        if np.hypot(dx - mx, dy - my).max() > OFFSET_TOL_M:
            return out
        for i in out.index[(out.status == "unresolved") & out.osm_id.notna()]:
            node = Q[Q.osm_id == out.loc[i, "osm_id"]].iloc[0]
            px, py = out.loc[i, "x_mastr"] + mx, out.loc[i, "y_mastr"] + my
            if np.hypot(node.osm_x - px, node.osm_y - py) <= OFFSET_HIT_M:
                out.loc[i, ["status", "x_utm32", "y_utm32", "lat", "lon", "coord_source"]] = \
                    ["corrected", node.osm_x, node.osm_y, node.osm_lat, node.osm_lon, f"osm:{node.osm_id}"]
                out.loc[i, "reason"] = (f"systematic park offset ({mx:.0f}, {my:.0f}) m shared by "
                                        f"{len(cor)} corrected units")
        return out


def is_placeholder(lat, lon) -> np.ndarray:
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return (np.abs(lat - np.round(lat)) < 1e-9) & (np.abs(lon - np.round(lon)) < 1e-9)


def apply_fixes(res: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    """Manually researched coordinates (coordinate_fixes.csv) override the match."""
    res = res.copy()
    for f in fixes.itertuples(index=False):
        hit = res["EinheitMastrNummer"] == f.EinheitMastrNummer
        if not hit.any():
            continue
        x, y = to_utm32(f.lon, f.lat)
        i = res.index[hit][0]
        d = float(np.hypot(res.loc[i, "x_mastr"] - x, res.loc[i, "y_mastr"] - y))
        res.loc[i, ["status", "lat", "lon", "x_utm32", "y_utm32", "osm_id", "d_osm_m", "coord_source", "reason"]] = \
            ["corrected_manual", f.lat, f.lon, float(x), float(y), f.osm_id, d, f"manual:{f.osm_id}", f.evidence]
    return res


def min_pair_spacing_d(x, y, rotor) -> float:
    """Smallest pairwise distance in rotor diameters (max of the pair); inf for n=1."""
    x, y, rotor = map(np.asarray, (x, y, rotor))
    if len(x) < 2:
        return np.inf
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    dmax = np.maximum(rotor[:, None], rotor[None, :])
    np.fill_diagonal(d, np.inf)
    return float((d / dmax).min())


def park_consistent(match: pd.DataFrame, rotor) -> bool:
    """Strict OSM consistency (used for replacement candidates)."""
    return bool((match["status"] == "ok").all()
                and min_pair_spacing_d(match.x_mastr, match.y_mastr, rotor) >= 1.0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("out")
    c = sub.add_parser("check"); c.add_argument("matched"); c.add_argument("osm"); c.add_argument("lokation", nargs="+")
    k = sub.add_parser("corrections"); k.add_argument("matched"); k.add_argument("osm"); k.add_argument("selection"); k.add_argument("out"); k.add_argument("--fixes")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_osm(a.out)
        return
    t = pd.read_csv(a.matched, low_memory=False)
    idx = OsmIndex(load_osm(a.osm))
    if a.cmd == "corrections":
        sel = pd.read_csv(a.selection)
        res = pd.concat([idx.match_park(t[t.lokation == lk]) for lk in sel.lokation], ignore_index=True)
        if a.fixes:
            res = apply_fixes(res, pd.read_csv(a.fixes))
        res.to_csv(a.out, index=False)
        print(res.status.value_counts().to_string())
        print(res[res.status != "ok"][["lokation", "EinheitMastrNummer", "status", "d_osm_m", "reason"]]
              .round(1).to_string(index=False))
        return
    for lk in a.lokation:
        m = idx.match_park(t[t.lokation == lk])
        print(m[["EinheitMastrNummer", "status", "d_osm_m", "osm_id", "reason"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
