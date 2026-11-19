#!/usr/bin/env python3
"""Build a self-contained HTML map of the selected real wind parks (one marker per turbine).

Usage: build_park_map.py <wind_turbines_matched.csv> <wind_park_selection_v0.csv> <geojson_DE> <stations.csv> <out.html> [template.html] [coords.csv]
  default template: park_map_template.html (offline SVG); park_map_osm_template.html = Leaflet + OSM tiles (needs internet)
  coords.csv (optional): per-unit coordinates to plot instead of MaStR's (EinheitMastrNummer, lat, lon),
  e.g. osm_corrections_v1_1.csv from osm_match.py
"""
import json, os, re, sys, numpy as np, pandas as pd


def douglas_peucker(pts, tol):
    """Iterative Douglas-Peucker; closed rings are split at the point farthest from the start."""
    pts = np.asarray(pts, float)[:, :2]
    if len(pts) < 3: return pts
    if np.allclose(pts[0], pts[-1]): pts = pts[:-1]
    far = int(np.argmax(((pts - pts[0]) ** 2).sum(1)))
    keep = np.zeros(len(pts), bool); keep[[0, far, len(pts) - 1]] = True; stack = [(0, far), (far, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1: continue
        p, q = pts[a], pts[b]; seg = q - p; L = float(np.hypot(*seg)) or 1e-12
        rel = pts[a + 1:b] - p; d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L; i = int(np.argmax(d))
        if d[i] > tol: keep[a + 1 + i] = True; stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def main(turb_csv, sel_csv, geojson, stations_csv, out, template=None, coords_csv=None):
    t = pd.read_csv(turb_csv, low_memory=False); sel = pd.read_csv(sel_csv)
    if coords_csv:
        c = pd.read_csv(coords_csv).set_index('EinheitMastrNummer')
        hit = t['EinheitMastrNummer'].isin(c.index)
        t.loc[hit, 'Breitengrad'] = t.loc[hit, 'EinheitMastrNummer'].map(c['lat'])
        t.loc[hit, 'Laengengrad'] = t.loc[hit, 'EinheitMastrNummer'].map(c['lon'])
    key = 'lokation' if 'lokation' in sel.columns else 'park_cluster'
    sel = sel.dropna(subset=[key]); meta = sel.set_index(key)
    if key == 'park_cluster': meta.index = meta.index.astype(int)
    slot_col = 'cls' if 'cls' in sel.columns else 'slot'
    tt = t[t[key].isin(meta.index)].copy()
    tt['client'] = tt[key].map(meta['client']); tt['slot'] = tt[key].map(meta[slot_col])
    tt['park_name'] = tt[key].map(meta['name']).fillna(''); tt['park_n'] = tt[key].map(meta['n'])
    turbines = [dict(id=r.EinheitMastrNummer, lat=round(r.Breitengrad, 5), lon=round(r.Laengengrad, 5), type=str(r.Typenbezeichnung), lib=str(r.lib_name),
                     hub=None if pd.isna(r.Nabenhoehe) else round(float(r.Nabenhoehe), 1), rotor=None if pd.isna(r.Rotordurchmesser) else round(float(r.Rotordurchmesser), 1),
                     kw=round(float(r.Nettonennleistung)), year=int(r.year), park=str(getattr(r, key)), park_name=str(r.park_name) if r.park_name else '',
                     park_n=int(r.park_n), client=str(r.client), slot=str(r.slot), wp=str(r.NameWindpark) if isinstance(r.NameWindpark, str) else '')
                for r in tt.itertuples()]
    parks = [dict(id=str(i), client=str(r.client), slot=str(r[slot_col]), name=str(r['name']) if isinstance(r['name'], str) else '', n=int(r.n), k=int(r.k), cap=float(r.cap_mw),
                  y0=int(r.y_min), y1=int(r.y_max), hubs=str(r.hubs), types=str(r.types), station=str(r.get('station', '-')), d=float(r['d_km']) if 'd_km' in r and pd.notna(r['d_km']) else -1.0,
                  region=int(r.region) if not pd.isna(r.region) else -1, ops=int(r['n_operators']) if 'n_operators' in r else 1)
             for i, r in meta.iterrows()]
    g = json.load(open(geojson)); polys = g['features'][0]['geometry']['coordinates']
    rings = []
    for poly in polys:
        ring = douglas_peucker(poly[0], 0.008)
        if len(ring) >= 4: rings.append([[round(x, 4), round(y, 4)] for x, y in ring])
    st = pd.read_csv(stations_csv, dtype={'station_id': str})
    stations = [dict(id=r.station_id, lat=round(r.latitude, 4), lon=round(r.longitude, 4), alt=int(r.station_height)) for r in st.itertuples()]
    data = dict(turbines=turbines, parks=parks, rings=rings, stations=stations)
    html = open(template or __file__.replace('build_park_map.py', 'park_map_template.html')).read().replace('/*__DATA__*/', 'const DATA = ' + json.dumps(data, ensure_ascii=False) + ';')
    m = re.search(r'selection_(v[0-9_]+)', os.path.basename(sel_csv))
    if m:  # title follows the selection version (templates say 'v1')
        html = html.replace('Windpark-Auswahl v1', 'Windpark-Auswahl ' + m.group(1).replace('_', '.'))
    open(out, 'w').write(html)
    print(f'{len(turbines)} turbines, {len(parks)} parks, {len(rings)} boundary rings ({sum(len(r) for r in rings)} pts), {len(stations)} stations -> {out} ({len(html)//1024} KB)')


if __name__ == '__main__':
    main(*sys.argv[1:8])
