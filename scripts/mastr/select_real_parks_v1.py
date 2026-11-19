#!/usr/bin/env python3
"""Selection v1: real MaStR wind parks (park = LokationMaStRNummer), classes B1-B5, 8 clients x 10 parks
(2 per class) + held-out ladder; 4 regional clients (k-means regions) + 4 nationwide clients
(parks spread over all regions); no station constraint (ERA5-driven targets).

Usage: select_real_parks_v1.py <wind_turbines_matched.csv> <out_dir> [--seed 42] [--regions 4]
"""
import argparse, numpy as np, pandas as pd
from sklearn.cluster import KMeans

CLASSES = [('B1_single', lambda n, k: n == 1), ('B2_small_homog', lambda n, k: 2 <= n <= 4 and k == 1),
           ('B3_large_homog', lambda n, k: n >= 5 and k == 1), ('B4_two_types', lambda n, k: k == 2), ('B5_three_plus', lambda n, k: k >= 3)]
COHORT = lambda y: '<=2004' if y <= 2004 else '2005-14' if y <= 2014 else '2015+'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('matched'); ap.add_argument('out'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--regions', type=int, default=4)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    t = pd.read_csv(a.matched, low_memory=False)
    t['type_key'] = t['Hersteller'].astype(str).str.split().str[0] + '|' + t['Rotordurchmesser'].round(0).astype(str) + '|' + t['Nettonennleistung'].round(-1).astype(str)
    g = t.groupby('lokation')
    p = g.agg(n=('EinheitMastrNummer', 'size'), k=('type_key', 'nunique'), ok=('lib_name', lambda s: s.notna().all()), ops=('operator', 'nunique'),
              y_min=('year', 'min'), y_max=('year', 'max'), cap_kw=('Nettonennleistung', 'sum'), lat=('Breitengrad', 'mean'), lon=('Laengengrad', 'mean'),
              name=('NameWindpark', lambda s: s.mode().iloc[0] if s.notna().any() else ''),
              types=('lib_name', lambda s: ' | '.join(f'{v}x {n}' for n, v in s.value_counts().items())),
              hubs=('Nabenhoehe', lambda s: '/'.join(str(int(x)) for x in sorted(s.dropna().unique()))))
    pool = p[p['ok']].copy()
    pool['cls'] = [next(c for c, f in CLASSES if f(r.n, r.k)) for r in pool.itertuples()]
    pool['cohort'] = pool['y_max'].map(COHORT)
    la, lo = np.deg2rad(pool['lat'].values), np.deg2rad(pool['lon'].values); xy = np.c_[6371 * np.cos(la) * lo, 6371 * la]
    km = KMeans(n_clusters=a.regions, n_init=10, random_state=a.seed).fit(xy); pool['region'] = km.labels_
    print(f'pool: {len(pool)} parks with curves (of {len(p)} Lokationen)'); print('per class:', pool['cls'].value_counts().reindex([c for c, _ in CLASSES]).to_dict())
    print('per region:', pool['region'].value_counts().sort_index().to_dict()); print('region centres (lat, lon):', [(round(float(np.degrees(y / 6371)), 2), round(float(np.degrees(x / 6371 / np.cos(np.radians(51)))), 2)) for x, y in km.cluster_centers_])
    used, rows = set(), []
    def pick(cands, have_cohorts):
        c = cands[~cands.index.isin(used)]
        if len(c) == 0: return None
        # prefer cohorts not yet used for this client+class, then random
        c = c.assign(score=c['cohort'].isin(have_cohorts).astype(int) + rng.random(len(c)) * 0.5)
        return c.sort_values('score').iloc[0]
    clients = [('R%d' % r, 'regional', r) for r in range(a.regions)] + [('N%d' % i, 'nationwide', None) for i in range(8 - a.regions)] + [('holdout', 'nationwide', None)]
    for cid, kind, reg in clients:
        rr = 0
        for cls, _ in CLASSES:
            have = set()
            for j in range(2):
                if kind == 'regional':
                    cand = pool[(pool['cls'] == cls) & (pool['region'] == reg)]
                else:  # nationwide: rotate regions so each client spreads over all regions
                    cand = pool[(pool['cls'] == cls) & (pool['region'] == rr % a.regions)]; rr += 1
                pk = pick(cand, have)
                if pk is None: rows.append({'client': cid, 'kind': kind, 'cls': cls, 'lokation': None}); continue
                used.add(pk.name); have.add(pk['cohort'])
                rows.append({'client': cid, 'kind': kind, 'cls': cls, 'lokation': pk.name, 'region': int(pk['region']), 'n': int(pk['n']), 'k': int(pk['k']), 'n_operators': int(pk['ops']),
                             'cap_mw': round(pk['cap_kw'] / 1e3, 2), 'y_min': int(pk['y_min']), 'y_max': int(pk['y_max']), 'cohort': pk['cohort'], 'lat': round(pk['lat'], 4), 'lon': round(pk['lon'], 4),
                             'hubs': pk['hubs'], 'types': pk['types'], 'name': pk['name']})
    sel = pd.DataFrame(rows); sel.to_csv(f'{a.out}/wind_park_selection_v1.csv', index=False); pool.to_csv(f'{a.out}/wind_park_pool_v1.csv')
    f = sel.dropna(subset=['lokation'])
    print(f'\nselected {len(f)}/{len(sel)} slots | turbines {int(f.n.sum())} | capacity {f.cap_mw.sum():.0f} MW | types {len({x.split("x ")[1] for s in f.types for x in s.split(" | ")})}')
    print('cohorts:', f['cohort'].value_counts().to_dict(), '| multi-operator parks:', int((f.n_operators > 1).sum()))
    print(f.groupby('client').agg(parks=('lokation', 'size'), turbines=('n', 'sum'), mw=('cap_mw', 'sum'), regions=('region', 'nunique')).round(0).to_string())


if __name__ == '__main__':
    main()
