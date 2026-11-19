#!/usr/bin/env python3
"""Select real MaStR wind parks (all turbine types present in the scraped power-curve library)
near the synthetic sites and assemble 8 clients x 10-park ladders + 10 held-out parks.

Usage: select_real_parks.py <wind_turbines_matched.csv> <stations_master.csv> <out_dir> [--seed 42]
Ladder slots (composition classes): see SLOTS. Each park is used once, one park per station.
Clients are regional (k-means on park coordinates, 8 regions); each client gets one park per slot,
chosen to spread commissioning cohorts. Held-out = one more ladder from the remaining pool.
"""
import argparse, numpy as np, pandas as pd
from sklearn.cluster import KMeans

SLOTS = [  # evaluated in order (first match wins); (name, test(n, n_types, span_years, minority_share))
    ('S10_four_plus_types', lambda n, k, sp, mi: n >= 6 and k >= 4),
    ('S7_three_types', lambda n, k, sp, mi: n >= 5 and k == 3),
    ('S8_extended', lambda n, k, sp, mi: n >= 4 and k == 2 and sp >= 4),
    ('S5_mid_mixed', lambda n, k, sp, mi: n >= 7 and k == 2),
    ('S3_small_mixed', lambda n, k, sp, mi: 2 <= n <= 6 and k == 2),
    ('S9_xl_homog', lambda n, k, sp, mi: n >= 13 and k == 1),
    ('S6_large_homog', lambda n, k, sp, mi: 9 <= n <= 12 and k == 1),
    ('S4_mid_homog', lambda n, k, sp, mi: 5 <= n <= 8 and k == 1),
    ('S2_small_homog', lambda n, k, sp, mi: 2 <= n <= 4 and k == 1),
    ('S1_single', lambda n, k, sp, mi: n == 1),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('matched'); ap.add_argument('stations'); ap.add_argument('out'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-km', type=float, default=15.0); a = ap.parse_args()
    t = pd.read_csv(a.matched, low_memory=False)
    st = pd.read_csv(a.stations, dtype={'station_id': str})
    g = t.groupby('park_cluster')
    parks = g.agg(n=('EinheitMastrNummer', 'size'), ok=('lib_name', lambda s: s.notna().all()), k=('lib_name', 'nunique'),
                  y_min=('year', 'min'), y_max=('year', 'max'), cap_kw=('Nettonennleistung', 'sum'), lat=('Breitengrad', 'mean'), lon=('Laengengrad', 'mean'),
                  types=('lib_name', lambda s: ' | '.join(f'{v}x {n}' for n, v in s.value_counts().items())),
                  hubs=('Nabenhoehe', lambda s: '/'.join(str(int(x)) for x in sorted(s.dropna().unique()))),
                  name=('NameWindpark', lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ''))
    parks['span'] = parks['y_max'] - parks['y_min']
    parks['minority'] = g['lib_name'].apply(lambda s: (1 - s.value_counts(normalize=True).iloc[0]) if s.notna().any() else np.nan)
    parks = parks[parks['ok']]
    # nearest station
    def xy(lat, lon): la, lo = np.deg2rad(lat), np.deg2rad(lon); return np.c_[6371 * np.cos(la) * lo, 6371 * la]
    d = np.sqrt(((xy(parks['lat'].values, parks['lon'].values)[:, None] - xy(st['latitude'].values, st['longitude'].values)[None]) ** 2).sum(-1))
    parks['station'] = st['station_id'].values[d.argmin(1)]; parks['d_km'] = d.min(1)
    pool = parks[parks['d_km'] <= a.max_km].copy()
    pool['slot'] = [next((nm for nm, f in SLOTS if f(r.n, r.k, r.span, r.minority)), None) for r in pool.itertuples()]
    pool['cohort'] = pd.cut(pool['y_max'], [0, 2004, 2009, 2014, 2019, 2030], labels=['<=2004', '2005-09', '2010-14', '2015-19', '2020+'])
    print(f'pool: {len(pool)} fully-matched parks within {a.max_km} km of {pool["station"].nunique()} stations')
    print('per slot:', pool['slot'].value_counts().reindex([s for s, _ in SLOTS]).fillna(0).astype(int).to_dict())
    print('per cohort (youngest group):', pool['cohort'].value_counts().sort_index().to_dict())
    pool = pool.dropna(subset=['slot'])
    # regions
    rng = np.random.default_rng(a.seed)
    km = KMeans(n_clusters=8, n_init=10, random_state=a.seed).fit(xy(pool['lat'].values, pool['lon'].values))
    pool['region'] = km.labels_
    print('parks per region:', pool['region'].value_counts().sort_index().to_dict())
    # greedy assignment: for each slot, for each client (region), pick an unused park from that region
    # (fallback: nearest other region), unique station, preferring under-represented cohorts and small distance
    used_parks, used_stations, rows = set(), set(), []
    cohort_count = {c: 0 for c in range(8)}
    region_centers = km.cluster_centers_
    for slot, _ in SLOTS:
        cand_all = pool[pool['slot'] == slot]
        for client in range(8):
            order = np.argsort(((region_centers - region_centers[client]) ** 2).sum(1))  # own region first
            picked = None
            for reg in order:
                c = cand_all[(cand_all['region'] == reg) & ~cand_all.index.isin(used_parks) & ~cand_all['station'].isin(used_stations)]
                if len(c) == 0: continue
                # prefer cohorts the client does not have yet, then closeness to station
                have = {r.get('cohort') for r in rows if r['client'] == client}
                c = c.assign(score=c['cohort'].astype(str).isin(have).astype(int) * 10 + c['d_km'] / a.max_km + (reg != client) * 5)
                picked = c.sort_values('score').iloc[0]; break
            if picked is None: rows.append({'client': client, 'slot': slot, 'park_cluster': None}); continue
            used_parks.add(picked.name); used_stations.add(picked['station'])
            rows.append({'client': client, 'slot': slot, 'park_cluster': picked.name, 'region': int(picked['region']), 'own_region': int(picked['region']) == client,
                         'station': picked['station'], 'd_km': round(picked['d_km'], 1), 'n': int(picked['n']), 'k': int(picked['k']), 'cap_mw': round(picked['cap_kw'] / 1e3, 1),
                         'y_min': int(picked['y_min']), 'y_max': int(picked['y_max']), 'cohort': str(picked['cohort']), 'hubs': picked['hubs'], 'types': picked['types'], 'name': picked['name']})
    # held-out ladder (client 8) from the remaining pool, any region
    for slot, _ in SLOTS:
        c = pool[(pool['slot'] == slot) & ~pool.index.isin(used_parks) & ~pool['station'].isin(used_stations)]
        if len(c) == 0: rows.append({'client': 'holdout', 'slot': slot, 'park_cluster': None}); continue
        picked = c.sort_values('d_km').iloc[0]; used_parks.add(picked.name); used_stations.add(picked['station'])
        rows.append({'client': 'holdout', 'slot': slot, 'park_cluster': picked.name, 'region': int(picked['region']), 'own_region': False, 'station': picked['station'], 'd_km': round(picked['d_km'], 1),
                     'n': int(picked['n']), 'k': int(picked['k']), 'cap_mw': round(picked['cap_kw'] / 1e3, 1), 'y_min': int(picked['y_min']), 'y_max': int(picked['y_max']), 'cohort': str(picked['cohort']),
                     'hubs': picked['hubs'], 'types': picked['types'], 'name': picked['name']})
    sel = pd.DataFrame(rows)
    sel.to_csv(f'{a.out}/wind_park_selection_v0.csv', index=False); pool.to_csv(f'{a.out}/wind_park_pool.csv')
    filled = sel['park_cluster'].notna()
    print(f'\nselection: {filled.sum()} of {len(sel)} slots filled; parks from own region: {sel.loc[filled, "own_region"].mean() * 100:.0f}%; distinct stations: {sel.loc[filled, "station"].nunique()}')
    print('unfilled slots:', sel.loc[~filled, ['client', 'slot']].values.tolist())
    print('cohort mix of selection:', sel.loc[filled, 'cohort'].value_counts().sort_index().to_dict())
    print('turbines total:', int(sel.loc[filled, 'n'].sum()), '| capacity MW:', round(sel.loc[filled, 'cap_mw'].sum()), '| distinct turbine types used:', len({x.split('x ')[1] for s in sel.loc[filled, 'types'] for x in s.split(' | ')}))
    print('\nclient 0 ladder example:'); print(sel[sel['client'] == 0][['slot', 'n', 'k', 'cap_mw', 'y_min', 'y_max', 'hubs', 'types']].to_string(index=False))


if __name__ == '__main__':
    main()
