#!/usr/bin/env python3
"""Empirical park composition on the full MaStR onshore fleet: per park n (turbines),
k (distinct types = manufacturer x rotor x rated kW), k_hub (distinct hub heights), span (years).

Usage: park_composition_stats.py <wind_turbines_matched.csv> <out_dir>
Writes park_composition.csv (one row per park) and prints distribution tables.
"""
import sys, numpy as np, pandas as pd

t = pd.read_csv(sys.argv[1], low_memory=False)
t['type_key'] = t['Hersteller'].astype(str).str.split().str[0] + '|' + t['Rotordurchmesser'].round(0).astype(str) + '|' + t['Nettonennleistung'].round(-1).astype(str)
g = t.groupby('park_cluster')
p = g.agg(n=('EinheitMastrNummer', 'size'), k=('type_key', 'nunique'), k_hub=('Nabenhoehe', lambda s: s.round(-1).nunique()),
          y_min=('year', 'min'), y_max=('year', 'max'), cap_mw=('Nettonennleistung', lambda s: s.sum() / 1e3),
          matched=('lib_name', lambda s: s.notna().all()), lat=('Breitengrad', 'mean'), lon=('Laengengrad', 'mean'))
p['span'] = p['y_max'] - p['y_min']
p['minority'] = g['type_key'].apply(lambda s: round(1 - s.value_counts(normalize=True).iloc[0], 2))
p.to_csv(f'{sys.argv[2]}/park_composition.csv')
nb = pd.cut(p['n'], [0, 1, 2, 4, 6, 8, 12, 20, 1000], labels=['1', '2', '3-4', '5-6', '7-8', '9-12', '13-20', '>20'])
kb = pd.cut(p['k'], [0, 1, 2, 3, 100], labels=['1', '2', '3', '4+'])
print(f'parks: {len(p)}  turbines: {p.n.sum()}  capacity: {p.cap_mw.sum()/1e3:.1f} GW  (all-types-matched parks: {p.matched.sum()})\n')
ct = pd.crosstab(nb, kb, margins=True); print('== parks by n (rows) x k types (cols), counts =='); print(ct.to_string())
print('\n== same, % of all parks =='); print((pd.crosstab(nb, kb) / len(p) * 100).round(1).to_string())
print('\n== same, % of all TURBINES (capacity-relevant view) =='); print((pd.crosstab(nb, kb, values=p['n'], aggfunc='sum') / p.n.sum() * 100).round(1).to_string())
print('\n== k distribution: parks %', (p['k'].clip(upper=5).value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print('== k_hub > k (same type, several hub heights): %.1f%% of parks' % ((p['k_hub'] > p['k']).mean() * 100))
print('== span >= 4 y among k>=2 parks: %.1f%%  | among all: %.1f%%' % ((p.loc[p.k >= 2, 'span'] >= 4).mean() * 100, (p['span'] >= 4).mean() * 100))
print('== minority share among k=2 parks: quantiles', p.loc[p.k == 2, 'minority'].quantile([.25, .5, .75]).to_dict())
print('\n== within the library-matched pool (all types have curves): n x k counts =='); pm = p[p.matched]
print(pd.crosstab(pd.cut(pm['n'], [0, 1, 2, 4, 6, 8, 12, 20, 1000], labels=['1', '2', '3-4', '5-6', '7-8', '9-12', '13-20', '>20']), pd.cut(pm['k'], [0, 1, 2, 3, 100], labels=['1', '2', '3', '4+']), margins=True).to_string())
