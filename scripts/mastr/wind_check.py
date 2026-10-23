"""Feasibility check: rebuild real German wind parks from the MaStR extract.

Questions: completeness of per-turbine fields, park grouping, size distribution,
distance to the 200 synthetic sites, and match rate of turbine types against the
scraped power-curve library (by manufacturer + rotor diameter + rated power).
"""
import re, numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

MASTR = 'data/Marktstammdatenregister_20250625_25/Wind_MaStR.csv'
w = pd.read_csv(MASTR)
import xml.etree.ElementTree as ET
kat = {int(k.find('Id').text): k.find('Wert').text for k in ET.parse('data/Marktstammdatenregister_20250625_25/Katalogwerte.xml').getroot()}
w['Hersteller'] = pd.to_numeric(w['Hersteller'], errors='coerce').map(kat)
for c in ['Nabenhoehe', 'Rotordurchmesser', 'Nettonennleistung', 'Breitengrad', 'Laengengrad']:
    w[c] = pd.to_numeric(w[c], errors='coerce')
print('rows total:', len(w)); 
print('Hersteller (decoded) top:', w['Hersteller'].value_counts().head(12).to_dict()); print('Typenbezeichnung sample:', w['Typenbezeichnung'].dropna().unique()[:10].tolist())
w = w[w['EinheitBetriebsstatus'] == 35]                      # in Betrieb
w = w[w['Kuestenentfernung'].isna()]                          # offshore units carry a coast distance
w = w[(w['Breitengrad'] < 55.2) & (w['Laengengrad'].between(5.5, 15.5))]
w = w[w['Nettonennleistung'] >= 100]
print('operating onshore >=100 kW:', len(w), ' capacity MW:', round(w['Nettonennleistung'].sum()/1e3))
for c in ['Breitengrad', 'Nabenhoehe', 'Rotordurchmesser', 'Typenbezeichnung', 'Hersteller', 'Inbetriebnahmedatum', 'NameWindpark']:
    print(f'  {c:20s} non-null {w[c].notna().mean()*100:5.1f}%')
w['year'] = pd.to_datetime(w['Inbetriebnahmedatum'], errors='coerce').dt.year
complete = w.dropna(subset=['Breitengrad', 'Laengengrad', 'Nabenhoehe', 'Rotordurchmesser', 'Typenbezeichnung', 'Hersteller', 'year'])
print('complete per-turbine records:', len(complete), f'({len(complete)/len(w)*100:.1f}%)')

# ---- park grouping: spatial DBSCAN (eps 1.2 km, min 1) as the primary key; NameWindpark as a check
lat0 = np.deg2rad(complete['Breitengrad'].values); lon0 = np.deg2rad(complete['Laengengrad'].values)
xy = np.c_[6371*np.cos(lat0)*lon0, 6371*lat0]                 # km, equirectangular
complete = complete.copy(); complete['park_cluster'] = DBSCAN(eps=1.2, min_samples=1).fit_predict(xy)
sizes = complete.groupby('park_cluster').size()
print('\nspatial parks (eps 1.2 km):', len(sizes))
print('turbines per park: median', sizes.median(), 'mean', round(sizes.mean(), 2), 'quantiles', sizes.quantile([.25, .5, .75, .9, .99]).round(1).to_dict())
print('share of parks with 1 turbine:', round((sizes == 1).mean(), 3), '; >=3:', round((sizes >= 3).mean(), 3), '; >=10:', round((sizes >= 10).mean(), 3))
named = complete['NameWindpark'].notna().mean(); print('NameWindpark filled:', round(named, 3))

# ---- turbine-type matching against the library
pc = pd.read_csv('power_curves/turbine_power.csv', sep=';', decimal=',')
pc = pc.loc[:, ~pc.columns.duplicated()]
specs = pd.read_csv('power_curves/turbine_specs.csv', sep=';', decimal=',')
lib = pd.DataFrame({'name': pc.columns[1:], 'rated_kw': pc.iloc[:, 1:].max().values})
rot = specs.groupby('Turbine')['Rotordurchmesser'].first()
lib['rotor'] = pd.to_numeric(lib['name'].map(rot), errors='coerce'); lib = lib.dropna(subset=['rotor'])
lib['mfr'] = lib['name'].str.split().str[0].str.lower()
MFR = {'enercon': 'enercon', 'vestas': 'vestas', 'nordex': 'nordex', 'senvion': 'senvion', 'repower': 'senvion', 'siemens': 'siemens',
       'gamesa': 'gamesa', 'ge': 'ge', 'eno': 'eno', 'fuhrl': 'fuhrl', 'vensys': 'vensys', 'neg': 'neg', 'bonus': 'an', 'an': 'an',
       'dewind': 'dewind', 'tacke': 'tacke', 'nordtank': 'nordtank', 'micon': 'micon', 'südwind': 'südwind', 'suedwind': 'südwind',
       'lagerwey': 'lagerwey', 'kenersys': 'kenersys', 'w2e': 'w2e', 'adwen': 'adwen', 'areva': 'areva', 'enron': 'enron', 'windtec': 'windtec',
       'powerwind': 'powerwind', 'fwt': 'fwt', 'leitwind': 'leitwind', 'aerodyn': 'aerodyn', 'windwinD': 'winwind'}
def mfr_key(s):
    s = str(s).lower()
    for k, v in MFR.items():
        if s.startswith(k) or k in s.split()[0]:
            return v
    return s.split()[0] if s.split() else s
lib['mk'] = lib['name'].map(mfr_key)
lib['mk'] = lib['mk'].replace({'siemens': 'siemens', 'gamesa': 'siemens'})  # Siemens Gamesa family
complete['mk'] = complete['Hersteller'].map(mfr_key).replace({'siemens': 'siemens', 'gamesa': 'siemens'})

def match(row):
    cand = lib[(lib['mk'] == row['mk']) & ((lib['rotor'] - row['Rotordurchmesser']).abs() <= 1.5)
               & ((lib['rated_kw'] - row['Nettonennleistung']).abs() <= 0.07 * row['Nettonennleistung'])]
    if len(cand) == 0:
        return None
    return cand.iloc[((cand['rated_kw'] - row['Nettonennleistung']).abs()).argmin()]['name']
keys = complete[['mk', 'Rotordurchmesser', 'Nettonennleistung']].drop_duplicates()
keys['lib_name'] = keys.apply(match, axis=1)
complete = complete.merge(keys, on=['mk', 'Rotordurchmesser', 'Nettonennleistung'], how='left')
m = complete['lib_name'].notna()
print(f'\nlibrary match: {m.mean()*100:.1f}% of turbines, {complete.loc[m, "Nettonennleistung"].sum()/complete["Nettonennleistung"].sum()*100:.1f}% of capacity')
for lo, hi in [(0, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]:
    s = complete[complete['year'].between(lo, hi)]
    print(f'  cohort {lo}-{hi}: n={len(s):6d}  matched {s["lib_name"].notna().mean()*100:5.1f}%  (cap-weighted {s.loc[s.lib_name.notna(),"Nettonennleistung"].sum()/max(s["Nettonennleistung"].sum(),1)*100:5.1f}%)')
un = complete[~m].groupby(['Hersteller', 'Typenbezeichnung', 'Rotordurchmesser', 'Nettonennleistung']).size().sort_values(ascending=False)
print('\ntop unmatched (count):'); print(un.head(25).to_string())

# ---- rebuildable parks: >=3 turbines, all matched, within 15 km of a synthetic site
st = pd.read_csv('/home/viktor/Work/stations_master.csv', dtype={'station_id': str})
slat = np.deg2rad(st['latitude'].values); slon = np.deg2rad(st['longitude'].values)
sxy = np.c_[6371*np.cos(slat)*slon, 6371*slat]
parks = complete.groupby('park_cluster').agg(n=('EinheitMastrNummer', 'size'), cap=('Nettonennleistung', 'sum'),
        matched=('lib_name', lambda s: s.notna().mean()), n_types=('lib_name', 'nunique'), n_hub=('Nabenhoehe', 'nunique'),
        y_min=('year', 'min'), y_max=('year', 'max'), lat=('Breitengrad', 'mean'), lon=('Laengengrad', 'mean'))
plat = np.deg2rad(parks['lat'].values); plon = np.deg2rad(parks['lon'].values)
pxy = np.c_[6371*np.cos(plat)*plon, 6371*plat]
d = np.sqrt(((pxy[:, None, :] - sxy[None, :, :])**2).sum(-1)); parks['d_station_km'] = d.min(1); parks['station'] = st['station_id'].values[d.argmin(1)]
cand = parks[(parks['n'] >= 3) & (parks['matched'] == 1.0)]
print(f'\nparks >=3 turbines & fully matched: {len(cand)} (of {len(parks[parks.n>=3])} parks with >=3)')
for r in [5, 10, 15, 25]:
    print(f'  within {r:2d} km of a synthetic site: {(cand["d_station_km"] <= r).sum()}  (>=5 turbines: {((cand["d_station_km"] <= r) & (cand["n"] >= 5)).sum()}; heterogeneous (>=2 types): {((cand["d_station_km"] <= r) & (cand["n_types"] >= 2)).sum()}; multi-hub: {((cand["d_station_km"] <= r) & (cand["n_hub"] >= 2)).sum()})')
c15 = cand[cand['d_station_km'] <= 15]
print('\nheterogeneity among candidates <=15 km: n_types distribution', c15['n_types'].value_counts().sort_index().to_dict())
print('size distribution', c15['n'].describe().round(1).to_dict())
print('commissioning span >= 4 y (extended parks):', (c15['y_max'] - c15['y_min'] >= 4).sum())
print('stations with >=1 candidate park within 15 km:', c15['station'].nunique())
cand.sort_values('d_station_km').to_csv('/home/viktor/Work/FL_Contribution/data/mastr/wind_candidate_parks.csv')
complete.to_csv('/home/viktor/Work/FL_Contribution/data/mastr/wind_turbines_matched.csv', index=False)

# ---- diagnose unmatched: missing in library vs name/spec mismatch; what to scrape next
print('\n--- library presence check for frequent unmatched types:')
full = pd.DataFrame({'name': pc.columns[1:], 'rated_kw': pc.iloc[:, 1:].max().values})
full['rotor_spec'] = pd.to_numeric(full['name'].map(rot), errors='coerce')
for q in ['V80', 'MM82', 'MM92', 'MD77', 'MD 77', 'E-66', 'E-138', 'E-160', 'E-115 EP3', 'GE 5.5', 'V150', 'N149', 'N163', 'V162', 'D6', 'NM64', 'F56', 'E-70', 'V90']:
    hit = full[full['name'].str.contains(q, case=False, regex=False)]
    print(f'  {q:10s}', [(n, r, rs) for n, r, rs in hit[['name', 'rated_kw', 'rotor_spec']].itertuples(index=False)][:5])
print('library turbines without rotor spec (dropped from matching):', full['rotor_spec'].isna().sum(), 'of', len(full))
un_cap = complete[~m].groupby(['Hersteller', 'Rotordurchmesser', 'Nettonennleistung']).agg(n=('EinheitMastrNummer', 'size'), cap_mw=('Nettonennleistung', lambda s: s.sum()/1e3), types=('Typenbezeichnung', lambda s: '|'.join(s.value_counts().index[:3])))
print('\ntop unmatched (manufacturer, rotor, kW) by capacity:'); print(un_cap.sort_values('cap_mw', ascending=False).head(30).to_string())
