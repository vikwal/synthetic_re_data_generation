"""Feasibility check: what MaStR tells us about PV parks (flattened CSV part)."""
import numpy as np, pandas as pd
AUS = {695:'N',696:'NE',697:'E',698:'SE',699:'S',700:'SW',701:'W',702:'NW',703:'tracking',704:'E-W'}
NEI = {806:'facade',807:'>60',808:'40-60',809:'20-40',810:'<20',811:'tracking'}
s = pd.read_csv('data/Marktstammdatenregister_20250625_25/Solar_MaStR.csv',
                usecols=['EinheitBetriebsstatus','Inbetriebnahmedatum','Breitengrad','Laengengrad','Nettonennleistung','AnzahlModule','Hauptausrichtung','HauptausrichtungNeigungswinkel'])
print('rows:', len(s)); s = s[s['EinheitBetriebsstatus'] == 35]
s['year'] = pd.to_datetime(s['Inbetriebnahmedatum'], errors='coerce').dt.year
s['az'] = s['Hauptausrichtung'].map(AUS); s['tilt'] = s['HauptausrichtungNeigungswinkel'].map(NEI)
for lo, hi, lab in [(0, 30, '<30 kW'), (30, 750, '30-750 kW'), (750, 5000, '750 kW-5 MW'), (5000, 1e9, '>=5 MW')]:
    g = s[s['Nettonennleistung'].between(lo, hi, inclusive='left')]
    print(f'\n== {lab}: n={len(g)}, cap={g["Nettonennleistung"].sum()/1e6:.1f} GW, coords {g["Breitengrad"].notna().mean()*100:.0f}%, az filled {g["az"].notna().mean()*100:.0f}%, modules filled {g["AnzahlModule"].notna().mean()*100:.0f}%')
    print('  orientation:', (g['az'].value_counts(normalize=True)*100).round(1).to_dict())
    print('  tilt class :', (g['tilt'].value_counts(normalize=True)*100).round(1).to_dict())
big = s[s['Nettonennleistung'] >= 750].copy()
big['w_per_module'] = big['Nettonennleistung']*1000/big['AnzahlModule']
big = big[big['w_per_module'].between(100, 800)]
print('\nAC W per module (Nettonennleistung/AnzahlModule) by commissioning year, parks >=750 kW:')
print(big.groupby('year')['w_per_module'].median().loc[2008:2025].round(0).to_dict())
# PV synthetic sites: which stations have solar output
import glob, os
pv_ids = sorted({os.path.basename(f)[6:11] for f in glob.glob('/mnt/nvme2/synthetic/solar/solar_hourly_noage_nosoil/synth_*.csv')})
st = pd.read_csv('/home/viktor/Work/stations_master.csv', dtype={'station_id': str}); st = st[st['station_id'].isin(pv_ids)]
print('\nPV synthetic sites:', len(st))
bb = big.dropna(subset=['Breitengrad'])
slat = np.deg2rad(st['latitude'].values); slon = np.deg2rad(st['longitude'].values); sxy = np.c_[6371*np.cos(slat)*slon, 6371*slat]
plat = np.deg2rad(bb['Breitengrad'].values); plon = np.deg2rad(bb['Laengengrad'].values); pxy = np.c_[6371*np.cos(plat)*plon, 6371*plat]
d = np.sqrt(((pxy[:, None, :] - sxy[None, :, :])**2).sum(-1)).min(1)
for r in [5, 10, 15]:
    print(f'  parks >=750 kW with coords within {r} km of a PV site: {(d <= r).sum()}  (>=5 MW: {((d <= r) & (bb["Nettonennleistung"].values >= 5000)).sum()})')
print('  parks >=750 kW with coordinates at all:', len(bb), 'of', len(big))
