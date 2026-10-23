"""Stream two MaStR solar XML files; keep units >= 750 kW; report the fields the flat CSV lacks
(Bruttoleistung vs Nettonennleistung vs inverter power, secondary orientation, mounting type,
power limitation)."""
import xml.etree.ElementTree as ET, pandas as pd, glob
import xml.etree.ElementTree as ET
kat = {int(k.find('Id').text): k.find('Wert').text for k in ET.parse('data/Marktstammdatenregister_20250625_25/Katalogwerte.xml').getroot()}
rows = []
for f in sorted(glob.glob('data/Marktstammdatenregister_20250625_25/EinheitenSolar_*.xml'))[:3]:
    for ev, el in ET.iterparse(f, events=('end',)):
        if el.tag == 'EinheitSolar':
            d = {c.tag: c.text for c in el}
            try:
                if float(d.get('Bruttoleistung', 0)) >= 750:
                    rows.append(d)
            except ValueError:
                pass
            el.clear()
df = pd.DataFrame(rows)
df.to_csv('/home/viktor/Work/FL_Contribution/data/mastr/solar_big_sample_raw.csv', index=False)
print('units >=750 kW in 3 of 54 files:', len(df)); print('fields present:', sorted(df.columns))
for c in ['Bruttoleistung', 'Nettonennleistung', 'ZugeordneteWirkleistungWechselrichter', 'AnzahlModule']:
    df[c] = pd.to_numeric(df.get(c), errors='coerce')
df = df[df['EinheitBetriebsstatus'] == '35']
print('\nmounting (ArtDerSolaranlage):', df['ArtDerSolaranlage'].map(lambda v: kat.get(int(v), v) if pd.notna(v) else v).value_counts().to_dict())
ff = df[df['ArtDerSolaranlage'].map(lambda v: kat.get(int(v), '') if pd.notna(v) else '') == 'Freifläche']
print('ground-mounted >=750 kW:', len(ff))
r = (ff['Bruttoleistung'] / ff['Nettonennleistung']); print('DC/AC = Brutto/Netto: quantiles', r.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
r2 = (ff['Bruttoleistung'] / ff['ZugeordneteWirkleistungWechselrichter']); print('DC/AC = Brutto/inverter: quantiles', r2.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(), 'filled', round(ff['ZugeordneteWirkleistungWechselrichter'].notna().mean(), 2))
ff = ff.copy(); ff['year'] = pd.to_datetime(ff['Inbetriebnahmedatum'], errors='coerce').dt.year
ff['wp_mod'] = ff['Bruttoleistung'] * 1000 / ff['AnzahlModule']
print('DC Wp per module by year (median):', ff.groupby('year')['wp_mod'].median().loc[2008:2025].round(0).to_dict())
print('DC/AC by year (median):', ff.groupby('year').apply(lambda g: (g['Bruttoleistung']/g['Nettonennleistung']).median()).loc[2008:2025].round(2).to_dict())
uni = ff['EinheitlicheAusrichtungUndNeigungswinkel'].value_counts(dropna=False).to_dict(); print('uniform orientation flag:', uni)
print('secondary orientation filled:', round(ff['Nebenausrichtung'].notna().mean(), 3) if 'Nebenausrichtung' in ff else 'field absent')
if 'Nebenausrichtung' in ff:
    combo = ff.dropna(subset=['Nebenausrichtung']).apply(lambda x: f"{kat.get(int(x['Hauptausrichtung']),'?')}/{kat.get(int(x['HauptausrichtungNeigungswinkel']),'?')} + {kat.get(int(x['Nebenausrichtung']),'?')}/{kat.get(int(float(x['NebenausrichtungNeigungswinkel'])),'?') if pd.notna(x.get('NebenausrichtungNeigungswinkel')) else '?'}", axis=1)
    print('top main+secondary combos:', combo.value_counts().head(10).to_dict())
print('main orientation (ground-mounted):', ff['Hauptausrichtung'].map(lambda v: kat.get(int(v), v)).value_counts(normalize=True).round(3).to_dict())
print('tilt class (ground-mounted):', ff['HauptausrichtungNeigungswinkel'].map(lambda v: kat.get(int(v), v)).value_counts(normalize=True).round(3).to_dict())
print('power limitation:', ff['Leistungsbegrenzung'].map(lambda v: kat.get(int(v), v) if pd.notna(v) else v).value_counts(dropna=False).to_dict())
print('tracking (Nachgefuehrt):', df.get('Nachgefuehrt', pd.Series(dtype=object)).value_counts(dropna=False).head(5).to_dict() if 'Nachgefuehrt' in df else 'no field')
print('coordinates present (>=750 kW):', round(df['Laengengrad'].notna().mean(), 3) if 'Laengengrad' in df else 'no coordinate field in XML')
df.to_csv('/home/viktor/Work/FL_Contribution/data/mastr/solar_big_sample.csv', index=False)
