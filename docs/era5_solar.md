# ERA5-Solardaten: von der Reanalyse zu instantanen PV-Eingangsgrößen

Wie aus stündlichen ERA5-Strahlungsdaten physikalisch gültige Momentanwerte
werden, die `pvlib` für die Transposition auf eine geneigte Fläche verarbeiten
kann — und warum der naive Weg falsche Ergebnisse liefert.

Stand: 2026-08-12

---

## 1. Datenlage

| Quelle | Inhalt |
|---|---|
| `public.era5_solar` (DB `WeatherDB`) | Rohwerte, 1184 Gitterzellen × 27048 Stunden, 2023-07-01 → 2026-07-31 |
| `/mnt/lambda1/nvme1/synthetic/era5_solar_cache/` | 202 Stationen, bilinear interpoliert, umgerechnet |

Erzeugt von `NWP/ECMWF/write_db_era5_solar.py` (GRIB → DB) und
`NWP/ECMWF/extract_era5_solar_parquet.py` (DB → Parquet, hier passiert die
Umrechnung).

### Spaltennamen

Die DB führt die GRIB-Kurznamen, der Parquet-Cache sprechende Namen:

| DB | Parquet | Bedeutung | Typ |
|---|---|---|---|
| `ssrd` | `ghi` | Globalstrahlung horizontal | akkumuliert |
| `fdir` | `bhi` | Direktstrahlung horizontal | akkumuliert |
| `ssrdc` | `ghi_clear` | Globalstrahlung, klarer Himmel | akkumuliert |
| `cdir` | `bhi_clear` | Direktstrahlung, klarer Himmel | akkumuliert |
| `tisr` | `toa` | Einstrahlung Atmosphärenoberkante | akkumuliert |
| `strd` / `strdc` | `lw_down` / `lw_down_clear` | langwellig abwärts, gesamt / klar | akkumuliert |
| `fal` | `albedo` | Albedo | instantan |
| `tcc` `lcc` `mcc` `hcc` | `cloud_total` `_low` `_mid` `_high` | Bewölkung | instantan |
| `tcwv` | `water_vapour` | Wasserdampfsäule | instantan |

Diffusstrahlung ist nicht gespeichert, sondern eine Differenz: `DHI = ghi - bhi`.

---

## 2. Was ERA5 eigentlich liefert

**Ein ERA5-Strahlungswert ist kein Momentanwert.** Der Wert mit Zeitstempel `t`
ist die Energiesumme über das Intervall `[t-1h, t]` in J m⁻². Geteilt durch 3600
ergibt das die **mittlere Leistung während dieser Stunde**, nicht die Leistung
zum Zeitpunkt `t`:

```
W/m² = J/m² / 3600        # bereits im Parquet-Cache angewandt
```

Zwei Konsequenzen, die alles Weitere bestimmen:

1. Der Zeitstempel markiert das **Intervallende**.
2. Innerhalb der Stunde hat ERA5 **keine** Information. Jede sub-stündliche
   Struktur muss von außen kommen.

> **Abgrenzung ECMWF HRES:** dort wird ab Vorhersagebeginn akkumuliert, der
> Stundenwert ist `(F[N] - F[N-1]) / 3600`. Der GRIB-Header meldet dort
> irreführend `stepType = instant`. ERA5 und HRES brauchen also
> unterschiedlichen Code — siehe `NWP/ECMWF/`.

---

## 3. Fall A: Lücken in 10-Minuten-Messwerten füllen

Der einfache Fall. Die DWD-10-Minuten-Werte sind ebenfalls Intervallwerte mit
Zeitstempel am Intervallende, liegen also auf derselben Konvention:

```python
FAKTOR = 10000 / 600            # J/cm² je 10 min -> W/m²
mess = pd.to_numeric(raw['ghi'], errors='coerce') * FAKTOR
mess_h = mess.resample('1h', label='right', closed='right').mean()
# mess_h.loc[t] und era5.loc[t] beschreiben jetzt dasselbe Intervall
```

Empirisch verifiziert (Station 00183, 12219 Tagstunden): `label='right'` ergibt
R² = 0.863, `label='left'` nur 0.791. Die falsche Konvention kostet 7
Prozentpunkte.

Als Modell-Feature gehört hier der **Clear-Sky-Index aus ERA5s eigenen Feldern**:

```python
k_c = ghi / ghi_clear
```

Zähler und Nenner stammen aus demselben Strahlungsschema und demselben
Intervall — der Sonnenstand kürzt sich exakt heraus, keine Korrektur nötig.

---

## 4. Fall B: synthetische PV-Erzeugung mit pvlib

Hier braucht es echte Momentanwerte, denn `pvlib` rechnet mit dem Sonnenstand zu
einem Zeitpunkt. Zwei Wege.

### Weg A — Sonnenstand auf die Intervallmitte schieben

Der schnelle Weg, den PVGIS und renewables.ninja gehen: Bestrahlungsstärken
unverändert lassen, nur den Sonnenstand verschieben.

```python
solpos = loc.get_solarposition(df.index - pd.Timedelta('30min'))
```

Man behandelt das Stundenmittel als Momentanwert zur Intervallmitte.

**Grenze:** in der Auf- und Untergangsstunde ist `cos(zenit)` innerhalb der
Stunde stark nichtlinear, und `DNI = BHI / cos(zenit)` läuft weg. Für
Tagessummen unkritisch, für eine saubere Lastkurve am Morgen nicht.

### Weg B — erst herunterskalieren, dann ist alles instantan

**Das ist der hier gewählte Weg.** Kern der Idee: nicht die W/m² interpolieren,
sondern die **dimensionslosen, glatten Verhältnisse** — und die absolute Höhe
aus einer analytischen Clear-Sky-Kurve holen, die beliebig fein auflösbar ist.

```python
import pvlib, pandas as pd, numpy as np

loc  = pvlib.location.Location(lat, lon, tz='UTC', altitude=hoehe)
idx10 = pd.date_range(start, end, freq='10min', tz='UTC')

# 1. Analytische Clear-Sky-Kurve auf 10 min - echte Momentanwerte
cs10 = loc.get_clearsky(idx10, model='ineichen')

# 2. DIESELBE Kurve auf das ERA5-Raster mitteln, Intervall [t-1h, t]
cs_h = cs10['ghi'].resample('1h', label='right', closed='right').mean()

# 3. Clear-Sky-Index - Zähler und Nenner auf demselben Intervall
k_c = (ghi_era5 / cs_h.reindex(ghi_era5.index)).clip(0, 1.2)

# 4. Interpolieren - k_c ist glatt und hat keinen Tagesgang
k_c10 = k_c.reindex(k_c.index.union(idx10)).interpolate('time').reindex(idx10)

# 5. Zurück zu absoluten Momentanwerten
ghi10 = k_c10 * cs10['ghi']
```

Analog für die Aufteilung in direkt und diffus — auch hier das **Verhältnis**
interpolieren, nicht die Absolutwerte:

```python
f_dir   = (bhi_era5 / ghi_era5).clip(0, 1).fillna(0)
f_dir10 = f_dir.reindex(f_dir.index.union(idx10)).interpolate('time').reindex(idx10)

solpos = loc.get_solarposition(idx10)          # keine Verschiebung nötig
bhi10  = ghi10 * f_dir10
dhi10  = ghi10 - bhi10
dni10  = bhi10 / np.cos(np.radians(solpos['zenith'].clip(upper=87)))
```

Transposition auf die geneigte Fläche:

```python
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=30, surface_azimuth=180,          # Süd, 30° Neigung
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth'],
    dni=dni10, ghi=ghi10, dhi=dhi10,
    albedo=albedo10, model='haydavies')
```

### Warum der Umweg über Schritt 2 der Kern ist

Weil dieselbe analytische Kurve einmal **gemittelt im Nenner** (Schritt 2/3) und
einmal **instantan im Zähler** (Schritt 5) steht, kürzt sie sich im Stundenmittel
exakt heraus. Die Rekonstruktion ist damit **energieerhaltend**: mittelt man
`ghi10` wieder über die Stunde, kommt `ghi_era5` zurück.

Interpoliert man stattdessen die W/m² direkt, gehen beide Eigenschaften
verloren — die Energieerhaltung und die Gültigkeit der Momentanwerte. Man
erfindet zusätzlich einen Tagesgang, den ERA5 innerhalb der Stunde gar nicht
kennt.

### Warum hier NICHT ERA5s `ghi_clear` verwendet wird

Naheliegend wäre, `k_c = ghi / ghi_clear` aus ERA5 zu bilden und mit der
pvlib-Kurve zu multiplizieren. **Das ist falsch.** Beide Clear-Sky-Modelle
beruhen auf verschiedenen Aerosol-Annahmen — ERA5 auf einer eigenen
Aerosolklimatologie, Ineichen auf der Linke-Trübung. Der Quotient aus dem einen
und das Produkt mit dem anderen verschiebt das Niveau systematisch.

Für Weg B muss die Referenz **konsistent dieselbe** sein, also durchgehend
pvlib. ERA5s `ghi_clear` bleibt trotzdem wertvoll:

- als Feature für die Lückenfüllung (Fall A), wo es aus demselben Schema wie
  `ghi` stammt und deshalb genau richtig ist;
- als **Kontrolle** der Trübungsannahme: `ghi_clear` gegen `cs_h` auftragen.
  Systematische Abweichungen zeigen, dass die Linke-Trübung standortspezifisch
  angepasst werden sollte.

---

## 5. Fallstricke

**Zeitzone der Datenbank.** Die Session-Zeitzone ist `Europe/Berlin`, nicht UTC.
`TIMESTAMPTZ` kommt dann mit wechselndem Offset über die Sommerzeit zurück
(+01:00 / +02:00), woran `pd.DatetimeIndex` bei gemischten Offsets hart
scheitert. Die gespeicherten Instants sind korrekt, nur die Darstellung nicht:

```python
cur.execute("SET TIME ZONE 'UTC';")          # direkt nach dem Connect
times = pd.to_datetime([r[0] for r in rows], utc=True)
```

Gleiches gilt für jede Uhrzeit-Extraktion in SQL:
```sql
EXTRACT(HOUR FROM (valid_time AT TIME ZONE 'UTC'))   -- richtig
EXTRACT(HOUR FROM valid_time)                        -- Session-TZ, falsch
```

**Nachtwerte.** GRIB-Packung erzeugt winzige negative Strahlungswerte. Der
Extraktor klippt bei 0. Zeilen mit Zenit > 90° gehören in Trainingsdaten
ohnehin ausgeschlossen.

**Auflösung.** ERA5 liegt bei ~31 km. Sub-Grid-Bewölkung ist nicht auflösbar —
die Obergrenze für stündliches R² gegen Punktmessungen liegt bei etwa 0.86 ohne
und wenig darüber mit stationsspezifischer Korrektur.

**Randstationen.** Der Gitterausschnitt reicht von 47.26 N / 5.86 E bis
55.01 N / 14.86 E. Görlitz (14.95 E) und List auf Sylt (55.011 N) liegen knapp
außerhalb und werden auf den Rand geklemmt — 6 km bzw. 100 m bei 31 km
Gitterweite, also unkritisch.

---

## 6. Kurzfassung

| Zweck | Zeitbasis | Vorgehen |
|---|---|---|
| Lücken in 10-min-Messwerten | Intervallmittel | Messung mit `label='right'` aggregieren, `k_c = ghi/ghi_clear` als Feature |
| PV-Simulation, grob | Intervallmittel | Sonnenstand bei `t - 30min` |
| PV-Simulation, sauber | Momentanwerte | Weg B: `k_c` und `f_dir` interpolieren, absolute Höhe aus pvlib-Clear-Sky |

Nie die W/m² direkt interpolieren.
