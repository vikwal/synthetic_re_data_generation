# SPDX-FileCopyrightText: Florian Maurer, Christian Rieke (Basis-Skript), Ergänzungen von Gemini
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import time
import os
import csv

import json5  # parse js-dict to python
import numpy as np
import pandas as pd
import requests
import yaml # Für Konfiguration aus Script 2
from bs4 import BeautifulSoup  # parse html
# from sqlalchemy import create_engine # Auskommentiert wie im Original
from tqdm import tqdm  # fancy for loop

# --- Konfiguration (aus Script 2 übernommen/angepasst) ---
def load_config(config_path):
    # Standardpfad relativ zum Skript, falls nicht absolut angegeben
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(__file__)
        # Geht davon aus, dass config.yaml im selben Verzeichnis wie das Skript liegt
        # oder im übergeordneten Verzeichnis, falls das Skript in einem Unterordner ist.
        # Passe dies ggf. an deine Struktur an.
        potential_path = os.path.abspath(os.path.join(script_dir, config_path))
        if not os.path.exists(potential_path):
            potential_path = os.path.abspath(os.path.join(script_dir, '..', config_path))

        config_path = potential_path


    log.info(f"Lade Konfiguration von: {config_path}")
    try:
        with open(config_path, "r", encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        log.error(f"FEHLER: Konfigurationsdatei nicht gefunden unter {config_path}")
        # Fallback auf Standardpfade, falls config.yaml fehlt
        return {
            'output_dir': 'output_data', # Standard-Ausgabeverzeichnis
            'data': {
                'turbine_power': 'turbine_power.csv',
                'turbine_specs': 'turbine_specs.csv',
                'turbine_cp': 'turbine_cp.csv',
                'turbine_ct': 'turbine_ct.csv'
            }
        }
    except Exception as e:
        log.error(f"FEHLER beim Laden der Konfiguration: {e}")
        # Fallback
        return {
            'output_dir': 'output_data',
            'data': {
                'turbine_power': 'turbine_power.csv',
                'turbine_specs': 'turbine_specs.csv',
                'turbine_cp': 'turbine_cp.csv',
                'turbine_ct': 'turbine_ct.csv'
            }
        }

# --- Logging ---
log = logging.getLogger("windmodel_combined")
log.setLevel(logging.INFO)
# Verhindere doppelte Handler, falls das Skript mehrmals importiert/ausgeführt wird
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


# --- Metadaten (aus Script 1) ---
metadata_info = {
    "schema_name": "windmodel",
    "data_date": time.strftime("%Y-%m-%d"), # Aktuelles Datum verwenden
    "data_source": "https://www.wind-turbine-models.com/", # Hauptseite als Quelle
    "license": "https://www.wind-turbine-models.com/terms",
    "description": "Wind turbine performance curves, specifications, Cp and Ct values scraped from wind-turbine-models.com.",
}

# --- Teil 1: Power Curve Scraping (Modifiziert für 2 Abfragen) ---

def get_turbines_with_power_curve():
    """Holt die Liste der Turbinen-IDs von der Powercurve-Vergleichsseite."""
    url = "https://www.wind-turbine-models.com/powercurves"
    log.info(f"Rufe Turbinenliste ab von {url}")
    try:
        page = requests.get(url, timeout=20)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        select_list = soup.find(class_="chosen-select")
        if not select_list:
            log.error("Konnte die Turbinen-Auswahlliste (chosen-select) nicht finden.")
            return []

        wind_turbines_with_curve = []
        for option in select_list.find_all("option"):
            value = option.get("value")
            if value: # Nur hinzufügen wenn ein Wert vorhanden ist
                wind_turbines_with_curve.append(value)
        log.info(f"{len(wind_turbines_with_curve)} Turbinen mit Power Curves gefunden.")
        return wind_turbines_with_curve
    except requests.exceptions.RequestException as e:
        log.error(f"Fehler beim Abrufen der Turbinenliste: {e}")
        return []

def download_turbine_curve(turbine_id, start=0, stop=25):
    """Lädt die Power Curve für EINE Turbine und EINEN Bereich.
       Gibt DF, Name und Detail-URL zurück."""
    url_post = "https://www.wind-turbine-models.com/powercurves"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "_action": "compare",
        "turbines[]": turbine_id,
        "windrange[]": [start, stop],
    }
    log.debug(f"  POST an {url_post} für Turbine {turbine_id}, Range [{start}-{stop}]")

    try:
        resp = requests.post(url_post, headers=headers, data=data, timeout=30)
        resp.raise_for_status()
        json_response = resp.json()
        strings = json_response.get("result", "")

        # Extraktion des JSON-Teils (wie zuvor)
        begin = strings.find("data: {")
        if begin == -1:
             begin = strings.find("data:")
             if begin != -1: begin += len("data:")
             else:
                  log.warning(f"  Konnte 'data:' nicht im Response für Turbine {turbine_id}, Range [{start}-{stop}] finden.")
                  return None, None, None

        open_brackets = 0
        current_pos = -1
        # Finde die öffnende Klammer '{' nach 'data:'
        start_data_obj = strings.find('{', begin)
        if start_data_obj == -1:
            log.warning(f"  Konnte öffnende Klammer '{{' nach 'data:' nicht finden (Turbine {turbine_id}, Range [{start}-{stop}]).")
            return None, None, None

        current_pos = start_data_obj
        end_data_object = -1
        in_string = False
        escape = False

        # Zähle Klammern, um das Ende des Objekts sicher zu finden
        while current_pos < len(strings):
            char = strings[current_pos]
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '{' and not in_string:
                open_brackets += 1
            elif char == '}' and not in_string:
                open_brackets -= 1
                if open_brackets == 0:
                    end_data_object = current_pos
                    break
            escape = char == '\\' and not escape
            current_pos += 1

        if end_data_object == -1:
             log.warning(f"  Konnte schließende Klammer '}}' für das data-Objekt nicht finden (Turbine {turbine_id}, Range [{start}-{stop}]).")
             return None, None, None

        relevant_js_text = strings[start_data_obj : end_data_object + 1] # Extrahiere das Objekt selbst

        # Versuche, es als JSON5 zu parsen
        curve_as_dict = json5.loads(relevant_js_text) # Kein Umklammern mehr nötig

        data_section = curve_as_dict # Das geparste Objekt ist direkt der 'data'-Teil
        labels = data_section.get("labels", [])
        datasets = data_section.get("datasets", [])

        if not labels or not datasets:
            log.warning(f"  Unvollständige Daten (labels oder datasets fehlt) in JSON für Turbine {turbine_id}, Range [{start}-{stop}]")
            return None, None, None

        first_dataset = datasets[0]
        x = labels
        y = first_dataset.get("data", [])
        label = first_dataset.get("label", f"Unknown_{turbine_id}")
        detail_url = first_dataset.get("url", None)

        if len(x) != len(y):
             log.warning(f"  Längen von Windgeschw. ({len(x)}) und Leistung ({len(y)}) stimmen nicht überein für {label} ({turbine_id}), Range [{start}-{stop}].")
             return None, None, None

        # Erstelle DataFrame mit robustem Index-Handling
        try:
            # Versuche direkte Konvertierung zu Float für den Index
            num_index = pd.to_numeric(x)
            df = pd.DataFrame(np.asarray(y, dtype=float), index=num_index, columns=[label])
            df.index.name = "wind_speed"
        except ValueError:
            log.warning(f"  Windgeschw.-Index konnte nicht zu numerisch konvertiert werden für {label}, Range [{start}-{stop}]. Verwende String-Index vorerst.")
            df = pd.DataFrame(np.asarray(y, dtype=float), index=x, columns=[label])
            df.index.name = "wind_speed_str" # Temporärer Name

        # Interpolation (nur wenn Index numerisch ist)
        if pd.api.types.is_numeric_dtype(df.index):
            try:
                df_interpolated = df.interpolate(method="linear") # Linear ist oft stabiler
                df = df_interpolated.fillna(0)
            except Exception as e:
                log.error(f"  Interpolationsfehler für {label}, Range [{start}-{stop}]: {e}. Verwende Originaldaten.")
                df = df.fillna(0)
        else:
             df = df.fillna(0) # Fülle NaNs ohne Interpolation bei String-Index

        df[df < 0] = 0 # Negative Werte auf 0 setzen

        log.debug(f"  Erfolgreich Daten für Range [{start}-{stop}] von {label} geladen.")
        return df, label, detail_url

    except requests.exceptions.RequestException as e:
        log.error(f"  Netzwerkfehler beim Download für Turbine {turbine_id}, Range [{start}-{stop}]: {e}")
        return None, None, None
    except json5.JSONDecodeError as e:
        log.error(f"  JSON(5)-Parsing-Fehler für Turbine {turbine_id}, Range [{start}-{stop}]: {e}")
        # log.debug(f"   Fehlerhafter JSON-String-Ausschnitt: {relevant_js_text[:500]}...")
        return None, None, None
    except Exception as e:
        log.error(f"  Unerwarteter Fehler in download_turbine_curve für {turbine_id}, Range [{start}-{stop}]: {e}")
        return None, None, None

# *** NEUE VERSION von download_all_power_curves ***
def download_all_power_curves():
    """Lädt alle Power Curves durch Abfrage von zwei Bereichen (0-21 und 22-25)
       und sammelt Turbinendetails (Name, ID, URL)."""
    wind_turbine_ids = get_turbines_with_power_curve()
    if not wind_turbine_ids:
        return pd.DataFrame(), []

    curves = []
    turbine_details_list = []
    log.info("Starte Download der Power Curves (in 2 Bereichen) für alle Turbinen...")

    for turbine_id in tqdm(wind_turbine_ids, desc="Downloading Power Curves"):
        log.debug(f"Verarbeite Turbine ID: {turbine_id}")
        final_turbine_df = None
        turbine_name = None
        detail_url = None
        df_list_for_concat = [] # Liste für die DFs dieser Turbine

        # --- Call 1: Low/Mid Range (0-25, Server liefert vermutlich nur bis ~21) ---
        log.debug(f"  Abfrage Bereich 1 (0-25) für {turbine_id}")
        curve_df_low, name_low, url_low = download_turbine_curve(turbine_id, start=0, stop=25)
        time.sleep(0.2) # Kurze Pause

        if curve_df_low is not None:
            # Name und URL aus der ersten (hoffentlich) erfolgreichen Abfrage speichern
            turbine_name = name_low
            detail_url = url_low
            # Daten < 22 m/s behalten
            # Sicherstellen, dass der Index numerisch ist für den Vergleich
            if pd.api.types.is_numeric_dtype(curve_df_low.index):
                 curve_df_low_filtered = curve_df_low[curve_df_low.index < 22]
                 if not curve_df_low_filtered.empty:
                     df_list_for_concat.append(curve_df_low_filtered)
                     log.debug(f"  Daten < 22 m/s aus Bereich 1 für {turbine_name} hinzugefügt.")
                 else:
                      log.debug(f"  Keine Daten < 22 m/s in Bereich 1 für {turbine_name} gefunden.")
            else:
                 log.warning(f"  Index für Bereich 1 von {turbine_name} nicht numerisch. Überspringe Filterung < 22.")
                 df_list_for_concat.append(curve_df_low) # Füge ungefiltert hinzu


        # --- Call 2: High Range (22-25) ---
        log.debug(f"  Abfrage Bereich 2 (22-25) für {turbine_id}")
        curve_df_high, name_high, url_high = download_turbine_curve(turbine_id, start=22, stop=25)
        time.sleep(0.2) # Kurze Pause

        if curve_df_high is not None:
             # Falls Name/URL im ersten Call fehlten, versuche sie hier zu bekommen
            if turbine_name is None: turbine_name = name_high
            if detail_url is None: detail_url = url_high
            df_list_for_concat.append(curve_df_high)
            log.debug(f"  Daten aus Bereich 2 (>=22 m/s) für {turbine_id} hinzugefügt.")


        # --- Kombiniere Ergebnisse für DIESE Turbine ---
        if not df_list_for_concat:
            log.warning(f"  Keine Power Curve Daten für Turbine {turbine_id} (Name: {turbine_name}) in beiden Bereichen gefunden.")
            continue # Nächste Turbine

        try:
            # Kombiniere die DataFrames aus der Liste
            final_turbine_df = pd.concat(df_list_for_concat, axis=0) # Axis 0, da wir Zeilen (Index) anfügen

            # Stelle sicher, dass der Index numerisch ist nach dem Concat
            if not pd.api.types.is_numeric_dtype(final_turbine_df.index):
                 final_turbine_df.index = pd.to_numeric(final_turbine_df.index)

            # Sortiere nach Index (Windgeschwindigkeit)
            final_turbine_df = final_turbine_df.sort_index()

            # Entferne Duplikate im Index (falls 21.x und 22.0 überlappen o.ä.)
            # Behalte den ersten Eintrag (aus dem niedrigeren Bereich, falls relevant)
            final_turbine_df = final_turbine_df[~final_turbine_df.index.duplicated(keep='first')]

            log.debug(f"  Kombinierte Daten für {turbine_name} (ID: {turbine_id}) erstellt.")

        except ValueError as e:
             log.error(f"  Fehler beim Konvertieren des Index zu numerisch für Turbine {turbine_id} nach Concat: {e}")
             continue # Überspringe diese Turbine
        except Exception as e:
             log.error(f"  Fehler beim Zusammenfügen der Daten für Turbine {turbine_id}: {e}")
             continue # Überspringe diese Turbine


        # --- Ergebnisse der Turbine speichern ---
        if final_turbine_df is not None and not final_turbine_df.empty:
            # Benenne die Spalte korrekt (könnte durch concat verloren gehen)
            final_turbine_df.columns = [turbine_name]

            curves.append(final_turbine_df)
            # Füge Details nur hinzu, wenn Name und URL vorhanden sind
            if turbine_name and detail_url:
                turbine_details_list.append({
                    'id': turbine_id,
                    'name': turbine_name,
                    'detail_url': detail_url
                })
            else:
                 log.warning(f"  Kein Name oder Detail-URL für Turbine {turbine_id} verfügbar, obwohl Daten gefunden wurden.")
        # Keine extra Pause hier, die kleinen Pausen nach jeder Anfrage reichen hoffentlich

    # --- Alle Turbinen verarbeitet, jetzt Gesamt-DF bauen ---
    log.info("Füge Power Curve DataFrames aller Turbinen zusammen...")
    if not curves:
        log.warning("Keine Power Curves konnten erfolgreich heruntergeladen und kombiniert werden.")
        return pd.DataFrame(), []

    # pd.concat mit axis=1 und outer join, um alle Windgeschwindigkeiten zu behalten
    # Fehlende Werte werden zu NaN
    combined_df = pd.concat(curves, axis=1, join='outer').sort_index()

    # Index ggf. zu Float machen (sollte durch obige Schritte schon numerisch sein)
    try:
        if not pd.api.types.is_numeric_dtype(combined_df.index):
             combined_df.index = combined_df.index.astype(float)
        # Stelle sicher, dass der Index-Name korrekt ist
        combined_df.index.name = "wind_speed"
    except ValueError:
        log.warning("Konnte finalen Power Curve Index nicht durchgehend in Float konvertieren.")

    log.info("Bereinige kombinierten Power Curve DataFrame...")
    # Entferne Zeilen, die NUR aus NaN bestehen (wichtig bei 'outer' join)
    all_turbines_trunc = combined_df.dropna(axis=0, how='all')
    # Fülle verbleibende NaNs mit 0 (nachdem sicher ist, dass die Zeile nicht komplett leer war)
    df_final = all_turbines_trunc.fillna(0)
    # Stelle sicher, dass keine negativen Werte vorhanden sind
    df_final[df_final < 0] = 0

    log.info(f"Power Curve Download abgeschlossen. {len(df_final)} Datenpunkte, {len(df_final.columns)} Turbinen.")
    log.info(f"{len(turbine_details_list)} Turbinen mit Detail-URLs für weitere Verarbeitung gefunden.")

    return df_final, turbine_details_list


# --- Teil 2: Specs, Cp, Ct Scraping (aus Script 2, angepasst - KEINE ÄNDERUNGEN HIER) ---
# Die Funktionen scrape_turbine_details und scrape_cp_ct bleiben unverändert zum vorherigen kombinierten Skript

def scrape_turbine_details(turbine_details_list):
    """Scraped Rotordurchmesser und Nabenhöhe von den Detailseiten."""
    detailed_turbine_data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    log.info("Starte Scraping der Turbinen-Spezifikationen (Durchmesser, Nabenhöhe)...")
    if not turbine_details_list:
        log.warning("Keine Turbinendetails zum Scrapen der Spezifikationen vorhanden.")
        return []

    for turbine_item in tqdm(turbine_details_list, desc="Scraping Specs"):
        turbine_name = turbine_item.get('name', 'Unbekannt') # Sicherer Zugriff
        detail_url = turbine_item.get('detail_url')

        if not detail_url:
            log.warning(f"Überspringe Detail-Scraping für {turbine_name} (ID: {turbine_item.get('id', 'N/A')}), da keine URL vorhanden.")
            continue

        log.debug(f"Scrape Specs von: {detail_url}")
        try:
            time.sleep(0.5 + np.random.rand() * 0.5) # Etwas längere, variablere Pause
            response = requests.get(detail_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extraktion (wie in Script 2, aber mit verbesserter Fehlertoleranz)
            h1_header = soup.find('h1', class_='page-header')
            scraped_name = h1_header.text.strip() if h1_header else turbine_name # Fallback auf Name aus Liste

            # Wortbasierte Suche für Rotordurchmesser
            rotor_diameter_text = "Nicht gefunden"
            durchmesser_label = soup.find(lambda tag: tag.name == 'div' and 'Durchmesser:' in tag.get_text(strip=True) and 'col-label' in tag.get('class', []))
            if durchmesser_label:
                rotor_diameter_div = durchmesser_label.find_next_sibling('div', class_=lambda x: x and 'col-right' in x)
                if rotor_diameter_div:
                    rotor_diameter_text = rotor_diameter_div.text.strip()

            # Wortbasierte Suche für Nabenhöhe
            hub_heights_text_list = []
            nabenhoehe_label = soup.find(lambda tag: tag.name == 'div' and 'Nabenhöhe:' in tag.get_text(strip=True) and 'col-label' in tag.get('class', []))
            if nabenhoehe_label:
                hub_height_div = nabenhoehe_label.find_next_sibling('div', class_=lambda x: x and 'col-right' in x)
                if hub_height_div:
                    hub_height_raw_text = hub_height_div.text.strip()
                    hub_heights_text_list = [
                        h.strip().replace(" m", "").replace("*", "")
                        for h in hub_height_raw_text.replace(" / ", "/").replace(" , ", "/").replace(",", "/").split('/') if h.strip() # Mehr Separatoren
                    ]

            # Bereinigung Rotordurchmesser
            rotor_diameter_cleaned = "Nicht gefunden"
            if rotor_diameter_text != "Nicht gefunden":
                 rotor_diameter_cleaned = rotor_diameter_text.replace(",", ".").replace(" m", "").strip()

            log.debug(f"  {scraped_name}: Durchmesser='{rotor_diameter_cleaned}', Nabenhöhe(n)='{hub_heights_text_list}'")

            # Erzeuge Einträge für jede Nabenhöhe
            if hub_heights_text_list:
                for hub_height in hub_heights_text_list:
                    # Versuche, Höhe zu Float zu konvertieren, sonst behalte String
                    try: hub_height_val = float(hub_height)
                    except ValueError: hub_height_val = hub_height

                    detailed_turbine_data.append({
                        'turbine_name': scraped_name,
                        'rotor_diameter': rotor_diameter_cleaned,
                        'hub_height': hub_height_val
                    })
            else: # Fallback, wenn keine Nabenhöhe gefunden wurde
                detailed_turbine_data.append({
                    'turbine_name': scraped_name,
                    'rotor_diameter': rotor_diameter_cleaned,
                    'hub_height': "Nicht gefunden"
                })

        except requests.exceptions.RequestException as e:
            log.error(f"Fehler beim Abrufen der Detail-URL {detail_url} für Specs: {e}")
            detailed_turbine_data.append({'turbine_name': turbine_name, 'rotor_diameter': f"Fehler: {e}", 'hub_height': f"Fehler: {e}"})
        except Exception as e:
             log.error(f"Unerwarteter Fehler beim Verarbeiten der Specs von {detail_url}: {e}")
             detailed_turbine_data.append({'turbine_name': turbine_name, 'rotor_diameter': f"Verarbeitungsfehler: {e}", 'hub_height': f"Verarbeitungsfehler: {e}"})

    log.info(f"Spezifikations-Scraping abgeschlossen. {len(detailed_turbine_data)} Einträge erstellt.")
    return detailed_turbine_data

def scrape_cp_ct(turbine_details_list):
    """Scraped CP- und CT-Daten von den Detailseiten."""
    # Verwende Turbinennamen aus der Liste als Index (nur die mit URL)
    turbine_names_index = [item['name'] for item in turbine_details_list if item.get('detail_url') and item.get('name')]
    if not turbine_names_index:
         log.warning("Keine Turbinennamen mit URLs für Cp/Ct Scraping gefunden.")
         return pd.DataFrame(), pd.DataFrame()

    # Entferne Duplikate, falls derselbe Name mehrmals vorkommt
    turbine_names_index = sorted(list(set(turbine_names_index)))

    df_cp = pd.DataFrame(index=turbine_names_index)
    df_ct = pd.DataFrame(index=turbine_names_index)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    log.info("Starte Scraping der Cp/Ct-Werte...")
    processed_urls = set() # Um doppelte Verarbeitung zu vermeiden, falls URLs mehrfach vorkommen

    for turbine_item in tqdm(turbine_details_list, desc="Scraping Cp/Ct"):
        turbine_name = turbine_item.get('name')
        detail_url = turbine_item.get('detail_url')

        if not detail_url or not turbine_name:
            continue # Überspringe Einträge ohne URL oder Name

        if detail_url in processed_urls:
             continue # Überspringe bereits verarbeitete URL
        processed_urls.add(detail_url)

        log.debug(f"Scrape Cp/Ct von URL: {detail_url} für Turbine: {turbine_name}")
        try:
            time.sleep(0.5 + np.random.rand() * 0.5) # Pause
            response = requests.get(detail_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Finde das Script-Tag (wie in Script 2)
            script_tag = soup.find('script', {'type': 'text/javascript'},
                                   string=lambda text: text and "tLeistungskurveChart" in text)
            if not script_tag or not script_tag.string:
                log.warning(f"Kein passendes Script-Tag für Cp/Ct gefunden auf {detail_url} ({turbine_name})")
                continue

            script_content = script_tag.string
            wind_speeds = None
            cp_values = None
            ct_values = None

            # Extrahiere Windgeschwindigkeiten (Labels)
            start_labels = script_content.find("labels: [")
            if start_labels != -1:
                end_labels = script_content.find("]", start_labels)
                if end_labels != -1:
                    labels_str = script_content[start_labels + len("labels: ["):end_labels].strip()
                    try:
                        wind_speeds = [float(x.strip().strip('"')) for x in labels_str.split(',') if x.strip()]
                    except ValueError as e:
                        log.warning(f"Fehler beim Konvertieren der Windgeschwindigkeiten für {turbine_name} von {detail_url}: {e}")
                        wind_speeds = None

            if not wind_speeds:
                 log.warning(f"Keine validen Windgeschwindigkeiten für Cp/Ct gefunden für {turbine_name} auf {detail_url}")
                 continue

            # Extrahiere CP-Daten
            start_cp_section = script_content.find('{"label":"cp"')
            if start_cp_section != -1:
                start_cp_data = script_content.find('"data":[', start_cp_section)
                if start_cp_data != -1:
                    end_cp_data = script_content.find(']', start_cp_data)
                    if end_cp_data != -1:
                        cp_values_str = script_content[start_cp_data + len('"data":['):end_cp_data].strip()
                        try:
                            cp_values = [float(x.strip().strip('"')) if x.strip().lower() != 'null' else np.nan for x in cp_values_str.split(',') if x.strip()] # Verwende np.nan
                            if len(cp_values) != len(wind_speeds):
                                log.warning(f"Längen-Mismatch CP ({len(cp_values)}) vs Wind ({len(wind_speeds)}) für {turbine_name}. Setze CP auf None.")
                                cp_values = None
                        except ValueError as e:
                            log.warning(f"Fehler beim Konvertieren der CP-Werte für {turbine_name}: {e}")
                            cp_values = None

            # Extrahiere CT-Daten
            start_ct_section = script_content.find('{"label":"ct"')
            if start_ct_section != -1:
                start_ct_data = script_content.find('"data":[', start_ct_section)
                if start_ct_data != -1:
                    end_ct_data = script_content.find(']', start_ct_data)
                    if end_ct_data != -1:
                        ct_values_str = script_content[start_ct_data + len('"data":['):end_ct_data].strip()
                        try:
                            ct_values = [float(x.strip().strip('"')) if x.strip().lower() != 'null' else np.nan for x in ct_values_str.split(',') if x.strip()] # Verwende np.nan
                            if len(ct_values) != len(wind_speeds):
                                log.warning(f"Längen-Mismatch CT ({len(ct_values)}) vs Wind ({len(wind_speeds)}) für {turbine_name}. Setze CT auf None.")
                                ct_values = None
                        except ValueError as e:
                            log.warning(f"Fehler beim Konvertieren der CT-Werte für {turbine_name}: {e}")
                            ct_values = None

            # Füge Daten zu DataFrames hinzu, wenn vorhanden
            # Verwende .loc zum Einfügen, was Spalten bei Bedarf erstellt
            if cp_values:
                for ws, cp_val in zip(wind_speeds, cp_values):
                    if pd.notna(cp_val): # Nur nicht-NaN Werte eintragen
                        df_cp.loc[turbine_name, float(ws)] = cp_val

            if ct_values:
                 for ws, ct_val in zip(wind_speeds, ct_values):
                    if pd.notna(ct_val):
                        df_ct.loc[turbine_name, float(ws)] = ct_val

        except requests.exceptions.RequestException as e:
            log.error(f"Fehler beim Abrufen der Detail-URL {detail_url} für Cp/Ct: {e}")
            # Mache nichts, die Zeile bleibt leer/NaN
        except Exception as e:
            log.error(f"Unerwarteter Fehler beim Verarbeiten von Cp/Ct von {detail_url} ({turbine_name}): {e}")
            # Mache nichts

    # Sortiere Spalten (Windgeschwindigkeiten) numerisch
    df_cp = df_cp.reindex(sorted(df_cp.columns), axis=1)
    df_ct = df_ct.reindex(sorted(df_ct.columns), axis=1)

    log.info("Cp/Ct-Scraping abgeschlossen.")
    return df_cp, df_ct

# --- Hauptfunktion (Kombiniert - KEINE ÄNDERUNGEN HIER) ---

def main():
    """Hauptablauf: Power Curves laden, dann Specs, Cp, Ct scrapen."""
    # Lade Konfiguration für Dateipfade
    config_file_path = 'config.yaml' # Passe den Pfad an
    config = load_config(config_file_path)

    output_dir = config.get('output_dir', 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    power_curve_path = os.path.join(output_dir, config['data'].get('turbine_power', 'turbine_power.csv'))
    specs_path = os.path.join(output_dir, config['data'].get('turbine_specs', 'turbine_specs.csv'))
    cp_path = os.path.join(output_dir, config['data'].get('turbine_cp', 'turbine_cp.csv'))
    ct_path = os.path.join(output_dir, config['data'].get('turbine_ct', 'turbine_ct.csv'))

    log.info("Starte kombinierten Scraping-Prozess...")

    # 1. Power Curves und Detail-URLs herunterladen (jetzt mit 2 Bereichen pro Turbine)
    power_curve_df, turbine_details_list = download_all_power_curves()

    if power_curve_df.empty:
        log.error("Keine Power Curve Daten heruntergeladen. Breche ab.")
        return

    # 2. Power Curves speichern
    log.info(f"Speichere Power Curves nach: {power_curve_path}")
    try:
        # Verwende ';' als Trenner und ',' als Dezimalzeichen für deutschsprachige Excel-Versionen
        power_curve_df.to_csv(power_curve_path, sep=';', decimal=',', encoding='utf-8-sig')
    except Exception as e:
        log.error(f"Fehler beim Speichern der Power Curves: {e}")

    if not turbine_details_list:
        log.warning("Keine Turbinen mit Detail-URLs gefunden. Specs, Cp, Ct können nicht gescraped werden.")
    else:
        # 3. Turbine Specifications (Specs) scrapen
        specs_data_list = scrape_turbine_details(turbine_details_list)

        if not specs_data_list:
             log.warning("Keine Spezifikationsdaten konnten gescraped werden.")
        else:
            # 4. Specs speichern
            df_specs = pd.DataFrame(specs_data_list)
            df_specs.rename(columns={'turbine_name': 'Turbine',
                                     'rotor_diameter': 'Rotordurchmesser',
                                     'hub_height': 'Nabenhöhe'}, inplace=True)
            log.info(f"Speichere Spezifikationen nach: {specs_path}")
            try:
                # Standard CSV-Format (Komma als Trenner)
                df_specs.to_csv(specs_path, index=False, sep=',', encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
            except Exception as e:
                log.error(f"Fehler beim Speichern der Spezifikationen: {e}")

        # 5. Cp und Ct Werte scrapen
        df_cp, df_ct = scrape_cp_ct(turbine_details_list)

        # 6. Cp Werte speichern (transponiert, Windgeschwindigkeit als Index)
        if not df_cp.empty:
            df_cp_transposed = df_cp.T
            df_cp_transposed.index.name = 'wind_speed'
            log.info(f"Speichere Cp-Werte nach: {cp_path}")
            try:
                 # Verwende ';' als Trenner und ',' als Dezimalzeichen
                 df_cp_transposed.to_csv(cp_path, sep=';', decimal=',', encoding='utf-8-sig')
            except Exception as e:
                log.error(f"Fehler beim Speichern der Cp-Werte: {e}")
        else:
            log.warning("Keine Cp-Daten zum Speichern vorhanden.")

        # 7. Ct Werte speichern (transponiert, Windgeschwindigkeit als Index)
        if not df_ct.empty:
            df_ct_transposed = df_ct.T
            df_ct_transposed.index.name = 'wind_speed'
            log.info(f"Speichere Ct-Werte nach: {ct_path}")
            try:
                # Verwende ';' als Trenner und ',' als Dezimalzeichen
                df_ct_transposed.to_csv(ct_path, sep=';', decimal=',', encoding='utf-8-sig')
            except Exception as e:
                 log.error(f"Fehler beim Speichern der Ct-Werte: {e}")
        else:
            log.warning("Keine Ct-Daten zum Speichern vorhanden.")

    log.info("Kombinierter Scraping-Prozess abgeschlossen.")


# --- Script-Ausführung ---
if __name__ == "__main__":
    main()