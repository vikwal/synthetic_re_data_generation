import yaml
import pandas as pd
import requests
import csv
import time
import os

from bs4 import BeautifulSoup

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
config = load_config(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml')))
turbine_power = config['data']['turbine_power']
turbine_specs = config['data']['turbine_specs']
turbine_cp = config['data']['turbine_cp']
turbine_ct = config['data']['turbine_ct']

turbines_df = pd.read_csv(turbine_power)
turbine_names_list = turbines_df.columns[1:].tolist()
turbines = pd.DataFrame(turbine_names_list, columns=["turbine_names"])

turbines["turbine_names"] = turbines["turbine_names"].str.replace(" ", "+")
turbines["turbine_names"] = turbines["turbine_names"].str.replace("(", "%28")
turbines["turbine_names"] = turbines["turbine_names"].str.replace(")", "%29")

turbines_list = turbines["turbine_names"].tolist()


def get_url(turbine_name):
    url = "https://www.wind-turbine-models.com/search?q="+turbine_name
    return url

def get_turbine_name_for_url_path(turbine_name):
    """Wandelt den Turbinennamen in URL-Pfad-konformes Format um (für Link-Ende) """
    url_path_name = turbine_name.lower()

    # *** VERBESSERTE ERSETZUNGSLOGIK ***
    url_path_name = url_path_name.replace(" ", "-")   # Leerzeichen zu Bindestrich
    url_path_name = url_path_name.replace("/", "-")   # Schrägstrich zu Bindestrich
    url_path_name = url_path_name.replace("+", "-")   # Pluszeichen zu Bindestrich
    url_path_name = url_path_name.replace("(", "")     # Klammern entfernen
    url_path_name = url_path_name.replace(")", "")     # Klammern entfernen
    url_path_name = url_path_name.replace("%2F", "-")   # URL-kodierten Schrägstrich zu Bindestrich (sicherheitshalber)
    url_path_name = url_path_name.replace("+", "-") # Doppelte Ersetzung von '+' (sicherheitshalber)


    # *** Entferne MEHRERE Bindestriche HINTEREINANDER (z.B. durch "+/") ***
    while "--" in url_path_name:
        url_path_name = url_path_name.replace("--", "-")

    # *** Entferne Bindestrich am ENDE, falls vorhanden (z.B. durch abschließendes "+") ***
    if url_path_name.endswith("-"):
        url_path_name = url_path_name[:-1] # Entferne das letzte Zeichen (Bindestrich)

    return url_path_name

def scrape_turbine_url(turbine_names):
    turbine_data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    original_turbine_names = [tn.replace("+", " ").replace("%2F", "/").replace("%28", "(").replace("%29", ")") for tn in turbine_names]

    for i, turbine_name in enumerate(turbine_names):
        url = get_url(turbine_name)
        print(f"Scrape URL (Suche): {url}")

        try:
            time.sleep(2)
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen der URL {url}: {e}")
            continue
        except requests.exceptions.Timeout as te:
            print(f"Timeout beim Abrufen der URL {url}: {te}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        link_tags = soup.find_all('a', class_='overview-link btn btn-default btn-block gd-link')
        turbine_link_validated = None
        turbine_link_fallback = None

        if link_tags:
            original_turbine_name = original_turbine_names[i]
            expected_url_path_name = get_turbine_name_for_url_path(original_turbine_name)
            print(f"DEBUG: Erwarteter URL-Pfad: {repr(expected_url_path_name)}")

            found_exact_match = False
            for link_tag in link_tags:
                turbine_link_url_raw = link_tag['href']
                print(f"DEBUG: Untersuche Link: {repr(turbine_link_url_raw)}")
                if turbine_link_url_raw.endswith(expected_url_path_name):
                    turbine_link_validated = turbine_link_url_raw
                    print(f"  EXAKTER Turbinen-Link gefunden UND URL-Pfad validiert: {turbine_link_validated}")
                    found_exact_match = True
                    break # Wir haben einen exakten Match gefunden, wir können aufhören zu suchen

            if not found_exact_match and link_tags:
                turbine_link_fallback = link_tags[0]['href'] # Verwende den ersten Link als Fallback, falls kein exakter Match
                print(f"  KEIN EXAKTER Turbinen-Link gefunden. Verwende stattdessen den ERSTEN gefundenen Link als Fallback: {turbine_link_fallback}")
            elif not link_tags:
                print(f"  Kein Turbinen-Link mit Klasse 'overview-link ...' gefunden.")

        else:
            print(f"  Kein Turbinen-Link mit Klasse 'overview-link ...' gefunden.")

        turbine_info = {
            'name': original_turbine_names[i],
            'url': url,
            'turbine_link_validated': turbine_link_validated,
            'turbine_link_fallback': turbine_link_fallback
        }
        turbine_data.append(turbine_info)
    return turbine_data

def scrape_turbine_details(turbine_data):
    detailed_turbine_data = []
    for turbine_item in turbine_data:
        # *** Priorisierte Link-Auswahl: Validiert > Fallback > Überspringen ***
        turbine_link_url = turbine_item.get('turbine_link_validated')
        if not turbine_link_url: # Wenn turbine_link_validated None ist ODER leer ist
            turbine_link_url = turbine_item.get('turbine_link_fallback')
            if not turbine_link_url: # Wenn auch turbine_link_fallback None ist ODER leer ist
                print(f"Kein validierter oder Fallback-Turbinen-Link für {turbine_item['name']} gefunden. Überspringe Detail-Scraping.")
                continue # Gehe zum nächsten Item über

        # *** Ab hier wird turbine_link_url garantiert einen Wert haben (validiert oder fallback) ***
        print(f"Scrape Turbine Detail URL: {turbine_link_url}")
        try:
            time.sleep(2)
            response = requests.get(turbine_link_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen der Detail-URL {turbine_link_url}: {e}")
            detailed_turbine_data.append(turbine_item)
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # *** Code zum Extrahieren der spezifischen Daten von der Detailseite (AKTUALISIERT - Wortbasierte Suche für Durchmesser UND Nabenhöhe) ***
        turbine_name_element = soup.find('h1', class_='page-header')

        # Wortbasierte Suche für Rotordurchmesser
        durchmesser_label_element = soup.find('div', string='Durchmesser:')
        rotor_diameter_element = None
        if durchmesser_label_element:
            rotor_diameter_element = durchmesser_label_element.find_next_sibling('div', class_='col-xs-6 row-col col-right')

        # Wortbasierte Suche für Nabenhöhe
        nabenhoehe_label_element = soup.find('div', string='Nabenhöhe:')
        hub_height_element = None
        hub_heights = [] # Liste für mehrere Nabenhöhen
        if nabenhoehe_label_element:
            hub_height_element = nabenhoehe_label_element.find_next_sibling('div', class_='col-xs-6 row-col col-right')
            if hub_height_element:
                hub_height_text = hub_height_element.text.strip()
                hub_heights = [h.strip() for h in hub_height_text.replace(" / ", "/").replace("/", "/").split('/')] # Split und Bereinigung verschiedener Separatoren

        turbine_name = turbine_name_element.text.strip() if turbine_name_element else "Name nicht gefunden"
        rotor_diameter_text = rotor_diameter_element.text.strip() if rotor_diameter_element else "Rotordurchmesser nicht gefunden"
        # hub_height wird jetzt unten pro Nabenhöhe behandelt

        print(f"  Turbinenname: {turbine_name}")
        print(f"  Rotordurchmesser (roh): {rotor_diameter_text}") # Rohdaten ausgeben
        print(f"  Nabenhöhe(n) (roh): {hub_heights}") # Rohdaten ausgeben

        # *** Datenbereinigung ***
        # Rotordurchmesser: Komma zu Punkt, " m" entfernen
        rotor_diameter_cleaned = rotor_diameter_text.replace(",", ".").replace(" m", "").replace("Rotordurchmesser nicht gefunden", "Rotordurchmesser nicht gefunden") if rotor_diameter_text != "Rotordurchmesser nicht gefunden" else "Rotordurchmesser nicht gefunden"

        # Nabenhöhe: " m" entfernen (für jede Höhe in der Liste)
        hub_heights_cleaned = [h.replace(" m", "").replace("*", "") for h in hub_heights]


        print(f"  Rotordurchmesser (bereinigt): {rotor_diameter_cleaned}") # Bereinigte Daten ausgeben
        print(f"  Nabenhöhe(n) (bereinigt): {hub_heights_cleaned}") # Bereinigte Daten ausgeben


        # Für jede Nabenhöhe eine Zeile in detailed_turbine_data erzeugen
        if hub_heights_cleaned:
            for hub_height in hub_heights_cleaned:
                detailed_turbine_data.append({
                    'turbine_name': turbine_name,
                    'rotor_diameter': rotor_diameter_cleaned, # Verwende bereinigten Wert
                    'hub_height': hub_height # Verwende bereinigte Werte
                })
        else: # Falls keine Nabenhöhe gefunden, trotzdem eine Zeile mit "Nabenhöhe nicht gefunden"
            detailed_turbine_data.append({
                'turbine_name': turbine_name,
                'rotor_diameter': rotor_diameter_cleaned, # Verwende bereinigten Wert
                'hub_height': "Nabenhöhe nicht gefunden"
            })


    return detailed_turbine_data


def scrape_cp_ct(turbine_data):
    """
    Scraped CP- und CT-Daten von den Detailseiten der Windkraftanlagen.
    Erstellt DataFrames mit Windgeschwindigkeiten als Spalten und Turbinennamen als Index.
    """
    turbine_names = [turbine['name'] for turbine in turbine_data]
    df_cp = pd.DataFrame(index=turbine_names)
    df_ct = pd.DataFrame(index=turbine_names)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                                    'Chrome/91.0.4472.124 Safari/537.36'}

    for turbine_item in turbine_data:
        turbine_name = turbine_item['name']

        # Priorisierte Link-Auswahl (validiert oder Fallback)
        turbine_link_url = turbine_item.get('turbine_link_validated') or turbine_item.get('turbine_link_fallback')
        if not turbine_link_url:
            print(f"Kein Link für {turbine_name}. CP/CT-Daten werden nicht gescraped.")
            continue

        print(f"Scrape CP/CT Daten von URL: {turbine_link_url}")
        try:
            time.sleep(2)
            response = requests.get(turbine_link_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Abrufen der Detail-URL {turbine_link_url} für CP/CT: {e}")
            continue
        except requests.exceptions.Timeout as te:
            print(f"Timeout beim Abrufen der Detail-URL {turbine_link_url} für CP/CT: {te}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Finde das <script>-Tag mit den Chart-Daten
        script_tag = soup.find('script', {'type': 'text/javascript'},
                                    string=lambda text: text and "tLeistungskurveChart" in text)
        if not script_tag:
            print(f"Keine Chart-Daten gefunden für {turbine_name}")
            continue

        script_content = script_tag.string

        # Extrahiere Windgeschwindigkeitsdaten (Labels)
        start_labels = script_content.find("labels: [")
        end_labels = script_content.find("]", start_labels)
        if start_labels != -1 and end_labels != -1:
            labels_str = script_content[start_labels + len("labels: ["):end_labels].strip()
            try:
                wind_speeds = [float(x.strip('"')) for x in labels_str.split(',')]
            except Exception as e:
                print(f"Fehler beim Konvertieren der Windgeschwindigkeitsdaten für {turbine_name}: {e}")
                wind_speeds = None
        else:
            print(f"Windgeschwindigkeitsdaten nicht gefunden für {turbine_name}")
            wind_speeds = None

        if wind_speeds:
            # Extrahiere CP-Daten
            start_cp = script_content.find('{"label":"cp"')
            cp_values = None
            if start_cp != -1:
                start_cp_data = script_content.find('"data":[', start_cp)
                end_cp_data = script_content.find(']', start_cp_data)
                if start_cp_data != -1 and end_cp_data != -1:
                    cp_values_str = script_content[start_cp_data + len('"data":['):end_cp_data].strip()
                    try:
                        cp_values = [float(x.strip().strip('"')) if x.strip().strip('"') != 'null' else None for x in cp_values_str.split(',')]
                    except Exception as e:
                        print(f"Fehler beim Konvertieren der CP-Daten für {turbine_name}: {e}")
                else:
                    print(f"CP-Daten nicht gefunden für {turbine_name}")
            else:
                print(f"CP-Abschnitt nicht gefunden für {turbine_name}")

            if cp_values and len(cp_values) == len(wind_speeds):
                # Füge die CP-Daten als Zeile zum DataFrame hinzu
                df_cp.loc[turbine_name, wind_speeds] = cp_values
            else:
                print(f"Anzahl CP-Werte/Windgeschwindigkeiten stimmt nicht für {turbine_name}")

            # Extrahiere CT-Daten
            start_ct = script_content.find('{"label":"ct"')
            ct_values = None
            if start_ct != -1:
                start_ct_data = script_content.find('"data":[', start_ct)
                end_ct_data = script_content.find(']', start_ct_data)
                if start_ct_data != -1 and end_ct_data != -1:
                    ct_values_str = script_content[start_ct_data + len('"data":['):end_ct_data].strip()
                    try:
                        ct_values = [float(x.strip().strip('"')) if x.strip().strip('"') != 'null' else None for x in ct_values_str.split(',')]
                    except Exception as e:
                        print(f"Fehler beim Konvertieren der CT-Daten für {turbine_name}: {e}")
                else:
                    print(f"CT-Daten nicht gefunden für {turbine_name}")
            else:
                print(f"CT-Abschnitt nicht gefunden für {turbine_name}")

            if ct_values and len(ct_values) == len(wind_speeds):
                # Füge die CT-Daten als Zeile zum DataFrame hinzu
                df_ct.loc[turbine_name, wind_speeds] = ct_values
            else:
                print(f"Anzahl CT-Werte/Windgeschwindigkeiten stimmt nicht für {turbine_name}")
        else:
            # Falls keine Windgeschwindigkeitsdaten gefunden wurden, wird nichts aktualisiert.
            continue

    return df_cp, df_ct


def main() -> None:
    turbine_urls_data = scrape_turbine_url(turbines_list)
    detailed_data = scrape_turbine_details(turbine_urls_data)
    detailed_cpct = scrape_cp_ct(turbine_urls_data)

    df_specs_data = pd.DataFrame(detailed_data)
    df_details_cp = detailed_cpct[0]
    df_details_ct = detailed_cpct[1]

    df_csv = df_specs_data[['turbine_name', 'rotor_diameter', 'hub_height']]
    df_csv.columns = ['Turbine', 'Rotordurchmesser', 'Nabenhöhe']

    df_details_cp_transposed = df_details_cp.T
    df_details_cp_transposed.index.name = 'wind_speed'
    df_details_cp_transposed.to_csv(turbine_cp, sep = ";")

    df_details_ct_transposed = df_details_ct.T
    df_details_ct_transposed.index.name = 'wind_speed'
    df_details_ct_transposed.to_csv(turbine_ct, sep = ";")

    df_specs_data.to_csv(turbine_specs, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, sep=',')


if __name__ == '__main__':
    main()