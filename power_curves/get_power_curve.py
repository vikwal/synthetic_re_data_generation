import yaml
import os
import logging
import time
import json5
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict
import re

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("windmodel")

# --- Load Config ---
def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config.yaml')
config = load_config(config_path)

# --- Paths ---
PATHS = {
    'power': config['data']['turbine_power'],
    'specs': config['data']['turbine_specs'],
    'cp': config['data']['turbine_cp'],
    'ct': config['data']['turbine_ct'],
    'names': config['data']['turbine_names']
}

# --- Constants ---
DATA_SOURCE = "https://www.wind-turbine-models.com/powercurves"
POWER_CURVE_ENDPOINT = "https://www.wind-turbine-models.com/turbines"
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/91.0.4472.124 Safari/537.36'
    )
}

# --- Utilities ---
def get_turbines_with_ids_and_names():
    try:
        resp = requests.get(DATA_SOURCE, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        select = soup.find(class_='chosen-select')
        return [{'id': opt['value'], 'name': opt.text.strip()} \
                for opt in select.find_all('option') if opt.get('value')]
    except Exception as e:
        log.error(f"Error fetching turbines list: {e}")
        return []

def handle_duplicates(items):
    counts = defaultdict(int)
    out = []
    for t in items:
        counts[t['name']] += 1
        name = t['name']
        unique = f"{name}_{counts[name]}" if counts[name] > 1 else name
        out.append({'id': t['id'], 'name': name, 'unique': unique})
    df = pd.DataFrame([{'original': i['name'], 'unique': i['unique']} for i in out])
    df.to_csv(PATHS['names'], index=False)
    return out

replacements = {" ": "-", "/": "-", "(": "", ")": "", "+": "-", "%28": "", "%29": "", ",": ""}
def to_url_path(name):
    s = name.lower()
    for old, new in replacements.items():
        s = s.replace(old, new)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip('-')

# --- Interpolation ---
def interpolate_series(series, order=3, resolution=0.01):
    if series.empty:
        return series
    nz = series[series > 0]
    if nz.empty:
        return pd.Series(0, index=np.arange(series.index.min(), series.index.max() + resolution, resolution))
    first = nz.index.min()
    last = nz.index.max()
    new_idx = np.arange(series.index.min(), series.index.max() + resolution, resolution)
    new_idx = np.round(new_idx, 2)
    s2 = series.reindex(new_idx).interpolate(method='polynomial', order=order, limit_direction='both')
    s2.loc[s2.index < first] = 0
    s2.loc[s2.index > last] = 0
    return s2.fillna(0).round(2)

# --- Chart JSON Extraction ---
def extract_chart_json(js_text: str) -> dict:
    # find start of data object
    m = re.search(r'data\s*:\s*\{', js_text)
    if not m:
        raise ValueError("Could not locate 'data:' in script")
    start = m.start(0) + js_text[m.start(0):].find('{')
    brace_count = 0
    for idx in range(start, len(js_text)):
        if js_text[idx] == '{': brace_count += 1
        elif js_text[idx] == '}': brace_count -= 1
        if brace_count == 0:
            end = idx
            break
    obj_text = js_text[start:end+1]
    # wrap as JSON with a key
    json_text = '{"data": ' + obj_text + '}'
    return json5.loads(json_text)

# --- Scrapers ---
def scrape_specs(soup):
    spec = {'rotor_diameter': None, 'hub_heights': [], 'cut_in': None, 'cut_out': None}
    def text_after(label):
        el = soup.find('div', string=label)
        if el:
            nxt = el.find_next_sibling('div')
            return nxt.text.strip() if nxt else None
        return None

    rd = text_after('Durchmesser:')
    if rd:
        try: spec['rotor_diameter'] = float(rd.replace(' m','').replace(',', '.'))
        except: spec['rotor_diameter'] = rd
    hh = text_after('Nabenhöhe:')
    if hh:
        parts = hh.replace(' / ','/').split('/')
        for p in parts:
            val = p.strip().replace(' m','')
            try: spec['hub_heights'].append(float(val.replace(',','.')))
            except: spec['hub_heights'].append(val)
    ci = text_after('Einschaltgeschwindigkeit:')
    if ci:
        try: spec['cut_in'] = float(ci.replace(' m/s','').replace(',', '.'))
        except: spec['cut_in'] = ci
    co = text_after('Abschaltgeschwindigkeit:')
    if co:
        try: spec['cut_out'] = float(co.replace(' m/s','').replace(',', '.'))
        except: spec['cut_out'] = co
    return spec

# --- Main Processing ---
def scrape_all():
    turbines = handle_duplicates(get_turbines_with_ids_and_names())
    power_df, cp_df, ct_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    specs_list = []

    for t in tqdm(turbines, desc='Turbines'):
        uid = t['unique']
        url = f"{POWER_CURVE_ENDPOINT}/{t['id']}-{to_url_path(t['name'])}"
        try:
            time.sleep(1)
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            sp = scrape_specs(soup)
            for hh in sp['hub_heights'] or [None]:
                specs_list.append({'Turbine': uid, 'Rotordurchmesser': sp['rotor_diameter'],
                                   'Nabenhöhe': hh, 'Einschaltgeschwindigkeit': sp['cut_in'],
                                   'Abschaltgeschwindigkeit': sp['cut_out']})
            tag = soup.find('script', string=lambda s: s and 'tLeistungskurveChart' in s)
            if not tag: continue
            chart = extract_chart_json(tag.string)
            speeds = [float(str(x).replace(',','.')) for x in chart['data']['labels']]
            for ds in chart['data']['datasets']:
                label = ds.get('label','').lower()
                vals = []
                for v in ds.get('data',[]):
                    if v in (None,'null'): vals.append(np.nan)
                    else:
                        try: vals.append(float(str(v).replace(',','.')))
                        except: vals.append(v)
                ser = pd.Series(vals, index=speeds, name=uid)
                if 'leistung' in label:
                    power_df = pd.concat([power_df, interpolate_series(ser)], axis=1)
                elif label == 'cp':
                    cp_df = pd.concat([cp_df, ser], axis=1)
                elif label == 'ct':
                    ct_df = pd.concat([ct_df, ser], axis=1)
        except Exception as e:
            log.error(f"Error for {uid}: {e}")
            continue

    cols = power_df.columns
    return {'power': power_df.sort_index(), 'cp': cp_df.reindex(columns=cols),
            'ct': ct_df.reindex(columns=cols), 'specs': pd.DataFrame(specs_list)}

# --- Save Functions ---
def save(df, path):
    df.to_csv(path, sep=';', index=True, index_label='wind_speed')

def main():
    log.info("Starting scrape...")
    out = scrape_all()
    log.info("Saving outputs...")
    save(out['power'], PATHS['power'])
    save(out['cp'], PATHS['cp'])
    save(out['ct'], PATHS['ct'])
    out['specs'].to_csv(PATHS['specs'], sep=';', index=False)
    log.info("Done.")

if __name__ == '__main__':
    main()
