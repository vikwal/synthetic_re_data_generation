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
        return [{'id': opt['value'], 'name': opt.text.strip()} 
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
def interpolate_series(series, order=3, resolution=0.01, clamp_mode='power'):
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
    
    if clamp_mode == 'power':
        s2.loc[s2.index < first] = 0
        s2.loc[s2.index > last] = 0
    elif clamp_mode == 'none':
        pass  # Keine Clamping für Cp/Ct
    
    return s2.fillna(0).round(2) if clamp_mode == 'power' else s2.round(2)

# --- Chart JSON Extraction ---
def extract_chart_json(js_text: str) -> dict:
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
    json_text = '{"data": ' + obj_text + '}'
    return json5.loads(json_text)

# --- Scrapers ---
def scrape_specs(soup):
    spec = {'rotor_diameter': None, 'hub_heights': [], 
            'cut_in': None, 'rated_speed': None, 'cut_out': None}
    
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
    
    # Korrektur: Deutsche Bezeichnung verwenden
    rs = text_after('Nennwindgeschwindigkeit:')
    if rs:
        try: spec['rated_speed'] = float(rs.replace(' m/s','').replace(',', '.'))
        except: spec['rated_speed'] = rs
    
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
            res = requests.get(url, headers=HEADERS, timeout=120)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            sp = scrape_specs(soup)
            
            for hh in sp['hub_heights'] or [None]:
                specs_list.append({
                    'Turbine': uid,
                    'Rotordurchmesser': sp['rotor_diameter'],
                    'Nabenhöhe': hh,
                    'Einschaltgeschwindigkeit': sp['cut_in'],
                    'Nennwindgeschwindigkeit': sp['rated_speed'],
                    'Abschaltgeschwindigkeit': sp['cut_out']
                })
            
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
                    power_series = interpolate_series(ser)
                    power_df = pd.concat([power_df, power_series], axis=1)
                elif label == 'cp':
                    cp_series = interpolate_series(ser, clamp_mode='none')  # Clamping deaktivieren
                    cp_df = pd.concat([cp_df, cp_series], axis=1)
                elif label == 'ct':
                    ct_series = interpolate_series(ser, clamp_mode='none')  # Clamping deaktivieren
                    ct_df = pd.concat([ct_df, ct_series], axis=1)
                    
        except Exception as e:
            log.error(f"Error for {uid}: {e}")
            continue

    cols = power_df.columns
    return {
        'power': power_df.sort_index(),
        'cp': cp_df.reindex(columns=cols),
        'ct': ct_df.reindex(columns=cols),
        'specs': pd.DataFrame(specs_list)
    }

# --- Post-processing ---
def post_process_data(out):
    cut_outs = []
    for speed in out['specs']['Abschaltgeschwindigkeit']:
        try:
            cut_outs.append(float(speed))
        except ValueError:
            continue
    max_cut_out = max(cut_outs) if cut_outs else 25
    
    unified_index = np.round(np.arange(0, max_cut_out + 0.01, 0.01), 2)
    
    for col in out['power'].columns:
        specs = out['specs'][out['specs']['Turbine'] == col]
        if specs.empty:
            continue
        
        try:
            cut_out = float(specs['Abschaltgeschwindigkeit'].iloc[0])
            cut_in = float(specs['Einschaltgeschwindigkeit'].iloc[0])
        except:
            continue
        
        # Power-Curve-Verarbeitung
        power_series = out['power'][col].copy()
        last_valid = power_series.last_valid_index()
        
        if last_valid and (last_valid < cut_out):
            new_speeds = np.round(np.arange(last_valid + 0.01, cut_out + 0.01, 0.01), 2)
            extension = pd.Series(power_series.loc[last_valid], index=new_speeds)
            power_series = power_series.combine_first(extension)
        
        first_non_zero = power_series[power_series > 0].index.min()
        if pd.notna(first_non_zero):
            power_series.loc[:first_non_zero] = 0
        power_series.clip(lower=0, inplace=True)
        power_series.loc[power_series.index > cut_out] = 0
        
        out['power'][col] = power_series.reindex(unified_index, fill_value=0)
        
        # Cp/Ct-Werte über Abschaltgeschwindigkeit auf NaN setzen
        mask = out['cp'].index > cut_out
        out['cp'].loc[mask, col] = np.nan
        out['ct'].loc[mask, col] = np.nan
    
    out['power'] = out['power'].reindex(unified_index, fill_value=0)
    out['cp'] = out['cp'].reindex(unified_index)
    out['ct'] = out['ct'].reindex(unified_index)
    
    return out

# --- Save Functions ---
def save(df, path):
    df.to_csv(path, sep=';', index=True, index_label='wind_speed')

def main():
    log.info("Starting scrape...")
    out = scrape_all()
    log.info("Post-processing...")
    out = post_process_data(out)
    log.info("Saving outputs...")
    save(out['power'], PATHS['power'])
    save(out['cp'], PATHS['cp'])
    save(out['ct'], PATHS['ct'])
    out['specs'].to_csv(PATHS['specs'], sep=';', index=False)
    log.info("Done.")

if __name__ == '__main__':
    main()