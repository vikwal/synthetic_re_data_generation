import os
import time
import yaml
import requests
from datetime import datetime
from bs4 import BeautifulSoup



def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def write_file(url: str, 
               href: str, 
               target_dir: str, 
               sleep=1, 
               verbose=False
    ):
    file_url = os.path.join(url, href)
    filename = os.path.join(target_dir, href)
    if os.path.exists(filename):
        if verbose: 
            print(f'{filename}_exists.')
        return
    
    file_response = requests.get(file_url, stream=True)
    file_response.raise_for_status()
    time.sleep(sleep/2)
    with open(filename, 'wb') as file:
        for chunk in file_response.iter_content(chunk_size=8192):
            file.write(chunk)
        if verbose:
            print(f'Download for {filename} completed.')

def download(url: str, 
             target_dir: str,
             check_date: str,
             sleep=1):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    
    existing_files = set(os.listdir(target_dir))

    rel_links = [link for link in links if '../' not in link.get('href')]
    check_date = datetime.strptime(check_date, '%Y%m%d')
    
    for link in rel_links:
        
        href = link.get('href')
        
        if href in existing_files:
            print(f'{href} already exists. Skipping.')
            continue
        
        split_link = href.split('_')
        if len(split_link) < 5:
            continue
        
        if 'txt' not in href:
            end_date = datetime.strptime(split_link[4], '%Y%m%d')
            if end_date < check_date:
                continue
        
        write_file(url=url, 
                   href=href, 
                   target_dir=target_dir, 
                   sleep=sleep, 
                   verbose=True)

def main():
    
    config_path = "config.yaml"
    config = load_config(config_path)
    sc_config = config['scraping']
    
    download_dir = config['download']['downloads_dir']
    download_url = sc_config['download_url']
    vars = sc_config['vars']
    check_date = sc_config['check_date']
    sleep = sc_config['sleep']
    
    for var in vars:
        target_dir = os.path.join(download_dir, var)
        os.makedirs(target_dir, exist_ok=True)
        url = os.path.join(download_url, var, 'historical')
        download(url=url, 
                target_dir=target_dir,
                check_date=check_date,
                sleep=sleep)
            
if __name__ == '__main__':
    main()