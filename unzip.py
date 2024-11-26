import os
import yaml
import zipfile


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def extract_files(directory, delete_after_extract=True):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_to = os.path.join(root, os.path.splitext(file)[0])
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                
                if delete_after_extract:
                    os.remove(zip_path)

def compress_files(directory, delete_after_compress=True):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            zip_path = file_path + '.zip'

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                zip_ref.write(file_path, arcname=file) 

            if delete_after_compress:
                os.remove(file_path)


def main():
    
    config_path = "config.yaml"
    config = load_config(config_path)
    
    base_directory = config['download']['downloads_dir']

    extract_files(base_directory, delete_after_extract=True)

    #compress_files(base_directory, delete_after_compress=True)

if __name__ == "__main__":
    main()