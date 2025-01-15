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
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_contents = zip_ref.namelist()
                    zip_ref.extract(zip_contents[0], root)
                
                if delete_after_extract:
                    os.remove(zip_path)


def main():
    
    config_path = "config.yaml"
    config = load_config(config_path)
    
    base_directory = config['data']['dir']

    extract_files(base_directory, delete_after_extract=True)

if __name__ == "__main__":
    main()