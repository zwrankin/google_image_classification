import os
import zipfile


RAW_DATA_DIR = '.../data/raw'

def unzip_data(data_dir=RAW_DATA_DIR):
    """Unzip all data and delete compressed files"""
    zip_files = [f for f in os.listdir(data_dir) if '.zip' in f]
    for f in zip_files:
        print(f)
        filepath = os.path.join(data_dir, f)
        with zipfile.ZipFile(filepath,"r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(filepath)


if __name__ == '__main__':
    !kaggle competitions download -c dog-breed-identification
    unzip_data()
    