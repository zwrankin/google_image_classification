import os
import zipfile
from google_images_download import google_images_download
from ..utils import utils

RAW_DATA_DIR = '.../data/raw'


def unzip_data(data_dir):
    """Unzip all data and delete compressed files"""
    zip_files = [f for f in os.listdir(data_dir) if '.zip' in f]
    for f in zip_files:
        print(f)
        filepath = os.path.join(data_dir, f)
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(filepath)


def download_google_images(things:list, project_dir:str, image_limit=50):
    response = google_images_download.googleimagesdownload()

    # We could randomly sample train/val, but for now we'll temporally separate
    train_range = '{"time_min":"01/01/2010","time_max":"01/01/2015"}'
    val_range = '{"time_min":"01/01/2015","time_max":"01/01/2018"}'

    for i, thing in enumerate(things):
        # Maintain thing order in directories to ensure proper labelling
        image_dir = str(i) + '_' + utils.format_string(thing)

        arguments = {"keywords": thing, "limit": image_limit, "print_urls": False,
                     'size': 'medium', 'format': 'jpg'}

        arguments.update({'output_directory': project_dir + '/train',
                          'image_directory': image_dir,
                          'time_range': train_range})
        paths = response.download(arguments)
        delete_non_jpegs(f'{project_dir}/train/{image_dir}')

        arguments.update({'output_directory': project_dir + '/val',
                          'image_directory': image_dir,
                          'time_range': val_range})
        paths = response.download(arguments)
        delete_non_jpegs(f'{project_dir}/val/{image_dir}')


def delete_non_jpegs(directory):
    non_jpgs = [f for f in os.listdir(directory) if f[-4:] != '.jpg']
    if len(non_jpgs) > 0:
        [os.remove(f'{directory}/{f}') for f in non_jpgs]


if __name__ == '__main__':
    pass
    # !kaggle competitions download -c dog-breed-identification
    # unzip_data(RAW_DATA_DIR)
