from tensorflow.python.keras.applications import ResNet50
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from ..utils import utils

RESNET_WEIGHTS_PATH = '../models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
IMAGE_SIZE = 224


resnet50 = ResNet50(weights=RESNET_WEIGHTS_PATH)


def read_and_prep_images(img_paths, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


def get_image_paths(image_directory, n=10, seed=1):
    img_paths = utils.list_all_files(image_directory)

    np.random.seed(seed)
    if len(img_paths) > n:
        img_paths = np.random.choice(img_paths, n, replace=False)

    return img_paths

