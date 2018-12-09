import os
import numpy as np
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense  # , Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping

from ..utils import utils
from ..data import download_data


DATA_ROOT = '../data/raw'
MODEL_ROOT = '../models'
NUM_CLASSES = 2
RESNET_WEIGHTS_PATH = '../models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_SIZE = 224


def distinguish_two_things(thing1, thing2):
    (data_dir, model_dir) = initialize_project(thing1, thing2)

    download_data.download_google_images([thing1, thing2], data_dir)

    model, history = fit_model(data_dir, model_dir)

    # To load model and make predictions interactively:
    if False:
        model = load_model(f'{model_dir}/model.h5')
        img_paths = get_val_image_paths([thing1, thing2], data_dir)
        predictions = make_predictions(model, img_paths, [thing1, thing2])


def initialize_project(thing1, thing2):
    # get folder-friendly names
    thing1_fmt = utils.format_string(thing1)
    thing2_fmt = utils.format_string(thing2)

    project_name = thing1_fmt + '_vs_' + thing2_fmt

    (data_dir, model_dir) = make_project_dirs(project_name)
    return data_dir, model_dir


def fit_model(data_dir, model_dir, max_epochs=10, batch_size=20):
    """
    Uses transfer learning to replace the last layer of ResNet50 to train a binary classifier
    Saves the model wihtin model_dir
    :param data_dir: Path to data root
    :param model_dir: Path to model root
    :param max_epochs: Maximum number of epochs to run
    :param batch_size: Number of images per batch
    :return: a tuple of model, history
    """
    # Specify Model
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=RESNET_WEIGHTS_PATH))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Say not to train first layer (ResNet) model. It is already trained
    model.layers[0].trainable = False

    # Compile Model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit Model
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = data_generator.flow_from_directory(
        f'{data_dir}/train',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        f'{data_dir}/val',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        class_mode='categorical')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01)
    history = model.fit_generator(
        train_generator,
        epochs=max_epochs,
        callbacks=[early_stopping],
        validation_data=validation_generator,
        validation_steps=1)

    model.save(f'{model_dir}/model.h5')
    # TODO - save history

    return model, history


def make_predictions(model, img_paths, things):
    imgs = read_and_prep_images(img_paths)
    preds = model.predict(imgs)
    pred_class = [things[0] if preds[i, 0] >= 0.5 else things[1] for i in range(0, len(preds))]
    class_prob = np.max(preds, axis=1)
    # import pdb; pdb.set_trace()
    return list(zip(pred_class, class_prob))


def make_project_dirs(project_name, existsok=True):
    data_dir = f'{DATA_ROOT}/{project_name}'
    model_dir = f'{MODEL_ROOT}/{project_name}'

    os.makedirs(data_dir, exist_ok=existsok)
    os.makedirs(f'{data_dir}/train', exist_ok=existsok)
    os.makedirs(f'{data_dir}/val', exist_ok=existsok)
    os.makedirs(model_dir, exist_ok=existsok)
    return data_dir, model_dir


def read_and_prep_images(img_paths, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


def get_val_image_paths(things, data_dir, n=10, seed=1):
    img_paths = []
    for i, thing in enumerate(things):
        image_dir = str(i) + '_' + utils.format_string(thing)
        img_paths += utils.list_all_files(f'{data_dir}/val/{image_dir}')

    np.random.seed(seed)
    img_paths = np.random.choice(img_paths, n, replace=False)
    return img_paths


if __name__ == '__main__':
    pass
