import warnings
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from settings import (
    DATASET_PATH, DATASET_SPLITTED_FOLDER,
    DATASET_TRAIN_FOLDER, DATASET_TEST_FOLDER
)

warnings.filterwarnings('ignore')

image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3
)

def genereate_image(path, file): 
    image = load_img(f'{path}/{file}')
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, ) + image_array.shape)

    i = 0

    for batch in image_datagen.flow(
        image_array,
        batch_size=1,
        save_format='jpg',
        save_prefix='aug',
        save_to_dir=path
    ):
        i += 1

        if i > 1:
            break


train_folders = os.listdir(f'{DATASET_PATH}/{DATASET_SPLITTED_FOLDER}/{DATASET_TRAIN_FOLDER}');

for folder in train_folders:
    FOLDER_PATH = f'{DATASET_PATH}/{DATASET_SPLITTED_FOLDER}/{DATASET_TRAIN_FOLDER}/{folder}'

    files = os.listdir(FOLDER_PATH)

    for file in files:
        genereate_image(FOLDER_PATH, file)


test_folders = os.listdir(f'{DATASET_PATH}/{DATASET_SPLITTED_FOLDER}/{DATASET_TEST_FOLDER}');

for folder in test_folders:
    FOLDER_PATH = f'{DATASET_PATH}/{DATASET_SPLITTED_FOLDER}/{DATASET_TEST_FOLDER}/{folder}'

    files = os.listdir(FOLDER_PATH)

    for file in files:
        genereate_image(FOLDER_PATH, file)