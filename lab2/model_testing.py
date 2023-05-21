import warnings
import tensorflow as tf
from settings import (
    DATASET_PATH, DATASET_SPLITTED_FOLDER, DATASET_TEST_FOLDER,
    MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE
)

warnings.filterwarnings('ignore')

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(f'{DATASET_PATH }/{DATASET_SPLITTED_FOLDER}/{DATASET_TEST_FOLDER}',
    seed=42,
    label_mode = 'int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

model = tf.keras.models.load_model(f'{MODEL_PATH}')

accuracy = model.evaluate(test_dataset)
print('Model test accuracy is:', accuracy[1])