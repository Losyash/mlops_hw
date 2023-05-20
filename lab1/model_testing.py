import tensorflow as tf

DATASET_PATH = 'lab1/data'

DATASET_RAW_FOLDER = 'faces_raw' 
DATASET_SPLITTED_FOLDER = 'faces_splitted'
DATASET_TRAIN_FOLDER = 'train';
DATASET_TEST_FOLDER = 'val';

MODEL_PATH = 'lab1/model'

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

BATCH_SIZE = 64
NUM_EPOCHS = 1
NUM_CLASSES = 2

test_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{DATASET_PATH }/{DATASET_SPLITTED_FOLDER}/{DATASET_TEST_FOLDER}',
    seed=42,
    label_mode = 'int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

model = tf.keras.models.load_model(f'{MODEL_PATH}')

accuracy = model.evaluate(test_ds)
print('Model test accuracy is:', accuracy[1])