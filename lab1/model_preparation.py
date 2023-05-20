# import pickle
import tensorflow as tf
from tensorflow.python import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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

train_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{DATASET_PATH }/{DATASET_SPLITTED_FOLDER}/{DATASET_TRAIN_FOLDER}',
    seed=42,
    label_mode = 'int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation = 'softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


history = model.fit(
    train_ds, epochs = NUM_EPOCHS,
    verbose = 1
)

model.save(f'{MODEL_PATH}')