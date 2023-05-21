import warnings
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from settings import (
    DATASET_PATH, DATASET_SPLITTED_FOLDER, DATASET_TRAIN_FOLDER,
    MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES
)

warnings.filterwarnings('ignore')

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(f'{DATASET_PATH }/{DATASET_SPLITTED_FOLDER}/{DATASET_TRAIN_FOLDER}',
    seed=42,
    label_mode = 'int',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
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

# model = Sequential([
#                   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width,3)),
#                   layers.Conv2D(32, 3, padding='same', activation='relu'),
#                   layers.MaxPooling2D(),
#                   layers.Conv2D(64, 3, padding='same', activation='relu'),
#                   layers.MaxPooling2D(),
#                   layers.Conv2D(128, 3, padding='same', activation='relu'),
#                   layers.MaxPooling2D(),
#                   layers.Dropout(0.2),
#                   layers.Conv2D(256, 3, padding='same', activation='relu'),
#                   layers.MaxPooling2D(),
#                   layers.Dropout(0.2),
#                   layers.Conv2D(512, 3, padding='same', activation='relu'),
#                   layers.MaxPooling2D(),
#                   layers.Dropout(0.2),
#                   layers.Flatten(),
#                   layers.Dense(256, activation='relu'),
#                   layers.Dropout(0.2),  
#                   layers.Dense(128, activation = 'relu'),
#                   layers.Dense(num_classes, activation = 'sigmoid')
# ])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


model.fit(
    train_dataset,
    epochs = NUM_EPOCHS,
    verbose = 1
)

model.save(f'{MODEL_PATH}')