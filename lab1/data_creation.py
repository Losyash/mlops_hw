import gdown
import zipfile
import splitfolders

DATA_URL = 'https://drive.google.com/file/d/18vilMtixEDJns6f-iVt6-xM9BhOAtNiC/view?usp=share_link'

DATASET_PATH = 'lab1/data'
DATASET_FILENAME = 'faces.zip'

DATASET_RAW_FOLDER = 'faces_raw' 
DATASET_SPLITTED_FOLDER = 'faces_splitted'

MODEL_PATH = 'lab1/model'

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

BATCH_SIZE = 64
NUM_EPOCHS = 1
NUM_CLASSES = 2

gdown.download(DATA_URL, f'{DATASET_PATH}/{DATASET_FILENAME}', quiet=False, fuzzy=True)

with zipfile.ZipFile(f'{DATASET_PATH}/{DATASET_FILENAME}', 'r') as zip_ref:
    zip_ref.extractall(f'{DATASET_PATH }/{DATASET_RAW_FOLDER}')

splitfolders.ratio(
    f'{DATASET_PATH }/{DATASET_RAW_FOLDER}',
    output=f'{DATASET_PATH }/{DATASET_SPLITTED_FOLDER}',
    seed=1337,
    ratio=(.8, .2),
    group_prefix=None,
    move=False
)