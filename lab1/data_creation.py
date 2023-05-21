import zipfile
import gdown
import splitfolders
from settings import DATA_URL, DATASET_PATH, DATASET_FILENAME, DATASET_RAW_FOLDER, DATASET_SPLITTED_FOLDER

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