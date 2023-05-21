#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ~/mlops/lab1/conda
pip install gdown zipfile36 split-folders tensorflow

cd ~/mlops/lab1

if ! [ -d data ]; then
    mkdir data
fi

python data_creation.py
python data_preprocessing.py
python model_preparation.py
python model_testing.py