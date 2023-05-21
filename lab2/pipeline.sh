#!/bin/bash

if conda env list | grep -q lab2; then echo "Environment already exists"; else conda create -y -n lab2 python=3.10; fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lab2

pip install gdown zipfile36 split-folders tensorflow pillow

cd ~/mlops/lab1

if ! [ -d data ]; then mkdir data; fi

echo 'Start data creation'
python data_creation.py
echo 'End data creation'

echo 'Start data prepocessing'
python data_preprocessing.py
echo 'End data prepocessing'

echo 'Start model fiting'
python model_preparation.py
echo 'End model fiting'

echo 'Start model testing'
python model_testing.py