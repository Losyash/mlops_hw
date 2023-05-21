#!/bin/bash

if conda env list | grep -q lab1; then echo "Environment already exists"; else conda create -y -n lab1 python=3.10; fi

conda activate lab1

cd ~/mlops/lab1


if ! [ -d data ]; then mkdir data fi

# python data_creation.py
# python data_preprocessing.py
# python model_preparation.py
# python model_testing.py