#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sagess

# Get dataset from environment variable
dataset=$DATASET

# print out config file
cat configs/dataset/${dataset}.yaml

python src/run_sagess.py dataset=${dataset}