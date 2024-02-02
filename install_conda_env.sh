#!/bin/bash

CONDA_ENV=sagess
# CUDA_VER={cu113}

conda env create -f environment.yml

# Install additional Python packages with pip
conda run -n $CONDA_ENV pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
conda run -n $CONDA_ENV pip install -r pyg_reqs.txt

# Clean up any unnecessary files
conda run -n $CONDA_ENV conda clean -y --all

# RUN THIS COMMAND AFTER ENV IS CREATED
# g++ -O2 -std=c++11 -o ./src/analysis/orca/orca ./src/analysis/orca/orca.cpp
