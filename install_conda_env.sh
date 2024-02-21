#!/bin/bash

CONDA_ENV=sagess
# CUDA_VER={cu113}

conda env create -f environment.yml

# Install additional Python packages with pip
conda run -n $CONDA_ENV pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
conda run -n $CONDA_ENV pip3 install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
conda run -n $CONDA_ENV pip3 install pytorch-lightning==2.1.4 torchmetrics==1.3.0.post0

# Clean up any unnecessary files
conda run -n $CONDA_ENV conda clean -y --all

# RUN THIS COMMAND AFTER ENV IS CREATED
# g++ -O2 -std=c++11 -o ./src/analysis/orca/orca ./src/analysis/orca/orca.cpp
