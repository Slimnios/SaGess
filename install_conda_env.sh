#!/bin/bash

conda env create -f environment.yml

source activate sagess 

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r pyg_reqs.txt -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html

conda clean -y --all

