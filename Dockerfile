# devel version needed for compilation 
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV WORKDIR=/root/workspace
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV CONDA_ENV=sagess

# Install essential packages and Miniconda
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY ./requirements.txt ./requirements.txt 

RUN apt-get update && \
    apt-get install -y libgdk-pixbuf2.0-0 libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgtk-3-0 libx11-6 libxcursor1 libxext6 libxrender1 libffi7 libgirepository-1.0-1 && \
    rm -rf /var/lib/apt/lists/* && \
    conda create -y -n $CONDA_ENV python=3.10 && \
    echo "source activate $CONDA_ENV" > ~/.bashrc && \
    conda run -n $CONDA_ENV conda install -c conda-forge graph-tool && \
    conda run -n $CONDA_ENV pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    conda run -n $CONDA_ENV pip install -r requirements.txt && \
    conda run -n $CONDA_ENV conda clean -y --all

COPY ./pyg_reqs.txt ./pyg_reqs.txt

RUN conda run -n $CONDA_ENV pip install -r pyg_reqs.txt -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html && \
    conda run -n $CONDA_ENV conda clean -y --all

WORKDIR $WORKDIR

CMD ["bash"]

# RUN THIS COMMAND AFTER CONTAINER IS CREATED
# g++ -O2 -std=c++11 -o ./src/analysis/orca/orca ./src/analysis/orca/orca.cpp

