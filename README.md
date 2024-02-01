# SaGess: Sampling Graph Discrete Denoising Diffusion Model

Official PyTorch implementation of [SaGess](https://arxiv.org/pdf/2306.16827.pdf).

SaGess is a discrete denoising diffusion model, which extends [DiGress](https://github.com/cvignac/DiGress) with a divide-and-conquer strategy to generate large synthetic networks by training on subgraph samples and reconstructing the overall graph.


## Setting up the environment

Create anaconda environment
```
chmod +x *.sh

./install_conda_env.sh
```

Activate the environment
```
conda activate sagess
```


## Setting up Wandb logs

By default, wandb stores the logs offline and would need to be synced after training.
Make sure to set the 'entity' parameter in the `setup_wandb()` function located in `src/run_sagess.py` to be able to sync the logs to your account.  
```
 'entity': 'wandb_username'
```


## Running the code
    
All code is currently launched with the following: 
``` python src\run_sagess.py dataset=Cora ```

4 datasets from `torch_geometric` are supported: Cora, Wiki, EmailEUCore, ego-facebook and one custom dataset loaded as a `.pkl` file. All the datasets are downloaded to or placed in the `data` folder. 

Dataset specific configuration resides in `configs/dataset/*.yaml` files. 

Other default parameters for DiGress are found in `configs/train/train_default.yaml`, `configs\model\discrete.yaml` and `configs\general\general_default.yaml`. 

The following parameters: dataset, number of subgraphs to train on, their size and sampling method are to be set manually in `src/dataset/large_graph_datasets.py`.

You can also find the evaluation pipeline in `src\evaluation.py`. 

Saved checkpoints, wandb log folder and other outputs can be found in the `outputs` folder. 


## Additional support for docker

To build and run the docker container, use `docker_build.sh` and `run_docker_container.sh` scripts respectively. 