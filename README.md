# Denoising diffusion models for graph generation


Warning: The code has been updated after experiments were run for the paper. If you don't manage to reproduce the 
paper results, please write to us so that we can investigate the issue.

For the conditional generation experiments, check the `guidance` branch. 

## Environment installation
  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit: `conda create -c conda-forge -n my-rdkit-env rdkit`
  - Install graph-tool (https://graph-tool.skewed.de/)
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`
  - Install mini-moses: `pip install git+https://github.com/igor-krawczuk/mini-moses@main`

## Download the data

  - QM9 and Guacamol should download by themselves when you run the code.
  - For the community, SBM and planar datasets, data can be found at https://github.com/KarolisMart/SPECTRE/tree/main/data
  - Moses dataset can be found at https://github.com/molecularsets/moses/tree/master/data
  


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available
    
## Checkpoints

We uploaded pretrained models for the Planar and SBM datasets. If you need other checkpoints, please write to us.

Planar: https://drive.switch.ch/index.php/s/tZCjJ6VXU2Z3FIh
SBM: https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j
    
## Cite the paper

```
@article{vignac2022digress,
  title={DiGress: Discrete Denoising diffusion for graph generation},
  author={Vignac, Clement and Krawczuk, Igor and Siraudin, Antoine and Wang, Bohan and Cevher, Volkan and Frossard, Pascal},
  journal={arXiv preprint arXiv:2209.14734},
  year={2022}
}
```
