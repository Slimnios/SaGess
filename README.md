# SaGess: Sampling Graph Discrete Denoising Diffusion Model


Here is the code to run SaGess, you will need to meet the requirements from the pyproject.toml file to run the model.





## Run the code
    
All code is currently launched with the following ``` python src\run_SaGess.py dataset=default ``` .

This will run with default parameters found in configs/train/train_default.yaml, configs\model\discrete.yaml and configs\general\general_default.yaml.
They set the different parameters for DiGress by default.

The following parameters: dataset, number of subgraphs to train on, their size and sampling method are to be set manually in src/dataset/large_graph_datasets.py.

You can also find the evaluation pipeline in src\evaluation.py

We apologize for the crudeness of this repo, it will be polished and clarified before being released publicly.
