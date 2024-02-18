import numpy as np
import glob

################################################################
from torch_geometric.data import Data
import torch_geometric as pyg
import pickle
import torch
import networkx as nx 
from collections import defaultdict
import pandas as pd 

from src.graph_statistics import compute_graph_statistics

    
def load_from_pickle(filename):
    with open(f'{filename}', 'rb') as f:
        edge_list = pickle.load(f)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    num_nodes = edge_index.max().item() + 1 
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return [data]

################################################################


method = 'full_run_sagess_mul_workers'

# dataset_name = 'EmailEUCore'
# dataset_name = 'Cora'
# dataset_name = 'Wiki'
# dataset_name = 'ego-facebook'
dataset_name = 'sbm'

print('Calculating graph statistics for dataset ' + dataset_name + ' - ')

filename = f'./runs_sagess/{method}/generated_graphs/0_{dataset_name}_edge_list.pkl'

dataset = load_from_pickle(filename)

data = dataset[0]

print('edges from data : ', data.edge_index.numpy().astype(int).shape)

if dataset_name == 'Cora':
    G = nx.from_edgelist(data.edge_index.numpy().astype(int).T, create_using=nx.DiGraph)
else: 
    G = nx.from_edgelist(data.edge_index.numpy().astype(int).T)

print('generated graph : ', G)


stats = compute_graph_statistics(G)

print(stats)

# Convert to pandas DataFrame
df = pd.DataFrame([stats])

# Save to .csv
df.to_csv(f'{"/".join(filename.split("/")[:-1])}/graph_statistics_{dataset_name}.csv', 
                                                                        index=False)

