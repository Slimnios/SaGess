import networkx as nx
from graspologic.models import DCSBMEstimator
import os
from torch_geometric.utils import subgraph, to_undirected
import numpy as np
from torch.utils.data import random_split, Dataset
from torch_geometric.datasets import EmailEUCore, Planetoid, Amazon, AttributedGraphDataset, SNAPDataset, StochasticBlockModelDataset
import torch_geometric.transforms as T
import torch

import pickle

base_path = os.path.join('data')

# graphs = EmailEUCore(base_path)



graphs = AttributedGraphDataset(base_path,"Wiki")
graphs = SNAPDataset(base_path,"ego-facebook").get(1)
# qgf



with open('sbm_edge_list.pkl','rb') as f:
    edge_list_syn = pickle.load(f)
# edge_list = to_undirected(graphs.data.edge_index)
# edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]



nx_graph = nx.from_edgelist(edge_list_syn)

# graphs = Planetoid(base_path, "Cora", transform=T.NormalizeFeatures())
#
# edge_list = graphs.data.edge_index
# edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
# nx_graph = nx.from_edgelist(edge_list_nx, create_using=nx.DiGraph)

dcsbme = DCSBMEstimator(directed=False)

adj = nx.to_numpy_array(nx_graph)

print(adj)

dcsbme.fit(adj)

A = dcsbme.sample()[0]

edge_list = nx.from_numpy_array(A)

edge_list_samples = list(edge_list.edges())

import pdb;pdb.set_trace()

import pickle
with open('DSBM_edge_list_sbm.pkl', 'wb') as f:
    pickle.dump(edge_list_samples, f)