import os
import math
import random

import networkx as nx

random.seed(10)
import copy
import numpy as np
import torch
torch.manual_seed(120)
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.datasets import EmailEUCore, Planetoid, Amazon, AttributedGraphDataset, SNAPDataset, StochasticBlockModelDataset
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import torch.nn.functional as F
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm


def random_walk(G, start_node, length):
    # G_copy = copy.copy(G)
    # G_copy = G_copy.to_undirected()
    prev_node = start_node
    rw = [prev_node]
    n_steps = 0
    while n_steps < length-1:
        node = random.choice(list(G.neighbors(prev_node)))
        rw.append(node)
        n_steps += 1
        prev_node = node
    return rw

class LargeGraphDataset(Dataset):
    def __init__(self, data_file, data_name="custom",sampling_method='ego'):
        if data_name == "custom":
            self.data_file = data_file
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            sbm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
            directed_graph = False

            if self.data_file == 'Wiki':
                graphs = AttributedGraphDataset(base_path, self.data_file)
            if self.data_file == 'ego-facebook':
                graphs = SNAPDataset(base_path, self.data_file).get(1)
                graphs.data = Data(x=graphs.x, edge_index=graphs.edge_index, 
                                   n_nodes=graphs.num_nodes)
            if self.data_file == 'EmailEUCore':
                graphs = EmailEUCore(base_path)
            if self.data_file == 'Cora':
                graphs = Planetoid(base_path, self.data_file, transform=T.NormalizeFeatures())
                directed_graph = True 
            if self.data_file == 'sbm':
                class SBMGraph:
                    def __init__(self):
                        self.data = None 
                        with open(base_path + '/sbm_graph.pkl', 'rb') as f:
                            self.data = pickle.load(f)
                graphs = SBMGraph()
            
            # print(graphs.data.has_isolated_nodes())
            
            if directed_graph == True:
                #edge_list = to_undirected(graphs.data.edge_index)

                edge_list = graphs.data.edge_index
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                nx_graph = nx.from_edgelist(edge_list_nx, create_using=nx.DiGraph)
            else:
                edge_list = to_undirected(graphs.data.edge_index)
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                nx_graph = nx.from_edgelist(edge_list_nx)

            #nx_graph = nx.k_core(nx_graph, k= 27)
            print('edges in graph : ', len(list(nx_graph.edges())))

            y = torch.zeros([1, 0]).float()
            n_nodes = len(list(nx_graph.nodes()))#graphs.data.num_nodes
            print('nodes in graph : ', n_nodes)
            subgraph_size = 20
            samples_sizes = []
            #int(n_nodes ** 2 * math.log(n_nodes) / 100)
            if sampling_method == 'mix':
                print('Sampling type: mix')
                G_copy = copy.copy(nx_graph)
                G_copy = G_copy.to_undirected()
                dataset_node_lists = []
                print('sampling nodes : ')
                for n in tqdm(nx_graph.nodes()):
                    # print(n)

                    for i in range(30):
                        final_sample = random_walk(G_copy, n, subgraph_size)
                        samples_sizes.append(subgraph_size)

                        dataset_node_lists.append(final_sample)

                print('sampling nodes : ')
                for n in tqdm(nx_graph.nodes()):
                    # print(n)
                    ego_net = nx.ego_graph(nx_graph.to_undirected(), n, radius=2)

                    temp_initial_size = len(list(ego_net.nodes()))
                    for i in range(7):
                        temp_ego = copy.deepcopy(ego_net)
                        temp_size = temp_initial_size
                        final_sample = list(temp_ego.nodes())
                        while temp_size > subgraph_size:
                            n_nodes_to_burn = int(temp_size / 2)
                            temp_burning = copy.deepcopy(list(temp_ego.nodes()))
                            temp_burning.remove(n)
                            burned_nodes = list(random.choices(temp_burning, k=n_nodes_to_burn))
                            temp_ego.remove_nodes_from(burned_nodes)
                            final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len)) + [n]))
                            temp_size = len(final_sample)

                        samples_sizes.append(temp_size)

                        dataset_node_lists.append(final_sample)
                n_samples = 7000
                samples_sizes += [subgraph_size for i in range(n_samples)]
                dataset_node_lists+=[list(random.choices(range(n_nodes), k=subgraph_size)) for i in range(n_samples)]

            if sampling_method == 'random walk':
                print('Sampling type: random walk')
                G_copy = copy.copy(nx_graph)
                G_copy = G_copy.to_undirected()
                dataset_node_lists = []
                print('sampling nodes : ')
                for n in tqdm(nx_graph.nodes()):
                    # print(n)

                    for i in range(40):
                        final_sample = random_walk(G_copy,n,subgraph_size)
                        samples_sizes.append(subgraph_size)

                        dataset_node_lists.append(final_sample)

            if sampling_method == 'ego':
                print('Sampling type: ego')
                dataset_node_lists = []
                print('sampling nodes : ')
                for n in tqdm(nx_graph.nodes()):
                    # print(n)
                    ego_net = nx.ego_graph(nx_graph.to_undirected(),n, radius = 2)

                    temp_initial_size = len(list(ego_net.nodes()))
                    for i in range(10):
                        temp_ego = copy.deepcopy(ego_net)
                        temp_size = temp_initial_size
                        final_sample = list(temp_ego.nodes())
                        while temp_size > subgraph_size:
                            n_nodes_to_burn = int(temp_size/2)
                            temp_burning = copy.deepcopy(list(temp_ego.nodes()))
                            temp_burning.remove(n)
                            burned_nodes = list(random.choices(temp_burning, k = n_nodes_to_burn))
                            temp_ego.remove_nodes_from(burned_nodes)
                            final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len))+[n]))
                            temp_size = len(final_sample)



                        samples_sizes.append(temp_size)

                        dataset_node_lists.append(final_sample)

            elif sampling_method == 'uniform':
                print('Sampling type: uniform')
                n_samples = 10000
                dataset_node_lists = [list(random.choices(range(n_nodes), k=subgraph_size)) for i in range(n_samples)]
                samples_sizes = [subgraph_size for i in range(n_samples)]

            print(f'We sampled {len(dataset_node_lists)} subgraphs')
            n_samples = len(dataset_node_lists)
            self.sample_size = len(dataset_node_lists)
            print(f'We need to sample {n_samples} subgraphs')
            # sampled graphs effectively for training: 90%
            dataset_samples_initialids = [(dataset_node_lists[i],
                                           subgraph(torch.tensor(dataset_node_lists[i]), edge_list)[0])
                                          for i in range(n_samples)]
            #samples_sizes = [len(sample) for sample in dataset_node_lists]
            dict_maps = [{dataset_samples_initialids[j][0][i]: i for i in range(samples_sizes[j])} for j in range(n_samples)]
            #print(samples_sizes)
            dataset_samples_wnmaps = [(torch.tensor([[x] for x in dataset_samples_initialids[i][0]]),
                                       dataset_samples_initialids[i][1].apply_(lambda x: dict_maps[i][x])) for i in
                                      range(n_samples)]

            Train_data = [Data(x=F.one_hot(torch.flatten(dataset_samples_wnmaps[i][0]),num_classes = n_nodes).float().to_sparse(), edge_index=dataset_samples_wnmaps[i][1],
                               edge_attr = torch.tensor([[0 for k in range(dataset_samples_wnmaps[i][1].size()[1])],[1 for n in range(dataset_samples_wnmaps[i][1].size()[1])]], dtype = torch.long).transpose(0,1), #np.ones(dataset_samples_wnmaps[i][1].size()[1])),
                               n_nodes=subgraph_size*torch.ones(1, dtype=torch.long), y = y)  for i in range(n_samples)]

            self.data = Train_data


        #""" This class can be used to load the comm20, sbm and planar datasets. """
        # base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        # filename = os.path.join(base_path, data_file)
        # self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(
        #     filename)
        # print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # adj = self.adjs[idx]
        # n = adj.shape[-1]
        # X = torch.ones(n, 1, dtype=torch.float)
        # y = torch.zeros([1, 0]).float()
        # edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        # edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        # edge_attr[:, 1] = 1
        # num_nodes = n * torch.ones(1, dtype=torch.long)
        # data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
        #                                  y=y, idx=idx, n_nodes=num_nodes)
        graph = self.data[idx]
        return graph


class LargeGraph(LargeGraphDataset):
    def __init__(self):
        super().__init__('enron')


class LargeGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs= 100000):
        super().__init__(cfg)
        self.n_graphs = n_graphs/2
        self.prepare_data()
        self.inner = self.train_dataloader()
    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs):
        #print(len(graphs))
        test_len = int(round(len(graphs) * 0.05))
        train_len = int(round((len(graphs) - test_len) * 0.95))
        #print(train_len)
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)


class LargeGraphModule(LargeGraphDataModule):

    def prepare_data(self):
        graphs = LargeGraphDataset(self.cfg.dataset.name)
        return super().prepare_data(graphs)



class LargeGraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()             # There are no node types
        print(self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
