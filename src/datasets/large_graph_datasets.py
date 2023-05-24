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
    def __init__(self, data_file, data_name="eucore",sampling_method='ego'):
        if data_name == "eucore":
            self.data_file = data_file
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            sbm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

            #graphs = AttributedGraphDataset(base_path,"Wiki")
            # graphs = SNAPDataset(base_path,"ego-facebook").get(1)

            graphs = EmailEUCore(base_path)
            #graphs = Planetoid(base_path, "Cora", transform=T.NormalizeFeatures())
            #graphs = Amazon(base_path,'Photo')
            # graphs = AttributedGraphDataset(base_path,"Wiki")
            # graphs.data.num_nodes = 2000

            # graphs = StochasticBlockModelDataset(base_path, block_sizes=[200, 200, 300, 300, 500, 500],
            #                                      edge_probs=[[0.1, 0.01, 0.01, 0.01, 0.01, 0.01],
            #                                                  [0.01, 0.2, 0.01, 0.01, 0.01, 0.01],
            #                                                  [0.01, 0.01, 0.05, 0.01, 0.01, 0.01],
            #                                                  [0.01, 0.01, 0.01, 0.1, 0.01, 0.01],
            #                                                  [0.01, 0.01, 0.01, 0.01, 0.04, 0.01],
            #                                                  [0.01, 0.01, 0.01, 0.01, 0.01, 0.04]], is_undirected=False)

            # graphs = torch.load(sbm_path + '\sbm_test_in_out_4_1600_nodes.pt')
            # print(graphs.data.has_isolated_nodes())
            directed_graph = False


            if directed_graph == True:
                #edge_list = to_undirected(graphs.data.edge_index)

                edge_list = graphs.data.edge_index
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                nx_graph = nx.from_edgelist(edge_list_nx, create_using=nx.DiGraph)
                #import pdb;pdb.set_trace()
            else:
                edge_list = to_undirected(graphs.data.edge_index)
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                nx_graph = nx.from_edgelist(edge_list_nx)

            #import pdb; pdb.set_trace()

            #nx_graph = nx.k_core(nx_graph, k= 27)
            print(len(list(nx_graph.edges())))

            y = torch.zeros([1, 0]).float()
            n_nodes = len(list(nx_graph.nodes()))#graphs.data.num_nodes
            print(n_nodes)
            subgraph_size = 40
            samples_sizes = []
            #int(n_nodes ** 2 * math.log(n_nodes) / 100)
            if sampling_method == 'mix':
                G_copy = copy.copy(nx_graph)
                G_copy = G_copy.to_undirected()
                dataset_node_lists = []
                for n in nx_graph.nodes():
                    print(n)

                    for i in range(30):
                        final_sample = random_walk(G_copy, n, subgraph_size)
                        samples_sizes.append(subgraph_size)

                        dataset_node_lists.append(final_sample)

                for n in nx_graph.nodes():
                    print(n)
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
                            # import pdb;pdb.set_trace()
                            final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len)) + [n]))
                            temp_size = len(final_sample)

                        samples_sizes.append(temp_size)

                        dataset_node_lists.append(final_sample)
                n_samples = 7000
                samples_sizes += [subgraph_size for i in range(n_samples)]
                dataset_node_lists+=[list(random.choices(range(n_nodes), k=subgraph_size)) for i in range(n_samples)]

            if sampling_method == 'random walk':
                G_copy = copy.copy(nx_graph)
                G_copy = G_copy.to_undirected()
                dataset_node_lists = []
                for n in nx_graph.nodes():
                    print(n)

                    for i in range(40):
                        final_sample = random_walk(G_copy,n,subgraph_size)
                        samples_sizes.append(subgraph_size)

                        dataset_node_lists.append(final_sample)

            if sampling_method == 'ego':
                dataset_node_lists = []
                for n in nx_graph.nodes():
                    print(n)
                    ego_net = nx.ego_graph(nx_graph.to_undirected(),n, radius = 2)

                    temp_initial_size = len(list(ego_net.nodes()))
                    for i in range(40):
                        temp_ego = copy.deepcopy(ego_net)
                        temp_size = temp_initial_size
                        final_sample = list(temp_ego.nodes())
                        while temp_size > subgraph_size:
                            n_nodes_to_burn = int(temp_size/2)
                            temp_burning = copy.deepcopy(list(temp_ego.nodes()))
                            temp_burning.remove(n)
                            burned_nodes = list(random.choices(temp_burning, k = n_nodes_to_burn))
                            temp_ego.remove_nodes_from(burned_nodes)
                            #import pdb;pdb.set_trace()
                            final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len))+[n]))
                            temp_size = len(final_sample)



                        samples_sizes.append(temp_size)

                        dataset_node_lists.append(final_sample)

            elif sampling_method == 'uniform':
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
        graphs = LargeGraphDataset('enron')
        #import pdb; pdb.set_trace()
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
