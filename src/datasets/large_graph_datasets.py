import os
import random
import time 
import networkx as nx

random.seed(10)
import copy
import torch
torch.manual_seed(120)
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.datasets import EmailEUCore, Planetoid, AttributedGraphDataset, SNAPDataset
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import torch.nn.functional as F
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm
import multiprocessing as mp 
from utils import random_walk, rw_task, ego_task, uniform_task

class LargeGraphDataset(Dataset):
    def __init__(self, cfg, data_name="custom",sampling_method='mix'):
        self.cfg = cfg 
        self.data_file = self.cfg.dataset.name
        if data_name == "custom":
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
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
            if self.data_file == 'deezer':
                class Deezer:
                    def __init__(self):
                        self.data = None 
                        with open(base_path + '/deezer_edge_lists.pkl', 'rb') as f:
                            graph_list = pickle.load(f)
                        self.data = []
                        for edge_list in graph_list:
                            undirected_edge_list = edge_list + [(j, i) for i, j in edge_list]
                            edge_index = torch.tensor(undirected_edge_list, dtype=torch.long).t().contiguous()
                            n = edge_index.max().item() + 1
                            
                            X = torch.ones(n, 1, dtype=torch.float)
                            y = torch.zeros([1, 0]).float()
                            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                            edge_attr[:, 1] = 1
                            num_nodes = n * torch.ones(1, dtype=torch.long)
                            
                            self.data.append(Data(x=X, 
                                                edge_index=edge_index, 
                                                edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes))
                        
                        self.data = self.data[-1]
                
                graphs = Deezer()

            
            # print(graphs.data.has_isolated_nodes())
            
            if directed_graph == True:
                edge_list = graphs.data.edge_index
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                self.nx_graph = nx.from_edgelist(edge_list_nx, create_using=nx.DiGraph)
            else:
                edge_list = to_undirected(graphs.data.edge_index)
                edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
                self.nx_graph = nx.from_edgelist(edge_list_nx)

            #nx_graph = nx.k_core(nx_graph, k= 27)
            print('edges in graph : ', len(list(self.nx_graph.edges())))

            y = torch.zeros([1, 0]).float()
            n_nodes = len(list(self.nx_graph.nodes()))
            print('nodes in graph : ', n_nodes)
            subgraph_size = 20

            self.samples_sizes = []
            self.dataset_node_lists = []

            sampling_start_time = time.time()

            if self.cfg.dataset.sampling_method == 'mix':
                print('Sampling type: mix')

                self.random_walk_sample(30, subgraph_size)
                self.uniform_sample(7000, subgraph_size)
                # self.ego_sample(7, subgraph_size)
                self.ego_sample_threaded(7, subgraph_size, 
                                         max_workers=self.cfg.dataset.sampling_threads)

            elif self.cfg.dataset.sampling_method == 'random_walk':
                print('Sampling type: random walk')

                self.random_walk_sample(40, subgraph_size)
                
                # self.random_walk_sample_threaded(40, subgraph_size, 
                #                                  max_workers=self.cfg.dataset.sampling_threads)

            elif self.cfg.dataset.sampling_method == 'ego':
                print('Sampling type: ego')

                # self.ego_sample(10, subgraph_size)

                self.ego_sample_threaded(10, subgraph_size, 
                                         max_workers=self.cfg.dataset.sampling_threads)

            elif self.cfg.dataset.sampling_method == 'uniform':
                print('Sampling type: uniform')

                self.uniform_sample(10000, subgraph_size)
            
            sampling_end_time = time.time()
            n_samples = len(self.dataset_node_lists)
            print(f'We need to sample {n_samples} subgraphs')
            print(f'We sampled {len(self.dataset_node_lists)} subgraphs')
            self.sample_size = len(self.dataset_node_lists)
            sampling_time = sampling_end_time - sampling_start_time
            minutes = sampling_time // 60
            seconds = sampling_time % 60
            print('sampling start time : ', sampling_start_time)
            print('sampling end time : ', sampling_end_time)
            print(f"Total sampling time: {int(minutes)} mins, {seconds:.2f} secs")
            
            # sampled graphs effectively for training: 90%
            dataset_samples_initialids = [(self.dataset_node_lists[i],
                                           subgraph(torch.tensor(self.dataset_node_lists[i]), edge_list)[0])
                                          for i in range(n_samples)]

            dict_maps = [{dataset_samples_initialids[j][0][i]: i for i in range(self.samples_sizes[j])} for j in range(n_samples)]
            dataset_samples_wnmaps = [(torch.tensor([[x] for x in dataset_samples_initialids[i][0]]),
                                       dataset_samples_initialids[i][1].apply_(lambda x: dict_maps[i][x])) for i in
                                      range(n_samples)]

            Train_data = [Data(x=F.one_hot(torch.flatten(dataset_samples_wnmaps[i][0]),num_classes = n_nodes).float().to_sparse(), 
                               edge_index=dataset_samples_wnmaps[i][1],
                               edge_attr = torch.tensor([[0 for k in range(dataset_samples_wnmaps[i][1].size()[1])],[1 for n in range(dataset_samples_wnmaps[i][1].size()[1])]], dtype = torch.long).transpose(0,1), #np.ones(dataset_samples_wnmaps[i][1].size()[1])),
                               n_nodes=subgraph_size*torch.ones(1, dtype=torch.long), y = y)  for i in range(n_samples)]

            self.data = Train_data


    def uniform_sample(self, n_samples=5000, subgraph_size=20):
        n_nodes = len(list(self.nx_graph.nodes()))
        self.samples_sizes += [subgraph_size for i in range(n_samples)]
        self.dataset_node_lists+=[list(random.choices(range(n_nodes), k=subgraph_size)) for i in tqdm(range(n_samples), desc="Uniform Sampling")]


    def random_walk_sample(self, loop=30, subgraph_size=20):
        G_copy = copy.copy(self.nx_graph)
        G_copy = G_copy.to_undirected()
        self.dataset_node_lists = []
        
        for n in tqdm(self.nx_graph.nodes(), desc="Random Walk Sampling"):
            for _ in range(loop):
                final_sample = random_walk(G_copy, n, subgraph_size)
                self.samples_sizes.append(subgraph_size)
                self.dataset_node_lists.append(final_sample)


    def ego_sample(self, loop=10, subgraph_size=20, radius=2):
        for n in tqdm(self.nx_graph.nodes(), desc="Ego Sampling"):
            ego_net = nx.ego_graph(self.nx_graph.to_undirected(), n, radius=radius)

            temp_initial_size = len(list(ego_net.nodes()))
            for _ in range(loop):
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

                self.samples_sizes.append(temp_size)

                self.dataset_node_lists.append(final_sample)


    def random_walk_sample_threaded(self, loop=30, subgraph_size=20, max_workers=2):
        G = copy.copy(self.nx_graph)

        args_list = [(n, G, loop, subgraph_size) for n in self.nx_graph.nodes()]

        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(rw_task, args_list), total=len(args_list), desc="Threaded Random Walk Sampling"))

        all_samples, all_sizes = zip(*results) 
        # Flatten the list of lists
        all_samples = [item for sublist in all_samples for item in sublist]  
        all_sizes = [item for sublist in all_sizes for item in sublist]
        
        self.dataset_node_lists.extend(all_samples)
        self.samples_sizes.extend(all_sizes)


    def ego_sample_threaded(self, loop=10, subgraph_size=20, radius=2, max_workers=2):
        G = copy.copy(self.nx_graph)

        args_list = [(n, G, radius, loop, subgraph_size) for n in self.nx_graph.nodes()]

        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(ego_task, args_list), total=len(args_list), desc="Threaded Ego Sampling"))

        all_samples, all_sizes = zip(*results) 
        # Flatten the list of lists
        all_samples = [item for sublist in all_samples for item in sublist]  
        all_sizes = [item for sublist in all_sizes for item in sublist]
        
        self.dataset_node_lists.extend(all_samples)
        self.samples_sizes.extend(all_sizes)


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


class LargeGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs= 100000):
        super().__init__(cfg)
        self.n_graphs = n_graphs/2
        self.graphs = LargeGraphDataset(cfg)
        self.prepare_data()
        self.inner = self.train_dataloader()
        
    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self):
        #print(len(self.graphs))
        test_len = int(round(len(self.graphs) * 0.05))
        train_len = int(round((len(self.graphs) - test_len) * 0.95))
        #print(train_len)
        val_len = len(self.graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(self.graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)


class LargeGraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()             # There are no node types
        print(self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
