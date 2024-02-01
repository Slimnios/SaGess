import os
import math
import random
random.seed(10)
import numpy as np
import matplotlib.pyplot as plt

import torch
import networkx as nx
torch.manual_seed(120)
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.datasets import EmailEUCore
# from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from torch_geometric.utils.convert import from_networkx
from itertools import combinations

import torch
from torch_geometric.datasets import Planetoid,SNAPDataset, StochasticBlockModelDataset, AttributedGraphDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from graph_statistics import compute_graph_statistics
from graph_statistics import power_law_alpha,gini
import pandas as pd
# import seaborn as sns
import pickle



def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def squares(G, nodes=None):
    r"""Borrowed from networkx square clustering, this could be made much more efficient.
    """
    if nodes is None:
        node_iter = G
    else:
        node_iter = G.nbunch_iter(nodes)
    squares = 0
    for v in node_iter:
        for u, w in combinations(G[v], 2):
            squares += len((set(G[u]) & set(G[w])) - {v})
    return squares/4

degrees = []
power_stats = []
edge_syn_lists = []
assortativity = []
cluster = []
gini_coef = []
for i in range(1,11,1):
    with open('outputs/email-eucore_stop_thresh_syngr_30ktrain_rw_200e_grsize40_batch32_right_progressive_percentage/15-51-32/' +str(i)+'_edge_list.pkl','rb') as f:
    # edge_list_syn = []
    # for edge in f.readlines():
    #     e = edge.decode().strip('\n').split(',')
    #
        edge_list_syn = pickle.load(f)
        edge_syn_lists.append(edge_list_syn)

    syn_graph = nx.from_edgelist(edge_list_syn, create_using=nx.Graph())
    deg =sorted((d for n, d in syn_graph.degree()), reverse=True)

    degrees.append(np.unique(deg, return_counts=True))

    # graphs.data.edge_index = to_undirected(graphs.data.edge_index)
    # edge_list_tensor = to_undirected(graphs.data.edge_index)
    # edge_list_tensor = edge_list_tensor.transpose(0, 1)
    # edge_list = [(int(i[0]), int(i[1])) for i in edge_list_tensor]
    # real_graph = nx.from_edgelist(edge_list)
    gini_coef.append(gini(nx.to_scipy_sparse_array(syn_graph)))
    power_stats.append(power_law_alpha(nx.to_scipy_sparse_array(syn_graph)))#sum(nx.triangles(syn_graph.to_undirected()).values()) / 3)
    assortativity.append(nx.degree_assortativity_coefficient(syn_graph))
    cluster.append(nx.average_clustering(syn_graph))

print(degrees[0])

with open('outputs/Synthetic_benchmark/DSBM_edge_list_sbm.pkl','rb') as f:
    edge_list_syn = pickle.load(f)
# plot lines
# plt.plot(list(range(10)), triangle_stats, label = "Unif 100 epochs", color ='blue',linestyle = 'dashed')

# df = pd.DataFrame({'assort.': assortativity,
#                    'real assort.':[-0.01099 for i in range(10)],
#                    'syn pow law':power_stats,
#                    'real pow law': [1.3613 for i in range(10)],
#                    'syn. clust. coef.':cluster,
#                    'real. clust. coef.':[0.39935 for i in range(10)],
#                    'gini coef.':gini_coef,
#                    'real gini coef.':[0.57105 for i in range(10)]})
#
#
#
#
# sns.set_style("darkgrid")
#
#
# fig,axs = plt.subplots(2,2)

# a=sns.histplot(degrees[0],ax=axs[0])

# g = sns.FacetGrid(data=df)#, palette=['red', 'red', 'blue', 'blue', 'purple', 'purple','green','green'])#,markers=True)
# g.map(plt.plot)




# g.set_xticks(range(len(df)))
# g.set_xticklabels([10*i for i in range(1,11,1)])
#
#
# axs[0,0].plot([10*i for i in range(1,11,1)], assortativity, label = "syn assort.", color ='blue', linestyle = 'dashed', marker = '*')
# axs[0,0].plot([10*i for i in range(1,11,1)], [-0.01099 for i in range(10)], label = "real assort.", color ='blue')
# axs[0,0].set_xlabel("Percentage of |E|")
# axs[0,0].set_ylabel('Assortativity')
# # axs[0,0].legend()
#
#
# axs[0,1].plot([10*i for i in range(1,11,1)], power_stats, label = "syn assort.", color ='purple', linestyle = 'dashed',marker = '*')
# axs[0,1].plot([10*i for i in range(1,11,1)], [1.3613 for i in range(10)], label = "real assort.", color ='purple')
# axs[0,1].set_xlabel("Percentage of |E|")
# axs[0,1].set_ylabel('Pow law alpha')
# # axs[0,1].legend()
#
# axs[1,0].plot([10*i for i in range(1,11,1)], cluster, label = "clust. coef.", color = 'orange',linestyle = 'dashed',marker = '*')
# axs[1,0].plot([10*i for i in range(1,11,1)], [0.39935 for i in range(10)], label = "real clust. coef.", color ='orange')
# axs[1,0].set_xlabel("Percentage of |E|")
# axs[1,0].set_ylabel('Clust. Coef.')
# # axs[1,0].legend()
#
# axs[1,1].plot([10*i for i in range(1,11,1)], gini_coef, label = "gini coef.", color = 'red',linestyle = 'dashed',marker = '*')
# axs[1,1].plot([10*i for i in range(1,11,1)], [0.57105 for i in range(10)], label = "real gini coef.", color ='red')
# axs[1,1].set_xlabel("Percentage of |E|")
# axs[1,1].set_ylabel('Gini Coef.')
# axs[1,1].legend()
#
# # plt.ylabel("metric")
# #


# #
# plt.legend(loc='upper left')
# plt.show()

edge_list_syn = np.load('outputs/Synthetic_benchmark/EmailEUCore_our_vae_generated_graph.npy')
         # edge_list_syn.append((int(e[0]),int(e[1])))
# edge_list_syn = nx.from_edgelist('outputs/Synthetic_benchmark/Cora_NetGAN_generated_graph.npy',create_using=nx.DiGraph())

#
base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')




# sbm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
#
# graphs = torch.load('datasets\data\direct_sbm_dataset.pt')

# # # #
# # #
# graphs = StochasticBlockModelDataset(base_path,block_sizes = [400,400,400,400],edge_probs = [[0.15,0.01,0.01,0.01],
#                                                                                              [0.01,0.15,0.01,0.01],
#                                                                                              [0.01,0.01,0.15,0.01],
#                                                                                              [0.01,0.01,0.01,0.15]])
# #
# graphs.data.num_nodes = 1600
# print(graphs.data)
# torch.save(graphs,'sbm_test_in_out_4_1600_nodes.pt')
# qgf
# graphs = torch.load('sbm_test_in_out_4_1600_nodes.pt')


#
# edge_list = [(int(i[0]), int(i[1])) for i in graphs.data.edge_index.transpose(0,1)]
#
# real_graph = nx.from_edgelist(edge_list, create_using=nx.Graph())
# graphs = Planetoid("\..", "Cora", transform=T.NormalizeFeatures())

# graphs = AttributedGraphDataset(base_path,"Wiki")
graphs = EmailEUCore(base_path)
# graphs = SNAPDataset(base_path,"ego-facebook").get(1)
#
#
#
# graphs.data = Data(x=graphs.x, edge_index=graphs.edge_index, n_nodes=graphs.num_nodes)
print(graphs.data)
# data = dataset[0]
# data.train_mask = data.val_mask = data.test_mask = data.y = None
# data = train_test_split_edges(data)

# print(data)

#sum(nx.triangles(G).values()) / 3

#syn_graph = nx.from_edgelist(random.choices(list(set(edge_list_syn)),k=27000), create_using=nx.Graph())
dir = False
if dir == True:

    syn_graph = nx.from_edgelist(edge_list_syn, create_using=nx.DiGraph())

    edge_list = [(int(i[0]), int(i[1])) for i in graphs.data.edge_index.transpose(0,1)]
    # syn_graph = nx.from_edgelist(edge_list)

    real_graph = nx.from_edgelist(edge_list, create_using=nx.DiGraph())

    print(len(list(syn_graph.nodes())))
    print(len(list(syn_graph.edges())))
if dir == False:

    syn_graph = nx.from_edgelist(edge_list_syn, create_using=nx.Graph())
    graphs.data.edge_index = to_undirected(graphs.data.edge_index)
    edge_list_tensor = to_undirected(graphs.data.edge_index)
    edge_list_tensor = edge_list_tensor.transpose(0,1)

    edge_list = [(int(i[0]),int(i[1])) for i in edge_list_tensor]
    # syn_graph = nx.from_edgelist(edge_list)
    real_graph = nx.from_edgelist(edge_list)
    print(graphs)
    print(len(list(syn_graph.nodes())))

    print(len(list(syn_graph.edges())))

print(compute_graph_statistics(real_graph))

print(compute_graph_statistics(syn_graph))



#
# print(f'Number of triangles in the real graph: {sum(nx.triangles(real_graph.to_undirected()).values()) / 3}')
# print(f'Number of triangles in the synthetic graph: {sum(nx.triangles(syn_graph.to_undirected()).values()) / 3}')
#
# print(f'Assortativity in the real graph: {nx.degree_assortativity_coefficient(real_graph)}')
# print(f'Assortativity in the synthetic graph: {nx.degree_assortativity_coefficient(syn_graph)}')
#
# print(f'Clustering coef in the real graph: {nx.average_clustering(real_graph)}')
# print(f'Clustering coef in the synthetic graph: {nx.average_clustering(syn_graph)}')
#
# degree_sequence_syn = sorted((d for n, d in syn_graph.degree()), reverse=True)
# degree_sequence_real = sorted((d for n, d in real_graph.degree()), reverse=True)
#
# #
# k_core_syn = sorted((d for n, d in nx.core_number(syn_graph.remove_edges_from(nx.selfloop_edges(syn_graph)))), reverse=True)
# k_core_real = sorted((d for n, d in nx.core_number(real_graph.remove_edges_from(nx.selfloop_edges(real_graph)))), reverse=True)
#
# print(f'core number in the real graph: {max(k_core_real)}')
# print(f'core number in the synthetic graph: {max(k_core_syn)}')
#
#
# print(f'number of squares in the real graph: {squares(real_graph)}')
# print(f'number of squares in the synthetic graph: {squares(syn_graph)}')
#
#
#
#
# print(f'Max degree coef in the real graph: {max(degree_sequence_real)}')
# print(f'Max degree coef in the synthetic graph: {max(degree_sequence_syn)}')





real_torch_graph = torch.tensor(np.array(list(real_graph.edges())).T,dtype=torch.long)


real_data = Data(edge_index = real_torch_graph, num_nodes = graphs.data.num_nodes)

degree_sequence_syn = sorted((d for n, d in syn_graph.degree()), reverse=True)
syn_distrib =  np.unique(degree_sequence_syn)*1./(sum(np.unique(degree_sequence_syn)))


degree_sequence_real = sorted((d for n, d in real_graph.degree()), reverse=True)
real_distrib = np.unique(degree_sequence_real) * 1. / (sum(np.unique(degree_sequence_real)))

#syn_torch_graph = from_networkx(syn_graph)

syn_torch_graph = torch.tensor(np.array(list(syn_graph.edges())).T,dtype=torch.long)

syn_torch_graph = Data(edge_index = syn_torch_graph, num_nodes = graphs.data.num_nodes)

#syn_torch_graph.edge_index = to_undirected(syn_torch_graph.edge_index)

# import numpy as np
# import matplotlib.pyplot as plt
# fig = plt.figure("Degree of a random graph", figsize=(8, 8))
#
# axgrid = fig.add_gridspec(5, 4)
# ax2 = fig.add_subplot(axgrid[0:3,:])
# ax2.bar(*np.unique(degree_sequence_syn, return_counts=True),color='red',label='Synthetic Graph')
# ax2.bar(*np.unique(degree_sequence_real, return_counts=True),label = 'Real Graph')
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("Number of Nodes")
# ax2.legend()
# fig.tight_layout()
# plt.show()

print(syn_torch_graph)


print(real_data)
syn_torch_graph = syn_torch_graph

syn_torch_graph.num_nodes=graphs.data.num_nodes

syn_torch_graph.x=torch.eye(graphs.data.num_nodes, dtype = torch.float32)#*1./graphs.data.num_nodes

data = train_test_split_edges(syn_torch_graph)

print(data.x)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


from torch_geometric.nn import VGAE
from torch_geometric.nn import GAE
#
# out_channels = 2
# num_features = 20
# epochs = 100
#
# # model
# model = GAE(GCNEncoder(num_features, out_channels))
#
# # move to GPU (if available)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# x = data.x.to(device)
# train_pos_edge_index = data.train_pos_edge_index.to(device)
#
# # inizialize the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)

    loss = loss + (1 / data.num_nodes) * model.kl_loss()  # new line
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

from torch.utils.tensorboard import SummaryWriter

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

out_channels = 2
num_features = graphs.data.num_nodes
epochs = 300
print('la')

model = VGAE(VariationalGCNEncoder(num_features, out_channels))  # new line

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

writer = SummaryWriter('runs/VGAE_experiment_' + '2d_100_epochs')

print('la')
graphs.data.x = torch.tensor([[1] for i in range(graphs.data.num_nodes)], dtype = torch.float32)


data_real = train_test_split_edges(real_data.to(device), test_ratio = 0.9)

for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    writer.add_scalar('auc train', auc, epoch)  # new line
    writer.add_scalar('ap train', ap, epoch)  # new line

auc, ap = test(data_real.test_pos_edge_index, data_real.test_neg_edge_index)
print('Final test vs real, AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))

qewf
a =  model.decoder.forward_all(model.encode(x,train_pos_edge_index))

uni = np.random.uniform(0,35,num_features*num_features).reshape(num_features,num_features)
b = torch.tensor(uni)<=a.cpu()
G =nx.from_numpy_matrix(np.array(b*1))
G.remove_edges_from(nx.selfloop_edges(G))
edge_list_samples = list(G.edges())



with open('vgae_eucore_edge_list.pkl', 'wb') as f:
    pickle.dump(edge_list_samples, f)

