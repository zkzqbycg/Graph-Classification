"""
    这里是处理节点后的起点
    改代码读取emb文件，将节点嵌入转换成列表，node_list长度为图的个数，node_list[i] = tensor[num_nodes,16]
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from position_embedding import PositionalEncodingTransform, LapPE

torch.set_printoptions(profile="full")

BN = False

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        if self.residual:
            x = x + previous_x
        return x


node_list = []
nodes = []
def load_node_embeddings_without_header(file_path):
    index = 0
    node_embeddings = {}
    emb_list = []
    with open(file_path, 'r') as f:
        node_embeddings = {}
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) <= 2:
                index = i
                emb_list.append(node_embeddings)
                node_embeddings = {}
                continue  # 跳过不包含有效嵌入向量的行
            if i > index:
                node = int(parts[0])  # 假设节点名称是整数形式
                embedding = [float(x) for x in parts[1:]]
                node_embeddings[node] = embedding
    return emb_list

def nodes_degree(dataset):
    global degree_tensor

    lin = Linear(in_features=1, out_features=8)

    graph_list = []
    for graph in dataset:
        degree_list = []
        # 将DGL图转换为NetworkX图
        nx_g = to_networkx(graph, to_undirected=True)

        # 计算每个节点的度
        node_degrees = dict(nx_g.degree())

        # 输出每个节点的度
        for node, degree in node_degrees.items():
            # print(f"节点 {node} 的度为: {degree}")
            degree = torch.tensor([degree]).float()
            degree = lin(degree)
            degree_list.append(degree)
        degree_tensor = torch.stack(degree_list)
        graph_list.append(degree_tensor)
    return graph_list


def new_x(name):
    global file_path, dataset
    if name == "MUTAG":
        file_path = "/home_b/zhaoke1/GT-K-FOLD-1.0/emb/mutag.emb"
        dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='MUTAG', use_node_attr=True)
    elif name == "IMDB-BINARY":
        file_path = "/home_b/zhaoke1/GT-K-FOLD-1.0/emb/imdbb.emb"
        dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='IMDB-BINARY', use_node_attr=True)
    elif name == "NCI1":
        file_path = "/home_b/zhaoke1/GT-K-FOLD-1.0/emb/nci1.emb"
        dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='NCI1', use_node_attr=True)
    elif name == "NCI109":
        file_path = "/home_b/zhaoke1/GT-K-FOLD-1.0/emb/nci109.emb"
        dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='NCI109', use_node_attr=True)
    elif name == "PTC_MR":
        file_path = "/home_b/zhaoke1/GT-K-FOLD-1.0/emb/ptc.emb"
        dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='PTC_MR', use_node_attr=True)
    # 加载节点嵌入文件并转换为张量（跳过第一行）
    node_embeddings_list = load_node_embeddings_without_header(file_path)

    for i in range(len(node_embeddings_list)):
        if i != 0:
            # 将节点嵌入字典转换为张量
            node_ids = sorted(node_embeddings_list[i].keys())  # 按节点序号排序
            node_embeddings_tensor = torch.tensor([node_embeddings_list[i][node] for node in node_ids])
            node_list.append(node_embeddings_tensor)

    embedding_dim = 32  # 设定嵌入维度
    embedding_layer = nn.Embedding(num_embeddings=dataset.x.size(1), embedding_dim=embedding_dim)

    # 拼接表征
    if 'x' in dataset[0]:
        for i in range(len(dataset)):
            indices = torch.argmax(dataset[i].x, dim=1).long()
            x = embedding_layer(indices)
            positional_encoding = LapPE(dataset[i].edge_index, 16, dataset[i].num_nodes)
            node_list[i] = torch.cat((x, node_list[i], positional_encoding),dim=1)
    else:
        graph_list = nodes_degree(dataset)
        for i in range(len(dataset)):
            node_list[i] = torch.cat((graph_list[i],node_list[i]),dim=1)
    return node_list

# list = new_x("MUTAG")
# print(dataset)
# print(node_list[0].shape)

# # 测试是否节点数目变化的图
# dataset = TUDataset(root='/home/zhaoke/struc2vec-master/TUDataset', name='PROTEINS', use_node_attr=True)
# print(len(node_list))
# for i in range(len(node_list)):
#     if node_list[i].size(0) != dataset[i].num_nodes:
#         print("nonono",i)