import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(out_channels, hidden_channels)
        # self.lin = nn.Linear(, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x=conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, data.edge_index)

        x = self.lin2(x)

        probabilities = F.softmax(x, dim=-1)  # 应用Softmax以输出概率分布

        entropy = self.calculate_entropy(probabilities)  # 计算熵

        weights = self.normalize_entropy_to_weights(entropy)
        weights = F.softmax(weights, dim=0)

        weights = weights.unsqueeze(1)
        graph = torch.sum(x * weights, dim=0, keepdim=True)
        graph = self.lin3(graph)

        return graph

    def calculate_entropy(self, probabilities):
        return -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # 计算熵

    def normalize_entropy_to_weights(self, entropy):
        sum_entropy = torch.sum(entropy)
        # 计算权重，熵越大权重越小
        epsilon = 1e-10
        weights = 1 / (entropy + epsilon)
        # Min-Max 归一化
        min_weight = torch.min(weights)
        max_weight = torch.max(weights)
        normalized_weights = (weights - min_weight) / (max_weight - min_weight)

        return normalized_weights

