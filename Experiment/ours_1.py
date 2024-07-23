import math
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GraphConv
from torch_geometric.utils import to_dense_adj

class CustomModel(nn.Module):
    def __init__(self, k):
        super(CustomModel, self).__init__()
        self.U = nn.Parameter(torch.randn(k, k))  # 可学习参数 U，维度为 k * d
        self.Z = nn.ReLU()  # 你可以选择其他激活函数

    def forward(self, A, H):
        A_t = A.t()  # A 的转置
        intermediate = torch.matmul(A_t, H)  # A^t * H
        Z_output = self.Z(torch.matmul(self.U, intermediate))  # Z(U * (A^t * H))
        R = Z_output + intermediate  # Z(U * (A^t * H)) + A^t * H
        return R
class GCN(torch.nn.Module):

    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        # self.q = torch.nn.Parameter(torch.randn(hidden_channels))  # 可学习的向量q
        self.lin2 = Linear(hidden_channels, 1)
        self.lin3 = Linear(num_classes, hidden_channels)
        # self.l1_norm = torch.nn.L1Loss(reduction='sum')

    def forward(self, subgraph):
        x = subgraph.x

        edge_index = subgraph.edge_index

        # batch = subgraph.batch

        x = self.conv1(x, edge_index)

        x = x.relu()

        x = self.conv2(x, edge_index)

        x = x.relu()

        # x = global_mean_pool(x, batch)
        return x


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]

    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)

    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]

    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):

        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)

        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data

        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))
        self.lin = nn.Linear(64, out_channels)
        self.GCN = GCN(64, in_channels, out_channels)
        self.R = CustomModel(3)
        self.w = nn.Parameter(torch.randn(hidden_channels, 3))

    def forward(self, data):
        # A = torch.matmul(data.x, self.w)
        # R = self.R(A, data.x)
        # S = torch.matmul(A, R)
        # data.x = S
        graph_x = self.GCN(data)
        data.x = graph_x


        global graph_tensor
        graph = []
        subgraph = data.sub_batch
        for i in range(len(subgraph)):
            sub_x = data[i].x[subgraph[i].original_indices]

            # print(sub_x.shape)
            # print(subgraph[i].batch)
            sub_x = global_mean_pool(sub_x, subgraph[i].batch)
            # sub_x = self.GCN(subgraph[i])
            x1 = self.trans_conv(sub_x)
            x1 = global_mean_pool(x1, None)
            graph.append(x1)
        graph_tensor = torch.cat(graph)
        p_x = self.lin(graph_tensor)
        p_x = F.softmax(p_x, dim=-1)
        if self.use_graph:
            x_gnn = self.gnn(data)
            x2 = x_gnn
            q_x = self.lin(x2)
            q_x = F.softmax(q_x, dim=-1)
            # x_gnn = self.lin(x_gnn)
            # x_sub = self.trans_conv(x_gnn)
            # x_sub = global_mean_pool(x_sub, batch=None)

            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * graph_tensor
            else:
                x = torch.cat((graph_tensor, x2), dim=1)

        else:
            x = 0

        x = self.fc(x)
        return x, p_x, q_x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


def accuracy(prediction, labels):
    _, indices = torch.max(prediction, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def val(model, data_loader, args):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list = []
    loss_list = []
    for data in data_loader:
        # loss_F1_graph = []
        pred_graph = []
        data = data.to(args.device)
        # pred = model(data)
        for i in range(len(data.sub_batch)):
            # subgraph = data[i].sub_batch
            # pred = model(data)
            out = model(data[i])
            pred_graph.append(out)
            # loss_F1_graph.append(loss_F1)
        pred_graph_batch = torch.cat(pred_graph, dim=0)
        acc_list.append(accuracy(pred_graph_batch, data.y))
        loss_list.append(loss_func(pred_graph_batch, data.y).detach().item())
    acc = np.average(acc_list)
    # loss = np.average(loss_list)
    loss = np.sum(loss_list)
    return acc, loss