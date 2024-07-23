import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, to_undirected, to_dense_adj, from_networkx
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

import random_walk
from new.cycle import CycleFinder
from new.graphlet import find_graphlets_batch, graphlet_to_data

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))  # 指定使用GPU设备

def KM(dataset, n_clusters, X_list, save_path):
    # 如果文件存在，加载并返回结果
    if os.path.exists(save_path):
        print(f"Loading km list from {save_path}...")
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        return results['centers_list'], results['delete_list']
    print("Generating km list and saving to cache...")
    centers_list = []
    delete_list = []

    for i in range(len(X_list)):
        if X_list[i].size(0) <= 10 or dataset[i].num_nodes != X_list[i].size(0):
            delete_list.append(i)
            centers_list.append(0)
            continue
        X = X_list[i].to(device)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X.cpu().detach().numpy())
        centers = torch.Tensor(kmeans.cluster_centers_).to(device)
        centers_list.append(centers)

    results = {'centers_list': centers_list, 'delete_list': delete_list}
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    return centers_list, delete_list
def kmeans(X, num_clusters, num_iters=100):
    # 随机初始化聚类中心
    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices]

    for i in range(num_iters):
        # 计算每个点到每个聚类中心的距离
        distances = torch.cdist(X, centroids)

        # 将每个点分配到最近的聚类中心
        cluster_assignments = distances.argmin(dim=1)

        # 重新计算聚类中心
        new_centroids = torch.stack([X[cluster_assignments == j].mean(dim=0) for j in range(num_clusters)])

        # 检查是否收敛
        if torch.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return cluster_assignments, centroids

def reindex_graph(original_data):
    # Step 1: Determine unique nodes in the subgraph
    unique_nodes = torch.unique(torch.cat((original_data.edge_index[0], original_data.edge_index[1])))

    # Step 2: Create a mapping from original node indices to new indices starting from 0
    node_map = {node.item(): index for index, node in enumerate(unique_nodes)}

    # Step 3: Update edge_index with new indices
    edge_index_reindexed = torch.tensor([[node_map[node.item()] for node in original_data.edge_index[0]],
                                         [node_map[node.item()] for node in original_data.edge_index[1]]],
                                        dtype=torch.long)

    edge_index_reindexed = to_undirected(edge_index_reindexed)
    # Step 4: Create a new Data object with reindexed nodes and edges
    reindexed_data = Data(x=original_data.x, edge_index=edge_index_reindexed)

    return reindexed_data

def found_subgraph(dataset, new_x, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading subgraph list from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print("Generating subgraph list and saving to cache...")

    subgraph_list = []  # 使用None占位符预先分配列表大小

    def process_data(i):
        try:
            data = dataset[i].to(device)
            x = new_x[i].to(device)
            G = to_networkx(data, to_undirected=True)
            G_copy = G.copy()
            cycle_finder = CycleFinder(data, x)
            cycle_data_list = cycle_finder.find_and_convert_cycles()
            # sampled_graphs = []
            # for n in range(G.number_of_nodes()):
            #     sample1 = random_walk.random_walk_sample(G, i, n, 10, x)
            #     # g_pyg = pyg_utils.from_networkx(sample1)
            #     sampled_graphs.append(sample1)
            # print(i, len(cycle_data_list))
            for cycle_data in cycle_data_list:
                # print(cycle_data.edge_index)
                # print(cycle_data.original_indices)
                # exit()
                cycle_graph = to_networkx(cycle_data, to_undirected=True)
                G.remove_nodes_from(cycle_graph.nodes)
            new_subgraph = []

            # isolates = list(nx.isolates(G))
            # if len(isolates) > 0:
            #     # print(x)
            #     for isolate in isolates:
            #         # 找到指定节点的邻居
            #         # 找到 isolate 节点的一阶邻居
            #         first_neighbors = list(G_copy.neighbors(isolate))
            #
            #         # # 找到二阶邻居
            #         # second_neighbors = set()
            #         # for neighbor in first_neighbors:
            #         #     second_neighbors.update(G_copy.neighbors(neighbor))
            #         #
            #         # # 移除原始节点和一阶邻居
            #         # second_neighbors.discard(isolate)
            #         # second_neighbors -= set(first_neighbors)
            #         if len(first_neighbors) == 0:
            #             continue
            #         # print(f"节点 {isolate} 的邻居节点:", neighbors)
            #         # second_neighbors_list = list(second_neighbors)
            #         # 包括指定节点和其邻居节点构成子图
            #         subgraph_nodes = [isolate] + first_neighbors
            #         # print(i, subgraph_nodes)
            #         x_sub = []
            #         for node in subgraph_nodes:
            #             x_sub.append(x[node])
            #
            #         x_tensor = torch.stack(x_sub, dim=0)
            #
            #         subgraph = G_copy.subgraph(subgraph_nodes)
            #         # print(subgraph)
            #
            #         # 打印子图的节点和边
            #         # print("子图的节点:", subgraph.nodes())
            #         # print("子图的边:", subgraph.edges())
            #         edge_index = torch.tensor(list(subgraph.edges), dtype=torch.long).t().contiguous()
            #         pyg_subgraph = Data(x=x_tensor, edge_index=edge_index)
            #         # print(pyg_subgraph.edge_index)
            #         refixed_graph = reindex_graph(pyg_subgraph)
            #         # print(refixed_graph.edge_index)
            #         new_subgraph.append(refixed_graph)


            all_subgraphs_1 = []

            all_subgraphs_2 = []
            for graphlet in find_graphlets_batch(G, 3, 500):
                original_indices = list(graphlet.nodes())
                original_indices = torch.tensor(original_indices, dtype=torch.long)
                data_sub = graphlet_to_data(graphlet, original_indices, x)
                all_subgraphs_1.append(data_sub)
                # print(len(sampled_graphs))



            if len(all_subgraphs_1) == 0:
                for graphlet in find_graphlets_batch(G, 2, 500):
                    original_indices = list(graphlet.nodes())
                    original_indices = torch.tensor(original_indices, dtype=torch.long)
                    data_sub = graphlet_to_data(graphlet, original_indices, x)
                    all_subgraphs_2.append(data_sub)

            # print(i,"all subgraphs 1", len(all_subgraphs_1))
            # print(i, "all subgraphs 2", len(all_subgraphs_2))
            all_subgraphs = all_subgraphs_1 + all_subgraphs_2

            for subgraph in all_subgraphs:

                new_subgraph.append(subgraph)

            for cycle_data in cycle_data_list:
                # refixed_cycle_graph = reindex_graph(cycle_data)
                # print(refixed_cycle_graph.edge_index)
                new_subgraph.append(cycle_data)

            # if len(new_subgraph) == 0:
            #     new_subgraph.append(data)

            return new_subgraph

        except Exception as e:
            print(f"Error processing data at index {i}: {e}")
            raise

        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     futures = {executor.submit(process_data, i): i for i in range(len(dataset))}
        #     process = 0
        #     for future in as_completed(futures):
        #         i = futures[future]
        #         try:
        #             subgraph_list[i] = future.result()
        #             process += 1
        #             print(f"Processed {process}/{len(dataset)}")  # 打印处理进度
        #         except Exception as exc:
        #             print(f"Graph {i} generated an exception: {exc}")
    for i in range(len(dataset)):
        subgraph = process_data(i)
        subgraph_list.append(subgraph)
        print(f"Processed {i+1}/{len(dataset)}")

        with open(cache_file, 'wb') as f:
            pickle.dump(subgraph_list, f)

        torch.cuda.empty_cache()

    return subgraph_list

def new_construct_subgraph(dataset, subgraph_list, node_list):
    all_graphs = []
    for n in range(len(subgraph_list)):
        print(n, len(subgraph_list[n]))
        one_all_subgraph = []
        for subgraph in subgraph_list[n]:
            subgraph.y = dataset[n].y
            one_all_subgraph.append(subgraph)
        batch = Batch.from_data_list(one_all_subgraph)
        data = Data(sub_batch=batch, y=dataset[n].y, x=node_list[n], edge_index=dataset[n].edge_index)
        all_graphs.append(data)
    return all_graphs

def entropy(subgraph_list, centers_list, delete_list, dataset, save_path): # 如果文件存在，加载并返回结果
    if os.path.exists(save_path):
        print(f"Loading entropy list from {save_path}...")
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        return results['sub_list'], results['y_list']
    print("Generating entropy list and saving to cache...")
    y_list = []
    sub_list = []

    for n in range(len(subgraph_list)):
        if n in delete_list:
            y_list.append(0)
            sub_list.append(0)
            continue

        remain_subgraph = []
        centers = centers_list[n].to(device)

        for subgraph in subgraph_list[n]:
            x1 = torch.mean(subgraph.x.to(device), dim=0)
            similarities = torch.cdist(x1.unsqueeze(0), centers, p=2)
            similarities = F.normalize(1 / similarities, p=2, dim=1)
            probabilities = F.softmax(similarities, dim=1)
            probabilities_np = probabilities.cpu().detach().numpy()
            print("probabilities:", probabilities_np)
            entropy_value = -np.sum(probabilities_np * np.log2(probabilities_np))
            entropy_value = 1 / (1 + np.exp(-entropy_value))
            print("entropy:", entropy_value)
            if entropy_value < 0.828:
                remain_subgraph.append(subgraph)
            else:
                print(subgraph.edge_index)

        sub_list.append(remain_subgraph)
        y_list.append(dataset[n].y)

    results = {'sub_list': sub_list, 'y_list': y_list}
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    return sub_list, y_list


def find_isolated_nodes(data):
    num_nodes = data.x.size(0)
    edge_index = data.edge_index

    # 创建一个零初始化的度数数组
    degree = torch.zeros(num_nodes, dtype=torch.long)

    # 计算每个节点的度数
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        degree[src] += 1
        degree[dst] += 1

    # 找到度数为零的节点
    isolated_nodes = (degree == 0).nonzero(as_tuple=True)[0]

    return isolated_nodes

def construct_subgraph(dataset, sub_list, y_list, delete_list):
    new_graph_list = []
    for n in range(len(dataset)):
        if n in delete_list or len(sub_list[n]) == 0:
            continue
        original_graph = dataset[n].to(device)
        print("construct_subgraph", n)

        new_node_features = []
        new_edges = []

        for subgraph in sub_list[n]:
            avg_features = torch.mean(subgraph.x.to(device), dim=0, keepdim=True)
            new_node_features.append(avg_features)

        new_node_features = torch.cat(new_node_features, dim=0)

        nodes_indices_list = [subgraph.edge_index.to(device).flatten().unique() for subgraph in sub_list[n]]

        for i, nodes_i in enumerate(nodes_indices_list):
            for j, nodes_j in enumerate(nodes_indices_list):
                if i >= j:
                    continue

                if len(set(nodes_i.tolist()).intersection(set(nodes_j.tolist()))) > 0:
                    new_edges.extend([[i, j], [j, i]])
                    continue

                for ni in nodes_i:
                    neighbors = torch.cat([original_graph.edge_index[1][original_graph.edge_index[0] == ni],
                                           original_graph.edge_index[0][original_graph.edge_index[1] == ni]]).unique()
                    if len(set(neighbors.tolist()).intersection(set(nodes_j.tolist()))) > 0:
                        new_edges.extend([[i, j], [j, i]])
                        break
        if len(new_edges) > 0:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
            # 计算每个节点的度
            edge_index = new_edge_index
            # edge_index 是形状为 [2, num_edges] 的张量
            # 我们需要统计每个节点在 edge_index 中出现的次数

            node_degrees = torch.zeros(new_node_features.size(0), dtype=torch.long)
            for edge in edge_index.t():
                node_degrees[edge[0]] += 1
                node_degrees[edge[1]] += 1

            # 计算平均度
            average_degree = node_degrees.float().mean().item()

            print(f"Average degree of the graph: {average_degree}")
            # if average_degree >= 15.0:
            #     num_edges = new_edge_index.size(1)
            #     num_edges_to_remove = int(0.2 * num_edges)
            #
            #     # 确保随机种子的一致性（可选）
            #     random.seed(42)
            #
            #     # 随机选择要删除的边的索引
            #     edges_indices = list(range(num_edges))
            #     edges_to_remove = random.sample(edges_indices, num_edges_to_remove)
            #
            #     # 创建新的 edge_index，移除选定的边
            #     mask = torch.ones(num_edges, dtype=torch.bool)
            #     mask[edges_to_remove] = False
            #
            #     remove_edge_index = new_edge_index[:, mask]
            #
            #     # 更新 new_graph 的 edge_index
            #     new_edge_index = remove_edge_index
            #
            new_edge_index = to_undirected(new_edge_index)
            y = y_list[n].to(device)
            new_graph = Data(x=new_node_features, edge_index=new_edge_index, y=y, raw_data=dataset[n])
            # adj_mat = to_dense_adj(new_graph.edge_index)
            new_graph_list.append(new_graph)

    return new_graph_list