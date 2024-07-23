from itertools import combinations
import networkx as nx
from collections import deque

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def find_connected_subgraphs(graph, k, new_x):
    # Initialize an empty list to store the connected subgraphs' nodes as tensors
    connected_subgraphs = []
    seen_subgraphs = set()  # Set to keep track of seen subgraphs

    # Perform BFS from each node in the graph
    for node in graph.nodes():
        visited = set()  # Set to keep track of visited nodes
        queue = deque([node])  # Queue for BFS traversal
        while queue:
            current_node = queue.popleft()
            visited.add(current_node)
            # Get all neighbors of the current node
            neighbors = list(graph.neighbors(current_node))
            # Add unvisited neighbors to the queue
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        # Check if the visited nodes form a connected subgraph of size k
        for subset in combinations(visited, k):
            subgraph = graph.subgraph(subset)
            if subgraph.number_of_nodes() == k and nx.is_connected(subgraph):
                # Sort the subset to create a unique representation
                sorted_subset = tuple(sorted(subset))
                if sorted_subset not in seen_subgraphs:
                    seen_subgraphs.add(sorted_subset)
                    # Convert the subset to a tensor and add to the list
                    nodes = list(subset)
                    edges = list(subgraph.edges())
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    original_indices = torch.tensor(nodes, dtype=torch.long)
                    x = new_x[original_indices]
                    connected_subgraphs.append(Data(edge_index=edge_index, original_indices=original_indices, x=x))

    return connected_subgraphs


def extract_connected_graphlets(G, n, node_features):
    graphlets = []
    combination_batch_size = 1000  # 每次处理的组合数量
    combination_count = 0

    for nodes in combinations(G.nodes(), n):
        subgraph = G.subgraph(nodes)

        if nx.is_connected(subgraph):
            edges = list(subgraph.edges())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            x = torch.stack([node_features[node] for node in nodes])
            original_indices = torch.tensor(nodes, dtype=torch.long)

            data = Data(edge_index=edge_index, original_indices=original_indices, x=x)
            graphlets.append(data)
            combination_count += 1

            # 批量处理，防止一次性加载所有数据
            if combination_count % combination_batch_size == 0:
                print(combination_count)
                # 检查显存占用情况
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved_memory = torch.cuda.memory_reserved(0)
                    allocated_memory = torch.cuda.memory_allocated(0)
                    free_memory = reserved_memory - allocated_memory
                    print(f"Total Memory: {total_memory}, Free Memory: {free_memory}")

    return graphlets

def batched_combinations(iterable, r, batch_size):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    while True:
        yield tuple(pool[i] for i in indices)
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        if len(indices) % batch_size == 0:
            yield None


def find_graphlets_batch(graph, num_nodes, batch_size):
    nodes = list(graph.nodes())
    visited_combos = set()  # 用于记录已经访问过的组合
    for combo in batched_combinations(nodes, num_nodes, batch_size):
        if combo is None:
            continue
        sorted_combo = tuple(sorted(combo))  # 对组合进行排序，确保组合的顺序不影响结果
        if sorted_combo in visited_combos:
            continue
        visited_combos.add(sorted_combo)

        subgraph = graph.subgraph(sorted_combo)  # 使用排序后的组合生成子图
        if nx.is_connected(subgraph):
            yield subgraph


def graphlet_to_data(graphlet, original_indices, node_features):
    edge_index = torch.tensor(list(graphlet.edges)).t().contiguous()
    x = torch.stack([node_features[node] for node in original_indices])
    return Data(edge_index=edge_index, original_indices=original_indices, x=x)
# def graphlet_to_data(graphlet, node_features):
#     # 创建新的节点索引
#     original_indices = list(graphlet.nodes)
#     new_indices = {original: new for new, original in enumerate(original_indices)}
#
#     # 创建新的边索引
#     edge_index = torch.tensor([(new_indices[u], new_indices[v]) for u, v in graphlet.edges],
#                               dtype=torch.long).t().contiguous()
#
#     # 根据新的节点索引获取节点特征
#     x = torch.stack([node_features[original] for original in original_indices])
#     edge_index = to_undirected(edge_index)
#     return Data(edge_index=edge_index, x=x)

def is_tensor_contained(tensor_a, tensor_b):
    """
    Check if tensor_a is completely contained within tensor_b.

    Args:
    - tensor_a (torch.Tensor): The tensor to check if it is contained.
    - tensor_b (torch.Tensor): The tensor to check against.

    Returns:
    - bool: True if tensor_a is completely contained within tensor_b, False otherwise.
    """
    set_a = set(tensor_a.tolist())
    set_b = set(tensor_b.tolist())

    return set_a.issubset(set_b)

# dataset = TUDataset(root='/home/zhaoke/struc2vec-k-fold/TUDataset', name='MUTAG', use_node_attr=True)
# data = dataset[0]
# G = to_networkx(data,to_undirected=True)
# cycle_finder = CycleFinder(data, config.node_list[0])
#
# # Find and convert cycles to Data objects
# cycle_data_list = cycle_finder.find_and_convert_cycles()
# # Find all connected subgraphs of size 3
# # subgraphs1 = find_connected_subgraphs(G, 3, config.node_list[0])
# subgraphs2 = find_connected_subgraphs(G, 4, config.node_list[0])
# new_subgraph = []
# for subgraph in subgraphs2:
#     # print(subgraph.original_indices)
#     # print(subgraph.x)
#     tensor_a = subgraph.original_indices
#     # print(tensor_a)
#     found_match = False
#     for cycle_data in cycle_data_list:
#         tensor_b = cycle_data.original_indices
#         # print(tensor_b)
#         result = is_tensor_contained(tensor_a, tensor_b)
#         if result:
#             found_match = True
#             break
#     if not found_match:
#         print("remain", subgraph.original_indices)
#         new_subgraph.append(subgraph)
# for cycle_data in cycle_data_list:
#     new_subgraph.append(cycle_data)
# print(len(new_subgraph))
