import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

import config


class CycleFinder:
    def __init__(self, data, new_x, lengths=[4, 5, 6, 7]):
        self.data = data
        self.lengths = lengths
        self.new_x = new_x
        self.graph = to_networkx(data, to_undirected=True)
        self.visited_edges = set()

    def find_cycles(self):
        cycles = []
        for length in self.lengths:
            cycles.extend(self._find_cycles_of_length(length))
        return cycles

    def _find_cycles_of_length(self, length):
        local_cycles = []

        def dfs(node, start, visited, path):
            if len(path) == length:
                if path[0] in self.graph[path[-1]] and tuple(sorted((path[-1], path[0]))) not in self.visited_edges:
                    local_cycles.append(path[:])
                    for i in range(length):
                        edge = tuple(sorted((path[i], path[(i + 1) % length])))
                        self.visited_edges.add(edge)
                return

            for neighbor in self.graph[node]:
                if neighbor == start and len(path) == length - 1:
                    if tuple(sorted((node, neighbor))) not in self.visited_edges:
                        path.append(neighbor)
                        local_cycles.append(path[:])
                        for i in range(length):
                            edge = tuple(sorted((path[i], path[(i + 1) % length])))
                            self.visited_edges.add(edge)
                        path.pop()
                    return
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, start, visited, path)
                    path.pop()
                    visited.remove(neighbor)

        for node in self.graph.nodes():
            visited = set([node])
            dfs(node, node, visited, [node])

        return local_cycles

    def convert_cycles_to_data(self, cycles):
        def cycle_to_data(cycle):
            nodes = list(cycle)
            x = self.new_x[nodes]

            # Create undirected edges using original indices
            edges = [(u, v) for u, v in zip(cycle, cycle[1:] + [cycle[0]])]
            undirected_edges = edges + [(v, u) for u, v in edges]
            edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()

            return Data(x=x, edge_index=edge_index, original_indices=torch.tensor(nodes))

        return [cycle_to_data(cycle) for cycle in cycles]
    # def convert_cycles_to_data(self, cycles):
    #     def cycle_to_data(cycle):
    #         # 生成新的节点索引
    #         num_nodes = len(cycle)
    #         new_indices = list(range(num_nodes))
    #
    #         # 使用新的节点索引获取节点特征
    #         x = self.new_x[new_indices]
    #
    #         # 创建无向边
    #         edges = [(u, v) for u, v in zip(new_indices, new_indices[1:] + [new_indices[0]])]
    #         undirected_edges = edges + [(v, u) for u, v in edges]
    #         edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    #
    #         return Data(x=x, edge_index=edge_index, new_indices=torch.tensor(new_indices))
    #
    #     return [cycle_to_data(cycle) for cycle in cycles]

    def find_and_convert_cycles(self):
        cycles = self.find_cycles()
        return self.convert_cycles_to_data(cycles)

# # Example usage
# # Create a sample graph
# dataset = TUDataset(root='/home/zhaoke/struc2vec-k-fold/TUDataset', name='MUTAG', use_node_attr=True)
# data = dataset[0]
# # Create an instance of the CycleFinder class
# cycle_finder = CycleFinder(data, config.node_list[0])
#
# # Find and convert cycles to Data objects
# cycle_data_list = cycle_finder.find_and_convert_cycles()
#
# # Print the data objects for the cycles
# for cycle_data in cycle_data_list:
#     print("Cycle Data Object:")
#     print("Nodes features:", cycle_data.x)
#     print("Edge index:", cycle_data.edge_index)
#     print("Original node indices:", cycle_data.original_indices)

#