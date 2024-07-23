import os

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, to_networkx
from torchvision.transforms import transforms

import config
from new.subgraph import found_subgraph, KM, entropy, construct_subgraph, new_construct_subgraph
from position_embedding import PositionalEncodingTransform


# 这里给出大家注释方便理解
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, file, name, km_file, cache_file, entropy_file, pre_transform=None, transform=None):
        self.file = file
        self.name = name
        self.km_file = km_file
        self.cache_file = cache_file
        self.entropy_file = entropy_file
        # self.pre_transform = pre_transform
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return []
    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        dataset = TUDataset(root=self.file, name=self.name, use_node_attr=True, pre_transform=self.pre_transform)
        node_list = config.new_x(self.name)

        # centers_list, delete_list = KM(dataset, 3, node_list, self.km_file)
        cache_file = self.cache_file
        subgraph_list = found_subgraph(dataset, node_list, cache_file)
        # sub_list, y_list = entropy(subgraph_list, centers_list, delete_list, dataset, self.entropy_file)
        # data_list = construct_subgraph(dataset, sub_list, y_list, delete_list)
        data_list = new_construct_subgraph(dataset, subgraph_list, node_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def get(self, idx):
    #     data = super().get(idx)
    #     processed_data = data
    #     raw_data = data.raw_data
    #     return processed_data, raw_data
def get_dataset(dataset, output_dir="./"):
    global data
    # pre_transform = PositionalEncodingTransform(
    #     lap_dim=8)
    print(f"Preprocessing {dataset}".upper())
    if not os.path.exists(os.path.join(output_dir, "dataset")):
        os.makedirs(os.path.join(output_dir, "dataset"))

    if dataset == "MUTAG":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/MUTAG", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='MUTAG', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_mutag.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_mutag.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_mutag.pkl')

    elif dataset == "ENZYMES":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/ENZYMES", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='ENZYMES', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_enzymes.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_enzymes.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_enzymes.pkl'
)
    elif dataset == "NCI1":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/NCI1", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='NCI1', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_nci1.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_nci1.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_nci1.pkl'
   )
    elif dataset == "IMDB-BINARY":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/IMDB-BINARY", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='IMDB-BINARY', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_imdbb.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_imdbb.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_imdbb.pkl'
            )
    elif dataset == "NCI109":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/NCI109", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='NCI109', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_nci109.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_nci109.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_nci109.pkl'
           )
    elif dataset == "PTC_MR":
        data = MyOwnDataset(
            "/home_b/zhaoke1/GT-K-FOLD-1.0/MYData/PTC", '/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='PTC_MR', km_file='/home_b/zhaoke1/GT-K-FOLD-1.0/km/km_ptc.pkl', cache_file='/home_b/zhaoke1/GT-K-FOLD-1.0/pickle/subgraph_cache_ptc.pkl', entropy_file='/home_b/zhaoke1/GT-K-FOLD-1.0/entropy/entropy_ptc.pkl'
        )
    return data
# def get_idx_split(data_size, train_size, valid_size, seed):
#     ids = shuffle(range(data_size), random_state=seed)
#     train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
#         ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
#     split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
#     ret
#     urn split_dict
#
# dataset = get_dataset("NCI1")
# dataset.process()
# split_idx = get_idx_split(len(dataset.data.y), train_size=math.floor(dataset.data.y.shape[0] * 0.8), valid_size=math.floor(dataset.data.y.shape[0] * 0.1), seed=20)
# train_loader = DataLoader(dataset[split_idx['train']], batch_size=64, shuffle=True)
# valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=64, shuffle=False)
# test_loader = DataLoader(dataset[split_idx['test']], batch_size=64, shuffle=False)
# for step, data in enumerate(train_loader):
#     adj_matrix_1 = to_dense_adj(data.edge_index)
#     print("adj",adj_matrix_1.shape)
#     print("x",data.x.shape)
# for data in dataset:
#     print(data)

# print(dataset.sub_list[0][0].x.size(1))
# print(dataset.sub_batch[0])
# for i,data in enumerate(dataset):
#     x = data.x
#     edge_index = data.edge_index
#     adj = to_dense_adj(edge_index)
#     # print(x.shape,adj.shape,edge_index.shape)
#     if x.size(0) != adj.size(1):
#         print(i)
#         print(data)
#         # print(x.shape, adj.shape, edge_index)
#         print("!!!")

# G1 = to_networkx(dataset[109], to_undirected=True)
# plt.figure(figsize=(8, 6))
# pos = nx.spring_layout(G1, seed=42)  # 使用 Spring layout 算法布局图形
# nx.draw(G1, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8)
# plt.title('Visualization of MUTAG Compound')
# plt.show()