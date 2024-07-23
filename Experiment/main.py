#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    输入源数据集，经过struc2vec得到新的节点嵌入。
    mutag，proteins，enzymes, imdb-binary, reddit-binary
    每次嵌入数据集： nohup  python src/main.py > 1.log 2>&1 &命令不变，只需要更换main-TUDataset-name和
                                                                            learn_embeddings-model.wv.save_word2vec_format路径
                                                                            arg.--dimensions更改得到的嵌入维度
    ？？？
    问题是：目前proteins数据集经过嵌入后有五张图左右节点数目变少？
"""
import argparse
import logging

import torch
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from torch_geometric.datasets import TUDataset

import graph
import struc2vec

logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

def parse_args():
    '''
	Parses the struc2vec arguments.
	'''
    parser = argparse.ArgumentParser(description="Run struc2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate-mirrored.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=16,
                        help='Number of dimensions. Default is 16.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool,
                        help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool,
                        help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                        help='optimization 3')
    return parser.parse_args()


def read_graph(graph_input):
    '''
	Reads the input network.
	'''
    logging.info(" - Loading graph...")
    G = graph.from_pyg(graph_input, undirected=True)
    logging.info(" - Graph loaded.")
    return G


def learn_embeddings():
    '''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                     workers=args.workers, epochs=args.iter)
    model.wv.save_word2vec_format('emb/Mutagenicity.emb', append=True)
    logging.info("Representations created.")

    return


def exec_struc2vec(graph_input, args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    if (args.OPT3):
        until_layer = args.until_layer
    else:
        until_layer = None

    G = read_graph(graph_input)
    G = struc2vec.Graph(G, args.directed, args.workers, untilLayer=until_layer)

    if (args.OPT1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if (args.OPT2):
        G.create_vectors()
        G.calc_distances(compactDegree=args.OPT1)
    else:
        G.calc_distances_all_vertices(compactDegree=args.OPT1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()

    G.simulate_walks(args.num_walks, args.walk_length)

    return G


def load_node_embeddings_without_header(file_path):
    node_embeddings = {}
    with open(file_path, 'r') as f:
        next(f)  # 跳过第一行
        for line in f:
            parts = line.strip().split()
            node = int(parts[0])  # 假设节点名称是整数形式
            embedding = [float(x) for x in parts[1:]]
            node_embeddings[node] = embedding
    return node_embeddings

def main(graph_input, args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    graph_input = graph_input.to(device)
    G = exec_struc2vec(graph_input, args)

    learn_embeddings()


if __name__ == "__main__":

    args = parse_args()
    dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='Mutagenicity', use_node_attr=True)
    # 对每一张图
    # for i in range(len(dataset)):
    #     # struc2vec进行嵌入
    #     data = dataset[i]
    #     G = nx.Graph()
    #
    #     # 使用 add_nodes_from 批处理的效率比 add_node 高
    #     G.add_nodes_from([i for i in range(data.x.shape[0])])
    #
    #     # 使用 add_edges_from 批处理的效率比 add_edge 高
    #     edges = np.array(data.edge_index.T, dtype=int)
    #     G.add_edges_from(edges)
    #     main(G, args)
    for data in dataset:
        G = data
        main(G,args)

    # IMDB-BINARY/REDDIT-BINARY
    # 对每一张图
    # for i in range(len(dataset)):
    #     # struc2vec进行嵌入
    #     data = dataset[i]
    #     G = nx.Graph()
    #
    #     # 使用 add_nodes_from 批处理的效率比 add_node 高
    #     G.add_nodes_from([i for i in range(data.num_nodes)])
    #
    #     # 使用 add_edges_from 批处理的效率比 add_edge 高
    #     edges = np.array(data.edge_index.T, dtype=int)
    #     G.add_edges_from(edges)
    #     main(G, args)
