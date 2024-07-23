import numpy as np
import torch
import torch_geometric
from scipy import sparse as sp


class PositionalEncodingTransform(object):
    def __init__(self, lap_dim=0):
        super().__init__()
        self.lap_dim = lap_dim

    def __call__(self, data):
        data.lap_pos_enc = LapPE(
            data.edge_index, self.lap_dim, data.num_nodes)
        return data


def LapPE(edge_index, pos_enc_dim, num_nodes):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.cpu().numpy().clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros

    return PE