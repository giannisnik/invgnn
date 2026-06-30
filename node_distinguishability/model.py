import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper, with bias.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        LU, pivots = torch.linalg.lu_factor(Q)
        P, L, U = torch.lu_unpack(LU, pivots)

        self.register_buffer("P", P)
        self.L = nn.Parameter(L)
        self.S = nn.Parameter(U.diag())
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

        # Bias term
        self.bias = nn.Parameter(torch.zeros(dim))

    def _assemble_W(self, device):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.dim, device=device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W(x.device)
        z = x @ W + self.bias
        return z

class InvGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, activation):
        super(InvGNN, self).__init__()
        self.n_layers = n_layers

        self.fc = nn.Linear(input_dim, hidden_dim)

        lst = list()
        for i in range(n_layers):
            lst.append(Invertible1x1Conv(hidden_dim))

        self.fcs = nn.ModuleList(lst)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        
    def forward(self, data):
        x, edge_index_exp, exp_adj_flat = data.x, data.edge_index_exp_adj, data.exp_adj_flat
        exp_adj = torch.sparse_coo_tensor(edge_index_exp, exp_adj_flat.squeeze(), torch.Size([x.size(0),x.size(0)])).to(x.device)    
        edge_index, exp_adj_flat = data.edge_index, data.exp_adj_flat
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=x.device), torch.Size([x.size(0),x.size(0)])).to(x.device)

        x = self.fc(x)
        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.act(x)
            x = torch.spmm(exp_adj, x)

        return x