import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        LU, pivots = torch.linalg.lu_factor(Q)
        P, L, U = torch.lu_unpack(LU, pivots)

        self.register_buffer("P", P)
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self, device):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W(x.device)
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z
      

class InvGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes, dropout):
        super(InvGNN, self).__init__()
        self.n_layers = n_layers

        self.fc = nn.Linear(input_dim, hidden_dim)

        lst = list()
        for i in range(n_layers):
            lst.append(nn.Sequential(Invertible1x1Conv(hidden_dim), nn.Sigmoid()))

        self.mlps = nn.ModuleList(lst)
        
        self.final_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, data):
        x, edge_index, exp_adj_flat = data.x, data.edge_index_exp_adj, data.exp_adj_flat
        exp_adj = torch.sparse_coo_tensor(edge_index, exp_adj_flat.squeeze(), torch.Size([x.size(0),x.size(0)])).to(x.device)    

        x = self.fc(x)
        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = torch.spmm(exp_adj, x)
            
        out = global_add_pool(x, data.batch)
        out = self.final_mlp(out)
        return F.log_softmax(out, dim=1)
