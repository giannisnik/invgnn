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
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

        self.bias = nn.Parameter(torch.zeros(dim))

    def _assemble_W(self, device):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W(x.device)
        z = x @ W + self.bias
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z

class InvGNN(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_classes, dropout):
        super(InvGNN, self).__init__()
        self.n_layers = n_layers

        self.atom_encoder = AtomEncoder(hidden_dim)

        lst = list()
        for i in range(n_layers):
            lst.append(nn.Sequential(Invertible1x1Conv(hidden_dim), nn.Sigmoid()))

        self.mlps = nn.ModuleList(lst)
        
        self.final_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, data):
        x, edge_index, exp_adj_flat = data.x, data.edge_index_exp_adj, data.exp_adj_flat
        exp_adj = torch.sparse_coo_tensor(edge_index, exp_adj_flat.squeeze(), torch.Size([x.size(0),x.size(0)])).to(x.device)    

        x = self.atom_encoder(x)
        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = torch.spmm(exp_adj, x)
            
        out = global_add_pool(x, data.batch)
        out = self.final_mlp(out)
        return out


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim, optional_full_atom_features_dims=None):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = nn.ModuleList()
        if optional_full_atom_features_dims is not None:
            full_atom_feature_dims = optional_full_atom_features_dims
        else:
            full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])
        return x_embedding


class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        full_bond_feature_dims = get_bond_feature_dims()
        self.bond_embedding_list = nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding 


# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))
