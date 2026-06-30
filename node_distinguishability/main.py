import argparse
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from model import InvGNN

def run_wl(dataset, n_wl_iters):
    """
    Computes Weisfeiler-Lehman (WL) node colors for every graph
    in the dataset.
    """
    wl_colors = []

    # Dictionary mapping WL signatures to unique integer colors
    # for each iteration
    colors = [dict() for i in range(n_wl_iters)]

    # Stores the parent-child relationships between colors
    # across WL iterations
    children = dict()

    # Process every graph independently
    for idx in range(len(dataset)):

        # Convert PyG graph into NetworkX graph
        G = to_networkx(dataset[idx], to_undirected=True)

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        # Tensor storing each node's color at every iteration
        x = torch.zeros(G.number_of_nodes(), n_wl_iters, dtype=torch.long)

        # Initially all nodes have the same color
        node_colors = {v: '0' for v in G.nodes()}

        # Perform WL color refinement
        for i in range(n_wl_iters):
            new_colors = {}

            # Update every node's color
            for j,node in enumerate(G.nodes()):

                # Collect neighboring colors
                color_list = []
                for nbr in G.neighbors(node):
                    color_list.append(node_colors[nbr])

                # WL signature:
                # current color + sorted multiset of neighbor colors
                color = node_colors[node] + '_' + ",".join(sorted(color_list))

                # Assign a new integer label to unseen signatures
                if color not in colors[i]:
                    colors[i][color] = str(len(colors[i]))

                    # Record the refinement tree
                    if str(i)+'-'+node_colors[node] not in children:
                        children[str(i)+'-'+node_colors[node]] = {str(i+1)+'-'+colors[i][color]: 0}
                    else:
                        children[str(i)+'-'+node_colors[node]][str(i+1)+'-'+colors[i][color]] = len(children[str(i)+'-'+node_colors[node]]) 
                
                # Save updated color
                new_colors[node] = colors[i][color]
                x[j,i] = int(colors[i][color])

            # Replace colors for next iteration
            node_colors = new_colors
        
        wl_colors.append(x)

    return wl_colors


class AddFeature:
    """
    Adds the precomputed WL colors to each graph object.
    """
    def __init__(self, wl_colors):
        self.wl_colors = wl_colors

    def __call__(self, data, idx):
        data.wl_colors = self.wl_colors[idx]
        return data


class GraphOperator(object):
    """
    Computes the matrix exponential of the graph adjacency matrix
    and initializes node features  to ones.
    """
    def __init__(self, norm_factor=1.):
        self.norm_factor = norm_factor

    def __call__(self, data):
        # Convert PyG graph into NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        # Normalize adjacency matrix
        A = nx.to_numpy_array(G)/self.norm_factor

        # Eigenvalue decomposition of adjacency matrix
        L,U = np.linalg.eigh(A)

        # Compute exp(A) = U exp(L) U^T
        exp_adj = np.linalg.multi_dot((U, np.diag(np.exp(L)), U.T))

        # Store only nonzero entries
        row, col = np.where(exp_adj>0)
        edge_index_exp_adj = torch.tensor(np.array([row, col]), dtype=torch.long)
        exp_adj = np.asarray(exp_adj)
        exp_adj_flat = torch.from_numpy(exp_adj[row,col]).unsqueeze(1).double()

        # Store transformed graph attributes
        data.edge_index_exp_adj = edge_index_exp_adj
        data.exp_adj_flat = exp_adj_flat

        # Initialize node features to ones
        data.x = torch.ones(G.number_of_nodes(), 1).double()

        return data


# Argument parser
parser = argparse.ArgumentParser(description='InvGNN')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--n-layers', type=int, default=4, help='Number of GNN lalyers')
parser.add_argument('--hidden-dim', type=int, default=4, help='Size of hidden layer of NN')
parser.add_argument('--activation', default='sigmoid', choices=['relu','leaky_relu', 'sigmoid', 'tanh', 'silu'], help='Activation function')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

norm_factor = 2.469
dataset = TUDataset(root='./datasets/MUTAG', name='MUTAG', transform=GraphOperator(norm_factor))

# Compute WL colors for every graph
wl_colors = run_wl(dataset, args.n_layers)

# Add WL colors to each graph
dataset = [AddFeature(wl_colors)(dataset[i], idx=i) for i in range(len(dataset))]
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
model = InvGNN(1, args.hidden_dim, args.n_layers, args.activation).to(device)
model = model.double()

model.eval()
wl_colors = []
z = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)

        # Compute node representations
        x = model(data)

        # Store node representations
        z.append(x.detach())

        # Store final WL colors.
        wl_colors.append(data.wl_colors[:,-1])

# Concatenate results from all batches
wl_colors = torch.cat(wl_colors, dim=0).unsqueeze(1)
z = torch.cat(z, dim=0)

# Upper-triangular indices avoid duplicate comparisons
rows, cols = torch.triu_indices(
    row=z.size(0),
    col=z.size(0),
    offset=1
)

# Indicator of whether two nodes have different final WL colors
wl_colors_diff = (wl_colors != wl_colors.T).long()
triu_wl_colors_diff = wl_colors_diff[rows,cols]

# Pairwise distances between learned embeddings
diff = torch.cdist(z, z)
z_diff = 1 - (diff <= 1e-7).long()
triu_z_diff = z_diff[rows,cols]

# Count pairs that are distinguished by both WL and the model
count = (triu_z_diff[triu_wl_colors_diff == 1] == 1).sum().item()

print((triu_wl_colors_diff== 1).sum().item() - count, 'missed pairs out of', (triu_wl_colors_diff== 1).sum().item(), 'pairs')
