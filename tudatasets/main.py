import os
import logging
import argparse
import json
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx

from model import GNN
 
unlabeled_datasets = ['IMDB-BINARY', 'IMDB-MULTI']

avg_lambda_max = {'MUTAG': [2.469, 2.466, 2.464, 2.471, 2.464, 2.468, 2.471, 2.473, 2.464, 2.471],
                'ENZYMES': [4.341, 4.343, 4.334, 4.338, 4.338, 4.346, 4.343, 4.345, 4.351, 4.338],
                'NCI1': [2.492, 2.491, 2.491, 2.491, 2.493, 2.492, 2.492, 2.49, 2.492, 2.491],
                'PROTEINS_full': [4.171, 4.174, 4.169, 4.17, 4.166, 4.156, 4.177, 4.167, 4.165, 4.174], 
                'IMDB-BINARY': [10.078, 9.841, 10.138, 9.985, 10.2, 10.062, 10.1, 9.926, 10.074, 9.949],
                'IMDB-MULTI': [8.653, 8.475, 8.672, 8.442, 8.561, 8.559, 8.677, 8.526, 8.558, 8.565]}

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0.


    def early_stop(self, validation_acc):
        if validation_acc >= self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Degree(object):
    def __call__(self, data):
        idx = data.edge_index[0]
        deg = degree(idx, data.num_nodes, dtype=torch.float)
        data.x = deg.unsqueeze(1)
        return data

class GraphOperator(object):
    def __init__(self, norm_factor=1.):
        self.norm_factor = norm_factor

    def __call__(self, data):
        G = to_networkx(data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_numpy_array(G)/self.norm_factor
        L,U = np.linalg.eigh(A)
        exp_adj = np.linalg.multi_dot((U, np.diag(np.exp(L)), U.T))
        row, col = np.where(exp_adj>0)
        edge_index_exp_adj = torch.tensor(np.array([row, col]), dtype=torch.long)
        exp_adj_flat = torch.from_numpy(exp_adj[row,col]).unsqueeze(1).float()
        data.edge_index_exp_adj = edge_index_exp_adj
        data.exp_adj_flat = exp_adj_flat
        return data

# Argument parser
parser = argparse.ArgumentParser(description='InvGNN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension size')
parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
parser.add_argument('--patience', default=100, help='Patience for early stopping')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        output = model(data)
        loss_all += F.nll_loss(output, data.y, reduction='sum').item()
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return loss_all / len(loader.dataset), correct / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


acc = []
for i in range(10):
    print('---------------- Split {} ----------------'.format(i))
    norm_factor = avg_lambda_max[args.dataset][i]

    if args.dataset in unlabeled_datasets:
        dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, transform=Compose([Degree(), GraphOperator(norm_factor)]))
    else:
        dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr, transform=GraphOperator(norm_factor))

    train_index = splits[i]['model_selection'][0]['train']
    val_index = splits[i]['model_selection'][0]['validation']
    test_index = splits[i]['test']

    test_dataset = dataset[test_index]
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = GNN(dataset.num_features, args.hidden_dim, args.n_layers, dataset.num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    early_stopper = EarlyStopper(patience=args.patience)
    best_val_acc, test_acc = None, 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch, train_loader, optimizer)
        val_loss, val_acc = val(val_loader)
        if best_val_acc is None or best_val_acc <= val_acc:
            best_val_acc = val_acc
            test_acc = test(test_loader)
        if epoch % 20 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                    epoch, train_loss, val_loss, test_acc))
    
        if early_stopper.early_stop(val_acc):
            break

    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
