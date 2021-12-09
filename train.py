"""Main training script."""

import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

import config
import data
import models
import util

# Load hyperparameters
args = config.parse()

# Define models
model = models.get_model(args)

# Define optimizer
optim = torch.optim.Adam(model.parameters(), args.lr)

# Create data objects
dataset = data.get_data(csv_file='ratings.csv', feat_dim=128)
mp_edge_index = dataset.edge_index[:, dataset.mp_mask]
sup_edge_index = dataset.edge_index[:, dataset.sup_mask]
train_edge_index = dataset.edge_index[:, torch.logical_or(dataset.mp_mask, dataset.sup_mask)]
num_nodes = dataset.x.shape[0]

# Start training
for it in range(args.sn_iter):
    out = model(dataset.x, mp_edge_index)

    pos_edges = util.sample_pos_edges(sup_edge_index, args.num_edges_per_iter)
    neg_edges = util.sample_neg_edges(train_edge_index, args.num_edges_per_iter)

    