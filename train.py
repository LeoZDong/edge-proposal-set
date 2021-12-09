"""Main training script."""

import torch
from torch.utils.data import DataLoader

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
graph = data.get_data(csv_file='ratings.csv', feat_dim=128)
train_dataset = data.get_dataset(graph, 'train', args.num_neg_per_user)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
mp_edge_index = graph.edge_index[:, graph.mp_mask]

# Start training
for it in range(args.n_iter):
    for batch in train_loader:
        out = model(graph.x, mp_edge_index)
        import ipdb; ipdb.set_trace()

    # pos_edges = util.sample_pos_edges(sup_edge_index, args.num_edges_per_iter)
    # neg_edges = util.sample_neg_edges(train_edge_index, args.num_edges_per_iter)
