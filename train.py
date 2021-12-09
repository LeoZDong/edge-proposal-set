"""Main training script."""

import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

import config
import data
import models

# Load hyperparameters
args = config.parse()

# Define models
model = models.get_model(args)

# Define optimizer
optim = torch.optim.Adam(model.parameters(), args.lr)

# Create data objects
dataset = data.get_data(csv_file='ratings.csv', feat_dim=128)

# Create data loader
# Here for completion only; there is only one graph and no batching is performed
loader = DataLoader([dataset])

for batch in loader:
    out = model(batch)
    import ipdb; ipdb.set_trace()
    print(out)