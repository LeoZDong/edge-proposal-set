"""Main training script."""

from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

import csv
import config
import data

# Load hyperparameters
args = config.parse()

# Create data objects
data = data.get_data(csv_file='ratings.csv', feat_dim=args.feat_dim)

# Create data loader
loader = DataLoader(data, 1, shuffle=True)

for batch in loader:
    import ipdb; ipdb.set_trace()