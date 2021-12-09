"""Main training script."""

import time

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
loss_it = []
it = 0
while it < args.n_iter:
    for batch in train_loader:
        t = time.time()
        node_feat = model(graph.x, mp_edge_index)

        # Loss calculation
        user, item_pos, item_neg = batch
        user_feat, item_pos_feat, item_neg_feat = node_feat[user], node_feat[item_pos], node_feat[item_neg]

        pred_pos = torch.sigmoid((user_feat * item_pos_feat).sum(-1))
        pred_neg = torch.sigmoid((user_feat * item_neg_feat).sum(-1))
        loss = -(pred_pos - pred_neg).log().mean()
        loss_it.append(loss.detach().cpu().item())
        loss.backward()
        optim.step()

        # Logging
        it += 1
        train_t = round(time.time() - t, 3)

        if it % args.log_interval == 0:
            print(f"It: {it}, loss={round(loss_it[-1], 3)}, time={train_t}")

        # if it % args.plot_interval:
        #     util.plot_record()

    # pos_edges = util.sample_pos_edges(sup_edge_index, args.num_edges_per_iter)
    # neg_edges = util.sample_neg_edges(train_edge_index, args.num_edges_per_iter)
