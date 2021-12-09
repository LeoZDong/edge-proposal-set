"""Main training script."""

import time

import torch
from torch.utils.data import DataLoader

import config
import data
import models
import util
import metrics

torch.manual_seed(0)
if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

# Load hyperparameters
args = config.parse()

# Create data objects
graph = data.get_data(csv_file='ratings.csv', feat_dim=128)
train_mask = torch.logical_or(graph.mp_mask, graph.sup_mask)
train_edge_index = graph.edge_index[:, train_mask]
val_edge_index = graph.edge_index[:, graph.val_mask]

# TEMP: use train to build adj_mat for eval!
val_adj_mat = data.build_adj_mat(train_edge_index, graph.num_user,
                                 graph.num_item)

num_user = graph.num_user
num_item = graph.num_item
train_dataset = data.get_dataset(graph, 'train', args.num_neg_per_user)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
mp_edge_index = graph.edge_index[:, graph.mp_mask]

val_mp_mask = torch.logical_or(graph.mp_mask, graph.sup_mask)
val_mp_edge_index = graph.edge_index[:, val_mp_mask]
val_edge_index = graph.edge_index[:, torch.logical_or(val_mp_mask, graph.val_mask)]
val_adj_mat = data.build_adj_mat(val_edge_index, num_user, num_item)

# test_mp_mask = torch.logical_or(val_mp_mask, graph.val_mask)
# test_edge_index = graph.edge_index[:, test_mp_mask]


# Define models
model = models.get_model(args, num_user + num_item)
if USE_CUDA:
    model = model.cuda()

# Define optimizer
optim = torch.optim.Adam(model.parameters(), args.lr)


# Start training
loss_it = []
it = 0
model.train()
while it < args.n_iter:
    for batch in train_loader:
        t = time.time()
        if USE_CUDA:
            node_feat = model(graph.x.cuda(), mp_edge_index.cuda())
        else:
            node_feat = model(graph.x, mp_edge_index)

        # Loss calculation
        optim.zero_grad()
        user, item_pos, item_neg = batch
        user_feat, item_pos_feat, item_neg_feat = node_feat[user], node_feat[item_pos], node_feat[item_neg]

        pred_pos = (user_feat * item_pos_feat).sum(-1)
        pred_neg = (user_feat * item_neg_feat).sum(-1)
        loss = -(pred_pos - pred_neg).sigmoid().log().mean()
        loss_it.append(loss.detach().cpu().item())
        loss.backward()
        optim.step()

        # Logging
        it += 1
        train_t = round(time.time() - t, 3)

        # if it == 20:
        # import ipdb; ipdb.set_trace()

        if it % args.log_interval == 0:
            print(f"It: {it}, loss={round(loss_it[-1], 3)}, time={train_t}")

        if it % args.eval_interval == 0:
            model.eval()
            # NOTE: Since metric is calculated on the entire known graph, perhaps
            # it is more fair to use test_edge_index (i.e. only test edges are heldout).
            if USE_CUDA:
                node_feat = model(graph.x.cuda(), val_mp_edge_index.cuda())
            else:
                node_feat = model(graph.x, val_mp_edge_index)

            userEmbeds = node_feat[:num_user]
            itemEmbeds = node_feat[num_user:]
            # precision, recall = metrics.metric_wrap(userEmbeds, itemEmbeds,
            #                                         args.k, graph.edge_set,
            #                                         'precision_recall')
            # print(f"Evaluation: precision={precision}, recall={recall}")
            hits_k = metrics.hits_k(userEmbeds, itemEmbeds, args.k, val_adj_mat, num_user)
            print(f"Evaluation: hits_k={round(hits_k, 3)}")
            model.train()

        if it % args.ckpt_interval == 0:
            file = f'models/model_{it}.pt'
            torch.save(model.state_dict(), file)


torch.save(model.state_dict(), 'models/model_final.pt')