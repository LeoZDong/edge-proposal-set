"""Main training script."""

import time

import torch
from torch.utils.data import DataLoader

import config
import data
import models
import util
import metrics
import pdb

torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 100000

def main(mode="pretrain", load=True):
    # Load hyperparameters
    args = config.parse()

    # Create data objects
    
    graph = data.get_data_cached(csv_file='/content/edge-proposal-set/ratings.csv', feat_dim=128)
    
    graph = graph.to(device)

    train_mask = torch.logical_or(graph.mp_mask, graph.sup_mask)
    train_edge_index = graph.edge_index[:, train_mask]
    val_edge_index = graph.edge_index[:, graph.val_mask]

    # TEMP: use train to build adj_mat for eval!
    # dz: why is this defined below again
    val_adj_mat = data.build_adj_mat(train_edge_index, graph.num_user,
                                     graph.num_item)

    num_user = graph.num_user
    num_item = graph.num_item
    train_dataset = data.get_dataset(graph, 'train', args.num_neg_per_user)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    mp_edge_index = graph.edge_index[:, graph.mp_mask]

    val_mp_mask = torch.logical_or(graph.mp_mask, graph.sup_mask)

    val_mp_edge_index = graph.edge_index[:, val_mp_mask]
    recall_exclude_edges = val_mp_edge_index.clone().T
    recall_exclude_edges[:, 1] -= num_user
    val_num_pos = util.count_pos(graph.edge_index[:, graph.val_mask], num_user)

    val_edge_index = graph.edge_index[:, torch.logical_or(val_mp_mask, graph.val_mask)]
    val_adj_mat = data.build_adj_mat(val_edge_index, num_user, num_item)

    test_mp_mask = torch.logical_or(val_mp_mask, graph.val_mask)
    test_mp_edge_index = graph.edge_index[:, test_mp_mask]
    test_recall_exclude_edges = test_mp_edge_index.clone().T
    test_recall_exclude_edges[:, 1] -= num_user
    test_num_pos = util.count_pos(graph.edge_index[:, graph.test_mask], num_user)
    
    test_edge_index = graph.edge_index
    test_adj_mat = data.build_adj_mat(test_edge_index, num_user, num_item)

    # Define models
    model = models.get_model(args, num_user + num_item)
    if USE_CUDA:
        model = model.cuda()
        
    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), args.lr)
    
    best_hits_k = 0
    if load:
        name = "pretrain" if mode == "train" else "none"
        state = util.load(name, "models/", "best", model, None)
        best_hits_k = state['best_hits_k']
        
    if mode == "train": #add new edges
        node_feat = model(graph.x.cuda(), val_mp_edge_index.cuda())
        userEmbeds = node_feat[:num_user]
        movieEmbeds = node_feat[num_user:]
        exclude_MP = mp_edge_index.clone().T
        exclude_MP[:, 1] -= num_user
        
        p_edges = util.top_P_edges(userEmbeds, movieEmbeds, P, exclude_MP)
        mp_edge_index = torch.cat([mp_edge_index.T, p_edges], dim=0).T #2 x E
    
    mp_edge_index = torch.cat([mp_edge_index, mp_edge_index.flip(0)], dim=1) #2 x 2E
    val_mp_edge_index = torch.cat([val_mp_edge_index, val_mp_edge_index.flip(0)], dim=1)
    test_edge_index = torch.cat([test_edge_index, test_edge_index.flip(0)], dim=1)
    
    # Start training
    loss_it = []
    it = 0
    model.train()
    
    while it < args.n_iter:
        for batch in train_loader:
            t = time.time()

            # X is the node embeds for all the users, MP_edge is the message passing edges used
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
                with torch.no_grad():
                    model.eval()
                    # NOTE: Since metric is calculated on the entire known graph, perhaps
                    # it is more fair to use test_edge_index (i.e. only test edges are heldout).
                    node_feat = model(graph.x, val_mp_edge_index)

                    userEmbeds = node_feat[:num_user]
                    itemEmbeds = node_feat[num_user:]
                    hits_k = metrics.hits_k(userEmbeds, itemEmbeds, args.k,
                                            val_adj_mat, num_user,
                                            recall_exclude_edges, val_num_pos)
                    print(f"Evaluation: hits_k={round(hits_k * 100, 2)}")

                    node_feat = model(graph.x, test_mp_edge_index)

                    hits_k_test = metrics.hits_k(userEmbeds, itemEmbeds, args.k,
                                            test_adj_mat, num_user,
                                            test_recall_exclude_edges, test_num_pos)
                    print(f"Test: hits_k={round(hits_k_test * 100, 2)}")

                    

                    if hits_k > best_hits_k:
                        best_hits_k = hits_k
                        print("Saving best model...")
                        util.save(mode, "models/", "best", model, optim, best_hits_k)

                    model.train()

            if it % args.ckpt_interval == 0 or it == args.n_iter - 1:
                # file = f'models/{mode}_{it}.pt'
                util.save(mode, "models/", it, model, optim)
                # torch.save(model.state_dict(), file)


if __name__ == "__main__":
    main("train", True)