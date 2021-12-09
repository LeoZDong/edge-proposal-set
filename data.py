"""Data loading and preprocessing."""
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import csv
import numpy as np
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data

THRESHOLD = 3

def build_adj_mat(edge_index, user_num, item_num):
    """Build sparse adjacency matrix for user-item bipartite graph."""
    adj_mat = sp.dok_matrix((user_num, item_num), dtype=bool)
    # load ratings as a dok matrix
    for x in edge_index.T:
        adj_mat[x[0], x[1]] = True

    return adj_mat


class BPRData(data.Dataset):
    """Build data object for BPR loss calculation. 
    Adopted from https://github.com/guoyang9/BPR-pytorch/blob/master/data_utils.py"""
    def __init__(self,
                 edge_index,
                 num_user,
                 num_item,
                 train_mat=None,
                 num_ng=0,
                 is_training=None):
        super(BPRData, self).__init__()
        """Note that the labels are only useful when training, we thus 
        add them in the ng_sample() function.
        Args:
            edge_index (2, num_edges): Edge list. This is the raw edge list
                where user and item nodes start at different indices.
            num_item (int): Number of items.
            train_mat (sparse matrix): (num_user, num_item) User-item iteraction matrix.
                `train_mat` does not have to correspond to edge_index.
                We use edge_index to return each data point, but reference
                `train_mat` to know which edges are negative.
            num_neg (int): Number of negative items to sample *per user*.
            is_training (bool): Whether we are in training model. If not, we do
                not perform negative sampling.
        """
        # Process edge index so user and item node indices both start at 1
        edge_index[1, :] -= num_user
        self.edge_index = edge_index.T
        self.num_user = num_user
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        if self.is_training:
            self.ng_sample()

    def ng_sample(self):
        assert self.is_training, "No need to sampling when testing"

        self.edge_index_neg = []
        for x in self.edge_index:
            u, i = x[0], x[1]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.edge_index_neg.append([u, i, j])
        self.edge_index_neg = np.array(self.edge_index_neg)

    def __len__(self):
        return self.num_ng * len(self.features) if \
          self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.edge_index_neg if \
          self.is_training else self.edge_index

        user = features[idx][0]
        item_i = features[idx][1]
        # Kind of a hack; force item_j = item_i when not training (i.e. no neg sampling)
        item_j = features[idx][2] if \
          self.is_training else features[idx][1]

        # Post-process item nodes back to start at num_user instead of 0
        item_i += self.num_user
        item_j += self.num_user
        return user, item_i, item_j


def get_dataset(data, split, num_neg_per_user):
    assert split in ['train', 'val', 'test'], "Split should be `train`, `val`, or `test`!"
    # mp_edge_index = data.edge_index[:, data.mp_mask]
    train_edge_index = data.edge_index[:, torch.logical_or(data.mp_mask, data.sup_mask)]
    num_user = data.num_user
    num_item = data.num_item
    train_mat = build_adj_mat(train_edge_index, num_user, num_item)

    is_training = False
    if split == 'train':
        edge_index = data.edge_index[:, data.sup_mask]
        is_training = True
    elif split == 'val':
        edge_index = data.edge_index[:, data.val_mask]
    else:
        edge_index = data.edge_index[:, data.test_mask]

    dataset = BPRData(edge_index, num_user, num_item, train_mat,
                      num_neg_per_user, is_training=is_training)
    return dataset


def process_row(row):
    userId, movieId, rating, timestamp = row
    return int(userId), int(movieId), float(rating)

def get_masks(length, seed=0):
    # NOTE: important to fix seed!
    torch.manual_seed(seed)
    mask_gen = torch.randint(low=0, high=10, size=(length,))
    mp_mask = torch.le(mask_gen, 5) #%60 chance
    sup_mask = torch.logical_or(torch.eq(mask_gen, 6), torch.eq(mask_gen, 7)) #%20 chance
    val_mask = torch.eq(mask_gen, 8) #%10 chance
    test_mask = torch.eq(mask_gen, 9) #%10 chance
    return mp_mask, sup_mask, val_mask, test_mask

def get_data(csv_file, feat_dim):
    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header = next(csvreader)
        userIds = {}
        movieIds = {}
        for row in csvreader:
            userId, movieId, rating = process_row(row)
            if userId not in userIds:
                userIds[userId] = len(userIds)
            if movieId not in movieIds:
                movieIds[movieId] = len(movieIds)
        for movieId, mappedId in movieIds.items():
            movieIds[movieId] += len(userIds)
        csvfile.seek(0)
        next(csvreader)

        edge_set = set()
        edge_index_lst = []
        for row in csvreader:
            userId, movieId, rating = process_row(row)
            edge = [userIds[userId], movieIds[movieId]]
            if rating > THRESHOLD:
                edge_set.add(tuple(edge))
                # Only append user->item edge
                edge_index_lst.append(edge)
        edge_index = torch.LongTensor(edge_index_lst).T

        mp_mask, sup_mask, val_mask, test_mask = get_masks(edge_index.shape[1])
        x = torch.ones(len(userIds) + len(movieIds), feat_dim)

        data = Data(x=x,
                    edge_index=edge_index,
                    mp_mask=mp_mask,
                    sup_mask=sup_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    edge_set=edge_set,
                    num_user=len(userIds),
                    num_item=len(movieIds))
        return data


def get_data_cached(csv_file='ratings.csv', feat_dim=128, write_new_file=False):
    file_name = f"data_and_masks_{csv_file}_{feat_dim}.pt"
    if not write_new_file and os.path.exists(file_name):
        print(f"File exists, loading {file_name}")
        data = torch.load(file_name)
        return data
    else:
        print(f"File does not exist: {file_name}, creating")
        data = get_data(csv_file, feat_dim)
        torch.save(data, file_name)
        return data


# def get_dataloader(args):
# Doesnt work because this is for datasets which have multiple "data"s, where we just have one data
#     data = get_data()
#     pdb.set_trace()
#     return DataLoader(data, batch_size=args.batch_size, shuffle=args.shuffle)

class Args():
    pass

import pdb
def main():
    # args = Args()
    # args.batch_size = 32
    # args.shuffle = True

    data = get_data_cached(write_new_file=True)
    print(data) # Data(x=[10334], edge_index=[201672, 2], edge_attr=[201672])
    pdb.set_trace()


if __name__ == "__main__":
    main()
