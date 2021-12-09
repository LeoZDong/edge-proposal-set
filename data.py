"""Data loading and preprocessing."""
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import csv
import numpy as np
import random

THRESHOLD = 3

def process_row(row):
    userId, movieId, rating, timestamp = row
    return int(userId), int(movieId), float(rating)

def get_data(csv_file='ratings.csv', feat_dim=128):
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

        edge_index_lst = []
        edge_attr_lst = []
        for row in csvreader:
            userId, movieId, rating = process_row(row)
            edge = [userIds[userId], movieIds[movieId]]
            edge_index_lst.append(edge)
            edge_index_lst.append(edge[::-1])
            edge_attr_lst.append(int(rating > THRESHOLD))
            edge_attr_lst.append(int(rating > THRESHOLD))
        edge_index = torch.LongTensor(edge_index_lst)
        
        
        edge_attr_tr, edge_attr_val, edge_attr_test = split_edge_labels(edge_attr_lst)
        
        x = torch.ones(len(userIds) + len(movieIds), feat_dim)
        
        tr_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tr)
        val_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_val)
        test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_test)
        
        return tr_data, val_data, test_data

def split_edge_labels(labels):
    tr_labels = []
    val_labels = []
    test_labels = []
    for label in labels:
        rand = random.random()
        if rand > 0.4 # 60% chance
            tr_labels.append(label)
            test_labels.append(-1)
            val_labels.append(-1)
        else if rand > 0.2:  # 20% chance
            val_labels.append(label)
            tr_labels.append(-1)
            test_labels.append(-1)
        else: # 20% chance
            test_labels.append(label)
            val_labels.append(-1)
            tr_labels.append(-1)
    
    f = lambda x : torch.LongTensor(x).unsqueeze(1)
    return f(tr_labels), f(val_labels), f(test_labels)

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
    data = get_data()
    pdb.set_trace()
    # DataLoader(data, batch_size=)
    print(data) # Data(x=[10334], edge_index=[201672, 2], edge_attr=[201672])
        
if __name__ == "__main__":
    main()