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
            if rating > THRESHOLD:
                edge_index_lst.append(edge)
                edge_index_lst.append(edge[::-1])
        edge_index = torch.LongTensor(edge_index_lst)
    
        mask_gen = torch.randint(low=0, high=10, size =(edge_index.shape[0],))
        mp_mask = torch.le(mask_gen, 5) #%60 chance
        sup_mask = torch.logical_or(torch.eq(mask_gen, 6), torch.eq(mask_gen, 7)) #%20 chance
        val_mask = torch.eq(mask_gen, 8) #%10 chance
        test_mask = torch.eq(mask_gen, 9) #%10 chance
        
        x = torch.ones(len(userIds) + len(movieIds), feat_dim)
        data = Data(x=x, edge_index=edge_index)
        
        return data, mp_mask, sup_mask, val_mask, test_mask

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
    data, mp_mask, sup_mask, val_mask, test_mask = get_data()
    pdb.set_trace()
    # DataLoader(data, batch_size=)
    print(data) # Data(x=[10334], edge_index=[201672, 2], edge_attr=[201672])
        
if __name__ == "__main__":
    main()
    