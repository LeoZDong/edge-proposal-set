import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import negative_sampling

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sample_pos_edges(edge_index, n_sample):
    n_total = edge_index.shape[1]
    idx = torch.randperm(n_total)[:n_sample]
    return edge_index[:, idx]

def sample_neg_edges(edge_index, n_sample, num_nodes=None):
    return negative_sampling(edge_index,
                             num_nodes=num_nodes,
                             num_neg_samples=n_sample)
