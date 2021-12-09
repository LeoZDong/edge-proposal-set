import os
import argparse

# import matplotlib.pyplot as plt
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

def count_pos(edge_index, num_user):
    count = np.zeros(num_user)
    for edge in edge_index.T:
        count[edge[0]] += 1
    return count