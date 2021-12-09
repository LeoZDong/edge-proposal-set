import os
import argparse

# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import negative_sampling

def load(name, log_dir, iteration, model, optimizer):
    """Loads a checkpoint.
    Args:
        iteration (int): iteration of checkpoint to load
    Raises:
        ValueError: if checkpoint for checkpoint_step is not found
    """
    target_path = (
        f'{os.path.join(log_dir, f"{name}_state")}'
        f'{iteration}.pt'
    )
    if os.path.isfile(target_path):
        state = torch.load(target_path)
        model.load_state_dict(state['network_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        print(f'Loaded checkpoint iteration {iteration}.')
        return state
    else:
        raise ValueError(
            f'No checkpoint for iteration {iteration} found.'
        )

def save(name, log_dir, iteration, model, optimizer):
    """Saves network and optimizer state_dicts as a checkpoint.
    Args:
        iteration (int): iteration to label checkpoint with
    """
    save_dict = dict(network_state=model.state_dict(),
             optimizer_state=optimizer.state_dict())
    torch.save(
        save_dict,
        f'{os.path.join(log_dir, f"{name}_state")}{iteration}.pt'
    )
    print(f'Saved checkpoint, step: {iteration}.')

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