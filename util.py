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

# def plot_record(record_name, record_vals, title, save_file):
#     plt.figure()
#     its = list(range(1:(len(record_vals) + 1)))
#     plt.plot(its, record_vals, alpha=1, linewidth=1.5)

#     # Plot the (normalized) adjusted closing stock prices
#     prices_norm = np.array(prices)[:, :, 0]
#     prices_norm /= prices_norm[0, :]
#     stock_codes = eval_env.envs[0].stock_codes

#     for i, stock_code in enumerate(stock_codes):
#         plt.plot(prices_norm[:, i] - 1, label=stock_code, alpha=0.5, linewidth=0.5)
#     plt.legend()
#     plt.xlabel('hour')
#     plt.ylabel('return')
#     plt.title("{} Environment Returns At Train Step {}".format(
#         'Train' if use_train else 'Evaluation', step))
#     plt.savefig(save_file, dpi=300)
#     plt.close()
