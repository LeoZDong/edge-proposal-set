"""Model definition."""
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

class GNNStack(torch.nn.Module):
    """Defines a stack of GNN layers."""
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = GAT
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GAT(MessagePassing):
    """Defines one GAT layer."""
    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        self.lin_l = nn.Linear(in_channels, out_channels * heads)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(torch.zeros(out_channels, heads))
        self.att_r = nn.Parameter(torch.zeros(out_channels, heads))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):

        H, C = self.heads, self.out_channels

        n_nodes, d = x.shape
        x_l = self.lin_l(x).reshape(n_nodes, H, C)
        x_r = self.lin_r(x).reshape(n_nodes, H, C)
        alpha_l = torch.diagonal(x_l @ self.att_l.unsqueeze(0), dim1=1, dim2=2)
        alpha_r = torch.diagonal(x_r @ self.att_r.unsqueeze(0), dim1=1, dim2=2)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        out = out.reshape(n_nodes, -1)

        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        out = None

        if ptr is not None:
            alpha = softmax(F.leaky_relu(alpha_i + alpha_j), ptr, num_nodes=size_i).unsqueeze(-1)
        else:
            alpha = softmax(F.leaky_relu(alpha_i + alpha_j), index, num_nodes=size_i).unsqueeze(-1)

        alpha = F.dropout(alpha, p=self.dropout,training=self.training)
        out = alpha * x_j

        return out


    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

        return out

def get_model(args):
    model = GNNStack(args.feat_dim, args.feat_dim, args.feat_dim, args, False)
    return model
    # input_dim, hidden_dim, output_dim, args, emb = False
