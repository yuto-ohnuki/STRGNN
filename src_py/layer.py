import os, sys, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor

from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

EPS = 1e-6


class NodeNorm(nn.Module):
    def __init__(self, unbiased=False, eps=EPS):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (
            torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
        ).sqrt()
        x = (x - mean) / std
        return x


class DropEdge(nn.Module):
    def __init__(self, ratio):
        super(DropEdge, self).__init__()
        self.ratio = ratio

    def dropedge(self, x, y):
        void_dt = np.dtype((np.void, x.dtype.itemsize * np.prod(x.shape[1:])))
        orig_dt = np.dtype((x.dtype, x.shape[1:]))

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_void = x.reshape(x.shape[0], -1).view(void_dt)
        y_void = y.reshape(y.shape[0], -1).view(void_dt)

        return np.setdiff1d(x_void, y_void).view(orig_dt).T

    def forward(self, edge_index):
        if self.ratio == 0:
            pass
        else:
            while True:
                rd = np.random.binomial(1, self.ratio, edge_index.shape[1])
                mask = rd.nonzero()[0]
                if mask.shape[0] != 0:
                    break
            drop_edge = edge_index[:, mask]
            rev_drop_edge = drop_edge.clone()
            rev_drop_edge[0, :], rev_drop_edge[1, :] = drop_edge[1, :], drop_edge[0, :]
            cat_drop_edge = torch.cat([drop_edge, rev_drop_edge], dim=1)
            X = edge_index.T.detach().cpu().numpy()
            Y = cat_drop_edge.T.detach().cpu().numpy()
            edge_index = Tensor(self.dropedge(X, Y)).long()
        return edge_index


class ECFPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, node_num, mask_index, conf):
        super(ECFPEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_num = node_num
        self.device = conf.device
        self.dropout_ratio = conf.dropout_ratio
        self.mask_index = mask_index

        self.fc1 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.in_dim // 2, self.in_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )
        self.fc3 = nn.Sequential(nn.Linear(self.in_dim // 4, out_dim))
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=-1)
        
        # mask for external node's embeddings
        if self.training:
            x[self.mask_index] = 0
            
        return x

class AminoSeqEncoder(nn.Module):
    def __init__(
        self,
        uniq_chars,
        seq_mxlen,
        node_num,
        out_dim,
        channel1,
        channel2,
        kernel_size,
        stride,
        mask_index,
        conf,
    ):
        super(AminoSeqEncoder, self).__init__()

        self.uniq_chars = uniq_chars
        self.seq_mxlen = seq_mxlen
        self.node_num = node_num
        self.out_dim = out_dim
        self.channel1 = channel1
        self.channel2 = channel2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride
        self.dropout = conf.dropout_ratio
        self.device = conf.device
        self.mask_index = mask_index
    
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.uniq_chars, self.channel1, self.kernel_size),
            nn.BatchNorm1d(self.channel1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.channel1, self.channel2, self.kernel_size),
            nn.BatchNorm1d(self.channel2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride),
        )
        
        self.densed_dim = self.calc_flattened_dim(self.seq_mxlen, self.channel2, [self.conv1, self.conv2])
        self.dense = nn.Sequential(
            nn.Linear(self.densed_dim, self.out_dim * 4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 4, self.out_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        
    def calc_flattened_dim(self, dim, channels, modules):
        for module in modules:
            conv = module[0]
            pool = module[-1]
            dim = math.floor((dim+2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0]-1)-1)/conv.stride[0] + 1)
            dim = math.floor((dim+2*pool.padding[0] - pool.kernel_size[0])/pool.stride[0] + 1)
        return dim*channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        
        # mask for external node's embeddings
        if self.training and (self.mask_index is not None):
            x[self.mask_index] = 0
        return x

class MonoEncoder(nn.Module):
    def __init__(self, dim, n_src, dropout_ratio, route_type):
        super(MonoEncoder, self).__init__()
        self.in_dim = dim
        self.out_dim = dim
        self.route_type = route_type
        self.fnn_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_ratio),
        )
        self.gnn_layer = GCNConv(self.in_dim, self.out_dim)
        self.node_norm = NodeNorm()

    def forward(self, x, edge_index, edge_weight=None):
        if self.route == "GNN":
            x = self.gnn_layer(x, edge_index, edge_weight=edge_weight)
        elif self.route == "FNN":
            x = self.fnn_layer(x)
        elif self.type == "SKIP":
            x_s = torch.clone(x)
            x = self.gnn_layer(x, edge_index, edge_weight=edge_weight)
            x = x + x_s
        elif self.type == "MIX":
            x_s = torch.clone(x)
            x = self.gnn_layer(x, edge_index, edge_weight)
            x_s = self.fnn_layer(x_s)
            x = x + x_s
        else:
            raise Exception("Encoder type error")

        x = self.node_norm(x)
        x = F.leaky_relu(x, inplace=True)
        return x


class BipartiteEncoder(nn.Module):
    def __init__(self, dim, n_src, n_tar, dropout_ratio, device, route_type):
        super(BipartiteEncoder, self).__init__()
        self.device = device
        self.in_dim = dim
        self.out_dim = dim
        self.n_src = n_src
        self.n_tar = n_tar
        self.fnn_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_ratio),
        )
        self.gnn_layer = GCNConv(self.in_dim, self.out_dim)
        self.node_norm = NodeNorm()

    def forward(self, x_src, x_tar, edge_index, edge_weight):

        x = torch.cat([x_src, x_tar], axis=0).to(self.device)

        if self.route == "GNN":
            x = self.gnn_layer(x, edge_index, edge_weight=edge_weight)
        elif self.route == "FNN":
            x = self.fnn_layer(x)
        elif self.route == "SKIP":
            x_s = torch.clone(x)
            x = self.gnn_layer(x, edge_index, edge_weight=edge_weight)
            x = x + x_s
        elif self.route == "MIX":
            x_s = torch.clone(x)
            x = self.gnn_layer(x, edge_index, edge_weight)
            x_s = self.fnn_layer(x_s)
            x = x + x_s
        else:
            raise Exception("Encoder type error")

        x = self.node_norm(x)
        x = F.leaky_relu(x, inplace=True)
        x_src = x[: self.n_src, :]
        x_tar = x[self.n_src :, :]
        return x_src, x_tar


class MLPDecoder(nn.Module):
    def __init__(self, dec_type, dim, dropout_ratio):
        super(MLPDecoder, self).__init__()
        self.dec_type = dec_type
        if self.dec_type == "CAT":
            self.in_dim = dim * 2
        elif self.dec_type == "MUL":
            self.in_dim = dim
        else:
            raise Exception("Decoder type error")
        self.fc1 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_ratio),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.in_dim // 2, self.in_dim // 4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_ratio),
        )
        self.fc3 = nn.Sequential(nn.Linear(self.in_dim // 4, 1), nn.Sigmoid())

    def forward(self, z_src, z_tar, edge_index):
        if self.dec_type == "CAT":
            z = torch.cat([z_src[edge_index[0]], z_tar[edge_index[1]]], dim=1)
        elif self.dec_type == "MUL":
            z = torch.mul(z_src[edge_index[0]], z_tar[edge_index[1]])
        else:
            raise Exception("Decoder type error")
        z = self.fc1(z)
        z = F.normalize(z, p=2, dim=-1)
        z = self.fc2(z)
        z = F.normalize(z, p=2, dim=-1)
        z = self.fc3(z)
        return z


class IPDDecoder(nn.Module):
    def __init__(self):
        super(IPDDecoder, self).__init__()

    def forward(self, z_src, z_tar, edge_index):
        z = (z_src[edge_index[0]] * z_tar[edge_index[1]]).sum(dim=1)
        z = torch.sigmoid(z)
        return z
