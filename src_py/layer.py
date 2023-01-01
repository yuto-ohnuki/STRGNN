import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor


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
    def __init__(self, in_dim, out_dim, node_num, conf):
        super(ECFPEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_num = node_num
        self.device = conf.device
        self.dropout_ratio = conf.dropout_ratio

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

        self.conv1 = nn.Sequenctial(
            nn.Conv1d(self.uniq_chars, self.channel1, self.kernel_size, self.padding),
            nn.BatchNorm1d(self.channel1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride),
        )

        self.conv2 = nn.Sequenctial(
            nn.Conv1d(self.uniq_chars, self.channel2, self.kernel_size, self.padding),
            nn.BatchNorm1d(self.channel2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride),
        )

        self.dense = nn.Sequential(
            nn.Linear(4335, self.out_dim * 4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 4, self.out_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class MonoEncoder(nn.Module):
    def __init__(self):
        super(MonoEncoder, self).__init__()

    def forward(self):
        pass


class BipartiteEncoder(nn.Module):
    def __init__(self):
        super(BipartiteEncoder, self).__init__()

    def forward(self):
        pass


class MLPDecoder(nn.Module):
    def __init__(self, dec_type, in_dim, dropout_ratio):
        super(MLPDecoder, self).__init__()

    def forward(self):
        pass


class IPDDecoder(nn.Module):
    def __init__(self):
        super(IPDDecoder, self).__init__()

    def forward(self):
        pass
