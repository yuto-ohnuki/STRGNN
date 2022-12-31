import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
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


class MyGAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
