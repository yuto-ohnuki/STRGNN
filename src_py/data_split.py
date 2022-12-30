import os, sys, glob
import scipy.sparse as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from collections import defaultdict, Counter


def split_drug_repositioning_edges(dz_edge_index, diz_df):
    train_edge_index = []
    test_edge_index = []

    # drug-disease pair
    dz_pairs = defaultdict(list)
    for src, tar in dz_edge_index.T:
        src = src.item()
        tar = tar.item()
        dz_pairs[src].append(tar)

    # drug: degree of drug-disease network >= 2
    drugs = dz_edge_index[0].tolist()
    deg_counts = Counter(drugs)
    test_drug_ids = [k for k, v in deg_counts.items() if v >= 2]

    # disease: no overlapping ICD-11 classification
    for did in test_drug_ids:
        diseases = dz_pairs[did]
        tmp = diz_df[diz_df["MyID"].isin(diseases)]
        diz_categories = tmp["ICD-11_Category"].values

        if len(set(diz_categories)) <= 1:
            continue
        else:
            picked_cat = np.random.choice(diz_categories)
            picked_diz = tmp[tmp["ICD-11_Category"] == picked_cat]
            for zid in picked_diz.MyID:
                test_edge_index.append([did, zid])

    # split train - test
    test_set = set([(x, y) for x, y in test_edge_index])
    for x, y in dz_edge_index.T:
        x = x.item()
        y = y.item()
        if (x, y) not in test_set:
            train_edge_index.append([x, y])
        else:
            continue

    train_edge_index = Tensor(np.array(train_edge_index)).T.long()
    test_edge_index = Tensor(np.array(test_edge_index)).T.long()
    return train_edge_index, test_edge_index


def split_drug_protein_edges(dp_edge_index, train_ratio):
    train_edge_index = []
    test_edge_index = []
    n_edge = dp_edge_index.shape[1]

    rd = np.random.binomial(1, train_ratio, n_edge)
    train_mask = rd.nonzero()[0]
    test_mask = (1 - rd).nonzero()[0]

    train_index = Tensor(np.array(dp_edge_index[:, train_mask])).long()
    test_index = Tensor(np.array(dp_edge_index[:, test_mask])).long()

    return train_index, test_index


def split_train_valid_edges(edge_index, train_ratio=0.8, cv=1):
    n_edge = edge_index.shape[1]
    train_edge_index = []
    valid_edge_index = []

    if cv == 1:
        rd = np.random.binomial(1, train_ratio, n_edge)
        train_mask = rd.nonzero()[0]
        valid_mask = (1 - rd).nonzero()[0]

        train_index = edge_index[:, train_mask]
        valid_index = edge_index[:, valid_mask]

        train_edge_index.append(train_index.long())
        valid_edge_index.append(valid_index.long())

    else:
        rd = np.random.permutation(np.arange(n_edge) % cv)
        for i in range(cv):
            train_index = edge_index[:, rd != i]
            valid_index = edge_index[:, rd == i]
            train_edge_index.append(train_index.long())
            valid_edge_index.append(valid_index.long())

    return train_edge_index, valid_edge_index


def split_link_prediction_datas(data, nodes, edge_symbols, weighted_edges, conf):
    src_node = conf.source_node
    tar_node = conf.target_node

    if conf.target_network == "drug_disease":
        train_edge_index, test_edge_index = split_drug_repositioning_edges(
            data.d_z_edge_index, nodes["disease"]
        )
        train_edge_index, valid_edge_index = split_train_valid_edges(
            train_edge_index, train_ratio=0.8, cv=conf.cv
        )

    elif conf.target_network == "drug_protein":
        train_edge_index, test_edge_index = split_drug_protein_edges(
            data.d_p_edge_index, train_ratio=0.8
        )
        train_edge_index, valid_edge_index = split_train_valid_edges(
            train_edge_index, train_ratio=0.8, cv=conf.cv
        )

    else:
        raise NotImplementedError("Target network is not considered")

    data.internal_src_index = np.array(
        [i for i in range(data["n_{}".format(src_node)])]
    )
    data.internal_tar_index = np.array(
        [i for i in range(data["n_{}".format(tar_node)])]
    )
    data.external_src_index = np.array([])
    data.external_tar_index = np.array([])

    data.train_edge_index = train_edge_index
    data.valid_edge_index = valid_edge_index
    data.test_edge_index = test_edge_index

    return data
