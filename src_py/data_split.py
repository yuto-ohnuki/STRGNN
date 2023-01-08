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


def remove_edge_from_mask(
    edge_index, edge_weight=None, src_mask_index=None, tar_mask_index=None
):
    ret_index, ret_weight = [], []
    is_mask_src = src_mask_index is not None
    is_mask_tar = tar_mask_index is not None

    # unweighted network
    if edge_weight is None:
        for pair in edge_index.T.numpy():
            fig = True
            if is_mask_src and pair[0] in src_mask_index:
                flg = False
            if is_mask_tar and pair[1] in tar_mask_index:
                flg = False
            if flg:
                ret_index.append(pair)
        ret_index = Tensor(np.array(ret_index).T).long()
        return ret_index

    # weighted network
    else:
        for pair, weight in zip(edge_index.T.numpy(), edge_weight.T):
            flg = True
            if is_mask_src and pair[0] in src_mask_index:
                flg = False
            if is_mask_tar and pair[1] in tar_mask_index:
                flg = False
            if flg:
                ret_index.append(pair)
                ret_weight.append(weight)
        ret_index = Tensor(np.array(ret_index).T).long()
        ret_weight = Tensor(np.array(ret_weight))
        return ret_index, ret_weight


def split_inductive_edges(data, edge_symbols, weighted_edge_names, unseen_ratio, conf):
    src_node = conf.source_node
    tar_node = conf.target_node

    # semi-inductive
    if conf.task_type == "semi_inductive":
        network = data["{}_edge_index".format(edge_symbols[conf.target_network])]
        src_dims = Counter(network[0].tolist())
        cand_src_ids = np.array([k for k, v in src_dims.items() if v >= 2])
        rd = np.random.binomial(1, unseen_ratio, len(cand_src_ids))
        external_src_index = cand_src_ids[rd.nonzero()[0]]
        external_src_set = set(external_src_index)
        internal_src_index = np.array(
            [
                x
                for x in range(data["n_{}".format(src_node)])
                if x not in external_src_set
            ]
        )
        internal_src_set = set(internal_src_index)

        internal_tar_index = np.array([x for x in range(data["n_{}".format(tar_node)])])
        internal_tar_set = set(internal_tar_index)
        external_tar_index = np.array([])
        external_tar_set = set(external_tar_index)

    # fully-inductive
    elif conf.task_type == "fully_inductive":
        src_dims = Counter(network[0].tolist())
        tar_dims = Counter(network[1].tolist())
        cand_src_ids = np.array([k for k, v in src_dims.items() if v >= 2])
        cand_tar_ids = np.array([k for k, v in tar_dims.items() if v >= 2])
        rd_src = np.random.binomial(1, unseen_ratio, len(cand_src_ids))
        rd_tar = np.random.binomial(1, unseen_ratio, len(cand_tar_ids))

        external_src_index = cand_src_ids[rd_src.nonzero()[0]]
        external_src_set = set(external_src_index)
        external_tar_index = cand_tar_ids[rd_tar.nonzero()[0]]
        external_tar_set = set(external_tar_index)

        internal_src_index = np.array(
            [
                x
                for x in range(data["n_{}".format(src_node)])
                if x not in external_src_set
            ]
        )
        internal_src_set = set(internal_src_index)
        internal_tar_index = np.array(
            [
                x
                for x in range(data["n_{}".format(tar_node)])
                if x not in external_tar_set
            ]
        )
        internal_tar_set = set(internal_tar_index)

    else:
        raise Exception("Task type error")

    # train-test split for target network
    train_edge_index, test_edge_index = [], []
    for pair in data[
        "{}_edge_index".format(edge_symbols[conf["target_network"]])
    ].T.numpy():
        if pair[0] in internal_src_set:
            if conf.task_type == "semi_inductive":
                train_edge_index.append(pair)
            else:
                if pair[1] in internal_tar_set:
                    train_edge_index.append(pair)
        elif pair[0] in external_src_set:
            if conf.task_type == "semi_inductive":
                test_edge_index.append(pair)
            else:
                if pair[1] in external_tar_set:
                    test_edge_index.append(pair)
        else:
            continue
    train_edge_index = np.array(train_edge_index)
    test_edge_index = np.array(test_edge_index)

    # remove edges including external nodes
    updated_edge_index = {}
    updated_edge_weight = {}
    mask_index = {src_node: external_src_index, tar_node: external_tar_index}
    for network, symbol in edge_symbols.items():
        types = network.split("_")
        if network == conf.target_network:
            updated_edge_index[symbol] = Tensor(train_edge_index).T.long()
        else:
            first, second = types[0], types[1]
            if first in mask_index.keys():
                first_mask = mask_index[first]
            else:
                first_mask = None

            if second in mask_index.keys():
                second_mask = mask_index[second]
            else:
                second_mask = mask_index[second]

            if (first_mask is None) and (second_mask is None):
                updated_edge_index[symbol] = data["{}_edge_index".format(symbol)]
                if network in weighted_edge_names:
                    updated_edge_weight[symbol] = data["{}_edge_weight".format(symbol)]
            else:
                if network in weighted_edge_names:
                    (
                        updated_edge_index[symbol],
                        updated_edge_weight[symbol],
                    ) = remove_edge_from_mask(
                        data["{}_edge_index".format(symbol)],
                        data["{}_edge_weight".format(symbol)],
                        src_mask_index=first_mask,
                        tar_mask_index=second_mask,
                    )
                else:
                    updated_edge_index[symbol] = remove_edge_from_mask(
                        data["{}_edge_index".format(symbol)],
                        src_mask_index=first_mask,
                        tar_mask_index=second_mask,
                    )

    # update edge_index and edge_weight
    for key, val in updated_edge_index.items():
        data["{}_edge_index".format(key)] = val
    for key, val in updated_edge_weight.items():
        data["{}_edge_weight".format(key)] = val

    data.internal_src_index = internal_src_index
    data.internal_tar_index = internal_tar_index
    data.external_src_index = external_src_index
    data.external_tar_index = external_tar_index

    train_edge_index = [Tensor(train_edge_index.T).long()]
    test_edge_index = Tensor(test_edge_index.T).long()
    return data, train_edge_index, test_edge_index


def split_link_prediction_datas(
    data, nodes, edge_symbols, weighted_edges, conf, unseen_ratio=0.5
):
    src_node = conf.source_node
    tar_node = conf.target_node
    if conf.task_type == "transductive":
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

    elif conf.task_type == "semi-inductive" or conf.task_type == "fully-inductive":
        data, train_edge_index, test_edge_index = split_inductive_edges(
            data, edge_symbols, weighted_edges, unseen_ratio=unseen_ratio, conf=conf
        )
        valid_edge_index, test_edge_index = split_train_valid_edges(
            test_edge_index, train_ratio=0.5
        )
        data.train_edge_index = train_edge_index
        data.valid_edge_index = valid_edge_index
        data.test_edge_index = test_edge_index

    else:
        raise Exception("Task type error")

    return data
