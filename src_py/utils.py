import os, sys, glob, copy
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import Tensor

from metrics import *


def get_parent_path(pth, depth=1):
    for _ in range(depth):
        pth = os.path.dirname(pth)
    return pth


def glob_files(pth):
    ret = [f for f in glob.glob(os.path.join(pth, "*"))]
    return ret


def get_pair(arr):
    return set([(l[0], l[1]) for l in arr.T.tolist()])


def union_set(s1, s2):
    return s1.union(s2)


def measure_time(txt, time_begin, time_fin):
    time = time_fin - time_begin
    print("{}: {:0.3f}".format(txt, time))


def count_unique_ids(var):
    if isinstance(var, pd.DataFrame):
        ret = len(var["MyID"].unique())
        return ret
    else:
        try:
            ret = len(set([x["id"] for x in var.values()]))
            return ret
        except:
            raise Exception("Dtype error: Node file")


def get_initial_feat(num_nodes, dim):
    x = Tensor(num_nodes, dim)
    x = x.data.normal_(std=1 / np.sqrt(dim))
    return x


def load_nodes(path, keys):
    nodes = dict()
    for key in keys:
        filepath = os.path.join(path, "Node", "{}.csv".format(key))
        assert os.path.exists(filepath)
        nodes[key] = pd.read_csv(filepath)
    return nodes


def load_edges(path, unweighted_keys, weighted_keys):
    edge_indexes = dict()
    edge_weights = dict()

    # load unweighted network
    for key in unweighted_keys:
        filepath = os.path.join(path, "Edge", "{}.npy".format(key))
        assert os.path.exists(filepath)
        edge = sp.coo_matrix(np.load(filepath))
        edge = Tensor(np.array([edge.row, edge.col])).long()
        edge_indexes[key] = edge

    # load weighted network
    for key in weighted_keys:
        filepath = os.path.join(path, "Edge", "{}.npy".format(key))
        assert os.path.exists(filepath)
        edge = np.load(filepath)
        row, col = edge.nonzero()
        edge_indexes[key] = Tensor([row, col]).long()
        edge_weights[key] = Tensor(np.array([edge[i][j] for i, j in zip(row, col)]))

    return edge_indexes, edge_weights


def load_attributes(nodes, keys, nums, MXLEN=1000):
    atts = dict()
    drug_dim = 2048
    prot_dim = MXLEN

    # drug
    if "drug_ecfp" in keys:
        mat = np.zeros((nums["drug"], drug_dim))
        mat = np.zeros((nums["drug"], drug_dim))
        for myid in tqdm(range(nums["drug"])):
            sm = nodes["drug"][nodes["drug"].MyID == myid]
            vec = sm["ECFP"].iloc[0][1:-1].replace(",", "").split()
            for x in vec:
                mat[myid][int(x)] = 1
        atts["drug_ecfp"] = mat

    # protein
    if "protein_seq" in keys:
        seqs = nodes["protein"]["Sequence"].values
        uniq_chars = set()
        char_to_id = {}
        for seq in seqs:
            uniq_chars = uniq_chars.union(set(list(seq)))
        for e, x in enumerate(list(uniq_chars)):
            char_to_id[x] = e
        mat = np.empty((len(seqs), len(char_to_id), prot_dim))
        for i, s in enumerate(tqdm(seqs)):
            if len(s) > MXLEN:
                s = s[:MXLEN]
            for j, c in enumerate(s):
                mat[i][char_to_id[c]][j] = 1
        atts["protein_seq"] = mat

    return atts


def negative_sampling_edge_index(
    pos_edge_index, n_src, n_tar, used, src_index, tar_index, conf
):
    pos = pos_edge_index.clone().cpu().numpy()
    pos_pair = set([(l[0], l[1]) for l in pos.T.tolist()])
    src_set = set(src_index)
    tar_set = set(tar_index)
    neg_srcs = [0] * pos.shape[1]
    neg_tars = [0] * pos.shape[1]

    # choice random source nodes
    if conf.task_type == "transductive":
        for i, tar in enumerate(pos[1]):
            while True:
                neg_src = np.random.randint(0, n_src)
                if neg_src not in src_set:
                    continue
                if not {(neg_src, tar)} <= used:
                    break
            neg_srcs[i] = neg_src

        # choice random target nodes
        for i, src in enumerate(pos[0]):
            while True:
                neg_tar = np.random.randint(0, n_tar)
                if neg_tar not in tar_set:
                    continue
                if not {(src, neg_tar)} <= used:
                    break
            neg_tars[i] = neg_tar

        neg_edge = Tensor(
            np.concatenate(
                [np.array([pos[0], neg_tars]), np.array([neg_srcs, pos[1]])], axis=1
            )
        ).long()
        return neg_edge

    else:
        raise Exception("not implemented")


def merge_used(used, edge_index):
    ret = copy.deepcopy(used)
    neg_pair = get_pair(edge_index)
    ret = union_set(ret, neg_pair)
    return ret


def to_undirected(edge_index, n_src):
    edge_index[1] += n_src
    rev_edge_index = edge_index.clone()
    rev_edge_index[0, :], rev_edge_index[1, :] = edge_index[1, :], edge_index[0, :]
    return torch.cat([edge_index, rev_edge_index], dim=1)


def to_bipartite_network(
    data, edge_symbols, symbol_to_nodename, weighted_edge_names, conf
):
    for network, symbol in edge_symbols.items():
        src, tar, *diff = symbol.split("_")
        if network == conf.target_network:
            continue

        elif src == tar:
            continue

        else:
            assert src != tar
            data["{}_edge_index".format(symbol)] = to_undirected(
                data["{}_edge_index".format(symbol)],
                data["n_{}".format(symbol_to_nodename[src])],
            )
            if network in weighted_edge_names:
                data["{}_edge_weight".format(symbol)] = data[
                    "{}_edge_weight".format(symbol)
                ].repeat(2)
    return data


def describe_dataset(data, nodes, edges, edge_symbols, conf):

    line = "#" * 40
    src_node, tar_node = conf.target_network.split("_")
    print(line)

    # Task type
    task_type = conf.task_type
    print("Task  : {}".format(task_type))
    print("Target: {}".format(conf.target_network))
    print(line)

    # Dataset
    print("Node counts >>")
    for key in nodes:
        print("\t{}: {}".format(key, data["n_{}".format(key)]))

    print("\nEdge counts >>")
    for key in edges:
        src, tar, *diff = key.split("_")
        if src == tar:
            print(
                "\t{}: {}".format(
                    key, data["{}_edge_index".format(edge_symbols[key])].shape[1]
                )
            )
        else:
            print(
                "\t{}: {}".format(
                    key, data["{}_edge_index".format(edge_symbols[key])].shape[1] // 2
                )
            )
    print(line)

    # Link prediction
    print("Node splits >> ")
    print("\tInternal {} nodes: {}".format(src_node, data.internal_src_index.shape[0]))
    print(
        "\tInternal {} nodes: {}\n".format(tar_node, data.internal_tar_index.shape[0])
    )
    print("\tExternal {} nodes: {}".format(src_node, data.external_src_index.shape[0]))
    print("\tExternal {} nodes: {}".format(tar_node, data.external_tar_index.shape[0]))
    print(line)

    # Train, Valid, Test
    print("Edge splits >> ")
    for i in range(conf.cv):
        print(
            "\tTrain (cv-{}): {} >> {}".format(
                i, "INT", data.train_edge_index[i].shape[1]
            )
        )
        print(
            "\tValid (cv-{}): {} >> {}\n".format(
                i,
                "INT" if task_type == "transductive" else "EXT",
                data.valid_edge_index[i].shape[1],
            )
        )
    print(
        "\tTest: {} >> {}".format(
            "INT" if task_type == "transductive" else "EXT",
            data.test_edge_index.shape[1],
        )
    )
    print(line)


def train(model, optimizer, feat, pos_edge_index, neg_edge_index, conf):
    model.train()
    optimizer.zero_grad()
    ret_feat, pos_score, neg_score = model.encoder(feat, pos_edge_index, neg_edge_index)

    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])

    score = torch.cat([pos_score, neg_score])
    target = torch.cat([pos_target, neg_target])

    loss = (
        -torch.log(pos_score + conf.eps).mean()
        - torch.log(1 - neg_score + conf.eps).mean()
    )

    if conf.norm_lambda != 0:
        if conf.encoder_type in ["MIX", "NN"]:
            l1_fnn = torch.sum(
                Tensor(
                    [
                        torch.norm(
                            model.encoder.net_encoder[key].fc[0].weight, conf.reg_type
                        )
                        * conf.norm_lambda
                        for key in model.encoder.net_encoder.keys()
                    ]
                )
            )
            loss += l1_fnn
        if conf.encoder_type in ["MIX", "GNN", "SKIP"]:
            l1_gnn = torch.sum(
                Tensor(
                    [
                        torch.norm(
                            model.encoder.net_encoder[key].conv.weight, conf.reg_type
                        )
                        * conf.norm_lambda
                        for key in model.encoder.net_encoder.keys()
                    ]
                )
            )
            loss += l1_gnn

    loss.backward()
    optimizer.step()

    metrics = evaluation(score, target)
    loss = loss.item()

    return ret_feat, loss, metrics


def valid_and_test(model, feat, pos_edge_index, neg_edge_index, conf):
    model.eval()
    z_src = feat["{}_feat".format(conf.source_node)]
    z_tar = feat["{}_feat".format(conf.target_node)]
    pos_score = model.decoder(z_src, z_tar, pos_edge_index)
    neg_score = model.decoder(z_src, z_tar, neg_edge_index)

    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])

    score = torch.cat([pos_score, neg_score])
    target = torch.cat([pos_target, neg_target])

    loss = (
        -torch.log(pos_score + conf.eps).mean()
        - torch.log(1 - neg_score + conf.eps).mean()
    )

    if conf.norm_lambda != 0:
        if conf.encoder_type in ["MIX", "NN"]:
            l1_fnn = torch.sum(
                Tensor(
                    [
                        torch.norm(
                            model.encoder.net_encoder[key].fc[0].weight, conf.reg_type
                        )
                        * conf.norm_lambda
                        for key in model.encoder.net_encoder.keys()
                    ]
                )
            )
            loss += l1_fnn
        if conf.encoder_type in ["MIX", "GNN", "SKIP"]:
            l1_gnn = torch.sum(
                Tensor(
                    [
                        torch.norm(
                            model.encoder.net_encoder[key].conv.weight, conf.reg_type
                        )
                        * conf.norm_lambda
                        for key in model.encoder.net_encoder.keys()
                    ]
                )
            )
            loss += l1_gnn

    metrics = evaluation(score, target)
    loss = loss.item()

    return loss, metrics
