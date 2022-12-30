import os, sys, glob
import scipy.sparse as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor


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
        mat = np.empty((len(seq), len(char_to_id), prot_dim))
        for i, s in enumerate(tqdm(seq)):
            if len(s) > MXLEN:
                s = s[:MXLEN]
            for j, c in enumerate(s):
                mat[i][char_to_id[c]][j] = 1
        atts["protein_seq"] = mat

    return atts
