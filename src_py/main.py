import os, sys, time, json, copy, glob, random, datetime
import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter as Param

from torch_geometric.nn.models import GAE
from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from utils import *
from arguments import *
from data_split import *


def main():
    cur_path = os.getcwd()
    dataset_path = os.path.join(get_parent_path(cur_path, 1), "Dataset")

    ############################################################
    ### configuration
    ############################################################
    conf = get_args()

    ############################################################
    ### load multimodal network datasets
    ############################################################
    data = Data.from_dict(dict())

    # load nodes
    node_names = ["drug", "disease", "protein", "mrna", "mirna", "metabolite"]
    node_symbols = {
        "drug": "d",
        "disease": "z",
        "protein": "p",
        "mrna": "m",
        "mirna": "mi",
        "metabolite": "b",
    }
    symbol_to_nodename = {v: k for k, v in node_symbols.items()}
    nodes = load_nodes(dataset_path, node_names)
    node_nums = {node: count_unique_ids(nodes[node]) for node in node_names}

    # load edges
    unweighted_edge_names = [
        "disease_metabolite",
        "disease_mirna",
        "disease_mrna_down",
        "disease_mrna_up",
        "disease_protein",
        "drug_disease",
        "drug_drug",
        "drug_mrna_down",
        "drug_mrna_up",
        "drug_metabolite_down",
        "drug_metabolite_up",
        "drug_protein",
        "protein_protein",
    ]
    weighted_edge_names = [
        "disease_disease",
        "drug_mirna",
        "protein_mrna",
        "protein_mirna",
        "mrna_mrna",
        "mrna_mirna",
        "mirna_mirna",
    ]
    edge_symbols = {
        "disease_disease": "z_z",
        "disease_metabolite": "z_b",
        "disease_mirna": "z_mi",
        "disease_mrna_down": "z_m_down",
        "disease_mrna_up": "z_m_up",
        "disease_protein": "z_p",
        "drug_disease": "d_z",
        "drug_drug": "d_d",
        "drug_mrna_down": "d_m_down",
        "drug_mrna_up": "d_m_up",
        "drug_metabolite_down": "d_b_down",
        "drug_metabolite_up": "d_b_up",
        "drug_protein": "d_p",
        "drug_mirna": "d_mi",
        "mirna_mirna": "mi_mi",
        "mrna_mirna": "m_mi",
        "mrna_mrna": "m_m",
        "protein_protein": "p_p",
        "protein_mirna": "p_mi",
        "protein_mrna": "p_m",
    }
    symbol_to_edgename = {v: k for k, v in edge_symbols.items()}
    edges, edge_weights = load_edges(
        dataset_path, unweighted_edge_names, weighted_edge_names
    )

    # load attributes
    att_names = ["drug_ecfp", "protein_seq"]
    att_symbols = {"drug": "drug_ecfp", "protein": "protein_seq"}
    atts = load_attributes(
        nodes, att_names, node_nums, MXLEN=nodes["protein"].SeqLength.max()
    )

    # load node counts
    for node in node_symbols.keys():
        data["n_{}".format(node)] = node_nums[node]

    # load edge indexes
    for network in edge_symbols.keys():
        data["{}_edge_index".format(edge_symbols[network])] = edges[network]

    # load edge weights
    for network in weighted_edge_names:
        data["{}_edge_weight".format(edge_symbols[network])] = edge_weights[network]

    # initial features
    for node in node_names:
        if node not in att_symbols.keys():
            data["{}_feat".format(node_symbols[node])] = get_initial_feat(
                data["n_{}".format(node)], conf.emb_dim
            )

        else:
            key = att_symbols[node]
            data[key] = Tensor(atts[key])
            data[key].requires_grad = False

    ############################################################
    ### data-split, negative-sampling
    ############################################################

    # split target network
    data = split_link_prediction_datas(
        data, nodes, edge_symbols, weighted_edge_names, conf
    )

    # negative sampling for valid/test datas
    raw_used = get_pair(data["{}_edge_index".format(edge_symbols[conf.target_network])])
    data.valid_neg_edge_index = []
    for i in range(conf.cv):
        valid_neg_edge_index = negative_sampling_edge_index(
            data.valid_edge_index[i],
            data["n_{}".format(conf.source_node)],
            data["n_{}".format(conf.target_node)],
            raw_used,
            data.internal_src_index,
            data.internal_tar_index,
            conf,
        )
        data.valid_neg_edge_index.append(valid_neg_edge_index)

    else:
        data.test_neg_edge_index = negative_sampling_edge_index(
            data.test_edge_index,
            data["n_{}".format(conf.source_node)],
            data["n_{}".format(conf.target_node)],
            raw_used,
            data.internal_src_index,
            data.internal_tar_index,
            conf,
        )

    # merge negative edges to used
    used = []
    for i in range(conf.cv):
        cv_used = merge_used(raw_used, data.valid_neg_edge_index[i])
        cv_used = merge_used(cv_used, data.test_edge_index)
        used.append(cv_used)

    # to bipartite networks
    data = to_bipartite_network(
        data, edge_symbols, symbol_to_nodename, weighted_edge_names, conf
    )

    describe_dataset(data, nodes, edges, edge_symbols, conf)

    ############################################################
    ### Model Training (link prediction task)
    ############################################################


if __name__ == "__main__":
    main()
