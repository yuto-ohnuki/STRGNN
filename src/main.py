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

from utils import *
from arguments import *
from data_split import *
from metrics import *
from layer import *
from model import *


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
        dataset_path, unweighted_edge_names, weighted_edge_names, conf
    )

    # load attributes
    att_names = ["drug_ecfp", "protein_seq"]
    att_symbols = {"drug": "drug_ecfp", "protein": "protein_seq"}
    atts = load_attributes(
        nodes, att_names, node_nums, MXLEN=nodes["protein"].SeqLength.max()
    )
    att_dims = {
        "drug_ecfp": atts["drug_ecfp"].shape[1],
        "protein_uniqchar": atts["protein_seq"].shape[1],
        "protein_mxlen": atts["protein_seq"].shape[2],
    }

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
    ).to(conf.device)

    describe_dataset(data, nodes, edges, edge_symbols, conf)

    ############################################################
    ### Model Training (link prediction task)
    ############################################################
    train_losses, valid_losses = defaultdict(list), defaultdict(list)
    train_records, valid_records = defaultdict(list), defaultdict(list)
    loss_dicts = defaultdict(list)
    best_records = defaultdict(list)
    best_states = dict()

    print("\t--- LINK PREDICTION TYPE: {}---".format(conf.task_type))
    print("\t--- LINK PREDICTION TYPE: {}---".format(conf.task_type))

    for cv in range(conf.cv):
        print("\n>>> cross validation: {} <<<".format(cv + 1))

        # define Model
        epoch_num = 0
        best_auprc = 0
        model = STRGNN(
            data, node_symbols, edge_symbols, weighted_edge_names, att_dims, conf
        )
        model = model.to(conf.device)
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)

        # model training
        for epoch in range(conf.epoch_num):
            time_begin = time.time()

            # initial feature
            feat = {}
            for key, value in node_symbols.items():
                if key in att_symbols.keys():
                    feat["{}_feat".format(value)] = data[att_symbols[key]]
                else:
                    feat["{}_feat".format(value)] = data["{}_feat".format(value)]

            feat = Data.from_dict(feat).to(conf.device)

            # negative sampling
            pos_edge_index = data.train_edge_index[cv].clone()
            neg_edge_index = negative_sampling_edge_index(
                pos_edge_index,
                data["n_{}".format(conf.source_node)],
                data["n_{}".format(conf.target_node)],
                used[cv],
                data.internal_src_index,
                data.internal_tar_index,
                conf,
            ).to(conf.device)

            # train
            feat, train_loss, train_metrics = train(
                model,
                optimizer,
                feat,
                pos_edge_index,
                neg_edge_index,
                node_symbols,
                att_symbols,
                data.internal_src_index,
                data.internal_tar_index,
                conf
            )

            train_auroc, train_auprc, train_acc = (
                train_metrics["AUROC"],
                train_metrics["AUPRC"],
                train_metrics["ACC"],
            )
            train_records[cv].append([train_auroc, train_auprc, train_acc])
            train_losses[cv].append(train_loss)

            if epoch % conf.verbose == conf.verbose - 1 or epoch == 0:
                print("Train >> ", end="")
                if epoch == 0:
                    print(
                        "EPOCH:{:3d}  AUROC:{:0.4f}  AUPRC:{:0.4f} ACC:{:0.4f}  TIME:{:0.2f}".format(
                            epoch + 1,
                            train_auroc,
                            train_auprc,
                            train_acc,
                            (time.time() - time_begin),
                        )
                    )
                else:
                    print(
                        "EPOCH:{:3d}  TRAIN_LOSS:{:0.4f}  AUROC:{:0.4f}  AUPRC:{:0.4f} ACC:{:0.4f}  TIME:{:0.2f}".format(
                            epoch + 1,
                            train_loss,
                            train_auroc,
                            train_auprc,
                            train_acc,
                            (time.time() - time_begin),
                        )
                    )

            # validation Model
            val_loss, val_metrics = valid_and_test(
                model,
                feat,
                data.valid_edge_index[cv],
                data.valid_neg_edge_index[cv],
                node_symbols,
                att_symbols,
                data.internal_src_index,
                data.internal_tar_index,
                conf,
            )
            val_auroc, val_auprc, val_acc = (
                val_metrics["AUROC"],
                val_metrics["AUPRC"],
                val_metrics["ACC"],
            )
            valid_records[cv].append([val_auroc, val_auprc, val_acc])
            valid_losses[cv].append(val_loss)
            if epoch % conf.verbose == conf.verbose - 1 or epoch == 0:
                print("Valid >> ", end="")
                if epoch == 0:
                    print(
                        "EPOCH:{:3d}  AUROC:{:0.4f}  AUPRC:{:0.4f} ACC:{:0.4f}  TIME:{:0.2f}".format(
                            epoch + 1,
                            val_auroc,
                            val_auprc,
                            val_acc,
                            (time.time() - time_begin),
                        )
                    )
                else:
                    print(
                        "EPOCH:{:3d}  VALID_LOSS:{:0.4f}  AUROC:{:0.4f}  AUPRC:{:0.4f} ACC:{:0.4f}  TIME:{:0.2f}".format(
                            epoch + 1,
                            val_loss,
                            val_auroc,
                            val_auprc,
                            val_acc,
                            (time.time() - time_begin),
                        )
                    )

            # test and update
            if val_auprc >= best_auprc:
                best_auprc = val_auprc
                print("\t-- UPDATE BEST MODEL --".format(epoch))
                state = {
                    "cv": cv,
                    "epoch": epoch_num,
                    "feat": feat,
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                test_loss, test_metrics = valid_and_test(
                    model,
                    feat,
                    data.test_edge_index,
                    data.test_neg_edge_index,
                    node_symbols,
                    att_symbols,
                    data.internal_src_index,
                    data.internal_tar_index,
                    conf,
                )
                test_auroc, test_auprc, test_acc = (
                    test_metrics["AUROC"],
                    test_metrics["AUPRC"],
                    test_metrics["ACC"],
                )
                best_records[cv].append(test_metrics)
        else:
            # save result
            best_states[cv] = state
    
    ############################################################
    ### results
    ############################################################
    
    epochs = [i for i in range(1, conf.epoch_num+1)]
    train_indexes, train_aurocs, train_auprcs, train_accs = get_record(train_records, conf)
    valid_indexes, valid_aurocs, valid_auprcs, valid_accs = get_record(valid_records, conf)


if __name__ == "__main__":
    main()
