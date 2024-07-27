import os, sys, glob, copy, datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch import Tensor
from torch.nn import CosineEmbeddingLoss
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

def save_models(state, txt):
    dt = datetime.datetime.now()
    stamp = dt.strftime("%Y-%m-%d_%H:%M:%S")
    torch.save(state["net"], "{}_{}_ModelStates.pth".format(stamp, txt))
    torch.save(state["optimizer"], "{}_{}_OptimizerStates.pth".format(stamp, txt))
    torch.save(state["feat"], "{}_{}_features.pt".format(stamp, txt))

def load_edges(path, unweighted_keys, weighted_keys, conf):
    edge_indexes = dict()
    edge_weights = dict()

    # load unweighted network
    for key in unweighted_keys:
        filepath = os.path.join(path, "Edge", "{}.npy".format(key))
        assert os.path.exists(filepath)
        edge = np.load(filepath)
        if conf.input_network_operation != "none":
            edge = randomize_network(key, edge, conf)
        edge = sp.coo_matrix(edge)
        edge = Tensor(np.array([edge.row, edge.col])).long()
        edge_indexes[key] = edge

    # load weighted network
    for key in weighted_keys:
        filepath = os.path.join(path, "Edge", "{}.npy".format(key))
        assert os.path.exists(filepath)
        edge = np.load(filepath)
        if conf.input_network_operation != "none":
            edge = randomize_network(key, edge, conf, weighted=True)
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

    elif conf.task_type == "semi-inductive":
        neg_edge = []
        negative_size = 2

        # only choice random target nodes
        for i, src in enumerate(pos[0]):
            for _ in range(negative_size):
                while True:
                    neg_tar = np.random.randint(0, n_tar)
                    if (used is not None) and ({(src, neg_tar)} <= used):
                        continue
                    if not {(src, neg_tar)} <= pos_pair:
                        break
                neg_edge.append([src, neg_tar])
        neg_edge = Tensor(np.array(neg_edge)).T.long()
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

        if src != tar:
            data["{}_edge_index".format(symbol)] = to_undirected(
                data["{}_edge_index".format(symbol)],
                data["n_{}".format(symbol_to_nodename[src])],
            )
            if network in weighted_edge_names:
                data["{}_edge_weight".format(symbol)] = data[
                    "{}_edge_weight".format(symbol)
                ].repeat(2)
        else:
            pass
    return data


def remove_or_insert_symmetric_matrix(matrix, operation, ratio=0.2, insert_value=None):
    triu_indexes = np.triu_indices(matrix.shape[0])
    paired_indexes = np.column_stack(triu_indexes)
    triu_values = matrix[triu_indexes]
    
    zero_indexes = paired_indexes[triu_values == 0]
    non_zero_indexes = paired_indexes[triu_values != 0]
    
    if operation == 'insert':
        insert_num = int(non_zero_indexes.shape[0] * ratio)
        insert_indexes = zero_indexes[np.random.choice(zero_indexes.shape[0], insert_num, replace=False)]
        
        if insert_value is not None:
            for i, j in insert_indexes:
                matrix[i][j] = 1
                matrix[j][i] = 1
        else:
            for i, j in insert_indexes:
                matrix[i][j] = insert_value
                matrix[j][i] = insert_value
        
        return matrix
    
    if operation == 'remove':
        remove_num = int(non_zero_indexes.shape[0] * ratio)
        remove_indexes = non_zero_indexes[np.random.choice(non_zero_indexes.shape[0], remove_num, replace=False)]
        
        for i, j in remove_indexes:
            matrix[i][j] = 0
            matrix[j][i] = 0
        
        return matrix
    
    else:
        raise ValueError("OperationError")


def remove_or_insert_asymmetric_matrix(matrix, operation, ratio=0.2, insert_value=None):
    
    zero_indexes = np.argwhere(matrix == 0)
    non_zero_indexes = np.argwhere(matrix != 0)
    
    if operation == 'insert':
        insert_num = int(non_zero_indexes.shape[0] * ratio)
        insert_indexes = zero_indexes[np.random.choice(zero_indexes.shape[0], insert_num, replace=False)]
        
        for i,j in insert_indexes:
            if insert_value is not None:
                matrix[i][j] = 1
            else:
                matrix[i][j] = insert_value
        
        return matrix

    elif operation == 'remove':
        remove_num = int(non_zero_indexes.shape[0] * ratio)
        remove_indexes = non_zero_indexes[np.random.choice(non_zero_indexes.shape[0], remove_num, replace=False)]
        
        for i,j in remove_indexes:
            matrix[i][j] = 0
        
        return matrix
    
    else:
        raise ValueError("Operation Error")


def randomize_network(key, edge, conf, weighted=False):
    
    # In advance, calculated from original dataset
    pre_calculated_average_weight = {
        'disease_disease': 0.1351,
        'drug_mirna': 0.3285,
        'protein_mrna': 0.1841,
        'protein_mirna': 0.1841,
        'mrna_mrna': 0.2564,
        'mrna_mirna': 0.2163,
        'mirna_mirna': 0.2908
    }
    
    src, tar, *diff = key.split("_")
    if key == conf.target_network:
        pass
    
    # Weighted network
    elif key in pre_calculated_average_weight.keys():
        if src == tar:
            edge = remove_or_insert_symmetric_matrix(
                edge, conf.input_network_operation, conf.network_randomize_ratio, pre_calculated_average_weight[key]
            )
        else:
            edge = remove_or_insert_asymmetric_matrix(
                edge, conf.input_network_operation, conf.network_randomize_ratio, pre_calculated_average_weight[key]
            )
    
    # Not weighted network
    else:
        if src == tar:
            edge = remove_or_insert_symmetric_matrix(
                edge, conf.input_network_operation, conf.network_randomize_ratio
            )
        else:
            edge = remove_or_insert_asymmetric_matrix(
                edge, conf.input_network_operation, conf.network_randomize_ratio
            )
            
    return edge


def heatmap_modelweights(model, figsize=(12,8), vmin=0.005, vmax=0.002):
    fnn_weight, gnn_weight = {}, {}
    for x in model.encoder.state_dict().keys():
        key = x.split('.')
        if key[0]=='net_encoder' and key[-1]=='weight':
            network = key[1]
            route = key[2]
            if route == 'fnn_layer':
                fnn_weight[network] = torch.pow(model.encoder.state_dict()[x], 2).mean().item()
            elif route == 'gnn_layer':
                gnn_weight[network] = torch.pow(model.encoder.state_dict()[x], 2).mean().item()
            else:
                pass
    fig = plt.figure(figsize=figsize)
    weights = np.array([list(fnn_weight.values()), list(gnn_weight.values())])
    heatmap = sns.heatmap(weights, cmap='Blues', yticklabels=['FNN','GNN'], xticklabels=list(fnn_weight.keys()), vmin=vmin, vmax=vmax)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("Networks", fontsize=18)
    plt.xticks(fontsize=16)
    plt.show()
    plt.close()


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
        if src == tar or key == conf.target_network:
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


def train(
    model,
    optimizer,
    feat,
    pos_edge_index,
    neg_edge_index,
    nsymbol,
    embsymbol,
    int_src_index,
    int_tar_index,
    conf,
):
    model.train()
    optimizer.zero_grad()
    src_symbol = nsymbol[conf.source_node]
    tar_symbol = nsymbol[conf.target_node]

    if conf.source_node in embsymbol.keys():
        feat["src_feat_init"] = feat["{}_feat".format(src_symbol)].clone()
    if conf.target_node in embsymbol.keys():
        feat["tar_feat_init"] = feat["{}_feat".format(tar_symbol)].clone()
    
    feat, pos_score, neg_score = model(feat, pos_edge_index, neg_edge_index)
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
            if conf.reg_type in ["l1", "elastic"]:
                l1_fnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].fnn_layer[0].weight,
                                1,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l1_fnn
            
            if conf.reg_type in ["l2", "elastic"]:
                l2_fnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].fnn_layer[0].weight,
                                2,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l2_fnn
                
        if conf.encoder_type in ["MIX", "GNN", "SKIP"]:
            if conf.reg_type in ["l1", "elastic"]:
                l1_gnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].gnn_layer.weight,
                                1,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l1_gnn
            
            if conf.reg_type in ["l2", "elastic"]:
                l2_gnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].gnn_layer.weight,
                                2,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l2_gnn            

    if conf.emb_loss == "cos":
        cosemb_loss = CosineEmbeddingLoss()
        if conf.task_type == "semi-inductive":
            label_src = torch.ones(len(int_src_index)).to(conf.device)
            emb_loss_src = cosemb_loss(
                feat["{}_feat".format(src_symbol)][int_src_index],
                feat["{}_feat_att".format(src_symbol)][int_src_index],
                label_src,
            )
            loss += emb_loss_src
        elif conf.task_type == "fully-inductive":
            label_src = torch.ones(len(int_src_index)).to(conf.device)
            label_tar = torch.ones(len(int_tar_index)).to(conf.device)
            emb_loss_src = cosemb_loss(
                feat["{}_feat".format(src_symbol)][int_src_index],
                feat["{}_feat_att".format(src_symbol)][int_src_index],
                label_src,
            )
            emb_loss_tar = cosemb_loss(
                feat["{}_feat".format(tar_symbol)][int_tar_index],
                feat["{}_feat_att".format(tar_symbol)][int_tar_index],
                label_tar,
            )
            loss += emb_loss_src + emb_loss_tar
    else:
        pass

    loss.backward()
    optimizer.step()

    metrics = evaluation(score, target)
    loss = loss.item()
    return feat, loss, metrics


def valid_and_test(
    model,
    feat,
    pos_edge_index,
    neg_edge_index,
    nsymbol,
    embsymbol,
    int_src_index,
    int_tar_index,
    conf,
):
    model.eval()
    src_symbol = nsymbol[conf.source_node]
    tar_symbol = nsymbol[conf.target_node]

    if conf.task_type == "transductive":
        z_src = feat["{}_feat".format(src_symbol)]
        z_tar = feat["{}_feat".format(tar_symbol)]
    elif conf.task_type == "semi-inductive":
        z_src = model.encoder.att_encoder[conf.source_node](feat["src_feat_init"]).to(
            conf.device
        )
        z_tar = feat["{}_feat".format(tar_symbol)]
    elif conf.task_type == "fully-inductive":
        z_src = model.encoder.att_encoder[conf.source_node](feat["src_feat_init"]).to(
            conf.device
        )
        z_tar = model.encoder.att_encoder[conf.target_node](feat["tar_feat_init"]).to(
            conf.device
        )
    else:
        raise Exception("Task type error")

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
            if conf.reg_type in ["l1", "elastic"]:
                l1_fnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].fnn_layer[0].weight,
                                1,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l1_fnn
            
            if conf.reg_type in ["l2", "elastic"]:
                l2_fnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].fnn_layer[0].weight,
                                2,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l2_fnn
                
        if conf.encoder_type in ["MIX", "GNN", "SKIP"]:
            if conf.reg_type in ["l1", "elastic"]:
                l1_gnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].gnn_layer.weight,
                                1,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l1_gnn
            
            if conf.reg_type in ["l2", "elastic"]:
                l2_gnn = torch.sum(
                    Tensor(
                        [
                            torch.norm(
                                model.encoder.net_encoder[key].gnn_layer.weight,
                                2,
                            )
                            * conf.norm_lambda
                            for key in model.encoder.net_encoder.keys()
                        ]
                    )
                )
                loss += l2_gnn          

    if conf.emb_loss == "cos":
        cosemb_loss = CosineEmbeddingLoss()
        if conf.task_type == "semi-inductive":
            label_src = torch.ones(len(int_src_index)).to(conf.device)
            emb_loss_src = cosemb_loss(
                feat["{}_feat".format(src_symbol)][int_src_index],
                feat["{}_feat_att".format(src_symbol)][int_src_index],
                label_src,
            )
            loss += emb_loss_src
        elif conf.task_type == "fully-inductive":
            label_src = torch.ones(len(int_src_index)).to(conf.device)
            label_tar = torch.ones(len(int_tar_index)).to(conf.device)
            emb_loss_src = cosemb_loss(
                feat["{}_feat".format(src_symbol)][int_src_index],
                feat["{}_feat_att".format(src_symbol)][int_src_index],
                label_src,
            )
            emb_loss_tar = cosemb_loss(
                feat["{}_feat".format(tar_symbol)][int_tar_index],
                feat["{}_feat_att".format(tar_symbol)][int_tar_index],
                label_tar,
            )
            loss += emb_loss_src + emb_loss_tar
    else:
        pass

    metrics = evaluation(score, target)
    loss = loss.item()

    return loss, metrics


def get_record(record, conf):
    ret_indexes, ret_aurocs, ret_auprcs, ret_accs = [], [], [], []
    for cv in range(conf.cv):
        cv_record = np.array(record[cv]).T
        best_index = np.argmax(cv_record)
        ret_indexes.append(best_index)
        ret_aurocs.append(cv_record[0][best_index])
        ret_auprcs.append(cv_record[1][best_index])
        ret_accs.append(cv_record[2][best_index])
    return ret_indexes, ret_aurocs, ret_auprcs, ret_accs