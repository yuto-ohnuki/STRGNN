import os, sys, glob
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from layer import *


class MyEncoder(nn.Module):
    def __init__(self, data, nsymbols, esymbols, weighted_networks, attributes, conf):
        super(MyEncoder, self).__init__()

        # conf
        self.device = conf.device
        self.route = conf.encoder_type
        self.cycle = conf.cycle_num
        self.network_order = conf.network_order
        self.target_network = conf.target_network

        # nodes
        self.n_nodes = {key: data["n_{}".format(key)] for key in nsymbols.keys()}
        self.node_symbols = nsymbols

        # edges
        self.edge_index = {
            network: data["{}_edge_index".format(network)]
            for network in esymbols.values()
        }
        self.edge_weight = {
            esymbols[network]: data["{}_edge_weight".format(esymbols[network])]
            for network in weighted_networks
        }
        self.edge_symbols = esymbols
        self.weighted_networks = set(
            [esymbols[network] for network in weighted_networks]
        )

        # Attribute Encoder
        self.add_encoder = nn.ModuleDict()
        self.att_encoder["drug"] = ECFPEncoder(
            attributes["drug_ecfp"], conf.emb_dim, data["n_drug"], conf
        )
        self.att_encoder["protein"] = AminoSeqEncoder(
            attributes["protein_seq"],
            attributes["protein_mxlen"],
            data["n_drug"],
            conf.emb_dim,
            channel1=5,
            channel2=5,
            kernel_size=3,
            stride=2,
            conf=conf,
        )

        # Network Encoder
        self.net_encoder = nn.ModuleDict()
        for network, symbol in esymbols.items():
            src, tar, *diff = network.split("_")
            if network == conf.target_network:
                pass
            elif src == tar:
                self.net_encoder[symbol] = MonoEncoder(
                    conf.emb_dim, data["n_{}".format(src)], self.dropout_ratio
                )
            else:
                self.net_encoder[symbol] = BipartiteEncoder(
                    conf.emb_dim,
                    data["n_{}".format(src)],
                    data["n_{}".format(tar)],
                    self.dropout_ratio,
                    conf.device,
                )

        # others
        self.drop_edge = DropEdge(conf.dropedge_ratio)

    def forward(self, feat):

        # Attribute Encoder
        d_feat_att = self.att_encoder["drug"](feat["d_feat"])
        p_feat_att = self.att_encoder["protein"](feat["p_feat"])
        feat["d_feat"] = d_feat_att.clone()
        feat["p_feat"] = p_feat_att.clone()

        # Network Encoder
        for _ in range(self.cycle_num):
            for network in self.network_order:
                src, tar, *diff = network.split("_")

                if src == tar:
                    if network not in self.weighted_networks:
                        edge_index = self.drop_edge(self.edge_index[network]).to(
                            self.device
                        )
                        feat["{}_feat".format(src)] = self.net_encoder[network](
                            feat["{}_feat".format(src)], edge_index, self.route
                        )
                    else:
                        edge_index = self.edge_index[network].to(self.device)
                        feat["{}_feat".format(src)] = self.net_encoder[network](
                            feat["{}_feat".format(src)],
                            edge_index,
                            self.route,
                            self.edge_weight[network],
                        )

                else:
                    if network not in self.weighted_networks:
                        edge_index = self.drop_edge(self.edge_index[network]).to(
                            self.device
                        )
                        (
                            feat["{}_feat".format(src)],
                            feat["{}_feat".format(tar)],
                        ) = self.net_encoder[network](
                            feat["{}_feat".format(src)],
                            feat["{}_feat".format(tar)],
                            edge_index,
                            self.route,
                        )
                    else:
                        edge_index = self.edge_index[network].to(self.device)
                        (
                            feat["{}_feat".format(src)],
                            feat["{}_feat".format(tar)],
                        ) = self.net_encoder[network](
                            feat["{}_feat".format(src)],
                            feat["{}_feat".format(tar)],
                            edge_index,
                            self.route,
                            self.edge_weight[network],
                        )

        ret_feat = {key: feat[key] for key in feat.keys}
        ret_feat["d_feat_att"] = d_feat_att
        ret_feat["p_feat_att"] = p_feat_att
        ret_feat = Data.from_dict(ret_feat)

        return ret_feat


class MyDecoder(nn.Module):
    def __init__(self, conf):
        super(MyDecoder, self).__init__()
        self.device = conf.device

        if conf.dec_type == "CAT" or conf.dec_type == "MUL":
            self.decoder = MLPDecoder(conf.dec_type, conf.emb_dim, conf.dropout_ratio)
        elif conf.dec_type == "IPD":
            self.decoder = IPDDecoder(conf.emb_dim, weighted=False)
        else:
            raise Exception("Decoder type error")

    def forward(self):
        pass


class STRGNN(nn.Module):
    def __init__(self, encoder, decoder):
        super(STRGNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder