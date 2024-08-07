import argparse
import random
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", "-g", type=int, default=1, help="GPU ID")
    parser.add_argument(
        "--task_type",
        "-type",
        type=str,
        default="transductive",
        choices=["transductive", "semi-inductive", "fully-inductive"],
    )
    parser.add_argument(
        "--target_network",
        "-tg",
        type=str,
        default="drug_disease",
        choices=["drug_disease", "drug_protein"],
        help="link prediction target network",
    )
    parser.add_argument(
        "--encoder_type", "-enc", default="MIX", choices=["MIX", "GNN", "NN", "SKIP"]
    )
    parser.add_argument(
        "--decoder_type", "-dec", default="IPD", choices=["CAT", "MUL", "IPD"]
    )
    parser.add_argument("--reg_type", "-rt", default='l1', choices=['l1', 'l2', 'elastic'])
    parser.add_argument("--shuffle_network_order", "-sno", default="none", choices=["every", "once", "none"])
    parser.add_argument(
        "--input_network_operation", "-nop", default='none', choices=["insert", "remove", "none"]
    )
    parser.add_argument("--network_randomize_ratio", "-nrr", default=0)
    parser.add_argument("--lr", "-lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", "-d", type=int, default=128)
    parser.add_argument(
        "--emb_loss", "-ls", type=str, default="none", choices=["cos", "none"]
    )
    parser.add_argument("--cycle_num", "-c", type=int, default=2)
    parser.add_argument("--norm_lambda", "-nl", type=float, default=1e-6)
    parser.add_argument("--channel_size", "-cs", type=int, default=5)
    parser.add_argument("--kernel_size", "-ks", type=int, default=3)
    parser.add_argument("--stride", "-s", type=int, default=2)
    parser.add_argument("--dropedge_ratio", "-de", type=float, default=0.2)
    parser.add_argument("--dropout_ratio", "-do", type=float, default=0.5)
    parser.add_argument("--cv", "-cv", type=int, default=1)
    parser.add_argument("--eps", "-eps", type=float, default=1e-6)
    parser.add_argument("--epoch_num", "-e", type=int, default=5000)
    parser.add_argument("--verbose", "-v", type=int, default=10)
    parser.add_argument("--save_model", "-save", type=str, default=False)
    args = parser.parse_args()
    check_and_update_args(args)
    return args


def check_and_update_args(args):
    line = "#" * 40

    # check: target network
    assert (
        args.target_network == "drug_disease" or args.target_network == "drug_protein"
    )
    args.source_node, args.target_node = args.target_network.split("_")

    # check: device
    args.device = torch.device(
        "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
    )

    # update: order of interaction networks to be trained
    args.network_order = [
        "disease_disease",
        "disease_metabolite",
        "disease_mirna",
        "disease_mrna_up",
        "disease_mrna_down",
        "disease_protein",
        "protein_protein",
        "protein_mirna",
        "protein_mrna",
        "mrna_mrna",
        "mrna_mirna",
        "mirna_mirna",
        "drug_mirna",
        "drug_mrna_up",
        "drug_mrna_down",
        "drug_metabolite_up",
        "drug_metabolite_down",
        "drug_protein",
        "drug_drug",
    ]
    if args.shuffle_network_order == "once":
        random.shuffle(args.network_order)
    assert args.target_network not in args.network_order

    # result
    print(line)
    print("Args: ")
    for key, value in vars(args).items():
        if key != "network_order":
            print("\t{}:{}".format(key, value))
        else:
            print("\t{}:".format(key))
            for e,name in enumerate(value):
                print("\t\t{}: {}".format(e+1, name))
    print(line)


def main():
    args = get_args()
    # print(args)


if __name__ == "__main__":
    main()
