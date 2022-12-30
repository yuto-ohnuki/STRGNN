import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", "-g", type=int, default=1, help="GPU ID")
    parser.add_argument(
        "--task_type",
        "-type",
        type=str,
        default="transductive",
        choices=["transductive", "inductive"],
    )
    parser.add_argument(
        "--target_network",
        "-tg",
        type=str,
        default="drug_disease",
        help="link prediction target network",
    )
    parser.add_argument(
        "--encoder_type", "-enc", default="MIX", choices=["MIX", "GNN", "NN", "SKIP"]
    )
    parser.add_argument(
        "--decoder_type", "-dec", default="IPD", choices=["CAT", "MUL", "IPD"]
    )
    parser.add_argument("--reg_type", "-rt", default=1)
    parser.add_argument("--lr", "-lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", "-d", type=int, default=128)
    parser.add_argument("--cycle_num", "-c", type=int, default=2)
    parser.add_argument("--norm_lambda", "-nl", type=float, default=1e-6)
    parser.add_argument("--dropedge_ratio", "-de", type=float, default=0.2)
    parser.add_argument("--dropout_ratio", "-do", type=float, default=0.5)
    parser.add_argument("--cv", "-cv", type=int, default=1)
    parser.add_argument("--eps", "-eps", type=float, default=1e-6)
    parser.add_argument("--epoch_num", "-e", type=int, default=5000)
    parser.add_argument("--verbose", "-v", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)


if __name__ == "__main__":
    main()
