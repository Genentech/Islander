from __future__ import absolute_import, division, print_function
import argparse, numpy as np, Data_Handler as dh


def Parser_Trainer():
    parser = argparse.ArgumentParser(description="Parser for scTrain")
    #
    """ === General === """
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="debug")
    parser.add_argument("--runname", type=str, default="debug")
    parser.add_argument("--savename", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="mixup",
        choices=["vanilla", "mixup", "scl", "triplet"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lung",
        choices=[
            "lung",
            "lung_fetal_donor",
            "lung_fetal_organoid",
            "brain",
            "breast",
            "heart",
            "eye",
            "gut_fetal",
            "skin",
            "COVID",
            "pancreas",
        ],
    )
    """ === Model HyperOpt === """
    parser.add_argument(
        "--type",
        type=str,
        default="ae-concept",
        choices=[
            "base-ae",
            "ae-concept",
        ],
    )
    parser.add_argument("--batchnorm", type=bool, default=True)
    parser.add_argument("--mlp_size", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--leakage", type=int, default=16)
    parser.add_argument("--dropout", type=bool, default=True)
    parser.add_argument("--batch1d", type=str, default="Vanilla", choices=["Vanilla", "scVI"])

    """ === Training HyperOpt === """
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--optimiser",
        type=str,
        default="adam",
        choices=["adam", "sgd", "asgd", "rmsprop"],
    )
    parser.add_argument("--w_rec", type=float, default=0)
    # parser.add_argument("--w_hsi", type=float, default=1)
    # parser.add_argument("--w_con", type=float, default=2e-2)
    # parser.add_argument("--w_inf", type=float, default=5e-5)
    parser.add_argument("--w_cet", type=float, default=1)
    parser.add_argument("--w_scl", type=float, default=1e-2)
    parser.add_argument("--w_psl", type=float, default=1e-2)

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--train_and_test", type=bool, default=False)
    parser.add_argument("--load_from_save", type=bool, default=False)
    parser.add_argument("--saved_checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args


MODEL_DIR = rf"{dh.MODEL_DIR}/_lung_"
B_PATH = rf"{MODEL_DIR}/MODE-mixup_LEAK-16_MLP-128 128"


def Parser_Benchmarker():
    parser = argparse.ArgumentParser(description="Parser for scBenchmarker")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_path", type=str, default=B_PATH)
    parser.add_argument("--batch_id", type=str, nargs="+", default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="lung",
        choices=[
            "lung",
            "lung_fetal_donor",
            "lung_fetal_organoid",
            "brain",
            "breast",
            "heart",
            "eye",
            "gut_fetal",
            "skin",
            "COVID",
            "pancreas",
        ],
    )
    parser.add_argument("--highvar", action="store_true")
    parser.add_argument("--highvar_n", type=int, default=2000)
    parser.add_argument("--savecsv", type=str, default=None)

    # === Integration Baselines ===
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--scvi", action="store_true")
    parser.add_argument("--bbknn", action="store_true")
    parser.add_argument("--scgen", action="store_true")
    parser.add_argument("--scpoli", action="store_true")
    parser.add_argument("--scanvi", action="store_true")
    parser.add_argument("--harmony", action="store_true")
    parser.add_argument("--fastmnn", action="store_true")
    parser.add_argument("--scanorama", action="store_true")
    #
    parser.add_argument("--islander", action="store_true")
    parser.add_argument("--obsm_keys", type=str, nargs="+", default=None)

    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--saveadata", action="store_true")

    #  === local debug ===
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    pass
