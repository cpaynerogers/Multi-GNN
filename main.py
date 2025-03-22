import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

from data_loading import get_data
from training import train_gnn
from inference import infer_gnn


def logger_setup():
    """
    Setup logging to a file in addition to stdout
    """
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )


def create_parser():
    parser = argparse.ArgumentParser()

    # Adaptations
    parser.add_argument(
        "--emlps",
        action="store_true",
        help="Use emlps in GNN training",
    )
    parser.add_argument(
        "--reverse_mp",
        action="store_true",
        help="Use reverse MP in GNN training",
    )
    parser.add_argument(
        "--ports",
        action="store_true",
        help="Use port numberings in GNN training",
    )
    parser.add_argument(
        "--tds",
        action="store_true",
        help="Use time deltas (i.e. the time between subsequent transactions) in GNN training",
    )
    parser.add_argument(
        "--ego",
        action="store_true",
        help="Use ego IDs in GNN training",
    )

    # Model parameters
    parser.add_argument(
        "--batch_size",
        default=8192,
        type=int,
        help="Select the batch size for GNN training",
    )
    parser.add_argument(
        "--n_epochs",
        default=100,
        type=int,
        help="Select the number of epochs for GNN training",
    )
    parser.add_argument(
        "--num_neighs",
        nargs="+",
        default=[100,100],
        help="Pass the number of neighbors to be sampled in each hop (descending).",
    )

    # Misc
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Select the random seed for reproducibility",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Use tqdm logging (when running interactively in terminal)",
    )
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        help="Select the AML dataset. Needs to be either small or medium.",
        required=True,
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]",
        required=True,
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Disable wandb logging while running the script in 'testing' mode.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the best model.",
    )
    parser.add_argument(
        "--unique_name",
        action="store_true",
        help="Unique name under which the model will be stored.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.",
    )

    return parser


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    # TODO: (cpr) setting 'PYTHONHASHSEED' after the program has
    # started won't do anything, unless the program is restarted, e.g.:
    #
    #   os.environ['PYTHONHASHSEED'] = str(seed)
    #   os.execv(sys.executable, [sys.executable] + sys.argv)
    #
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open("data_config.json", "r") as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    # set seed
    set_seed(args.seed)

    # get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        # Training
        logging.info("Running Training")
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
    else:
        # Inference
        logging.info("Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)


if __name__ == "__main__":
    main()
