""" Contains the main experiment for Project 1
The goal for this is to train a SAT model with and without data augmentation and provide standard language task performance metrics

"""

import argparse
import os
from os.path import join
from models.utils import load_model_dict, save_model_dict
from models.sat_model import SATDecoder, SATEncoder
import torch

RESULTS_DIRECTORY = "../results/project1"

EPOCHS = 20
EMBED = 1024
ATTENTION = 1024
DECODER = 1024
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 1 Main Experiment")
    parser.add_argument(
        "--skip_training", action="store_true", required=False
    )  # if model is already trained, and you just want to regenerate results
    parser.add_argument(
        "--transfer_weights", action="store_true", required=False
    )  # use transfer learning for SAT Model
    parser.add_argument(
        "--augment_data", action="store_true", default=False, required=False
    )  # set option to augment data
    parser.add_argument("--freeze_encoder", action="store_true", default=False, required=False)
    parser.add_argument("--unfreeze_last", action="store", type=int, default=0, required=False)
    parser.add_argument("--data_directory", action="store", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true", type=bool, default=False, required=False)
    return parser.parse_args()

def train_model():
    raise NotImplementedError
def evaluate_model():
    raise NotImplementedError
def main():
    """Main experiment
    The following describes the pipeline:

    1. Construct encoder and load pretrained weights if applicable
    2. Construct decoder and compose the two components
    3. Load dataset and split into train, validation, and test sets
    4. Train model and store best weights
    5. Load best weights
    6. Generate test results and metrics
    7. Generate Class Activation maps on the best performing test set images
    """

    # parse arguments 
    args = parse_args()

    # create results directory
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    best_checkpoint_path = join(
        RESULTS_DIRECTORY,
        f"best-{'augmented' if args.augment_data else 'base'}-weights.pt",
    )

    # Construct Model
    encoder = SATEncoder(pretrained=args.transfer_weights, freeze=args.freeze_encoder, unfreeze_last=args.unfreeze_last)
    decoder = SATDecoder()

    # Load data
    if args.augment_data:
        # load augmented dataset
        raise NotImplementedError
    else:
        # no augmentation
        pass

    if not args.skip_training:
        # train the model
        raise NotImplementedError

    # Load best model
    model = load_model_dict(model, best_checkpoint_path)

    # test set evaluation

    # generate class attention maps


if __name__ == "__main__":
    main()
