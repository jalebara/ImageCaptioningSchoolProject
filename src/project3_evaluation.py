#!/bin/env python3.8
"""Evaluates a trained meshed memory transformer.
Using the test set, this script generates the following NLP metrics:
BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE, and METEOR. It also generates
captioned images, attention maps, and Class Activation Maps for the best 
test image results.
"""
""" Performs the main training for the meshed memory transformer.
There are two phases of training for the MMT. First, we perform supervised 
training using negative log likelihood (Cross Entropy Loss).The second phase 
of training involves fine tuning using Reinforement Learning. Due to the decorrelation 
of the loss and the natural languange metrics in convnetional deep learning training 
methods, it is necessary to directly involve the desired metrics through 
reinforcement learning.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
import warnings
from data.augmentation import Flickr30KFeatures

from models.Configuration import *
from models.meshed_memory import MeshedMemoryTransformer
from models.model_utils import count_parameters

import argparse
from gc import callbacks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from data.augmentation import Flickr30KFeatures

from models.Configuration import *
from models.meshed_memory import MeshedMemoryTransformer
from models.model_utils import count_parameters

from utils import Flickr30KMetricsCallback, TextMessageUpdateCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 2 Training")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--data_dir", action="store", type=str, default="../flickr30k.exdir")
    parser.add_argument("--checkpoint", action="store", type=str, default="./best_project2_checkpoint.ckpt")
    parser.add_argument("--num_workers", action="store", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    smoke_test = args.smoke_test
    checkpoint = args.checkpoint
    num_workers = args.num_workers

    # Load Config
    config = BayesianMemoryTinyTransformerConfiguration()

    # Load Data
    test = Flickr30KFeatures(
        root=data_dir,
        max_detections=config["max_detections"],
        feature_mode="region",
        smoke_test=smoke_test,
        mode="test",
    )
    testloader = DataLoader(test, batch_size=1, num_workers=num_workers)

    test_captions = test.annotations

    # Load Model
    lightning_model = MeshedMemoryTransformer(config, beam_size=5)
    trainable, total = count_parameters(lightning_model)
    metric_callback = Flickr30KMetricsCallback(test.inv_word_map, test.annotations)

    # Text Message Updates
    if os.environ.get("TWILIO_ACCOUNT_SID", None) is not None:
        callbacks = [
            metric_callback,
            TextMessageUpdateCallback(
                os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"], os.environ["SMS_RECIPIENT"]
            ),
        ]
    else:
        callbacks = [
            metric_callback,
        ]
    # Trainer
    trainer = pl.Trainer(max_epochs=config["epochs"], fast_dev_run=smoke_test, callbacks=callbacks)
    trainer.test(model=lightning_model, dataloaders=testloader, ckpt_path=checkpoint, verbose=True)


if __name__ == "__main__":
    main()
