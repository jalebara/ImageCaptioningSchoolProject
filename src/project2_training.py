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
import warnings
from data.augmentation import Flickr30KFeatures

from models.Configuration import *
from models.meshed_memory import MeshedMemoryTransformer
from models.model_utils import count_parameters

from utils import Flickr30KMetricsCallback, TextMessageUpdateCallback
import os
from multiprocessing import cpu_count

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 2 Training")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--golden_debug_1", action="store_true")
    parser.add_argument("--data_dir", action="store", type=str, default="/home/jalexbox/Code/school/ece763/class_project/ImageCaptioningProject/flickr30k.exdir")
    parser.add_argument("--num_workers", action="store", type=int, default=cpu_count())
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    smoke_test = args.smoke_test
    gold_overfit = args.golden_debug_1
    num_workers = args.num_workers
    
    # Load Config
    config = MemoryLessTinyTransformerConfiguration()
    
    # Load Data
    train = Flickr30KFeatures( root=data_dir, max_detections=config["max_detections"], feature_mode="region", smoke_test=smoke_test or gold_overfit, mode="train")
    valid = Flickr30KFeatures( root=data_dir, max_detections=config["max_detections"], feature_mode="region", smoke_test=smoke_test or gold_overfit, mode="valid")
    
    trainloader = DataLoader(train, batch_size=config["batch_size"], num_workers=num_workers)
    valloader = DataLoader(valid, batch_size=config["batch_size"], num_workers=num_workers)
    
    # Load Model
    lightning_model = MeshedMemoryTransformer(config)
    trainable, total = count_parameters(lightning_model)
    
    # Model Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="bleu4", filename='{epoch}-{bleu4:.4f}', mode="max")
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    
    # Language Metric Aggregation
    metric_callback = Flickr30KMetricsCallback(valid.inv_word_map, valid.annotations)
    
    # Text Message Updates
    if os.environ.get("TWILIO_ACCOUNT_SID", None) is not None and not gold_overfit:
        callbacks = [ 
            checkpoint_callback,
            metric_callback,
            lr_monitor_callback,
            TextMessageUpdateCallback(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"], os.environ["SMS_RECIPIENT"])
        ]
    else:
        callbacks = [ 
            checkpoint_callback,
            metric_callback,
            lr_monitor_callback,
        ]

    # Cross Entropy Training
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="auto", fast_dev_run=smoke_test, gpus=1, callbacks=callbacks)
    trainer.fit(lightning_model, trainloader, valloader)

    # Testing
    test = Flickr30KFeatures( root=data_dir, max_detections=config["max_detections"], feature_mode="region", smoke_test=smoke_test or gold_overfit, mode="test")
    testloader = DataLoader(test, batch_size=config["batch_size"], num_workers=10)
    trainer.test(testloader)

if __name__ == "__main__":
    main()
