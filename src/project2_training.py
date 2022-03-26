""" Performs the main training for the meshed memory transformer.
There are two phases of training for the MMT. First, we perform supervised 
training using negative log likelihood (Cross Entropy Loss).The second phase 
of training involves fine tuning using Reinforement Learning. Due to the decorrelation 
of the loss and the natural languange metrics in convnetional deep learning training 
methods, it is necessary to directly involve the desired metrics through 
reinforcement learning.
"""
import argparse
from gc import callbacks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from data.augmentation import Flickr30KRegionalFeatures

from models.Configuration import *
from models.meshed_memory import MeshedMemoryTransformer
from models.model_utils import count_parameters

from utils import Flickr30KMetricsCallback

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 2 Training")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--data_dir", action="store", type=str, default="/home/jalexbox/Code/school/ece763/class_project/ImageCaptioningProject/flickr30k.exdir")
    return parser.parse_args()
def main():
    args = parse_args()
    data_dir = args.data_dir
    smoke_test = args.smoke_test
    
    # Load Config
    config = MediumDroppyTransformerConfiguration()
    
    # Load Data
    train = Flickr30KRegionalFeatures( root=data_dir, max_detections=config["max_detections"], smoke_test=smoke_test, mode="train")
    valid = Flickr30KRegionalFeatures( root=data_dir, max_detections=config["max_detections"], smoke_test=smoke_test, mode="valid")
    
    trainloader = DataLoader(train, batch_size=config["batch_size"], num_workers=10)
    valloader = DataLoader(valid, batch_size=config["batch_size"], num_workers=10)
    
    # Load Model
    lightning_model = MeshedMemoryTransformer(config)
    trainable, total = count_parameters(lightning_model)
    
    # Model Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", filename='{epoch}-{val_loss:.2f}', mode="min")
    # Language Metric Aggregation
    metric_callback = Flickr30KMetricsCallback(valid.inv_word_map, valid.annotations)
    # Cross Entropy Training
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="auto", fast_dev_run=smoke_test, gpus=1, callbacks=[checkpoint_callback, metric_callback])
    trainer.fit(lightning_model, trainloader, valloader)
                                                    

if __name__ == "__main__":
    main()
