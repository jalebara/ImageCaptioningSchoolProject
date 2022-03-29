"""Evaluates a trained meshed memory transformer.
Using the test set, this script generates the following NLP metrics:
BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE, and METEOR. It also generates
captioned images, attention maps, and Class Activation Maps for the best 
test image results.
"""

import argparse
from gc import callbacks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from data.augmentation import Flickr30KRegionalFeatures

from models.Configuration import MediumDroppyTransformerConfiguration
from models.meshed_memory import MeshedMemoryTransformer
from models.model_utils import count_parameters

from utils import Flickr30KMetricsCallback

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 2 Training")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--data_dir", action="store", type=str, default="../flickr30k.exdir")
    parser.add_argument("--checkpoint", action="store", type=str, default="../epoch=57-val_loss=2.91.ckpt")
    parser.add_argument("--num_workers", action="store", type=int, default=12)
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    smoke_test = args.smoke_test
    checkpoint= args.checkpoint
    num_workers = args.num_workers
    
    # Load Config
    config = MediumDroppyTransformerConfiguration()
    
    # Load Data
    test = Flickr30KRegionalFeatures( root=data_dir, max_detections=config["max_detections"], smoke_test=smoke_test, mode="test")
    testloader = DataLoader(test, batch_size=1, num_workers=num_workers)

    test_captions = test.annotations
    
    # Load Model
    lightning_model = MeshedMemoryTransformer(config, beam_size=5, inv_word_map=test.inv_word_map, reference_captions=test_captions)
    trainable, total = count_parameters(lightning_model)
    
    # Trainer
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="auto", fast_dev_run=smoke_test)
    trainer.test(model = lightning_model, dataloaders=testloader, ckpt_path=checkpoint, verbose=True)

if __name__ == "__main__":
    main()
