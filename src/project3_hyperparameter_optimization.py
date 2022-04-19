
""" Performs hyperparameter optimization using the Ray Tune library. This code eases the burden of manually 
finding the best hyperparameters, leaving more time for working on other aspects of the learning pipeline.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCheckpointCallback, get_tune_resources
from ray import tune
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

def xe_parameter_optimization(config):
    # Load Data
    train = Flickr30KFeatures( root=config["data_dir"], max_detections=config["max_detections"], feature_mode="region",  mode="train")
    valid = Flickr30KFeatures( root=config["data_dir"], max_detections=config["max_detections"], feature_mode="region", mode="valid")
    
    trainloader = DataLoader(train, batch_size=config["batch_size"], lazy_cache=True, num_workers=cpu_count())
    valloader = DataLoader(valid, batch_size=config["batch_size"], lazy_cache=True, num_workers=cpu_count())
    
    # Load Model
    lightning_model = MeshedMemoryTransformer(TransformerConfiguration(config))
    trainable, total = count_parameters(lightning_model)
    
    # Model Checkpointing
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    
    # Language Metric Aggregation
    metric_callback = Flickr30KMetricsCallback(valid.inv_word_map, valid.annotations)
    
    # Text Message Updates
    if os.environ.get("TWILIO_ACCOUNT_SID", None) is not None:
        callbacks = [ 
            metric_callback,
            lr_monitor_callback,
            TextMessageUpdateCallback(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"], os.environ["SMS_RECIPIENT"])
        ]
    else:
        callbacks = [ 
            metric_callback,
            lr_monitor_callback,
        ]
    metrics = {"loss": "ptl/val_loss", "meteor":"ptl/meteor"}
    callbacks.append(TuneReportCheckpointCallback(metrics, on="validation_end"))

    # Cross Entropy Training
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="auto", callbacks=callbacks)
    trainer.fit(lightning_model, trainloader, valloader)


def main():
    args = parse_args()
    data_dir = args.data_dir
    
    # Load Config
    config = {
        # Constant across runs
        "data_dir": data_dir,
        "vocabulary_size": 2004,
        "pad_token": 0,
        "max_sequence_length": 30,
        "data_size": 1024,
        "end_token": 1,
        "start_token": 3,
        "max_detections":50,
        # parameters to vary
        "key_size": tune.randint(8,64),
        "value_size": tune.randint(8,64),
        "dropout_rate": tune.uniform(0.1, 0.2),
        "num_encoder_layers": tune.randint(2,5),
        "num_decoder_layers": tune.randint(2,5),
        "out_size": tune.randint(64,512),
        "feedforward_size": tune.randint(54,512),
        "num_heads": tune.randint(2,16),
        "num_memory_slots": tune.randint(4,20),
        "batch_size": tune.randint(8,64),
    }
    asha_scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        metric='meteor',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )
    analysis = tune.run(
        xe_parameter_optimization,
        config=config,
        num_samples=100,
        resources_per_trial={"cpu":5, "gpu":0.5},
        scheduler=asha_scheduler,
        name="xe_param_opt"
    )
    print(f"Best result: {analysis.best_result}")
    print(f"Best configuration: {analysis.best_config}")

if __name__ == "__main__":
    main()
