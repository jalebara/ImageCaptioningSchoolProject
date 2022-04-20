""" Performs hyperparameter optimization using the Ray Tune library. This code eases the burden of manually 
finding the best hyperparameters, leaving more time for working on other aspects of the learning pipeline.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import tune
import uuid

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
    parser.add_argument(
        "--data_dir",
        action="store",
        type=str,
        default="/home/jalexbox/Code/school/ece763/class_project/ImageCaptioningProject/flickr30k.exdir",
    )
    parser.add_argument("--num_workers", action="store", type=int, default=8)
    return parser.parse_args()

def trial_name_generator(trial):
    return f"{str(uuid.uuid4())}"

def xe_parameter_optimization(config, **train_params):
    # Load Data
    print("Starting New Trial")
    train = Flickr30KFeatures(
        root=train_params["data_dir"],
        max_detections=train_params["max_detections"],
        smoke_test=train_params["smoke_test"],
        lazy_cache=True,
        feature_mode="region",
        mode="train",
        disable_progress_bar=True,
        num_processes=10,
    )
    valid = Flickr30KFeatures(
        root=train_params["data_dir"],
        max_detections=train_params["max_detections"],
        smoke_test=train_params["smoke_test"],
        lazy_cache=True,
        feature_mode="region",
        mode="valid",
        disable_progress_bar=True,
        num_processes=10
    )
    print("Configuring Dataloaders")
    trainloader = DataLoader(train, batch_size=config["batch_size"], num_workers=train_params["num_data_workers"])
    valloader = DataLoader(valid, batch_size=config["batch_size"], num_workers=train_params["num_data_workers"])
    config.update(train_params)
    
    print("Constructing Model")
    # Load Model
    config = TransformerConfiguration(config)
    lightning_model = MeshedMemoryTransformer(config)

    print("Constructing Callbacks")
    # Model Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="bleu4", filename="{epoch}-{bleu4:.4f}", mode="max")
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()

    # Language Metric Aggregation
    metric_callback = Flickr30KMetricsCallback(valid.inv_word_map, valid.annotations)

    # Text Message Updates
    if os.environ.get("TWILIO_ACCOUNT_SID", None) is not None:
        print("Texting Enabled")
        callbacks = [
            metric_callback,
            checkpoint_callback,
            lr_monitor_callback,
            TextMessageUpdateCallback(
                os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"], os.environ["SMS_RECIPIENT"]
            ),
        ]
    else:
        callbacks = [
            metric_callback,
            checkpoint_callback,
            lr_monitor_callback,
        ]
    print("Creating Tune Callback")
    metrics = {"meteor": "meteor"}
    callbacks.append(TuneReportCheckpointCallback(metrics, on="validation_end"))
    # Plugins

    # Cross Entropy Training
    print("Building Trainer")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        gpus=1,
        progress_bar_refresh_rate=100,
    )
    print("Training Model")
    trainer.fit(lightning_model, trainloader, valloader)


def main():
    args = parse_args()
    data_dir = args.data_dir
    smoke_test = args.smoke_test
    # Load Config
    constant_configs = {
        # Testing vars
        "smoke_test": smoke_test,
        # Constant across runs
        "data_dir": data_dir,
        "vocabulary_size": 2004,
        "pad_token": 0,
        "max_sequence_length": 30,
        "data_size": 1024,
        "end_token": 1,
        "start_token": 3,
        "max_detections": 50,
        "num_data_workers": 10
    }
    config = {
        # parameters to vary
        "key_size": tune.randint(8, 64),
        "value_size": tune.randint(8, 64),
        "dropout_rate": tune.uniform(0.1, 0.2),
        "num_encoder_layers": tune.randint(2, 5),
        "num_decoder_layers": tune.randint(2, 5),
        "out_size": tune.choice([2**x for x in range(7, 9)]),
        "feedforward_size": tune.choice([2**x for x in range(6, 9)]),
        "num_heads": tune.choice([2**x for x in range(1, 4)]),
        "num_memory_slots": tune.randint(4, 20),
        "batch_size": tune.randint(8, 64),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
    }
    asha_scheduler = tune.schedulers.ASHAScheduler(
        time_attr="training_iteration",
        metric="meteor",
        mode="max",
        max_t=100,
        grace_period=1,
        reduction_factor=2,
    )
    xe_opt = tune.with_parameters(xe_parameter_optimization, **constant_configs)
    analysis = tune.run(
        xe_opt,
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=100,
        scheduler=asha_scheduler,
        name="xe_param_opt",
        trial_dirname_creator=trial_name_generator
    )
    print(f"Best result: {analysis.best_result}")
    print(f"Best configuration: {analysis.best_config}")


if __name__ == "__main__":
    main()
