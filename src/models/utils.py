"""Provides utilities for model training"""
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import typing
from tqdm import tqdm
from time import time


def save_model_dict(model: nn.Module, path: str) -> typing.NoReturn:
    """Wrapper function for saving model state
    Args:
        model (nn.Module): The model to save
        path (str): The path to the checkpoint location
    """
    torch.save(model.state_dict(), path)


def load_model_dict(model: nn.Module, path: str) -> nn.Module:
    """Wrapper function for loading the model state
    Args:
        model (nn.Module): The model to save
        path (str): The path to the checkpoint location
    """
    model.load_state_dict(torch.load(path))
    return model


def nlp_metrics(predicted: str, targets: list) -> dict:
    """Computes standard NLP metrics on predicted sentences
    Metrics that should be computed here
     - BLEU-1
     - BLEU-2
     - BLEU-3
     - BLEU-4
     - CIDEr
     - ROUGE
     - METEOR

    Args:
        predicted (str): Predicted string
        targets (list): A list of possible representative sentences
    Returns:
        (dict): A dictionary where the keys indicate the metric, and the values are the metric results
    """
    raise NotImplementedError


class ModelComposition(nn.Module):
    """Wrapper class to compose model components"""

    def __init__(self, models: list) -> typing.NoReturn:
        super().__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)


class AverageMeter(object):
    """Simple class to compute a running average of some tracked value and can be printed"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._sum = 0
        self._count = 0

    def get_average(self) -> float:
        return self._sum / self._count

    def update(self, x: float) -> typing.NoReturn:
        self._sum += x
        self._count += 1

    def __str__(self) -> str:
        return f"{self._name}: {self.get_average():.4f}"


class Trainer(object):
    """Defines a simple model training class
    This class should run a training loop, track average training losses,
    calculate remaining time for training, and log losses and metrics on tensorboard
    """

    def __init__(
        self,
        train_func: typing.Callable,
        val_func: typing.Callable,
        epochs: int,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        device: str = "cpu",
    ) -> None:
        self._train = train_func
        self._validate = val_func
        self._epochs = epochs
        self._optimizer = optimizer
        self._trainloader = train_dataloader
        self._valloader = val_dataloader
        self._criterion = criterion
        self._validation_results = []

    def training_loop(self, model) -> typing.NoReturn:
        average_time = AverageMeter()
        for i in range(self._epochs):
            train_loss = AverageMeter()
            average_time = AverageMeter()  # average time in seconds
            start = time()
            metrics = {}
            for x, y in tqdm(self._trainloader, desc=f"Epoch {i+1} Train Progress "):
                tl = self._train(model, x, y, self._criterion, self._optimizer)
                train_loss.update(tl)
                metrics["train_loss"] = train_loss.get_average()
            for x, y in tqdm(self._valloader, desc=f"Epoch {i+1} validation_progress "):
                m = self._validate(model, x, y, self._criterion)
                metrics.update(m)
            average_time.update(time() - start)
            estimated_time_remaining = (self._epochs - i - 1) * average_time.get_average()
