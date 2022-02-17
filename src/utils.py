import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import typing
from tqdm import tqdm
from time import time
import numpy as np


class ModelComposition(nn.Module):
    """Wrapper class to compose model components"""

    def __init__(self, models: list) -> typing.NoReturn:
        super().__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)


class AverageMeter(object):
    """Simple class to compute a running average of some tracked value and can be printed"""

    def __init__(self, name: str="mean") -> None:
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


class EarlyStopping(object):
    def __init__(
        self,
        checkpoint_path: str,
        delta: float = 0.005,
        mode: str = "max",
        patience: int = 8,
        report_func: typing.Callable = print,
    ) -> None:
        self.patience = patience
        self.checkpoint = checkpoint_path
        self.mode = mode
        self.counter = 0
        self.delta = delta
        self.report = report_func
        self.stop = False
        self.best_metric = np.inf if mode == "min" else -np.inf

    def __call__(self, metric: float, state: dict):

        metric  = abs(metric)

        if (self.mode == "min" and self.best_metric - self.delta <= metric) or (self.mode == "max" and self.best_metric + self.delta >= metric):
            self.counter += 1
            self.report(f"Model did not improve {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_metric = metric
            torch.save(state, self.checkpoint)
            self.counter = 0


def calc_time(t: float) -> str:
    hours = int(t) // 3600
    minutes = int(t - hours * 3600) // 60
    seconds = int(t - hours * 3600 - minutes * 60)
    return f"{hours}h{minutes}m{seconds}s"


def topk_accuracy(yhat: torch.Tensor, y: torch.Tensor, k: int):
    batch_size = y.size(0)
    _, idx = yhat.topk(k, 1, True, True)
    correct = idx.eq(y.view(-1, 1).expand_as(idx))
    correct = correct.view(-1).float().sum()
    return correct.item() / batch_size * 100.0
