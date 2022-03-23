import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import typing
from tqdm import tqdm
from time import time
import numpy as np
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from functools import wraps


class NLPMetricAggregator(object):
    """Class for aggregating caption hypotheses and generating NLP metrics"""

    def __init__(self) -> None:
        self._predicted_captions = []
        self._reference_captions = []
        self.bleu1 = BLEUScore(1)
        self.bleu2 = BLEUScore(2)
        self.bleu3 = BLEUScore(3)
        self.bleu4 = BLEUScore(4)
        self.rouge = ROUGEScore()

    def update(self, predicted: list, reference: list):
        """Store predictions and references"""
        self._predicted_captions.extend(predicted)
        self._reference_captions.extend(reference)

    def generate_metric_summaries(self):
        """Retrieves NLP metrics
        Returns:
            (dict): A dictionary of NLP metrics generated from stored hypotheses and references
        """
        meteor_meter = AverageMeter("Meteor Mean")
        for pred, refs in zip(self._predicted_captions, self._reference_captions):
            meteor_meter.update(meteor_score([word_tokenize(r) for r in refs], word_tokenize(pred)))
        return {
            "bleu1": self.bleu1(self._predicted_captions, self._reference_captions),
            "bleu2": self.bleu2(self._predicted_captions, self._reference_captions),
            "bleu3": self.bleu3(self._predicted_captions, self._reference_captions),
            "bleu4": self.bleu4(self._predicted_captions, self._reference_captions),
            "rouge_fmeasure": self.rouge(self._predicted_captions, self._reference_captions)["rouge1_fmeasure"],
            "meteor": np.round(meteor_meter.get_average(), 4),
        }


class AverageMeter(object):
    """Simple class to compute a running average of some tracked value and can be printed"""

    def __init__(self, name: str = "mean") -> None:
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
        patience: int = 5,
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

        metric = abs(metric)

        if (self.mode == "min" and self.best_metric - self.delta <= metric) or (
            self.mode == "max" and self.best_metric + self.delta >= metric
        ):
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


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
