import torch
import torch.nn as nn
import pytorch_lightning as pl

import typing


class CaptioningModel(pl.LightningModule):
    def __init__(self) -> typing.NoReturn:
        super().__init__()

    def get_construction_parameters(self) -> dict:
        raise NotImplementedError
