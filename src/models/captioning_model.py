import torch
import torch.nn as nn

import typing


class CaptioningModel(nn.Module):
    def __init__(self) -> typing.NoReturn:
        super().__init__()

    def get_construction_parameters(self) -> dict:
        raise NotImplementedError
