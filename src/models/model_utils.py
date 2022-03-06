"""Provides utilities for model training"""
import torch
import torch.nn as nn
import torch.optim as optim
import typing
from tqdm import tqdm
from time import time
from typing import Optional
from .captioning_model import CaptioningModel


class ModelComposition(CaptioningModel):
    """Wrapper class to compose model components"""

    def __init__(self, models: list) -> typing.NoReturn:
        super().__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)


def save_project_state(
    path: str,
    encoder: CaptioningModel,
    decoder: CaptioningModel,
    encoder_optimizer: Optional[optim.Optimizer] = None,
    decoder_optimizer: Optional[optim.Optimizer] = None,
    epoch: Optional[int] = None,
) -> typing.NoReturn:
    """Wrapper function for saving the project state
    Args:
        model (nn.Module): The model to save
        path (str): The path to the checkpoint location
    """
    state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "encoder_construct": encoder.get_construction_parameters(),
        "decoder_construct": decoder.get_construction_parameters(),
    }
    if encoder_optimizer is not None:
        state["encoder_optimizer"] = encoder_optimizer.state_dict()
    if decoder_optimizer is not None:
        state["decoder_optimizer"] = decoder_optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    torch.save(state, path)


def load_project_state(
    path: str,
    encoder: CaptioningModel,
    decoder: CaptioningModel,
    encoder_optimizer: Optional[optim.Optimizer] = None,
    decoder_optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
) -> tuple:
    """Wrapper function for loading the project state
    Args:
        model (nn.Module): The model to save
        path (str): The path to the checkpoint location
    Returns:
        (tuple): The encoder, decoder, encoder optimizer, decoder optimizer, and epoch number
    """
    state = torch.load(path, map_location=device)
    encoder = encoder(**state["encoder_construct"])
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    decoder = decoder(**state["decoder_construct"])
    epoch = 0
    if encoder_optimizer is not None:
        encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        decoder_optimizer.load_state_dict(state["decoder_optimizer"])
    if "epoch" in state.keys():
        epoch = state["epoch"]
    return encoder, decoder, encoder_optimizer, decoder_optimizer, epoch


def count_parameters(model: nn.Module):
    """Used for determining the model size
    Args:
        model (nn.Module): the model to examine
    Returns:
        (tuple): A tuple of the number of the trainable and total parameters, respectively.
    """
    total = sum([p.numel() for p in model.parameters()])
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return trainable, total
