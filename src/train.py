""" Training and validation functions for the models

"""
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter, calc_time, topk_accuracy


# Constants
C_ALPHA = 1.0  # regularization parameter to ensure that model attends to the whole image


def train_sat_epoch(
    epoch: int,
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
):
    """Trains a single epoch for the Show, Attend, and Tell Model."""
    loss_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    n = len(dataloader)
    stats = {
        "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
        "loss": f"{loss_meter.get_average():.4f}",
        "time t-minus": "unknown",
    }
    encoder.train()
    decoder.train()

    prev_time = 0
    start_time = time.time()
    for i, (images, captions, caption_lengths) in enumerate(
        pbar := tqdm(dataloader, f"Epoch {epoch+1} Train Progress ", postfix=stats)
    ):

        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # Feed Forward
        images = encoder(images)
        predictions, sorted_captions, decoded_lengths, alphas, _ = decoder(images, captions, caption_lengths, True)

        # remove <start> token for backpropagation
        y = sorted_captions[:, 1:, :]

        # remove unnecessary padding
        yhat, _ = pack_padded_sequence(predictions, decoded_lengths, batch_first=True)
        y, _ = pack_padded_sequence(y, decoded_lengths, batch_first=True)

        # compute loss and doubly stochastic regularization
        loss = criterion(yhat, y)
        loss += C_ALPHA * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # back propagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update statistics
        end_time = time.time()
        batch_time = end_time - (prev_time if prev_time > 0 else start_time)
        prev_time = end_time
        batch_time_meter.update(batch_time)
        time_remaining = calc_time(batch_time_meter.get_average() * (n - i))
        top5_acc_meter.update(topk_accuracy(yhat, y, 5))
        loss_meter.update(loss.cpu().item())

        stats = {
            "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
            "loss": f"{loss_meter.get_average():.4f}",
            "t-minus": time_remaining,
        }
        pbar.set_postfix(stats)
    return {
        "top 5 acc": top5_acc_meter.get_average(),
        "train_loss": loss_meter.get_average(),
        "epoch_time": time.time() - start_time,
    }


def validate_sat_epoch(
    epoch: int,
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
):
    """Validates a SAT epoch"""

    loss_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    bleu1_meter = AverageMeter()
    bleu2_meter = AverageMeter()
    bleu3_meter = AverageMeter()
    bleu4_meter = AverageMeter()
    meteor_meter = AverageMeter()
    cider_meter = AverageMeter()
    rouge_meter = AverageMeter()
    
    n = len(dataloader)
    stats = {
        "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
        "loss": f"{loss_meter.get_average():.4f}",
        "t-minus": "unknown",
    }
    encoder.eval()
    decoder.eval()

    prev_time = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(
            pbar := tqdm(dataloader, f"Epoch {epoch+1} Validation Progress ", postfix=stats)
        ):
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            # Feed Forward
            images = encoder(images)
            predictions, sorted_captions, decoded_lengths, alphas, _ = decoder(images, captions, caption_lengths, True)

            # remove <start> token for backpropagation
            y = sorted_captions[:, 1:, :]

            # remove unnecessary padding
            yhat, _ = pack_padded_sequence(predictions, decoded_lengths, batch_first=True)
            y, _ = pack_padded_sequence(y, decoded_lengths, batch_first=True)

            # compute loss and doubly stochastic regularization
            loss = criterion(yhat, y)
            loss += C_ALPHA * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # Compute statistics
            loss_meter.update(loss.item()/sum(decoded_lengths))
            # Learning Metrics



