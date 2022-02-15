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
from torchmetrics import BLEUScore


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
    loss_meter = AverageMeter("Loss")
    top5_acc_meter = AverageMeter("Top5Acc")
    batch_time_meter = AverageMeter("BatchTime")
    n = len(dataloader)
    stats = {
        "top 5 acc": f"{0:.4f}",
        "loss": f"{0:.4f}",
        "time t-minus": "unknown",
    }
    encoder.train()
    decoder.train()

    prev_time = 0
    start_time = time.time()
    for i, (images, captions, caption_lengths, _) in enumerate(
        pbar := tqdm(dataloader, f"Epoch {epoch+1} Train Progress ", postfix=stats)
    ):

        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # Feed Forward
        images = encoder(images)
        predictions, alphas = decoder(images, captions, caption_lengths, True)

        # remove <start> token for backpropagation
        y = captions[:, 1:]

        # remove unnecessary padding
        yhat = pack_padded_sequence(predictions, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]
        y = pack_padded_sequence(y, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]

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
        "loss": loss_meter.get_average(),
        "epoch_time": time.time() - start_time,
    }


def validate_sat_epoch(
    epoch: int,
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    word_map:dict,
    device: str = "cpu",
):
    """Validates a SAT epoch"""

    loss_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    bleu4 = BLEUScore(4)
    bleu4_meter = AverageMeter()
    inv_word_map = {v:k for k,v in word_map.items()}


    n = len(dataloader)
    stats = {
        "top 5 acc": "0",
        "loss": "?",
        "t-minus": "unknown",
    }
    encoder.eval()
    decoder.eval()

    prev_time = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (images, captions, caption_lengths, all_captions) in enumerate(
            pbar := tqdm(dataloader, f"Epoch {epoch+1} Validation Progress ", postfix=stats)
        ):
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            # Feed Forward
            images = encoder(images)
            predictions, alphas = decoder(images, captions, caption_lengths, True)

            # remove <start> token for backpropagation
            y = captions[:, 1:]

            # remove unnecessary padding
            yhat = pack_padded_sequence(predictions, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]
            y = pack_padded_sequence(y, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]

            # compute loss and doubly stochastic regularization
            loss = criterion(yhat, y)
            loss += C_ALPHA * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # Compute statistics
            loss_meter.update(loss.item() / sum(caption_lengths).item())

            # get  reference captions without additional characters
            references = []
            for  i in range(all_captions.shape[0]):#iterate over batches
                ref_caps = all_captions[i].tolist()
                temp_ref = []
                for ref in ref_caps: # iterate over available captions for image
                    # strip unnecessary tokens
                    tmp = [f"{inv_word_map[t]} " for t in ref if t not in [word_map["<pad>"], word_map["<start>"]]]
                    temp_ref.append("".join(tmp))
                references.append(temp_ref)
            
            _, preds = torch.max(predictions, dim=2)
            preds = preds.tolist()
            predicted_captions = []
            for i in range(len(preds)):
                p = preds[i]
                temp = [f"{inv_word_map[t]} " for t in p if t not in [word_map["<pad>"], word_map["<start>"]]]
                predicted_captions.append("".join(temp))
            
            assert len(predicted_captions) == len(references)
            end_time = time.time()
            batch_time = end_time - (prev_time if prev_time > 0 else start_time)
            prev_time = end_time
            batch_time_meter.update(batch_time)
            time_remaining = calc_time(batch_time_meter.get_average() * (n - i))
            top5_acc_meter.update(topk_accuracy(yhat, y, 5))
            bleu4_score = bleu4(predicted_captions, references)
            bleu4_meter.update(bleu4_score)
            pbar.set_postfix( { 
                "bleu4": f"{bleu4_meter.get_average():.4f}",
                "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
                "loss": f"{loss_meter.get_average():.4f}",
                "t-minus": time_remaining,
            })
        return {
            "bleu4": bleu4_meter.get_average(),
            "top 5 acc": top5_acc_meter.get_average(),
            "loss": loss_meter.get_average()
        }


            
