"""Provides utilities for model training"""
import torch
import torch.nn as nn
import torch.optim as optim
import typing
from tqdm import tqdm
from time import time
from typing import Optional, OrderedDict
from .captioning_model import CaptioningModel
import pytorch_lightning as pl


def BeamSearch(model: pl.LightningModule, inputs, beam_size: int):
    # Searches for best sequence with beam search, by default beam_size=5
    # Get all the captions
    best_sequences = torch.full([model.beam_size, model.max_sequence_length], model.pad_token).to(inputs.device)
    # Fill first column with start tokens
    best_sequences[:, 0] = torch.full([model.beam_size], model.start_token).to(inputs.device)
    # Keep list of best_scores
    best_scores = torch.full([model.beam_size], 0.0).to(inputs.device)

    # Do the first set of words manually, since each of the 5 best sequences are the same thing
    first_sequence = model(inputs, best_sequences[0, :].unsqueeze(0)).squeeze(0).to(inputs.device)
    # The network doesn't output a <start> token
    # so the first word is actually the 0th row, not the 1st
    first_words_under_consideration = first_sequence[0, :]
    first_words_under_consideration = torch.nn.functional.softmax(first_words_under_consideration, dim=-1)
    first_scores, first_best_words = torch.topk(first_words_under_consideration, model.beam_size)
    # print(first_scores)
    # first_scores originally is 1x1x5, this makes it 5
    first_scores = torch.log(first_scores).squeeze(0).squeeze(0)

    best_sequences[:, 1] = first_best_words
    best_scores += first_scores

    for i in range(2, model.max_sequence_length):
        scores = torch.full([model.beam_size * model.beam_size], 0.0).to(inputs.device)
        words = torch.full([model.beam_size * model.beam_size], model.pad_token).to(inputs.device)
        for j in range(0, model.beam_size):
            # First, if the sequence is finished (end_token encountered), we don't want to consider any successors to this node
            # We can check this by seeing if the sequence contains the end_token
            if model.end_token in best_sequences[j]:
                # To "disable" this node we set the first temp_score to the best_score for this node
                scores[model.beam_size * j] = best_scores[j]
                # We don't need to anything with the corresponding word since words was set to pad_token originally anyways
                # The rest of the scores to 0 (so they won't be chosen by topk)
                # Probability is 0, so log probability is -inf
                scores[range(model.beam_size * j + 1, model.beam_size * j + model.beam_size)] = -float("Inf")
            # Otherwise continue as normal
            else:
                # temp_seq is max_sequence_length x vocab_size (30 x 2004)
                temp_seq = model.forward(inputs, best_sequences[j, :].unsqueeze(0)).squeeze(0)

                # Only care about the words in the i-1th (because no start token) column, get this and turn it into row
                words_under_consideration = temp_seq[i - 1, :]
                words_under_consideration = torch.nn.functional.softmax(words_under_consideration, dim=-1)
                # Get the beam_size best words in words_under_consideration
                temp_scores, temp_best_words = torch.topk(words_under_consideration, model.beam_size)
                # Add the score of the jth best sequence to temp_scores
                temp_scores = torch.log(temp_scores) + best_scores[j]

                # Put the temp_scores and temp_best_words into the scores and words tensor
                scores[range(model.beam_size * j, model.beam_size * j + model.beam_size)] = temp_scores
                words[range(model.beam_size * j, model.beam_size * j + model.beam_size)] = temp_best_words
        # words now contains the beam_size^2 best successor words, i.e. the beam_size successors to each of the beam_size nodes
        # scores now contains the corresponding scores
        # Compute the 5 best choices for successor words (indices)...
        best_candidate_scores, best_candidate_words_idxs = torch.topk(scores, model.beam_size)
        # ... find the sequences we are tacking them on to
        sequences_for_best_candidate_words = torch.div(
            best_candidate_words_idxs, model.beam_size, rounding_mode="floor"
        )
        # ... and finally the words themselves
        best_candidate_words = words[best_candidate_words_idxs]
        new_best_sequences = torch.full([model.beam_size, model.max_sequence_length], model.pad_token)
        new_best_scores = torch.full([model.beam_size], 0.0)
        for k in range(0, model.beam_size):
            new_best_sequences[k, :] = best_sequences[sequences_for_best_candidate_words[k]]
            new_best_sequences[k, i] = best_candidate_words[k]
            new_best_scores[k] = best_candidate_scores[k]
        best_sequences = new_best_sequences
        best_scores = new_best_scores

    # Now, we have the 5 best sequences overall
    # Find the best one
    best_sequence_score, best_sequence_idx = torch.max(best_scores, dim=0, keepdim=True)
    best_sequence = best_sequences[best_sequence_idx, :].squeeze(0)
    result = {
        "test_batch_preds": best_sequence,
        # "test_image_ids": filename
    }
    result = OrderedDict(result)
    return result


class ModelComposition(CaptioningModel):
    """Wrapper class to compose model components"""

    def __init__(self, models: list) -> typing.NoReturn:
        super().__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)


def save_model_dict(
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


def load_model_dict(
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
