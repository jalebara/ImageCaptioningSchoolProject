"""
This module contains the attention modules used in this project. At the end of the project,
This module should contain the following

- Single Headed Visual Attention
- Multi Headed Visual Attention
- Bayesian Single Headed Visual Attention
- Bayesian Multi Headed Visual Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SATAttention(nn.Module):
    def __init__(self, encoder_size: int, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.feature_shaper = nn.Linear(encoder_size, attention_size)
        self.hidden_state_shaper = nn.Linear(hidden_size, attention_size)
        self.attention_model = nn.Linear(
            attention_size, 1
        )  # attention for each annotation vector in aᵢ for i = 1 ... L
        self.feature_vector_size = encoder_size

    def forward(self, feature_vectors, hidden_state):

        # Shape vectors so I can add them together
        fv_shaped = self.feature_shaper(feature_vectors)
        hidden_state_shaped = self.hidden_state_shaper(hidden_state).unsqueeze(1)

        # Compute e in the paper
        e = self.attention_model(F.relu(fv_shaped + hidden_state_shaped)).squeeze(2)

        # alpha = softmax(e)
        alpha = F.softmax(e, dim=1)

        # z = sum alpha_i a_i
        zhat = (feature_vectors * alpha.unsqueeze(2)).sum(dim=1)

        # Return values
        return (zhat, alpha)


class Attention(nn.Module):
    """Implements Scaled Dot Product Attention"""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class MultiHeadedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class BayesianAttention(Attention):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class BayesianMultiHeadedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


def train_single_epoch():
    raise NotImplementedError
