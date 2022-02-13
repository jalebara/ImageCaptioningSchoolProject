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


class Attention(nn.Module):
    def __init__(self, feature_vector_size, intermediate_network_size=100) -> None:
        super().__init__()
        self.feature_shaper = nn.Linear(feature_vector_size, intermediate_network_size)
        self.hidden_state_shaper = nn.Linear(intermediate_network_size, intermediate_network_size)
        self.attention_model = nn.Linear(intermediate_network_size, feature_vector_size)
        self.feature_vector_size = feature_vector_size

    def forward(self, feature_vectors, hidden_state):

        # Shape vectors so I can add them together
        fv_shaped = self.feature_shaper(feature_vectors)
        hidden_state_shaped = self.hidden_state_shaper(hidden_state)

        # Compute e in the paper
        e = self.attention_model(feature_vectors + hidden_state)

        # alpha = softmax(e)
        alpha = nn.Softmax(e)

        # z = sum alpha_i a_i
        z = torch.zeros(feature_vector_size)
        for i in range(0,feature_vector_size-1):
            z = z + alpha[i]*feature_vectors[:][i]

        # Return values
        return (z, alpha)

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
