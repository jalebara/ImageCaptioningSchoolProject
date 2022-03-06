"""
This module contains the attention modules used in this project. At the end of the project,
This module should contain the following

- Single Headed Visual Attention
- Multi Headed Visual Attention
- Bayesian Single Headed Visual Attention
- Bayesian Multi Headed Visual Attention
"""

from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NoReturn, Optional
import math


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

    def __init__(self, vocab_size: int, key_size: int, value_size: int, num_heads: int) -> NoReturn:
        """Initializer function for scaled dot product attention
        Args:
            vocab_size (int):  the number of words in the model's vocabulary
            key_size (int): Key dimension
            value_size (int): size of feature array
            num_heads (int): The number of heads in attention
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1/math.sqrt(key_size)

        # Layers to reshape inputs and generate multiheaded subspaces
        # Linear layers represent the flattened attention heads
        self.keygen = nn.Linear(vocab_size, num_heads * key_size)
        self.querygen = nn.Linear(vocab_size, num_heads * key_size)
        self.valuegen = nn.Linear(vocab_size, num_heads * value_size)
        self.output = nn.Linear(num_heads * value_size, vocab_size)

    def _initialize_weights(self):
        """Initializes the model weights to a uniform distribution.
        The main reason behind this is that using the uniform distribution
        to initialize the model weights encourages faster convergence.
        """
        raise NotImplementedError

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        """ Performs the forward pass of the Scaled Dot Product Attention.

        """
        num_queries = queries.size(1)
        num_keys = keys.size(1)
        batch_size = keys.size(0)
        
        # Flattened keys queries, and values
        queries = self.querygen(queries)
        keys = self.keygen(keys)
        values = self.valuegen(values)

        # Unflatten keys, queries, and values
        # shape should be (batch_size, heads, *, *)
        queries = queries.view(batch_size, num_queries, self.num_heads, self.key_size)
        queries = queries.permute(0,2,1,3)

        keys = keys.view(batch_size, num_keys, self.num_heads, self.key_size)
        keys = keys.permute(0,2,1,3)

        values = values.view(batch_size, )
        
        return 11, 11


class MultiHeadedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

class MemoryEnhancedAttention(nn.Module):
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
