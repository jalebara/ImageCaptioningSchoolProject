"""
This module contains the code for the meshed memory transformer. 
The meshed memory transformer consists of encoder and decoder stages.


At the end of the project, this module should contain the following:

- Meshed Memory Encoder
- Meshed Memory Decoder
- Bayesian Meshed Memory Encoder
- Bayesian Meshed Memory Decoder
"""

import torch.nn as nn
import torch
from attention import Attention # Scaled dot product attention


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


# Note: References to equations (like (2)) refer to numbered equations in Meshed-Memory Transformer for Image Captioning paper

# Code is written using the following assumptions of dimensions, they could be (probably are) wrong
# Dimensions for reference (TODO correct if wrong, change code if needed)
# --------------------------------------------------
# Xbar_i: (encoded_size x 1), num_encoder_layers of these
# Y: (vocab_size x 1)
# C(Xbar_i, Y): (vocab_size x 1), the cross-attention operation
# alpha_i: vocab_size x 1

class DecoderLayer(nn.Module):
    # See comment above forward() for explanation of num_encoder_layers and encoded_size
    def __init__(self, num_encoder_layers, encoded_size, vocab_size, num_heads):
        super().__init__()

        # TODO: I guessed at what the arguments should be. Fix this, as idk what key_size and value_size should be
        # Scaled dot product attention
        self.sdp_attention = Attention(out_size=vocab_size, key_size=1, value_size=num_encoder_layers, num_heads=num_heads)
        
        # These represent W_q, W_k, and W_v in (2) of Meshed-Memory Transformer paper
        # "proj" because the paper describes these as projections
        self.Wq_proj = nn.Linear(num_encoder_layers, encoded_size, bias=False)
        self.Wk_proj = nn.Linear(num_encoder_layers, encoded_size, bias=False)
        self.Wv_proj = nn.Linear(num_encoder_layers, encoded_size, bias=False)

        # This should be such that W is 2d x d, but I can't find what "d" is in the paper lol
        # TODO figure out what "d" is (I think it's vocab_size?)
        self.fully_connected = nn.Linear(2*vocab_size, vocab_size, bias=True)

        # Sigmoid function
        self.sigmoid = nn.Sigmoid()

    # TODO Unsure of what this is in comparison to self-attention...
    # For now, assuming this is the same as self-attention, defined in (2)
    # Idk where this is used, so it's probably not even necessary
    def masked_self_attention(self, Y):
        return self.sdp_attention(self.Wq_proj(Y), self.Wk_proj(Y), self.Wv_proj(Y))

    def cross_attention(self, Xbar_i, Y):
        return self.sdp_attention(self.Wq_proj(Y), self.Wk_proj(Xbar_i), self.Wv_proj(Xbar_i))

    # X has num_encoder_layers cols, encoder_size rows (TODO change encoder_size to better name)
    def forward(self, X, Y):
        # This part is in figure (2) but absent in the text description
        # Assuming it is not needed
        # Pass Y through masked self-attention to generate Q
        # Q = self.masked_self_attention(Y)

        # From the paper...
        # (6) Ybar = M_mesh(Xbar, Y) = sum_{i=1}^{num_encoder_layers} alpha_i * C(Xbar_i, Y)
        # (7) C(Xbar_i, Y) = Attention(Wq*Y, Wk*Xbar_i, Wv*Xbar_i) [note: self.cross_attention implements this]
        # (8) alpha_i = Sigmoid(Wi*[Y, C(Xbar_i, Y)] + b_i)

        alpha = []
        Ybar = torch.tensor(torch.zeros([self.vocab_size, 1]))
        # For each Xbar_i
        for i in range(0, self.num_encoder_layers):
            C = self.cross_attention(X[i,:], Y) # Cross attention
            concat = torch.cat(Y, C)
            projected = self.fully_connected(concat)
            alpha[i] = self.sigmoid(projected)
            Ybar += alpha[i] * C

        return Ybar

    
class Decoder(nn.Module):
    def __init__(self, num_layers, encoded_size, vocab_size, num_heads) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.decoder_layers = []
        for i in range(0, num_layers):
            self.decoder_layers.append(DecoderLayer(num_layers, encoded_size, vocab_size, num_heads))
        

    def forward(self, X, Y):
        Y_ = Y
        for i in range(0, self.num_layers):
            Y_ = self.decoder_layers[i](X, Y_)

        return Y_
            


class MeshedMemoryTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class BayesianEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class BayesianDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class BayesianMeshedMemoryTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
