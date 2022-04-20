"""
This module contains the code for the meshed memory transformer. 
The meshed memory transformer consists of encoder and decoder stages.


At the end of the project, this module should contain the following:

- Meshed Memory Encoder
- Meshed Memory Decoder
- Bayesian Meshed Memory Encoder
- Bayesian Meshed Memory Decoder
"""

import numpy as np
from typing import Optional, NoReturn, OrderedDict
import torch.nn as nn
import torch.optim as optim
import torch

from .attention import AttentionLayer  # scaled dot product attention
from .Configuration import Configuration
import pytorch_lightning as pl
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any
from .model_utils import BeamSearch


class PWFeedForward(nn.Module):
    def __init__(self, att_size: int, feedforward_size: int, dropout_rate: float) -> NoReturn:
        """Implements the position wise feed forward map defined in the meshed memory paper.

        Args:
            att_size (int): output size of attention
            feedforward_size (int): intermediate size
            dropout_rate (float): dropout rate of PWFF layer
        """
        super().__init__()
        self.affine_inner = nn.Linear(att_size, feedforward_size)
        self.affine_outer = nn.Linear(feedforward_size, att_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(att_size)
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.affine_inner.weight)
        nn.init.xavier_uniform_(self.affine_outer.weight)

    def forward(self, att):
        x = self.relu(self.affine_inner(att))
        x = self.dropout(x)
        x = self.dropout(self.affine_outer(x))
        x = self.layer_norm(att + x)  # residual connection
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        key_size: int,
        value_size: int,
        num_heads: int,
        dropout_rate: float,
        feedforward_size: int,
        num_mem_slots: Optional[int] = None,
        pad_token: int = 0,
    ) -> NoReturn:
        """Implements a single encoding layer for the transformer as defined in the Meshed Memory paper

        Args:
            out_size (int): _description_
            key_size (int): _description_
            value_size (int): _description_
            num_heads (int): _description_
            dropout_rate (float): _description_
            feedforward_size (int): _description_
            num_mem_slots (Optional[int], optional): _description_. Defaults to None.
            pad_token (int, optional): _description_. Defaults to 0.

        Returns:
            NoReturn: _description_
        """
        super().__init__()
        self.pad_token = pad_token
        self.attention = AttentionLayer(
            out_size=out_size,
            key_size=key_size,
            value_size=value_size,
            dropout=dropout_rate,
            num_heads=num_heads,
            num_memory_slots=num_mem_slots,
        )
        self.pw_feedforward = PWFeedForward(
            att_size=out_size, feedforward_size=feedforward_size, dropout_rate=dropout_rate
        )

    def forward(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attention(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            attention_weights=attention_weights,
        )
        x = self.pw_feedforward(x)
        return x


class MeshedMemoryEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        num_layers: int,
        out_size: int,
        key_size: int,
        value_size: int,
        num_heads: int,
        dropout_rate: float,
        feedforward_size: int,
        num_mem_slots: Optional[int] = None,
        pad_token: int = 0,
    ) -> NoReturn:
        """Generates all of the encoder layers and adds additional layer norms before the encoder layer

        Args:
            num_layers (int): number of EncoderLayers to generate
            out_size (int): model output size
            key_size (int): number of keys
            value_size (int): number of values
            num_heads (int): number of heads
            dropout_rate (float): value in [0,1] determining dropout rate
            feedforward_size (int): size of positionwise feed foward layer size
            num_mem_slots (Optional[int], optional): number of memory slots to include. Defaults to None.
        """
        super().__init__()
        self.pad_token = pad_token
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    out_size=out_size,
                    key_size=key_size,
                    value_size=value_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    feedforward_size=feedforward_size,
                    num_mem_slots=num_mem_slots,
                )
                for _ in range(num_layers)
            ]
        )
        self.input_project = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_size)

        # initialization
        nn.init.xavier_uniform_(self.input_project.weight)

    def forward(self, x, attention_weights: Optional[torch.Tensor] = None):

        # (batch_size, 1, 1, sequence_length)
        x = self.relu(self.input_project(x))
        x = self.dropout(x)
        x = self.layer_norm(x)
        # generate masks to prevent data leaks. This also preserves the autoregressive
        # property of transformers. See Attention is All You Need paper
        mask = (
            (torch.sum(x, -1) == self.pad_token).unsqueeze(1).unsqueeze(1)
        )  # mask over sequence (batch_size, 1 , 1 , sequence_length)
        encoded_output = []
        for enc_layer in self.encoder_layers:
            x = enc_layer(keys=x, queries=x, values=x, attention_mask=mask, attention_weights=attention_weights)
            encoded_output.append(x.unsqueeze(1))

        # (batch_size, num_layers, seq_len, input_size)
        encoded_output = torch.cat(encoded_output, axis=1)
        return encoded_output, mask


# Note: References to equations (like (2)) refer to numbered equations in Meshed-Memory Transformer for Image Captioning paper

# Code is written using the following assumptions of dimensions, they could be (probably are) wrong
# Dimensions for reference (TODO correct if wrong, change code if needed)
# --------------------------------------------------
# Xbar_i: (encoded_size x 1), num_encoder_layers of these
# Y: (vocab_size x 1)
# C(Xbar_i, Y): (vocab_size x 1), the cross-attention operation
# alpha_i: vocab_size x 1


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        encoded_size: int,
        out_size: int,
        key_size: int,
        value_size: int,
        feedforward_size: int,
        num_heads: int,
        dropout_rate: float,
    ):
        """Implements a single Decoder layer for a meshed memory transformer

        Args:
            num_encoder_layers (int): number of encoder layers
            encoded_size (int): encoder output size
            out_size (int): model output size
            key_size (int): key size
            value_size (int): value size
            feed_forward_size (int): position wise feed forward layer output size
            num_heads (int): number of heads to use for attention
        """
        super().__init__()
        self.num_encoder_layers = num_encoder_layers

        # Self Attention Layer
        self.self_attention = AttentionLayer(
            out_size=out_size, key_size=key_size, value_size=value_size, num_heads=num_heads, dropout=dropout_rate
        )

        # Cross Attention Layer
        self.cross_attention = AttentionLayer(
            out_size=out_size, key_size=key_size, value_size=value_size, num_heads=num_heads
        )

        # Encode position into data
        self.pw_feedforward = PWFeedForward(out_size, feedforward_size=feedforward_size, dropout_rate=dropout_rate)

        # Fully Connected Layers
        self.fully_connected = nn.ModuleList([nn.Linear(2 * out_size, out_size) for _ in range(num_encoder_layers)])
        # Sigmoid function
        self.sigmoid = nn.Sigmoid()

        # Initialize Weights
        for layer in self.fully_connected:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask_pad: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attention = self.self_attention(x, x, x, self_attention_mask) * mask_pad

        out = None
        for i in range(self.num_encoder_layers):
            if i == 0:
                cross = self.cross_attention(
                    self_attention, encoder_output[:, i], encoder_output[:, i], cross_attention_mask
                )
                linear = self.sigmoid(self.fully_connected[i](torch.cat([self_attention, cross], -1)))
                out = cross * linear
            else:
                cross = self.cross_attention(
                    self_attention, encoder_output[:, i], encoder_output[:, i], cross_attention_mask
                )
                linear = self.sigmoid(self.fully_connected[i](torch.cat([self_attention, cross], -1)))
                out = out + cross * linear
        out = (out / np.sqrt(self.num_encoder_layers)) * mask_pad
        out = self.pw_feedforward(out) * mask_pad
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_layers: int,
        max_sequence_len: int,
        pad_token: int,
        out_size: int,
        key_size: int,
        value_size: int,
        feedforward_size: int,
        encoded_size: int,
        vocab_size: int,
        num_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_sequence_len
        self.pad_token = pad_token
        self.num_layers = num_layers
        self.encoded_size = encoded_size
        self.vocab_embedding = nn.Embedding(vocab_size, out_size, padding_idx=pad_token)

        # From Attention is All You Need, we need to compute sinusoidal embeddings to encode the
        # position of the tokens.
        p = torch.arange(max_sequence_len + 1, dtype=torch.float32).view(-1, 1)
        d = torch.arange(out_size // 2, dtype=torch.float32).view(1, -1)
        sine = torch.sin(p / 10000 ** (2 * d / out_size))
        cosine = torch.cos(p / 10000 ** (2 * d / out_size))
        pos_emb = torch.zeros((p.size(0), out_size))
        pos_emb[:, ::2] = sine
        pos_emb[:, 1::2] = cosine
        pos_emb[pad_token] = 0
        self.position_embedding = nn.Embedding.from_pretrained(pos_emb)
        self.output = nn.Linear(out_size, vocab_size)
        # Modules must be initialized in a Module List. Otherwise, the parameters won't be registered in PyTorch
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    num_encoder_layers=num_encoder_layers,
                    encoded_size=out_size,
                    out_size=out_size,
                    key_size=key_size,
                    value_size=value_size,
                    feedforward_size=feedforward_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        # initializatios
        nn.init.xavier_uniform_(self.output.weight)
        self.vocab_embedding.weight.data.uniform_(-1, 1)

    def generate_masks(self, y, seq_len):
        masks = (y != self.pad_token).unsqueeze(-1).float().to(y.device)
        # we need to block the left context to avoid cheating with the ground truth
        self_attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=y.device), diagonal=1)
        self_attention_mask = self_attention_mask.unsqueeze(0).unsqueeze(0)
        self_attention_mask = self_attention_mask.to(y.device)
        self_attention_mask = self_attention_mask + (y == self.pad_token).unsqueeze(1).unsqueeze(1).byte().to(y.device)
        self_attention_mask = self_attention_mask.gt(0)
        return masks.to(y.device), self_attention_mask.to(y.device)

    def forward(self, y, encoded, encoder_mask):
        batch_size, seq_len = y.size(0), y.size(1)
        masks, self_attention_mask = self.generate_masks(y, seq_len)
        pos = torch.arange(1, seq_len + 1).view(1, -1).expand(batch_size, -1).to(y.device)
        pos = pos.masked_fill(masks.squeeze(-1) == 0, 0).to(y.device)
        vocab_embedding = self.vocab_embedding(y)
        pos_embedding = self.position_embedding(pos)
        out = vocab_embedding + pos_embedding
        for decode in self.decoder_layers:
            out = decode(out, encoded, masks, self_attention_mask, encoder_mask)
        out = self.output(out)
        return out


class MeshedMemoryTransformer(pl.LightningModule):
    def __init__(
        self, config: Configuration, beam_size=5, inv_word_map: dict = None, reference_captions: dict = None
    ) -> None:
        self.comp_device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        # Test params/stuff
        self.beam_size = beam_size
        self.max_sequence_length = config["max_sequence_length"]
        self.previous_image = ""
        # Training Params
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.loss_func = config["loss_function"]
        self.end_token = config["end_token"]
        self.start_token = config["start_token"]
        self.pad_token = config["pad_token"]
        self.vocab_size = config["vocabulary_size"]
        self.example_input_array = (
            torch.randn(1, 50, 1024).to(self.comp_device),
            torch.randint(0, 1504, (1, 30)).to(self.comp_device),
        )
        # Construct Encoder
        self.encoder = MeshedMemoryEncoder(
            in_size=config["data_size"],
            num_layers=config["num_encoder_layers"],
            out_size=config["out_size"],
            key_size=config["key_size"],
            value_size=config["value_size"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"],
            feedforward_size=config["feedforward_size"],
            num_mem_slots=config["num_memory_slots"],
        )

        # Construct Decoder
        self.decoder = Decoder(
            num_encoder_layers=config["num_encoder_layers"],
            num_layers=config["num_decoder_layers"],
            max_sequence_len=config["max_sequence_length"],
            pad_token=config["pad_token"],
            out_size=config["out_size"],
            key_size=config["key_size"],
            value_size=config["value_size"],
            feedforward_size=config["feedforward_size"],
            encoded_size=config["out_size"],
            vocab_size=config["vocabulary_size"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"],
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, captions):
        encoded, masks = self.encoder(data)
        out = self.decoder(captions, encoded, masks)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 10000, 117659)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        self.lr_schedulers().step()
        inputs, captions, _ = batch
        out = self(inputs, captions)
        # remove start token for backpropagation
        y = captions[:, 1:].contiguous()
        y = y.view(-1)
        out = out[:, :-1].contiguous()
        out = out.view(-1, self.vocab_size)
        loss = self.loss_func(out, y, ignore_index=self.pad_token)
        self.log("train_loss", loss.detach(), batch_size=self.batch_size)
        tqdm_dict = {"train_loss": loss.detach()}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def on_train_start(self) -> None:
        self.logger.experiment.add_graph(self, self.example_input_array)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        inputs, caption, _ = batch
        out = self(inputs, caption)

        # remove start token for backpropagation
        y = caption[:, 1:].contiguous()
        y = y.view(-1)
        out = out[:, :-1].contiguous()
        out = out.view(-1, self.vocab_size)
        loss = self.loss_func(out, y, ignore_index=self.pad_token)
        tqdm_dict = {"val_loss": loss.detach()}
        self.log("val_loss", loss.detach(), prog_bar=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        output = OrderedDict({"val_loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict, "val_batch_preds": out})
        return output

    # Searches for best sequence with beam search, by default beam_size=5
    def test_step(self, batch, batch_idx):
        inputs, _, filename = batch
        filename = filename[0]
        # Avoid dupes
        if filename == self.previous_image:
            return
        self.previous_image = filename
        result = BeamSearch(self, inputs, 5)
        result["test_image_ids"] = filename
        return result


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
