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


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


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
