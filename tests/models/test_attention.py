""" Unit tests for scaled dot product attention
"""
import numpy as np
import torch

from models.attention import BayesianAttention


def test_bayesian_attention():
    image_tokens = torch.tensor(np.random.randn(10,50))
    cap_samples = torch.tensor(np.random.randint(0, 2000, size=(10,30)))
    
    attention_model = BayesianAttention(0.5, )
    
