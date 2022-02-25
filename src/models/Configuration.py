# Just a convenient class to consolidate all our configurations
# If using a new configuration, please create a new class here and inherit from Configuration
# This will hopefully prevent size mismatches in the future when using pretrained weights
# Configuration should be stored when saving weights

import torch

class Configuration:
    def __init__(self):
        self.embedding_size = 512
        self.scheduled_sampling_convergence = 0.2
        self.learning_rate = 1e-4
        self.epochs = 60
        self.hidden_size = 512
        self.attention_size = 512
        self.dropout_rate = 0.75
        self.encoder_size = 1408
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64

        # Encoder variables
        self.encoded_size = 7
        self.pretrained = True
        self.freeze = True
        self.unfreeze_last = 0

        self.vocab_size = 2004

class ConfigurationForWeightsFile(Configuration):
    def __init__(self):
        super().__init__()
        self.encoder_size = 2048
        self.vocab_size=1004
