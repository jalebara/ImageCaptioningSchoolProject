# Just a convenient class to consolidate all our configurations
# If using a new configuration, please create a new class here and inherit from Configuration
# This will hopefully prevent size mismatches in the future when using pretrained weights
# Configuration should be stored when saving weights

import torch


class Configuration:
    def __init__(self, config_dict: dict = None):
        if config_dict is None:
            self.config_dict = {
                "embedding_size": 512,
                "scheduled_sampling_convergence": 0.2,
                "learning_rate": 1e-4,
                "epochs": 60,
                "hidden_size": 512,
                "attention_size": 512,
                "dropout_rate": 0.75,
                "encoder_size": 1408,
                "batch_size": 64,
                "encoded_size": 7,
                "pretrained": True,
                "freeze": True,
                "unfreeze_last": 0,
                #'vocabulary_size': 2004
                "vocabulary_size": 1004,
            }
        else:
            self.config_dict = config_dict


class ConfigurationForWeightsFile(Configuration):
    def __init__(self):
        super().__init__()
        self.config_dict["encoder_size"] = 2048
        self.config_dict["vocabulary_size"] = 1004
