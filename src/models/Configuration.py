# Just a convenient class to consolidate all our configurations
# If using a new configuration, please create a new class here and inherit from Configuration
# This will hopefully prevent size mismatches in the future when using pretrained weights
# Configuration should be stored when saving weights

from typing import NoReturn, Optional
import torch.optim as optim

import torch.nn.functional as F

class Configuration:
    def __init__(self, config_dict: Optional[dict] = None):
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
                "loss_function": F.cross_entropy,
                "optimizer": optim.Adam,
            }
        else:
            self.config_dict = config_dict
    def __getitem__(self, idx):
        return self.config_dict[idx]
    
class ConfigurationForWeightsFile(Configuration):
    def __init__(self):
        super().__init__()
        self.config_dict["encoder_size"] = 2048
        self.config_dict["vocabulary_size"] = 1004

class TransformerConfiguration(Configuration):
    def __init__(self, config_dict: Optional[dict] = None) -> NoReturn:
        super().__init__(config_dict)
        if config_dict is None:
            update_dict = {
                "key_size": 32,
                "value_size": 32,
                "dropout_rate": 0.1,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "out_size": 256,
                "feedforward_size": 256,
                "num_heads": 8,
                "vocabulary_size": 2004,
                "pad_token": 0,
                "max_sequence_length": 30,
                "data_size": 1024,
                "end_token": 1,
                "start_token": 3,
                "num_memory_slots": 8,
                "batch_size": 64,
                "max_detections":50
            }
        else:
            update_dict = config_dict
        self.config_dict.update(update_dict)

class MediumTransformerConfiguration(Configuration):
    def __init__(self, config_dict: Optional[dict] = None) -> NoReturn:
        super().__init__(config_dict)
        if config_dict is None:
            update_dict = {
                "key_size": 32,
                "value_size": 32,
                "dropout_rate": 0.1,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "out_size": 256,
                "feedforward_size": 256,
                "num_heads": 8,
                "vocabulary_size": 2004,
                "pad_token": 0,
                "max_sequence_length": 30,
                "data_size": 1024,
                "end_token": 1,
                "start_token": 3,
                "num_memory_slots": 8,
                "batch_size": 64,
                "max_detections":50
            }
        else:
            update_dict = config_dict
        self.config_dict.update(update_dict)
        
class MediumDroppyTransformerConfiguration(Configuration):
    def __init__(self, config_dict: Optional[dict] = None) -> NoReturn:
        super().__init__(config_dict)
        if config_dict is None:
            update_dict = {
                "key_size": 32,
                "value_size": 32,
                "dropout_rate": 0.15,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "out_size": 256,
                "feedforward_size": 256,
                "num_heads": 8,
                "vocabulary_size": 2004,
                "pad_token": 0,
                "max_sequence_length": 30,
                "data_size": 1024,
                "end_token": 1,
                "start_token": 3,
                "num_memory_slots": 8,
                "batch_size": 64,
                "max_detections":50
            }
        else:
            update_dict = config_dict
        self.config_dict.update(update_dict)
               
class MediumDeepTransformerConfiguration(Configuration):
    def __init__(self, config_dict: Optional[dict] = None) -> NoReturn:
        super().__init__(config_dict)
        if config_dict is None:
            update_dict = {
                "key_size": 32,
                "value_size": 32,
                "dropout_rate": 0.1,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "out_size": 256,
                "feedforward_size": 256,
                "num_heads": 8,
                "vocabulary_size": 2004,
                "pad_token": 0,
                "max_sequence_length": 30,
                "data_size": 1024,
                "end_token": 1,
                "start_token": 3,
                "num_memory_slots": 8,
                "batch_size": 64,
                "max_detections":50
            }
        else:
            update_dict = config_dict
        self.config_dict.update(update_dict)
        
class BigTransformerConfiguration(Configuration):
    def __init__(self, config_dict: Optional[dict] = None) -> NoReturn:
        super().__init__(config_dict)
        if config_dict is None:
            update_dict = {
                "key_size": 64,
                "value_size": 64,
                "dropout_rate": 0.2,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "out_size": 512,
                "feedforward_size": 512,
                "num_heads": 8,
                "vocabulary_size": 2004,
                "pad_token": 0,
                "max_sequence_length": 30,
                "data_size": 1024,
                "end_token": 1,
                "start_token": 3,
                "num_memory_slots": 8,
                "batch_size": 64,
                "max_detections":50
            }
        else:
            update_dict = config_dict
        self.config_dict.update(update_dict)