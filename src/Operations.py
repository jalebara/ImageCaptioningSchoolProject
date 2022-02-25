# Class for operations: training and evaluating

from models.sat_model import SATModel
from data.augmentation import Flickr30k
from models.Configuration import Configuration
import os.path
import torch
import torch.nn as nn
class Trainer:
    # model_type currently unused, will be used with Bayesian SAT
    def __init__(self, config: Configuration, weights_file: str, exdir_data_location, smoke_test=False, fast_test=False, model_type="SAT"):
        self.config = config
        self.weights_file = weights_file
        self.smoke_test = smoke_test
        self.fast_test = fast_test
        self.model_type = model_type
        self.data_loader = Flickr30k(exdir_data_location, mode="test", smoke_test=True, fast_test=fast_test)
        self.vocabulary_size = config.vocab_size
        self.max_caption_size = self.data_loader.max_cap_len
        
        self.model = SATModel(self.config, self.vocabulary_size, self.max_caption_size)
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.encoder.to(self.config.device)
        self.model.decoder.to(self.config.device)

        self.criterion = nn.CrossEntropyLoss

        # Load the weights, if the weights file exists
        if os.path.exists(weights_file):
            print("Weights file exists, loading")
            state = torch.load(weights_file, config.device)
            self.model.encoder.load_state_dict(state["encoder"])
            self.model.decoder.load_state_dict(state["decoder"])
            self.model.decoder_optimizer.load_state_dict(state["optimizer"])

# class Evaluator:
    
