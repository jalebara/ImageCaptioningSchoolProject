# Class for operations: training and evaluating

from models.sat_model import SATModel
from data.augmentation import Flickr30k
from models.Configuration import Configuration
import os.path
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from utils import AverageMeter, calc_time, topk_accuracy
from torchmetrics import BLEUScore
import time
from torch.utils.data import DataLoader


# model_type in ctor currently unused, will be used with Bayesian SAT
class Trainer:
    # If a config is given, use that and create a new weights file
    # If a config is not given, check that the weights file exists and if it does, load that
    def __init__(self, weights_file: str, exdir_data_location, smoke_test=False, fast_test=False, model_type="SAT", config: Configuration=None, batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_file = weights_file
        self.smoke_test = smoke_test
        self.fast_test = fast_test
        self.model_type = model_type
        # TODO: once done debugging, change mode to "train"
        self.data = Flickr30k(exdir_data_location, mode="train", smoke_test=smoke_test, fast_test=fast_test)
        self.data_loader = DataLoader(self.data, num_workers=8, batch_size=batch_size)
        self.max_caption_size = self.data.max_cap_len
        self.epoch = 1

        # Meters
        self.loss_meter = AverageMeter("Loss")
        self.top5_acc_meter = AverageMeter("Top5Acc")
        self.batch_time_meter = AverageMeter("BatchTime")
        self.n = len(self.data_loader)
        self.bleu4 = BLEUScore(4)
        self.bleu4_meter = AverageMeter()
        self.inv_word_map = {v: k for k, v in self.data.word_map.items()}
        
        if (config is not None):
            if (not os.path.exists(weights_file)):
                self.config = config
                self.model = SATModel(self.config, self.max_caption_size, self.device)
                self.model.encoder.train()
                self.model.decoder.train()
                self.model.encoder.to(self.device)
                self.model.decoder.to(self.device)
            else:
                print("Weights file exists, but configuration was given")
        else:
            # Load the weights, if the weights file exists
            if os.path.exists(weights_file):
                print("Weights file exists, loading...")
                state = torch.load(weights_file, self.device)
                self.config = state["config"]
                self.model = SATModel(self.config, self.max_caption_size, self.device)
                self.model.encoder.load_state_dict(state["encoder"])
                self.model.decoder.load_state_dict(state["decoder"])
                self.model.decoder_optimizer.load_state_dict(state["optimizer"])
                self.epoch = 1 if "epoch" not in state else state["epoch"]
            else:
                print("Weights file does not exist, and no configuration was given")
        print("Trainer successfully loaded!")

    def train_one_epoch(self):
        self.model.encoder.train()
        self.model.decoder.train()
        stats = {
            "top 5 acc": f"{0:.4f}",
            "loss": f"{0:.4f}",
            "time t-minus": "unknown",
        }
        start_time = time.time()
        prev_time = time.time()
        for i, (images, captions, caption_lengths, all_captions, _) in enumerate(
            pbar := tqdm(self.data_loader, f"Epoch {self.epoch+1} Train Progress ", postfix=stats)
        ):
            # Forward
            predictions,alphas=self.model.forward(images, captions, caption_lengths)

            # remove <start> token for backpropagation
            y = captions[:, 1:]

            # remove unnecessary padding
            yhat = pack_padded_sequence(
                predictions, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False
            )[0]
            y = pack_padded_sequence(y, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]

            # Backward
            loss = self.model.backward(yhat, y)

            # Processing captions and predictions
            # get  reference captions without additional characters
            references = []
            for j in range(all_captions.shape[0]):  # iterate over batches
                ref_caps = all_captions[j].tolist()
                temp_ref = []
                for ref in ref_caps:  # iterate over available captions for image
                    # strip unnecessary tokens
                    tmp = [f"{self.inv_word_map[t]} " for t in ref if t not in [self.word_map["<pad>"], self.word_map["<start>"]]]
                    temp_ref.append("".join(tmp))
                references.append(temp_ref)

            _, preds = torch.max(predictions, dim=2)
            preds = preds.tolist()
            predicted_captions = []
            for k in range(len(preds)):
                p = preds[k]
                temp = [f"{self.inv_word_map[t]} " for t in p if t not in [self.word_map["<pad>"], self.word_map["<start>"]]]
                predicted_captions.append("".join(temp))

            assert len(predicted_captions) == len(references)
            
            # Metrics and progress bar updates
            self.top5_acc_meter.update(topk_accuracy(yhat, y, 5))
            self.loss_meter.update(loss.cpu().item())
            bleu4_score = self.bleu4(predicted_captions, references)
            self.bleu4_meter.update(bleu4_score)
            end_time = time.time()
            batch_time = end_time - prev_time
            prev_time = end_time
            self.batch_time_meter.update(batch_time)
            time_remaining = calc_time(self.batch_time_meter.get_average() * (self.n - i))
            pbar.set_postfix(
                {
                    "bleu4": f"{self.bleu4_meter.get_average():.4f}",
                    "top 5 acc": f"{self.top5_acc_meter.get_average():.4f}",
                    "loss": f"{self.loss_meter.get_average():.4f}",
                    "t-minus": time_remaining,
                }
            )
            
    

# class Evaluator:
    
