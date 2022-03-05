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
    def __init__(self, weights_file: str, exdir_data_location, smoke_test=False, fast_test=False, model_type="SAT", config: Configuration=None, criterion=nn.CrossEntropyLoss(), batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights_file = weights_file
        self.smoke_test = smoke_test
        self.fast_test = fast_test
        self.model_type = model_type
        
        self.data = Flickr30k(exdir_data_location, mode="train", smoke_test=smoke_test, fast_test=fast_test)
        self.data_loader = DataLoader(self.data, num_workers=0, batch_size=batch_size)

        self.validate_data = Flickr30k(exdir_data_location, mode="valid", smoke_test=smoke_test, fast_test=fast_test)
        self.validate_data_loader = DataLoader(self.validate_data, num_workers=8, batch_size=batch_size)
        
        self.max_caption_size = self.data.max_cap_len
        self.epoch = 0

        self.n = len(self.data_loader)
        self.bleu4 = BLEUScore(4)

        self.criterion = criterion
        
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
                self.epoch = 0 if "epoch" not in state else state["epoch"]
            else:
                print("Weights file does not exist, and no configuration was given")
        print("Trainer successfully loaded!")

    def train_one_epoch(self):
        # Meters
        loss_meter = AverageMeter("Loss")
        top5_acc_meter = AverageMeter("Top5Acc")
        batch_time_meter = AverageMeter("BatchTime")
        bleu4_meter = AverageMeter()
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

            y = self.remove_caption_padding(captions, caption_lengths, True)
            yhat = self.remove_caption_padding(predictions, caption_lengths, False)

            # Updates
            loss = self.update(yhat, y, alphas)

            # Processing captions and predictions
            # get  reference captions without additional characters
            references = []
            for j in range(all_captions.shape[0]):  # iterate over batches
                ref_caps = all_captions[j]
                references.append(self.caption_numbers_to_words(ref_caps))

            #_, preds = torch.max(predictions, dim=2)
            preds = self.get_best_prediction(predictions)
            predicted_captions = self.caption_numbers_to_words(preds)

            assert len(predicted_captions) == len(references)

            # TODO: When I integrate the evaluation class, do all this with that
            # Metrics and progress bar updates
            top5_acc_meter.update(topk_accuracy(yhat, y, 5))
            loss_meter.update(loss.cpu().item())
            bleu4_score = self.bleu4(predicted_captions, references)
            bleu4_meter.update(bleu4_score)
            end_time = time.time()
            batch_time = end_time - prev_time
            prev_time = end_time
            batch_time_meter.update(batch_time)
            time_remaining = calc_time(batch_time_meter.get_average() * (self.n - i))
            pbar.set_postfix(
                {
                    "bleu4": f"{bleu4_meter.get_average():.4f}",
                    "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
                    "loss": f"{loss_meter.get_average():.4f}",
                    "t-minus": time_remaining,
                }
            )
            # If training for real, set overwrite to true
            # To save in a different location, set alternate_location
            # I just don't want a file overwritten accidentally
            self.save_state(overwrite=False, alternate_location=None)
        
        return {
            "top 5 acc": top5_acc_meter.get_average(),
            "loss": loss_meter.get_average(),
            "epoch_time": time.time() - start_time,
        }

    # 12 chosen as default because Jeffrey said it converged pretty well at 12 epochs
    def train(self, epochs=12):
        # Keep training stats just in case
        train_stats = []
        validate_stats = []
        
        for i in range(0, epochs):
            train_stats.append(self.train_one_epoch())
            validate_stats.append(self.validate_epoch())

        return train_stats, validate_stats

    def validate_epoch(self):
        loss_meter = AverageMeter()
        top5_acc_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        bleu4_meter = AverageMeter()
        
        self.model.encoder.eval()
        self.model.decoder.eval()

        stats = {
            "top 5 acc": f"{0:.4f}",
            "loss": f"{0:.4f}",
            "time t-minus": "unknown",
        }

        prev_time = time.time()
        best_bleu = 0.0
        best_img = None
        best_caption = None
        
        with torch.no_grad():
            for i, (images, captions, caption_lengths, all_captions, orig_imgs) in enumerate(
                pbar := tqdm(self.validate_data_loader, f"Epoch {self.epoch+1} Validate Progress ", postfix=stats)
            ):
                # Forward
                predictions,alphas=self.model.forward(images, captions, caption_lengths)

                # Clean captions and predictions
                y = self.remove_caption_padding(captions, caption_lengths, True)
                yhat = self.remove_caption_padding(predictions, caption_lengths, False)

                # Get loss (loss only, no update)
                loss = self.update(yhat, y, alphas, loss_only=True)
                loss_meter.update(loss.item())
                
                # Caption/prediction numbers to words
                references = []
                for j in range(all_captions.shape[0]):  # iterate over batches
                    ref_caps = all_captions[j]
                    references.append(self.caption_numbers_to_words(ref_caps, validate=True))

                preds = self.get_best_prediction(predictions)
                predicted_captions = self.caption_numbers_to_words(preds, validate=True)
                
                assert len(predicted_captions) == len(references)

                end_time = time.time()
                batch_time = end_time - prev_time
                prev_time = end_time
                batch_time_meter.update(batch_time)
                time_remaining = calc_time(batch_time_meter.get_average() * (self.n - i))
                top5_acc_meter.update(topk_accuracy(yhat, y, 5))
                bleu4_score = self.bleu4(predicted_captions, references)
                if best_bleu <= bleu4_score:
                    best_bleu = bleu4_score
                    best_img = orig_imgs[0]
                    best_caption = predicted_captions[0]
                    actual_reference = references[0][0]
                bleu4_meter.update(bleu4_score)
                pbar.set_postfix(
                    {
                        "bleu4": f"{bleu4_meter.get_average():.4f}",
                        "top 5 acc": f"{top5_acc_meter.get_average():.4f}",
                        "loss": f"{loss_meter.get_average():.4f}",
                        "t-minus": time_remaining,
                    }
            )

            return (
                {
                    "bleu4": bleu4_meter.get_average(),
                    "top 5 acc": top5_acc_meter.get_average(),
                    "loss": loss_meter.get_average(),
                },
                best_img.cpu(),
                best_caption,
                actual_reference,
            )    
        
    def update(self, yhat, y, alphas, loss_only = False):
        loss = self.criterion(yhat, y)
        loss += 1.0 * ((1.0-alphas.sum(dim=1))**2).mean()
        if loss_only == False:
            self.model.decoder_optimizer.zero_grad()
            loss.backward()
            self.model.decoder_optimizer.step()
        return loss

    # remove_start_token should be true for y, false for yhat
    def remove_caption_padding(self, captions, caption_lengths, remove_start_token: bool = False):
        # remove <start> token for backpropagation
        if remove_start_token:
            y = captions[:, 1:]
        else:
            y = captions

        # Remove extra padding
        y = pack_padded_sequence(y, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]
        return y

    # Captions should be a tensor of captions, not a list
    # idk how to put this in python so I'm leaving this in a comment
    # Either way, "captions is list" returns false so I can't do that
    def caption_numbers_to_words(self, captions, validate=False):
        captions = captions.tolist()
        captions_words = []
        if validate == False:
            for k in range(len(captions)):
                p = captions[k]
                temp = [f"{self.data.inv_word_map[t]} " for t in p if t not in [self.data.word_map["<pad>"], self.data.word_map["<start>"]]]
                captions_words.append("".join(temp))

        else:
            for k in range(len(captions)):
                p = captions[k]
                temp = [f"{self.validate_data.inv_word_map[t]} " for t in p if t not in [self.validate_data.word_map["<pad>"], self.validate_data.word_map["<start>"]]]
                captions_words.append("".join(temp))        

        return captions_words

    # When implementing beam search, inherit this class and modify this function
    def get_best_prediction(self, predictions):
        _, preds = torch.max(predictions, dim=2)
        return preds

    def save_state(self, overwrite: bool = False, alternate_location: str=None):
        location = self.weights_file if alternate_location is None else alternate_location
        state = {
            "encoder": self.model.encoder.state_dict(),
            "decoder": self.model.decoder.state_dict(),
            "optimizer": self.model.decoder_optimizer.state_dict(),
            "config": self.config
        }
        if not overwrite and os.path.exists(self.weights_file):
            return False
        torch.save(state, location)
        return True

# TODO
# class Evaluator:
