""" Contains the main experiment for Project 1
The goal for this is to train a SAT model with and without data augmentation and provide standard language task performance metrics

"""

import argparse
import os
from os.path import join

import numpy as np
from models.utils import load_model_dict, save_model_dict
from models.sat_model import SATDecoder, SATEncoder
from models.attention import SATAttention
from data.augmentation import Flickr30k, AugmentedFlickrDataset
from train import train_sat_epoch, validate_sat_epoch
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import EarlyStopping,AverageMeter

import metrics_for_imagecaption
import gradcam

RESULTS_DIRECTORY = os.path.abspath("results/project1")
DATA_DIRECTORY = os.path.abspath("flickr30k/flickr30k.exdir")

SCHEDULED_SAMPLING_CONVERGENCE = 1/5
LEARNING_RATE = 5e-3
EPOCHS = 60
EMBED = 512
HIDDEN = 512
ATTENTION = 512
DROP = 0.5
ENCODER = 2048
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project 1 Main Experiment")
    parser.add_argument(
        "--skip_training", action="store_true", required=False
    )  # if model is already trained, and you just want to regenerate results
    # skip evaluation
    parser.add_argument("--skip_evaluation", action="store_true", default=False, required=False)
    parser.add_argument(
        "--transfer_weights", action="store_true", required=False
    )  # use transfer learning for SAT Model
    parser.add_argument(
        "--augment_data", action="store_true", default=False, required=False
    )  # set option to augment data
    parser.add_argument("--freeze_encoder", action="store_true", default=False, required=False)
    parser.add_argument("--unfreeze_last", action="store", type=int, default=0, required=False)
    parser.add_argument("--data_directory", action="store", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true", default=False, required=False)
    parser.add_argument("--fast_test", action="store_true", default=False, required=False)
    return parser.parse_args()


def train_model(
    encoder: nn.Module,
    decoder: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    results_path: str,
    word_map:dict,
    checkpoint_name: str,
    logger: logging.Logger,
):
    """Starts/Resumes training session"""
    # initialize variables
    checkpoint_path = os.path.exists(join(results_path, checkpoint_name))
    early_stopping_checkpoint = join(results_path, "early_stopping.pt")
    writer = SummaryWriter(os.path.abspath("results/project1/runs"))
    # Early Stopping
    early_stop = EarlyStopping(checkpoint_path=early_stopping_checkpoint, report_func=logger.info, delta=0.001)

    # Model Paramerers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=decoder.parameters(), lr=LEARNING_RATE)

    # Load existing checkpoint if any
    if checkpoint_path:
        # load checkpoint
        pass
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    for epoch in range(EPOCHS):
        if early_stop.stop:
            break  # stop training when the model fails to learn for too long
        if epoch > 30 and epoch % 2 == 1:
            decoder.update_scheduled_sampling_rate(SCHEDULED_SAMPLING_CONVERGENCE)
        train_metrics = train_sat_epoch(epoch, encoder, decoder, trainloader, optimizer, criterion, word_map, DEVICE)
        val_metrics, best_img, best_caption, actual_caption = validate_sat_epoch(
            epoch, encoder, decoder, valloader, criterion, word_map, DEVICE
        )

        # save state
        bleu4 = val_metrics["bleu4"]
        state = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, join(RESULTS_DIRECTORY, checkpoint_name))

        # update early stopping
        early_stop(bleu4, state)

        # report metrics to TensorBoard
        writer.add_scalars("Train-Val Loss/ Epoch", {
            "Train Loss": train_metrics["loss"],
            "Val Loss": val_metrics["loss"]
        }, global_step=epoch)
        writer.add_scalar("BLEU4", val_metrics["bleu4"], global_step=epoch)
        writer.add_scalars("Top 5 Acc", {
            "train top 5 acc": train_metrics["top 5 acc"],
            "val top 5 acc": val_metrics["top 5 acc"]
        }, global_step=epoch)
        writer.add_image(f"Epoch {epoch}", best_img/255 )
        writer.add_text(f"Epoch {epoch}", f"Predicted caption: {best_caption}", 0)
        writer.add_text(f"Epoch {epoch}", f"Actual caption: {actual_caption}", 1)
        # report best image and caption to TensorBoard


def evaluate_model(model: nn.Module, test_data_loader: DataLoader):
    global HIDDEN
    global ENCODER
    global DEVICE
    global EMBED
    # 1. Create encoder, decoder, and stuff that is passed into validate_sat_epoch
    encoder = 0
    decoder = 0
    for name,param in model.named_parameters:
        if name in ['encoder']:
            encoder = SATEncoder(params=param)
            break
        if name in ['decoder']:
            decoder = SATDecoder(
                params=param,
                embedding_size=EMBED,
                vocabulary_size=len(train_data.word_map),
                max_caption_size=train_data.max_cap_len,
                hidden_size=HIDDEN,
                attention=attention,
                encoder_size=ENCODER,
                device=DEVICE,
            )
            break
    # Ensure the model has params for encoder, and decoder
    if not (encoder is SATEncoder and decoder is SATDecoder):
            raise ValueError("Malformed model argument to evaluate_model()")


    encoder.to(DEVICE)
    decoder.to(DEVICE)

    bleu_1_avg = AverageMeter("bleu_1")
    bleu_2_avg = AverageMeter("bleu_2")
    bleu_3_avg = AverageMeter("bleu_3")
    bleu_4_avg = AverageMeter("bleu_4")
    rouge_avg = AverageMeter("rouge")
    meteor_avg = AverageMeter("meteor")
    
    # 2. Forward propagate through network, capture output, and pass to Calculate_metrics function

    # This part is mostly adapted from train.train_sat_epoch()
    for i, (images, captions, caption_lengths, _) in enumerate(
        pbar := tqdm(test_data_loader, f"Evaluation Progress ")
    ):
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths=caption_lengths.to(device)

        # Forward propagate
        images_encoded = encoder(images)
        predictions, alphas = decoder(images_encoded, captions, caption_lengths, True)

        # remove <start> token for backpropagation
        y = captions[:, 1:]

        # remove unnecessary padding
        yhat = pack_padded_sequence(predictions, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]
        y = pack_padded_sequence(y, caption_lengths.cpu().squeeze(), batch_first=True, enforce_sorted=False)[0]

        # Average all the scores
        metrics = Calculate_metrics(y, yhat)
        bleu_1_avg.update(metrics['bleu_1'])
        bleu_2_avg.update(metrics['bleu_2'])
        bleu_3_avg.update(metrics['bleu_3'])
        bleu_4_avg.update(metrics['bleu_4'])
        rouge_avg.update(metrics['rouge'])
        meteor_avg.update(metrics['meteor'])
        
        # GradCAM
        
        gcm = gradcam.GradCamModel(model, model.last_layer)
        images_gcm = gcm(images)
        # TODO: I don't really understand the gradcam class so someone else should fill in the rest


        # END TODO
    
    to_return['bleu_1'] = bleu_1_avg.get_average()
    to_return['bleu_2'] = bleu_2_avg.get_average()
    to_return['bleu_3'] = bleu_3_avg.get_average()
    to_return['bleu_4'] = bleu_4_avg.get_average()
    to_return['rouge']= rouge_avg.get_average()
    to_return['meteor']= meteor_avg.get_average()
    return to_return

def main():
    """Main experiment
    The following describes the pipeline:

    1. Construct encoder and load pretrained weights if applicable
    2. Construct decoder and compose the two components
    3. Load dataset and split into train, validation, and test sets
    4. Train model and store best weights
    5. Load best weights
    6. Generate test results and metrics
    7. Generate Class Activation maps on the best performing test set images
    """
    global EPOCHS
    # parse arguments
    args = parse_args()
    logger = logging.getLogger("sat_model")
    logger.setLevel(logging.INFO)

    # create results directory
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    best_checkpoint_path = join(
        RESULTS_DIRECTORY,
        f"best-{'augmented' if args.augment_data else 'base'}-SAT-weights.pt",
    )

    if not args.skip_training:
        # train the model
        if args.augment_data:
            # load augmented dataset
            train_data = AugmentedFlickrDataset(DATA_DIRECTORY, mode="train", smoke_test=args.smoke_test, fast_test=args.fast_test)
        else:
            train_data = Flickr30k(DATA_DIRECTORY, mode="train", smoke_test=args.smoke_test, fast_test=args.fast_test)
        # no augmentation on validation set
        valid_data = Flickr30k(DATA_DIRECTORY, mode="valid", smoke_test=args.smoke_test, fast_test=args.fast_test)

    # Construct the model
    encoder = SATEncoder()
    decoder = SATDecoder(
        embedding_size=EMBED,
        vocabulary_size=len(train_data.word_map),
        max_caption_size=train_data.max_cap_len,
        hidden_size=HIDDEN,
        attention_size=ATTENTION,
        encoder_size=ENCODER,
        device=DEVICE,
        dropout_rate=DROP
    )

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    if args.smoke_test:
        EPOCHS = 10
        # check to make sure that the model actually works
        tensor = torch.Tensor(np.random.uniform(0, 255, (1, 3, 224, 224))).to(DEVICE)

        encoded = encoder(tensor)
        decoded = decoder(encoded)

    if not args.skip_training:
        # train the model
        trainloader = DataLoader(train_data, num_workers=8, batch_size=BATCH_SIZE)
        valloader = DataLoader(valid_data, num_workers=8, batch_size=BATCH_SIZE)
        train_model(encoder, decoder, trainloader, valloader, RESULTS_DIRECTORY, train_data.word_map, "checkpoint.pt", logger)

    if not args.skip_evaluation:
        # no augmentation on test set
        test_data = Flickr30k(DATA_DIRECTORY, mode="test",num_workers=8, smoke_test=args.smoke_test)
        testloader = DataLoader(test_data, batch_size=BATCH_SIZE)
        model = load_model_dict(model, best_checkpoint_path)

        # test set evaluation
        evaluate_model(model, testloader)
        # generate class attention maps


if __name__ == "__main__":
    main()
