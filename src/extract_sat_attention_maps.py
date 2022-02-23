from cgi import test
from data.augmentation import Flickr30k
from models.sat_model import SATEncoder, SATDecoder
from tqdm import tqdm

import numpy as np
import torch 
import skimage
import torchvision
from torchmetrics import BLEUScore
import matplotlib.pyplot as plt

SCHEDULED_SAMPLING_CONVERGENCE = 1/5
SCHEDULED_SAMPLING = False
LEARNING_RATE = 1e-4
EPOCHS = 60
EMBED = 512
HIDDEN = 512
ATTENTION = 512
DROP = 0.75
ENCODER = 2048
DEVICE = "cpu"
BATCH_SIZE = 64
SMOKE_TEST=False

best_checkpoint_path = "/home/jeff-laptop/Code/school/ece763/class_project/ECE763-Class-Project-Image-Captioning/src/best_checkpoint1.pt"

def main():
    # load data
    test_data = Flickr30k(root="/home/jeff-laptop/Code/school/ece763/class_project/ECE763-Class-Project-Image-Captioning/flickr30k/flickr30k.exdir",mode="test", smoke_test=SMOKE_TEST)
    word_map = test_data.word_map
    vocab_size = len(test_data.word_map.keys())

    inv_word_map = {v:k for k,v in word_map.items()}
    # load the model
    state_dicts = torch.load(best_checkpoint_path, map_location="cpu")
    encoder = SATEncoder()
    decoder = SATDecoder(
        embedding_size=EMBED,
        vocabulary_size=len(test_data.word_map.keys()),
        max_caption_size=test_data.max_cap_len,
        hidden_size=HIDDEN,
        attention_size=ATTENTION,
        encoder_size=ENCODER,
        device=DEVICE,
        dropout_rate=DROP
    )
    encoder.load_state_dict(state_dicts["encoder"]) # enforce the correct backbone
    decoder.load_state_dict(state_dicts["decoder"])
    encoder.eval()
    decoder.eval()
    blue_score = BLEUScore()
    best_results = []
    for _, i in enumerate(pbar :=  tqdm(range(len(test_data)))):
        if i % 5 != 0:
            continue
        mod, target, lengths, all_captions, img = test_data[i]
        all_captions = all_captions[None, :, :]
        mod = mod[None,:,:,:]
        target = target[None,:]
        lengths = lengths[None,:]
        ann = encoder(mod)
        predictions, alphas = decoder(ann, target, lengths, SCHEDULED_SAMPLING)

        # get  reference captions without additional characters
        references = []
        for  j in range(all_captions.shape[0]):#iterate over batches
            ref_caps = all_captions[j].tolist()
            temp_ref = []
            for ref in ref_caps: # iterate over available captions for image
                # strip unnecessary tokens
                tmp = [f"{inv_word_map[t]} " for t in ref if t not in [word_map["<pad>"], word_map["<start>"]]]
                temp_ref.append("".join(tmp))
            references.append(temp_ref)
        
        _, preds = torch.max(predictions, dim=2)
        preds = preds.tolist()
        predicted_captions = []
        for k in range(len(preds)):
            p = preds[k]
            temp = [f"{inv_word_map[t]} " for t in p if t not in [word_map["<pad>"], word_map["<start>"]]]
            predicted_captions.append("".join(temp).split(".")[0])
        

        score = blue_score(predicted_captions, references)
        pbar.set_postfix({"bleu4":score.numpy()})
        best_results.append( (score, predicted_captions, all_captions, img, alphas.detach().numpy(), mod) )
        best_results.sort(key=lambda x: x[0].numpy(), reverse=True)
        if len(best_results) > 5:
            best_results = [best_results[i] for i in range(5)] # keep only top 5
    
    ## Get attention maps
    for res in best_results:
        score, predicted, allcaps, img, alphas, x = res
        print(allcaps)
        print(score)
        print(predicted)

        x = encoder(x)
        enc_img_side = x.size(1)

        batch_size = x.size(0)
        encoded_size = x.size(-1)
        scheduled_sampling = SCHEDULED_SAMPLING

        # Reshape encoded image into a set of annotation vectors.
        # we can compress the image into a vector and treat encoded_size as the number of annotation vectors
        x = x.view(batch_size, -1, encoded_size)

        # The LSTM expects tensors in (batch_ size, sequence length, number of sequences)
        # x = x.permute(0, 2, 1)

        # initialize hidden states
        h, c = decoder.initialize_hidden_states(x)

        if scheduled_sampling:
            # embed the ground truth for teacher forcing
            embedded_captions = decoder.embedding(target)  # (batch_size, caption_length, embedding dim)

        # our predictions will be the size of the largest encoding (batch_size, largest_encoding, vocab_size)
        # each entry of this tensor will have a score for each batch entry, position in encoding, and vocabulary word candidate
        predictions = torch.zeros(batch_size, decoder._max_cap_size, vocab_size).to(decoder._device)  # predictions set to <pad>
        prev_words = torch.zeros((batch_size,)).long().to(decoder._device)
        alphas = torch.zeros(
            batch_size, decoder._max_cap_size, x.size(1)
        )  # attention generated weights stored for Doubly Stochastic Regularization
        for i in range(decoder._max_cap_size):
            # For each token, determine if we apply teacher forcing
            if scheduled_sampling and np.random.uniform(0, 1) < decoder._teacher_forcing_rate:
                # In teacher forcing we know which captions have a specified length, so we can reduce wasteful
                # computation by only applying the model on valid captions
                if i > max(lengths[0]):
                    break  # no more captions left at requested size
                zhat, α = decoder.attention(x, h)


                # gate
                gate = decoder.sigmoid(decoder.f_beta(h))
                zhat = gate * zhat
                # get the next hidden state and memory state of the lstm
                h, c = decoder.recurrent(
                    # conditioning the LSTM on the previous state's ground truth.
                    # On i=0 this is just the start token
                    torch.cat([embedded_captions[:, i, :], zhat], dim=1),
                    # truncated hidden and memory states
                    (h, c),
                )
                scores = decoder.deep_output(decoder.dropout(h))  # assign a score to potential vocabulary candidtates
                predictions[:, i, :] = scores  # append predictions for the i-th token
                prev_words = torch.argmax(scores, dim=1)
                alphas[:, i] = α  # store attention weights for doubly stochastic regularization
            else:
                # No teacher forcing done here. We just do the standard LSTM calculations
                zhat, α = decoder.attention(x, h)  # apply attention
                embedded = decoder.embedding(prev_words)  # condition on zero
                # Gate
                gate = decoder.sigmoid(decoder.f_beta(h))
                zhat = gate * zhat
                h, c = decoder.recurrent(
                    # Conditioning on previous predicted scores
                    torch.cat([embedded, zhat], dim=1),
                    (h, c),
                )
                scores = decoder.deep_output(decoder.dropout(h)) # assign a score to potential vocabulary candidtates
                prev_words = torch.argmax(scores, dim=1)
                predictions[:, i, :] = scores  # append predictions for the i-th token
                alphas[:, i, :] = α  # store attention weights for doubly stochastic regularization
            # generate attention visualization
            square_att = α.view(-1, enc_img_side, enc_img_side).detach().numpy()
            square_att = skimage.transform.pyramid_expand(square_att[0], upscale=32, sigma=8)
            plt.imshow(img.permute(1,2,0).numpy()/255)
            plt.imshow(square_att, alpha=0.8,cmap="gray", interpolation=None)
            plt.title(inv_word_map[prev_words.cpu().item()])
            plt.tight_layout()
            plt.axis("off")
            plt.show()
            if inv_word_map[prev_words.cpu().item()] == '.':
                break


if __name__ == "__main__":
    main()
