"""Script to convert raw Flickr 30K data to a useable format.
This script creates an exdir dataset that holds the images for Flickr30K,
a word map for the n most popular words, and the mapped captions.
"""
import argparse
from collections import defaultdict
import copy
import enum
from lib2to3.pgen2 import token
import exdir
import nltk
import numpy as np
import os
from tqdm import tqdm
from data.augmentation import  Flickr30k

TOPK = 2000

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flickr_dir", action="store")
    return parser.parse_args()

def main():
    args = parse_args()
    root = args.flickr_dir
    assert os.path.exists(root)
    img_path = os.path.join(root, "flickr30k-images/")
    assert os.path.exists(img_path)
    ann_path = os.path.join(root, "results_20130124.token")

    dataset = Flickr30k(img_path, ann_path)

    # Create exdir archive
    archive = exdir.File(os.path.join(root, "flickr30k.exdir"))
    train_archive = archive.require_group("train")
    valid_archive = archive.require_group("valid")
    test_archive = archive.require_group("test")
    
    img_ids = dataset.ids
    train_ids = img_ids[:int(len(img_ids)*0.8)]
    val_ids = img_ids[int(len(img_ids)*0.8):int(len(img_ids)*0.8)+ int(len(img_ids)*0.15)]
    test_ids = img_ids[int(len(img_ids)*0.8)+int(len(img_ids)*0.15):]
    train_captions = defaultdict(list)
    val_captions = defaultdict(list)
    test_captions =defaultdict(list)

    print(f"# Train Examples {len(train_ids)}")
    print(f"# Validation Examples {len(val_ids)}")
    print(f"# Test examples {len(test_ids)}")

    # build word count map and load images
    words = defaultdict(lambda: 0)
    max_caption_length = 0
    for i, img_id in enumerate(tqdm(img_ids, desc="Loading Data")):
        image, captions = dataset[i]
        image = np.asarray(image)
        if img_id in train_ids:
            store = train_archive
            cap_store = train_captions
        elif img_id in val_ids:
            store = valid_archive
            cap_store = val_captions
        else:
            store = test_archive
            cap_store = test_captions
        #store.require_dataset(img_id, data=image)
        for cap in captions:
            cap_store[img_id].append(cap)
            tokens = nltk.tokenize.word_tokenize(cap.lower())
            if len(tokens) > max_caption_length:
                max_caption_length = len(tokens)
            for toke in tokens:
                words[toke] += 1
    max_caption_length += 2 # take start and end token into account
    
    # build token-map
    word_freqs = [(k, v) for k, v in words.items()]
    word_freqs.sort(key=lambda x : x[1], reverse=True)
    topk = word_freqs[:TOPK] # We select top K words to be the model's vocabulary
    topk = [w for w, _ in topk] # only care about the words
    token_map = {token:i for i, token in enumerate(topk)}
    token_map["<start>"] = len(token_map)
    token_map["<end>"] = len(token_map)
    token_map["<unc>"] = len(token_map)
    token_map["<pad>"] = len(token_map)

    # store tokens for dataset
    for img_id in tqdm(img_ids, desc="Processing Captions"):
        if img_id in train_ids:
            store = train_archive
            cap_store = train_captions
        elif img_id in val_ids:
            store = valid_archive
            cap_store = val_captions
        else:
            store = test_archive
            cap_store = test_captions
        tokenized_captions = []
        for cap in cap_store[img_id]:
            tokens = nltk.tokenize.word_tokenize(cap.lower())
            # replace uncommon words
            res = copy.copy(tokens)
            for i, toke in enumerate(tokens):
                if toke not in token_map.keys():
                    res[i] = token_map["<unc>"]
                else:
                    res[i] = token_map[toke]
            res.append(token_map["<end>"])
            res.insert(0, token_map["<start>"])

            # ensure equal length for all captions
            while len(res) < max_caption_length: 
                res.append(token_map["<pad>"])
            tokenized_captions.append(res)

        # store caption
        store[img_id].attrs["captions"] = tokenized_captions
    archive.attrs["word_map"] = token_map
             
if __name__ == "__main__":
    main()
