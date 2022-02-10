"""
This module contains classes to extend the data loading classes to
perform data augmentation. By the end of the project, the
following should be implemented
 - Child Class that performs data augmentation on Flicker30K dataset
"""

#from data.data import FlickrDataset

from torchvision.datasets import Flickr30k
import torchvision.transforms as transforms
from PIL import Image

class AugmentedFlickrDataset(Flickr30k):
    def __init__(self, root="../../flickr30k", annotations_file="../../flickr30k/results_20130124.token") -> None:
        super().__init__(root, annotations_file, transform=transforms.Resize([224, 224]))
        image = self[0][0]
        print(image)
        image.save("./0_224x224_downsample.jpg")

if __name__=="__main__":
    afd = AugmentedFlickrDataset()
