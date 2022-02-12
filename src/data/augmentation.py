"""
This module contains classes to extend the data loading classes to
perform data augmentation. By the end of the project, the
following should be implemented
 - Child Class that performs data augmentation on Flicker30K dataset
"""

from torchvision.datasets import Flickr30k
import torchvision.transforms as transforms

class AugmentedFlickrDataset(Flickr30k):
    def __init__(self,
                 # Root directory of images
                 root="../../flickr30k",
                 # Location of annotations file
                 annotations_file="../../flickr30k/results_20130124.token",
                 # Tuple that is resize height and width (default 224x224)
                 resize=[224, 224],
                 # Tuple that is range of degrees the image should be randomly rotated (default 0-360)
                 degrees=[0, 360],
                 # Tuple that is range for random translation of image (default translate at most 1/5 in x and y direction)
                 translate=[0.2, 0.2],
                 # Mean values of the Gaussian blur kernel size (default x=5, y=5)
                 blur_kernel_mean=(5, 5),
                 # Range of std deviation of the Gaussian blur kernel size (default (0.1, 5))
                 blur_kernel_std=(0.1, 5),
                 # Transform brightness of image randomly from 1-brightness_factor to 1+brightness_factor (default 0.5)
                 brightness_factor=0.5
    ) -> None:
        super().__init__(root,
                         annotations_file,
                         transform=transforms.Compose([
                             # Convert to tensor
                             transforms.ToTensor(),
                             transforms.Resize(resize),
                             transforms.RandomAffine(degrees=degrees, translate=translate),
                             transforms.GaussianBlur(kernel_size=blur_kernel_mean, sigma=blur_kernel_std),
                             transforms.ColorJitter(brightness=brightness_factor)
                             ]))
