"""
This module contains classes to extend the data loading classes to
perform data augmentation. By the end of the project, the
following should be implemented
 - Child Class that performs data augmentation on Flicker30K dataset
"""

from torchvision.datasets import Flickr30k
import torchvision.transforms as transforms
from torch import float32


class AugmentedFlickrDataset(Flickr30k):
    def __init__(
        self,
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
        brightness_factor=0.5,
    ) -> None:
        super().__init__(
            root,
            annotations_file,
            transform=transforms.Compose(
                [
                    # Convert to tensor, uint8_t [0, 255]
                    transforms.PILToTensor(),
                    transforms.Resize(resize),
                    transforms.RandomAffine(degrees=degrees, translate=translate),
                    transforms.GaussianBlur(kernel_size=blur_kernel_mean, sigma=blur_kernel_std),
                    transforms.ColorJitter(brightness=brightness_factor),
                    # Convert to tensor, float [0.0, 255.0]
                    # transforms.ConvertImageDtype(float32)
                ]
            ),
        )

    # EfficientNet requires a float tensor with intensities of [0.0, 255.0]
    # AFAIK, Pytorch doesn't have a transform that can accomplish this
    # Closest thing is ConvertImageDtype but that autoscales floats to [0.0, 1.0]
    # Hence, this function has to be created I guess
    # If we continue with this format, we can rewrite self.__getitem__ to just
    # return the value of this function so we can access items with array syntax
    def __getitem__(self, index):
        # Tuples are immutable so convert to string
        item = list(super().__getitem__(index))
        # item[0] is the image itself (rest are annotations), converts to float
        item[0] = item[0].float()
        # Return converted to tuple
        return tuple(item)
