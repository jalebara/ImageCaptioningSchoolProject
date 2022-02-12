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
                 # Tuple that is resize height and width
                 resize=[224, 224],
                 # Tuple that is range of degrees the image should be randomly rotated
                 degrees=[0, 360],
                 # Tuple that is range for random translation of image
                 translate=[0.2, 0.2],
                 # Mean values of the Gaussian blur kernel size
                 blur_kernel_mean=(5, 5),
                 # Range of std deviation of the Gaussian blur kernel size
                 blur_kernel_std=(0.1, 5),
                 # Transform brightness of image randomly from 1-brightness_factor to 1+brightness_factor
                 brightness_factor=0.5
    ) -> None:
        super().__init__(root,
                         annotations_file,
                         transform=transforms.Compose([
                             # Convert to tensor
                             transforms.ToTensor(),
                             # Downsample to 224x224
                             transforms.Resize([224, 224]),
                             # Rotate from 0-360 deg, translate up to 1/5 of the image
                             transforms.RandomAffine(degrees=360, translate=(0.2, 0.2)),
                             # Blur with kernel a x b where a,b mean=5, std dev is uniform on [0.1, 5]
                             transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 5)),
                             # Change brightness from 0.5 to 1.5
                             transforms.ColorJitter(brightness=0.5)
                             ]))
