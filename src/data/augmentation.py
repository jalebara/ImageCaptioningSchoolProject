"""
This module contains classes to extend the data loading classes to
perform data augmentation. By the end of the project, the
following should be implemented
 - Child Class that performs data augmentation on Flicker30K dataset
"""
from collections import defaultdict
import numpy as np
import exdir
from tqdm import tqdm
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from torch import float32
from typing import Any, Callable, Optional, Tuple
from multiprocessing import Pool
import PIL

def load_metadata(path, key, mode):
    archive = exdir.File(path, mode="r")
    archive = archive.require_group(mode)
    return {key: archive[key].attrs["captions"]}, {key: archive[key].attrs["lengths"]}

class Flickr30k(VisionDataset):
    """ """

    def __init__(
        self,
        root: str = "../../flickr30k",
        transform: Callable = transforms.Compose(
            [
                transforms.ConvertImageDtype(float32),
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
            ]
        ),
        target_transform: Optional[Callable] = None,
        mode: str = "test",
        smoke_test=False,
        fast_test=False
    ) -> None:
        super(Flickr30k, self).__init__(root, transform=transform, target_transform=target_transform)
        archive = exdir.File(root, mode="r")
        self.archive = archive.require_group(mode)
        data_keys = list(self.archive.keys())
        if smoke_test:
            data_keys = data_keys[:10]
        elif fast_test:
            data_keys = data_keys[:int(len(data_keys)*0.3)]
        # Read tokenized captions and store in dict
        self.annotations = defaultdict(list)
        self.lengths = defaultdict(list)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        with Pool(processes=10) as pool:
            zx =  list(zip([root for _ in range(len(data_keys))], data_keys, [mode for _ in range(len(data_keys))]))
            jobs = [pool.apply_async(func=load_metadata, args=(*argument,))  for argument in zx]
            for job in tqdm(jobs, desc=f"Loading {mode} data"):
                a, l = job.get()
                self.annotations.update(a)
                self.lengths.update(l)
        self.ids = list(sorted(self.annotations.keys()))
        self.word_map = archive.attrs["word_map"].to_dict()
        self.max_cap_len = archive.attrs["max_cap_len"]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index//5]

        # Image
        img = torch.Tensor(np.copy(self.archive[img_id][:]))
        img = img.permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        mod = self.normalize(img)

        # Captions
        target = self.annotations[img_id][ index % 5 ]
        target = torch.Tensor(target).long()
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Caption lengths
        lengths = self.lengths[img_id][ index % 5 ]
        lengths = torch.Tensor([lengths]).long()
        
        all_caps = torch.Tensor(self.annotations[img_id]).long()
        return mod, target, lengths, all_caps, img

    def __len__(self) -> int:
        return len(self.ids)*5


class AugmentedFlickrDataset(Flickr30k):
    def __init__(
        self,
        # Root directory of images
        root="../../flickr30k",
        # Tuple that is resize height and width (default 224x224)
        resize=[256, 256],
        # Tuple that is range of degrees the image should be randomly rotated (default 0-360)
        degrees=[0, 60],
        # Tuple that is range for random translation of image (default translate at most 1/5 in x and y direction)
        translate=[0.1, 0.2],
        # Mean values of the Gaussian blur kernel size (default x=5, y=5)
        blur_kernel_mean=(5, 5),
        # Range of std deviation of the Gaussian blur kernel size (default (0.1, 5))
        blur_kernel_std=(0.1, 3),
        # Transform brightness of image randomly from 1-brightness_factor to 1+brightness_factor (default 0.5)
        brightness_factor=0.2,
        # mode
        mode="test",
        smoke_test=False,
        fast_test=False
    ) -> None:
        super().__init__(
            root,
            transform=transforms.Compose(
                [
                    # Convert to tensor, uint8_t [0, 255]
                    transforms.ConvertImageDtype(float32),
                    transforms.Resize(resize),
                    transforms.CenterCrop(224),
                    #transforms.RandomAffine(degrees=degrees, translate=translate),
                    transforms.GaussianBlur(kernel_size=blur_kernel_mean, sigma=blur_kernel_std),
                    transforms.ColorJitter(brightness=brightness_factor),
                    # Convert to tensor, float [0.0, 255.0]
                ]
            ),
            mode=mode,
            smoke_test=smoke_test,
            fast_test=fast_test
        )

    # EfficientNet requires a float tensor with intensities of [0.0, 255.0]
    # AFAIK, Pytorch doesn't have a transform that can accomplish this
    # Closest thing is ConvertImageDtype but that autoscales floats to [0.0, 1.0]
    # Hence, this function has to be created I guess
    # If we continue with this format, we can rewrite self.__getitem__ to just
    # return the value of this function so we can access items with array syntax
    def __getitem__(self, index):
        return super().__getitem__(index)
