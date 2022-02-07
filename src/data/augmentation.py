"""
This module contains classes to extend the data loading classes to
perform data augmentation. By the end of the project, the
following should be implemented
 - Child Class that performs data augmentation on Flicker30K dataset
"""

from data.data import FlickerDataset

class AugmentedFlickerDataset(FlickerDataset):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError