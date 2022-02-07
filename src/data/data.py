"""
This module contains the encapsuling code for pulling and iterating though the datasets.
The code here should be consistent across subprojects, accessible, and easy to add
additional functionality to.
"""

from torch.utils.data import Dataset

class FlickerDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, index):
        raise NotImplementedError
