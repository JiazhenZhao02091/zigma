import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .remote_sensing import RemoteSensingDataset


class ImageFolder_FakeWrapper(Dataset):
    def __init__(self, *args, **kwargs):
        print("ImageFolder_FakeWrapper initialized")

    def __getitem__(self, index):
        x = torch.rand(3, 256, 256)
        y = 0
        # Add additional processing here
        return x, y

    def __len__(self):
        return 10086


DATASET_REGISTRY = {
    "imagefolder_fake": ImageFolder_FakeWrapper,
    "remote_sensing": RemoteSensingDataset,
}


def get_dataset(name, *args, **kwargs):
    """Retrieve a dataset from the registry."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](*args, **kwargs)
