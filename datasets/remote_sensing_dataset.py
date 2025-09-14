import os
from typing import Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RemoteSensingDataset(Dataset):
    """Simple dataset for paired remote sensing images.

    Expects a directory structure::

        root/
            input/   # conditioning images
            target/  # ground truth images

    Both folders must contain the same number of files with matching
    filenames. Images are loaded and mapped to ``[-1, 1]`` range.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.input_dir = os.path.join(root, "input")
        self.target_dir = os.path.join(root, "target")

        self.input_files = sorted(os.listdir(self.input_dir))
        self.target_files = sorted(os.listdir(self.target_dir))
        if len(self.input_files) != len(self.target_files):
            raise ValueError("Input and target folders must contain the same number of files")

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
        self.target_transform = target_transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.input_files)

    def _load_image(self, directory: str, filename: str) -> Image.Image:
        path = os.path.join(directory, filename)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):  # type: ignore[override]
        inp_name = self.input_files[idx]
        tar_name = self.target_files[idx]
        inp = self._load_image(self.input_dir, inp_name)
        tar = self._load_image(self.target_dir, tar_name)

        if self.transform:
            inp = self.transform(inp)
        if self.target_transform:
            tar = self.target_transform(tar)

        # ``image`` key keeps compatibility with existing code paths
        return {"input": inp, "target": tar, "image": tar}
