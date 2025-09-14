import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class RemoteSensingDataset(Dataset):
    """Dataset for paired MODIS and Landsat observations.

    Each sample is expected to be stored in an ``.npz`` file under ``root`` with
    the following arrays: ``modis_refs``, ``landsat_refs``, ``modis_target`` and
    ``landsat_gt``. Arrays are converted to :class:`torch.Tensor` and optional
    cropping/normalization is applied.
    """

    def __init__(self, root, crop_size=None, normalize=True):
        """Construct the dataset.

        Args:
            root (str): Directory containing ``.npz`` files for each sample.
            crop_size (tuple or int, optional): Center crop size ``(h, w)`` or
                single integer for a square crop.
            normalize (bool): If ``True``, scale pixel values to ``[0, 1]``.
        """
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.crop_size = crop_size
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def _process_array(self, arr):
        tensor = torch.from_numpy(arr)
        if self.crop_size is not None:
            if isinstance(self.crop_size, (list, tuple)):
                th, tw = self.crop_size
            else:
                th = tw = self.crop_size
            h, w = tensor.shape[-2:]
            top = max(0, (h - th) // 2)
            left = max(0, (w - tw) // 2)
            tensor = tensor[..., top:top + th, left:left + tw]
        tensor = tensor.float()
        if self.normalize:
            tensor = tensor / 255.0
        return tensor

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        modis_refs = [self._process_array(x) for x in data["modis_refs"]]
        landsat_refs = [self._process_array(x) for x in data["landsat_refs"]]
        modis_target = self._process_array(data["modis_target"])
        landsat_gt = self._process_array(data["landsat_gt"])
        return {
            "modis_refs": modis_refs,
            "landsat_refs": landsat_refs,
            "modis_target": modis_target,
            "landsat_gt": landsat_gt,
        }
