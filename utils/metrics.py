import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted images in ``[0, data_range]`` with shape ``(N,C,H,W)``.
        target: Ground truth images in ``[0, data_range]``.
        data_range: The maximum value in the input images.

    Returns:
        PSNR value for each image averaged over the batch.
    """
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    return peak_signal_noise_ratio(pred, target, data_range=data_range)

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute Structural Similarity (SSIM).

    Args:
        pred: Predicted images in ``[0, data_range]`` with shape ``(N,C,H,W)``.
        target: Ground truth images in ``[0, data_range]``.
        data_range: The maximum value in the input images.

    Returns:
        SSIM value averaged over the batch.
    """
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    return structural_similarity_index_measure(pred, target, data_range=data_range)

def calculate_sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral Angle Mapper (SAM) for multi-spectral remote sensing images.

    Args:
        pred: Predicted images with shape ``(N,C,H,W)``.
        target: Ground truth images with shape ``(N,C,H,W)``.
        eps: Small number to avoid division by zero.

    Returns:
        Average spectral angle (radians) over the batch and spatial dimensions.
    """
    pred = pred.double().view(pred.shape[0], pred.shape[1], -1)
    target = target.double().view(target.shape[0], target.shape[1], -1)
    dot = (pred * target).sum(dim=1)
    norm_pred = pred.norm(dim=1)
    norm_target = target.norm(dim=1)
    cos = dot / (norm_pred * norm_target + eps)
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)
    return angle.mean()
