"""Point-grid helpers for dense detectors."""

import torch


def build_points(
    feat_h: int,
    feat_w: int,
    stride: int,
    device,
    dtype,
) -> torch.Tensor:
    """Return FCOS-style point centers for a feature map."""
    shifts_x = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * stride
    shifts_y = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
