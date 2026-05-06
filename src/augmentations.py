"""
Image augmentations for Procgen RL experiments.

All functions:
  - Accept batched float32 tensors of shape (B, C, H, W) in [0, 1]
  - Return a tensor of the same shape
  - Are differentiable (pure PyTorch ops — no OpenCV, no PIL)
    so DrAC can backprop through the augmentation path
  - Can be applied to CPU or CUDA tensors

Usage::

    aug_fn = get_augmentation("crop")
    aug_obs = aug_fn(obs_batch)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Random Crop
# ---------------------------------------------------------------------------

def random_crop(x: Tensor, pad: int = 4) -> Tensor:
    """
    Pad by `pad` pixels (reflection) then randomly crop back to original size.

    This is the single most consistently effective augmentation on Procgen
    (Laskin et al. 2020, Raileanu & Fergus 2021).  Intuitively, it teaches the
    network to be invariant to small translations of the scene.
    """
    b, c, h, w = x.shape
    # Pad: reflection avoids black borders that the network could learn to detect.
    x_pad = F.pad(x, [pad] * 4, mode="reflect")
    # Sample a random crop origin for each image in the batch.
    h_pad, w_pad = h + 2 * pad, w + 2 * pad
    top  = torch.randint(0, h_pad - h + 1, (b,), device=x.device)
    left = torch.randint(0, w_pad - w + 1, (b,), device=x.device)

    # Gather crops using grid_sample for differentiability.
    # Build sampling grid: for item i, shift grid by (top[i], left[i]).
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1).clone()

    # Convert pixel offsets to [-1, 1] normalized coords.
    offset_y = (top.float()  / (h_pad - 1)) * 2   # shift in normalized coords
    offset_x = (left.float() / (w_pad - 1)) * 2

    # The padded image spans [-1 - offset_scale, 1 + offset_scale] in original coords.
    # Map grid from [crop_start, crop_start+H] in padded space to [-1,1].
    scale_y = h / h_pad
    scale_x = w / w_pad
    center_y = (top.float()  + h / 2) / h_pad * 2 - 1   # center of crop in padded coords
    center_x = (left.float() + w / 2) / w_pad * 2 - 1

    grid[..., 0] = grid[..., 0] * scale_x + center_x.view(b, 1, 1)
    grid[..., 1] = grid[..., 1] * scale_y + center_y.view(b, 1, 1)

    return F.grid_sample(x_pad, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# 2. Color Jitter
# ---------------------------------------------------------------------------

def color_jitter(
    x: Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> Tensor:
    """
    Randomly perturb brightness, contrast, saturation, and hue.

    All operations are in-place differentiable.  Hue shift requires HSV
    conversion which is done via RGB → HSV → RGB.
    """
    b = x.shape[0]

    # Brightness: multiply by random scalar in [1-brightness, 1+brightness]
    if brightness > 0:
        factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * brightness
        x = (x * factor).clamp(0, 1)

    # Contrast: blend toward mean luminance
    if contrast > 0:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * contrast
        x = (mean + factor * (x - mean)).clamp(0, 1)

    # Saturation: blend toward grayscale
    if saturation > 0:
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * saturation
        x = (gray + factor * (x - gray)).clamp(0, 1)

    # Hue: shift hue channel in HSV space
    if hue > 0:
        x = _rgb_to_hsv(x)
        shift = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * hue
        x[:, 0:1] = (x[:, 0:1] + shift) % 1.0
        x = _hsv_to_rgb(x)

    return x.clamp(0, 1)


def _rgb_to_hsv(img: Tensor) -> Tensor:
    """Convert (B, 3, H, W) RGB [0,1] → HSV [0,1]."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc = img.max(dim=1).values
    minc = img.min(dim=1).values
    v = maxc
    s = torch.where(maxc != 0, (maxc - minc) / (maxc + 1e-8), torch.zeros_like(maxc))
    rc = (maxc - r) / (maxc - minc + 1e-8)
    gc = (maxc - g) / (maxc - minc + 1e-8)
    bc = (maxc - b) / (maxc - minc + 1e-8)
    h = torch.where(r == maxc, bc - gc,
        torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(img: Tensor) -> Tensor:
    """Convert (B, 3, H, W) HSV [0,1] → RGB [0,1]."""
    h, s, v = img[:, 0], img[:, 1], img[:, 2]
    i = (h * 6.0).long()
    f = (h * 6.0) - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6
    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(
        i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(
        i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(
        i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))
    return torch.stack([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# 3. Random Grayscale
# ---------------------------------------------------------------------------

def random_grayscale(x: Tensor, prob: float = 0.2) -> Tensor:
    """Convert to grayscale with probability `prob`, independently per image."""
    b = x.shape[0]
    mask = (torch.rand(b, device=x.device) < prob).float().view(b, 1, 1, 1)
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).expand_as(x)
    return mask * gray + (1 - mask) * x


# ---------------------------------------------------------------------------
# 4. Random Cutout
# ---------------------------------------------------------------------------

def random_cutout(x: Tensor, patch_size: int = 16) -> Tensor:
    """Zero out a randomly located `patch_size × patch_size` square patch."""
    b, c, h, w = x.shape
    # Sample top-left corner uniformly so the patch fits within the image.
    top  = torch.randint(0, h - patch_size + 1, (b,))
    left = torch.randint(0, w - patch_size + 1, (b,))

    # Build a mask (1 = keep, 0 = zero) — do this on CPU then move to device.
    mask = torch.ones(b, 1, h, w, device=x.device)
    for i in range(b):
        mask[i, :, top[i]:top[i] + patch_size, left[i]:left[i] + patch_size] = 0.0

    return x * mask


# ---------------------------------------------------------------------------
# 5. Random Horizontal Flip
# ---------------------------------------------------------------------------

def random_flip(x: Tensor, prob: float = 0.5) -> Tensor:
    """Horizontally flip each image independently with probability `prob`."""
    b = x.shape[0]
    flip_mask = (torch.rand(b, device=x.device) < prob)
    out = x.clone()
    if flip_mask.any():
        out[flip_mask] = x[flip_mask].flip(-1)
    return out


# ---------------------------------------------------------------------------
# 6. Random Rotation
# ---------------------------------------------------------------------------

def random_rotation(x: Tensor, max_angle: float = 15.0) -> Tensor:
    """Rotate each image by a random angle in [-max_angle, max_angle] degrees."""
    b = x.shape[0]
    angles = (torch.rand(b, device=x.device) * 2 - 1) * max_angle   # degrees
    angles_rad = angles * (math.pi / 180.0)

    cos_a = angles_rad.cos()
    sin_a = angles_rad.sin()

    # Build affine theta matrix (B, 2, 3) for grid_sample
    zeros = torch.zeros(b, device=x.device)
    theta = torch.stack([
        cos_a, -sin_a, zeros,
        sin_a,  cos_a, zeros,
    ], dim=1).view(b, 2, 3)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


# ---------------------------------------------------------------------------
# 7. Random Convolution (Network Randomization)
# ---------------------------------------------------------------------------

def random_convolution(x: Tensor, mixing: float = 0.5) -> Tensor:
    """
    Apply a randomly-initialised 3×3 convolution and blend with original.

    From Lee et al. (2020) "Network Randomization: A Simple Technique for
    Generalization in Deep Reinforcement Learning".

    A new random filter is drawn every call (every mini-batch), so the network
    learns features that are invariant to low-level texture statistics.
    The mixing coefficient controls the blend: output = mixing * conv(x) + (1-mixing) * x.
    """
    b, c, h, w = x.shape
    # One random conv per batch item would be ideal but is slow — use one per call,
    # which is the standard practice in the literature.
    weight = torch.randn(c, c, 3, 3, device=x.device) / (c * 9) ** 0.5
    conv_x = F.conv2d(x, weight, padding=1)
    # Normalize conv output to [0,1] to avoid blowing up activations.
    conv_x = (conv_x - conv_x.min()) / (conv_x.max() - conv_x.min() + 1e-8)
    return (mixing * conv_x + (1 - mixing) * x).clamp(0, 1)


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------

# Maps config string → function
_AUG_REGISTRY: dict[str, Callable[[Tensor], Tensor]] = {
    "crop":        random_crop,
    "color_jitter": color_jitter,
    "grayscale":   random_grayscale,
    "cutout":      random_cutout,
    "flip":        random_flip,
    "rotate":      random_rotation,
    "random_conv": random_convolution,
    "none":        lambda x: x,
}

ALL_AUGMENTATIONS = [k for k in _AUG_REGISTRY if k != "none"]


def get_augmentation(name: str) -> Callable[[Tensor], Tensor]:
    """Return the augmentation function for a given string key."""
    if name not in _AUG_REGISTRY:
        raise ValueError(
            f"Unknown augmentation '{name}'.  "
            f"Available: {list(_AUG_REGISTRY.keys())}"
        )
    return _AUG_REGISTRY[name]
