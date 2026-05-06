"""
RAD — Reinforcement Learning with Augmented Data (Laskin et al., 2020).

Conceptually the simplest augmentation baseline: replace the observations fed
to the PPO update with augmented versions.  No extra loss terms.

Usage inside the PPO update loop::

    rad = RAD(aug_type="crop")
    # Inside the minibatch loop, before computing policy/value:
    mb_obs_aug = rad.augment(mb_obs)   # use this instead of mb_obs
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.augmentations import get_augmentation


class RAD:
    """
    Applies a fixed augmentation to minibatch observations during PPO updates.

    The augmentation is applied ONLY during training (inside the PPO update),
    not at environment-step time and not during evaluation.  This means the
    replay buffer stores raw observations and the augmentation is freshly
    applied each time a minibatch is drawn — providing a different augmentation
    per epoch, acting as data multiplier.
    """

    def __init__(self, aug_type: str) -> None:
        """
        Parameters
        ----------
        aug_type : str
            Name of augmentation to apply (must be a key in augmentations.py registry).
        """
        self.aug_type = aug_type
        self._augment_fn = get_augmentation(aug_type)

    def augment(self, obs: Tensor) -> Tensor:
        """
        Augment a minibatch of observations.

        Parameters
        ----------
        obs : Tensor
            uint8 (B, H, W, C) OR float32 (B, C, H, W) — handles both.

        Returns
        -------
        Tensor
            Float32 (B, C, H, W) augmented observations in [0, 1].
        """
        obs_float = _to_float_chw(obs)
        return self._augment_fn(obs_float)

    def __repr__(self) -> str:
        return f"RAD(aug_type={self.aug_type!r})"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _to_float_chw(obs: Tensor) -> Tensor:
    """
    Convert observations to float32 CHW [0,1] regardless of input format.

    Handles:
      - uint8  (B, H, W, C)  — Procgen native format from rollout buffer
      - float32 (B, C, H, W) — already converted (e.g., second pass through)
    """
    if obs.dtype == torch.uint8:
        return obs.permute(0, 3, 1, 2).float() / 255.0
    # If already float and CHW, pass through.
    if obs.ndim == 4 and obs.shape[1] in (1, 3):
        return obs.float()
    # Float but HWC — transpose.
    return obs.permute(0, 3, 1, 2).float()
