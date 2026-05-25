"""
DrAC — Data-Regularized Actor-Critic (Raileanu & Fergus, 2021).

Adds two regularization losses to the standard PPO objective:
  - Policy regularization:  KL( π(·|aug(s)) || π(·|s) )
  - Value regularization:   MSE( V(aug(s)), V(s) )

The augmented observations are used ONLY for the regularization terms.
The main PPO surrogate loss is computed on the ORIGINAL observations.

Usage inside the PPO minibatch update::

    drac = DrAC(model, aug_type="crop", aug_coef=0.1, mode="full")
    drac_loss = drac.compute_loss(mb_obs)
    total_loss = ppo_loss + drac_loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.augmentations import get_augmentation
from src.rad import _to_float_chw


class DrAC:
    """
    Computes the DrAC regularization loss for a single minibatch.

    Parameters
    ----------
    model      : ActorCritic — the policy/value network
    aug_type   : str         — augmentation to apply
    aug_coef   : float       — weight λ on the regularization loss
    mode       : str         — "full" | "actor_only" | "critic_only"
                               Controls which regularization term is included.
                               Used for the ablation study (drac_mode in config).
    """

    def __init__(
        self,
        model: "ActorCritic",  # noqa: F821 — forward reference
        aug_type: str,
        aug_coef: float = 0.1,
        mode: str = "full",
    ) -> None:
        self.model = model
        self.aug_coef = aug_coef
        self.mode = mode
        self._augment_fn = get_augmentation(aug_type)

    def compute_loss(self, obs: Tensor) -> Tensor:
        """
        Compute the DrAC regularization loss for a minibatch.

        Parameters
        ----------
        obs : Tensor
            uint8 (B, H, W, C) — original observations (already on device).

        Returns
        -------
        Tensor
            Scalar regularization loss (already weighted by aug_coef).
        """
        obs_float = _to_float_chw(obs)           # (B, C, H, W) float32

        # Augmented observations — gradients flow through the augmentation.
        obs_aug = self._augment_fn(obs_float)    # (B, C, H, W) float32

        loss = torch.tensor(0.0, device=obs.device)

        with torch.no_grad():
            # One encoder pass for original obs — cache features for both heads.
            orig_features = self.model.encoder(obs_float)
            orig_logits   = self.model.policy_head(orig_features)
            orig_value    = self.model.value_head(orig_features).squeeze(-1)

        # One encoder pass for augmented obs — cache features for both heads.
        aug_features = self.model.encoder(obs_aug)

        if self.mode in ("full", "actor_only"):
            # Policy regularization: KL( π_aug || π_orig )
            # Using π_orig as the fixed target.
            aug_logits = self.model.policy_head(aug_features)
            # KL(P||Q) = sum P * (log P - log Q)  where P = aug policy
            # Using F.kl_div which expects log-probs for input and probs for target.
            log_p_aug = F.log_softmax(aug_logits, dim=-1)
            p_orig    = F.softmax(orig_logits,   dim=-1)
            policy_reg = F.kl_div(log_p_aug, p_orig, reduction="batchmean")
            loss = loss + policy_reg

        if self.mode in ("full", "critic_only"):
            # Value regularization: MSE( V(aug), V(orig) )
            aug_value = self.model.value_head(aug_features).squeeze(-1)
            value_reg = F.mse_loss(aug_value, orig_value)
            loss = loss + value_reg

        return self.aug_coef * loss

    def __repr__(self) -> str:
        return (
            f"DrAC(aug_type={self._augment_fn.__name__!r}, "
            f"aug_coef={self.aug_coef}, mode={self.mode!r})"
        )
