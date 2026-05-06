"""
UCB-DrAC — UCB1 bandit for automatic augmentation selection in DrAC.

Raileanu & Fergus (2021), Section 4.3.

Maintains a multi-armed bandit over the set of available augmentations.
At each PPO update, selects the augmentation with the highest UCB1 score,
applies DrAC with that augmentation, then updates the bandit with the
observed episodic return as the reward signal.

Usage::

    bandit = UCBDrAC(
        model=model,
        aug_names=["crop", "color_jitter", "grayscale", "cutout", "flip", "rotate", "random_conv"],
        aug_coef=0.1,
        ucb_c=0.1,
    )
    # Inside update loop:
    drac_loss, selected_aug = bandit.compute_loss_and_select(mb_obs)
    total_loss = ppo_loss + drac_loss
    # After computing mean episodic return for this update:
    bandit.update(selected_aug, observed_mean_return)
    # Log bandit stats to W&B:
    wandb.log(bandit.get_stats())
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from src.drac import DrAC
from src.augmentations import ALL_AUGMENTATIONS


class UCBDrAC:
    """
    UCB1 bandit that selects the DrAC augmentation to use each PPO update.

    Parameters
    ----------
    model       : ActorCritic
    aug_names   : list of augmentation name strings to consider
    aug_coef    : DrAC regularization weight (shared across all arms)
    ucb_c       : UCB1 exploration coefficient (default 0.1 from paper)
    drac_mode   : "full" | "actor_only" | "critic_only"
    """

    def __init__(
        self,
        model: "ActorCritic",  # noqa: F821
        aug_names: list[str] | None = None,
        aug_coef: float = 0.1,
        ucb_c: float = 0.1,
        drac_mode: str = "full",
    ) -> None:
        self.model = model
        self.aug_names: list[str] = aug_names if aug_names is not None else ALL_AUGMENTATIONS
        self.ucb_c = ucb_c

        # One DrAC instance per arm (pre-built to avoid repeated augmentation lookups).
        self._arms: list[DrAC] = [
            DrAC(model=model, aug_type=name, aug_coef=aug_coef, mode=drac_mode)
            for name in self.aug_names
        ]
        n = len(self.aug_names)
        # Bandit statistics — initialise with one virtual pull to avoid div/0.
        self._counts  = [1.0] * n          # number of times each arm was pulled
        self._rewards = [0.0] * n          # cumulative reward per arm
        self._total   = float(n)           # total pulls

    def _ucb_score(self, arm_idx: int) -> float:
        """UCB1 score for arm i."""
        mean  = self._rewards[arm_idx] / self._counts[arm_idx]
        bonus = self.ucb_c * math.sqrt(math.log(self._total) / self._counts[arm_idx])
        return mean + bonus

    def select_arm(self) -> int:
        """Return the index of the arm with the highest UCB1 score."""
        return max(range(len(self.aug_names)), key=self._ucb_score)

    def compute_loss_and_select(self, obs: Tensor) -> tuple[Tensor, str]:
        """
        Select augmentation via UCB1, compute DrAC loss, return (loss, aug_name).

        Call update() afterwards with the observed episode return.
        """
        arm_idx = self.select_arm()
        loss = self._arms[arm_idx].compute_loss(obs)
        return loss, self.aug_names[arm_idx]

    def update(self, aug_name: str, observed_return: float) -> None:
        """
        Update bandit statistics after observing a reward.

        Parameters
        ----------
        aug_name        : name of the augmentation that was used
        observed_return : mean episodic return for this update (reward signal)
        """
        idx = self.aug_names.index(aug_name)
        self._counts[idx]  += 1.0
        self._rewards[idx] += observed_return
        self._total        += 1.0

    def get_stats(self) -> dict[str, Any]:
        """Return per-arm statistics for W&B logging."""
        stats: dict[str, Any] = {}
        for i, name in enumerate(self.aug_names):
            mean = self._rewards[i] / self._counts[i]
            stats[f"ucb_bandit/count_{name}"]  = int(self._counts[i])
            stats[f"ucb_bandit/reward_{name}"] = mean
            stats[f"ucb_bandit/ucb_{name}"]    = self._ucb_score(i)
        selected = self.aug_names[self.select_arm()]
        stats["ucb_bandit/selected_aug"] = selected
        return stats

    def get_selected_aug_name(self) -> str:
        """Return the name of the currently preferred augmentation."""
        return self.aug_names[self.select_arm()]

    def __repr__(self) -> str:
        return (
            f"UCBDrAC(n_arms={len(self.aug_names)}, "
            f"ucb_c={self.ucb_c}, "
            f"arms={self.aug_names})"
        )
