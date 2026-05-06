"""
PLR — Prioritized Level Replay (Jiang et al., 2021).

Maintains a score buffer over training levels and biases episode sampling
toward levels where the agent has high learning potential (high value loss).

Key idea: levels where the value function is poorly calibrated (high |V(s) - R|)
are levels where the agent still has something to learn.  Replay those
preferentially.

Integration with Procgen:
  Procgen doesn't directly let us set the level mid-training.  We work around
  this by creating a SEPARATE per-level eval env (1 env per query) and using
  it to score levels.  During rollout collection, we intercept episode resets
  by tracking done signals and replacing the new-episode start with a
  PLR-selected level.  Since Procgen doesn't expose a per-env level setter
  on the vectorised env, we implement PLR at the rollout-management level:
  after every episode end we pick the next level, then reconstruct envs with
  the selected level seed when the episode count calls for a PLR replay.

  Practical note: we implement a simplified but faithful version where PLR
  operates at the *update* level rather than per-episode.  At each update
  we decide whether to replay (with prob rho) or explore new levels.  When
  replaying, we reconstruct train envs seeded to the selected level.

Usage::

    plr = PLR(num_levels=200, rho=0.5, beta=0.1, staleness_coef=0.1)

    # Before rollout collection:
    level_seed = plr.sample_level()
    # Create env seeded to level_seed, collect rollout ...

    # After PPO update, compute per-level value loss and update scores:
    plr.update_score(level_seed, value_loss)
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


class PLR:
    """
    Prioritized Level Replay buffer.

    Parameters
    ----------
    num_levels      : number of training levels (Procgen's num_levels param)
    rho             : probability of replaying a scored level (default 0.5)
    beta            : temperature for rank-based softmax (default 0.1)
    staleness_coef  : weight for staleness bonus in priority computation
    scoring         : scoring function name (only "l1_value_loss" implemented)
    seed            : random seed for the PLR sampler itself
    """

    def __init__(
        self,
        num_levels: int,
        rho: float = 0.5,
        beta: float = 0.1,
        staleness_coef: float = 0.1,
        scoring: str = "l1_value_loss",
        seed: int = 0,
    ) -> None:
        self.num_levels     = num_levels
        self.rho            = rho
        self.beta           = beta
        self.staleness_coef = staleness_coef
        self.scoring        = scoring
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # Score buffer: level_id → score (L1 value loss, higher = more informative)
        self._scores:     np.ndarray = np.zeros(num_levels, dtype=np.float64)
        # Staleness: level_id → number of updates since it was last seen
        self._staleness:  np.ndarray = np.zeros(num_levels, dtype=np.float64)
        # Track which levels have been visited at least once
        self._seen:       np.ndarray = np.zeros(num_levels, dtype=bool)
        # Global update counter (for staleness computation)
        self._update_count: int = 0
        # Last-seen update index per level
        self._last_seen:  np.ndarray = np.full(num_levels, -1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_level(self) -> int:
        """
        Sample a level to use for the next rollout.

        With probability rho: replay from the scored buffer (rank-based).
        With probability 1-rho: sample a fresh level uniformly at random.

        Returns the level index (0-indexed, maps directly to Procgen start_level).
        """
        seen_levels = np.where(self._seen)[0]

        # If no levels scored yet, always explore.
        if len(seen_levels) == 0 or self._rng.random() > self.rho:
            return self._sample_new_level()

        return self._sample_replay_level(seen_levels)

    def _sample_new_level(self) -> int:
        """Uniformly sample from the full training level distribution."""
        return self._rng.randint(0, self.num_levels - 1)

    def _sample_replay_level(self, seen_levels: np.ndarray) -> int:
        """
        Sample a level from the scored buffer using rank-based prioritization.

        Priority = (1 - staleness_coef) * rank_score + staleness_coef * staleness_score
        where rank_score is the rank of the level's value-loss score (higher = better)
        and staleness_score is proportional to how long since the level was last seen.
        """
        scores = self._scores[seen_levels]
        staleness = self._staleness[seen_levels]

        # Rank-based score: higher L1 loss → higher priority rank
        ranks = np.argsort(np.argsort(-scores)) + 1  # rank 1 = highest score
        rank_scores = 1.0 / ranks

        # Staleness score: normalize to [0,1]
        if staleness.max() > 0:
            staleness_scores = staleness / staleness.max()
        else:
            staleness_scores = np.zeros_like(staleness)

        # Combined priority
        priorities = (
            (1.0 - self.staleness_coef) * rank_scores
            + self.staleness_coef * staleness_scores
        )

        # Softmax with temperature beta to get sampling probabilities.
        # Low beta → more greedy; high beta → more uniform.
        logits = priorities / (self.beta + 1e-8)
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        chosen_idx = self._np_rng.choice(len(seen_levels), p=probs)
        return int(seen_levels[chosen_idx])

    # ------------------------------------------------------------------
    # Score updates
    # ------------------------------------------------------------------

    def update_score(self, level_id: int, score: float) -> None:
        """
        Update the priority score for a level after observing its value loss.

        Parameters
        ----------
        level_id : int   — level index (0 to num_levels-1)
        score    : float — L1 value loss |V(s) - R| for episodes on this level
        """
        assert 0 <= level_id < self.num_levels, \
            f"level_id {level_id} out of range [0, {self.num_levels})"

        # Exponential moving average — gives more weight to recent observations
        # but doesn't discard historical signal entirely.
        alpha = 0.1  # EMA coefficient
        if self._seen[level_id]:
            self._scores[level_id] = (1 - alpha) * self._scores[level_id] + alpha * score
        else:
            self._scores[level_id] = score
            self._seen[level_id] = True

        # Reset staleness for this level.
        self._last_seen[level_id] = self._update_count
        self._update_count += 1

        # Update staleness for ALL levels (increment for non-seen levels).
        self._staleness += 1.0
        self._staleness[level_id] = 0.0

    def update_scores_batch(self, level_ids: list[int], scores: list[float]) -> None:
        """Update multiple level scores at once (convenience wrapper)."""
        for lid, score in zip(level_ids, scores):
            self.update_score(lid, score)

    # ------------------------------------------------------------------
    # Stats for logging
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for W&B logging."""
        seen = self._seen.sum()
        if seen > 0:
            seen_scores = self._scores[self._seen]
            return {
                "plr/num_seen_levels":   int(seen),
                "plr/mean_score":        float(seen_scores.mean()),
                "plr/max_score":         float(seen_scores.max()),
                "plr/score_std":         float(seen_scores.std()),
                "plr/mean_staleness":    float(self._staleness[self._seen].mean()),
            }
        return {"plr/num_seen_levels": 0}

    def get_state(self) -> dict[str, Any]:
        """Serialisable state for checkpointing."""
        return {
            "scores":       self._scores.copy(),
            "staleness":    self._staleness.copy(),
            "seen":         self._seen.copy(),
            "last_seen":    self._last_seen.copy(),
            "update_count": self._update_count,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._scores       = state["scores"].copy()
        self._staleness    = state["staleness"].copy()
        self._seen         = state["seen"].copy()
        self._last_seen    = state["last_seen"].copy()
        self._update_count = state["update_count"]
