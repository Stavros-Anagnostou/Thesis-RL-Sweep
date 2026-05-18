"""
Neural network architectures for Procgen RL experiments.

Three encoder backbones:
  - IMPALA-CNN  : ResNet-style CNN from Espeholt et al. (2018) / Cobbe et al. (2020).
  - Nature-CNN  : Three-layer CNN from Mnih et al. (2015) / DQN.
  - SmallCNN    : Deliberately undersized 2-layer CNN for capacity ablation.

Two head wrappers:
  - ActorCritic : Policy + value heads (for PPO).
  - QNetwork    : Q-value head (for DQN).

All encoders accept uint8 (B, H, W, C) and normalise internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Module, std: float = 0.01, bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight init + constant bias — standard practice in PPO implementations."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _obs_to_float(obs: Tensor) -> Tensor:
    """Convert observations to float32 CHW [0, 1] regardless of input format.

    Handles both (B, H, W, C) uint8 from the rollout buffer and
    (B, C, H, W) float32 from augmentation pipelines (e.g. DrAC, RAD).
    """
    if obs.dtype == torch.uint8:
        return obs.permute(0, 3, 1, 2).float() / 255.0
    # Already float: if channel dim is small (1 or 3), assume CHW.
    if obs.ndim == 4 and obs.shape[1] in (1, 3):
        return obs.float()
    # Float but HWC layout — transpose.
    return obs.permute(0, 3, 1, 2).float()


# ---------------------------------------------------------------------------
# IMPALA-CNN (ResNet-style, as used in the Procgen benchmark)
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Two-layer residual block with same-padding to preserve spatial dims."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return x + residual


class _IMPALABlock(nn.Module):
    """One IMPALA conv-pool-residual-residual block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class IMPALAEncoder(nn.Module):
    """
    IMPALA-CNN encoder.

    Architecture matches `common/models.py` in the openai/train-procgen repository
    and CleanRL's Procgen IMPALA encoder.  Output feature dimension: 256.
    """

    # Channel widths from Cobbe et al. (2020)
    _CHANNEL_WIDTHS = [16, 32, 32]
    OUT_DIM = 256

    def __init__(self) -> None:
        super().__init__()
        channels = [3] + self._CHANNEL_WIDTHS
        self.blocks = nn.Sequential(
            *[_IMPALABlock(channels[i], channels[i + 1]) for i in range(len(self._CHANNEL_WIDTHS))]
        )
        # After three blocks with stride-2 pooling: 64 → 32 → 16 → 8
        # AdaptiveAvgPool makes the code robust if input size ever changes.
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Flatten(),
            _layer_init(nn.Linear(32 * 8 * 8, self.OUT_DIM), std=1.0),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """obs: (B, H, W, C) uint8  →  features: (B, 256)"""
        x = _obs_to_float(obs)
        x = self.blocks(x)
        x = self.pool(x)
        x = nn.functional.relu(x)   # ReLU after last block, before projection
        return self.fc(x)


# ---------------------------------------------------------------------------
# Nature-CNN (DQN)
# ---------------------------------------------------------------------------

class NatureEncoder(nn.Module):
    """
    Three-layer CNN from Mnih et al. (2015).  Output feature dimension: 512.

    Note: Procgen observations are 64×64, not 84×84 (Atari).  The first conv
    with kernel 8 stride 4 still works fine at this resolution (output 15×15).
    """

    OUT_DIM = 512

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4), std=1.0),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), std=1.0),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), std=1.0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Determine flattened size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            flat_dim = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(flat_dim, self.OUT_DIM), std=1.0),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """obs: (B, H, W, C) uint8  →  features: (B, 512)"""
        x = _obs_to_float(obs)
        return self.fc(self.cnn(x))


# ---------------------------------------------------------------------------
# Actor-Critic wrapper
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Wraps an encoder backbone with a policy head and value head.

    Usage::

        model = ActorCritic(encoder="impala", num_actions=15)
        action, log_prob, entropy, value = model.get_action_and_value(obs)
    """

    def __init__(self, encoder: str, num_actions: int) -> None:
        super().__init__()
        if encoder == "impala":
            self.encoder = IMPALAEncoder()
            hidden_dim = IMPALAEncoder.OUT_DIM
        elif encoder == "nature":
            self.encoder = NatureEncoder()
            hidden_dim = NatureEncoder.OUT_DIM
        elif encoder == "small":
            self.encoder = SmallCNNEncoder()
            hidden_dim = SmallCNNEncoder.OUT_DIM
        else:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose 'impala', 'nature', or 'small'.")

        # Orthogonal init with small std for policy head (common in PPO implementations)
        self.policy_head = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        # Value head — std=1 keeps values in a reasonable range before learning
        self.value_head = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def get_value(self, obs: Tensor) -> Tensor:
        """Return scalar value estimate for a batch of observations."""
        return self.value_head(self.encoder(obs))

    def get_action_and_value(
        self,
        obs: Tensor,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample an action (or evaluate a supplied one) and return all PPO quantities.

        Returns
        -------
        action     : (B,) int64
        log_prob   : (B,) float32
        entropy    : (B,) float32  — per-sample entropy for logging
        value      : (B, 1) float32
        """
        features = self.encoder(obs)
        logits = self.policy_head(features)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(features)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# SmallCNN — deliberately undersized encoder for capacity ablation
# ---------------------------------------------------------------------------

class SmallCNNEncoder(nn.Module):
    """
    Two-layer CNN encoder.  Deliberately small (128-dim features) to test
    whether reduced capacity forces better generalization by preventing
    overfitting to training-level textures, or simply underperforms.

    Architecture:
      Conv(16, 5×5, stride=2) → ReLU → Conv(32, 3×3, stride=2) → ReLU
      → Flatten → Linear(*, 128) → ReLU

    Output feature dimension: 128.
    """

    OUT_DIM = 128

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(3, 16, kernel_size=5, stride=2), std=1.0),
            nn.ReLU(),
            _layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2), std=1.0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            flat_dim = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(flat_dim, self.OUT_DIM), std=1.0),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """obs: (B, H, W, C) uint8  →  features: (B, 128)"""
        x = _obs_to_float(obs)
        return self.fc(self.cnn(x))


# ---------------------------------------------------------------------------
# QNetwork wrapper (for DQN)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Wraps an encoder with a single Q-value head for DQN.

    Usage::

        q_net = QNetwork(encoder="impala", num_actions=15)
        q_values = q_net.get_q_values(obs)   # (B, num_actions)
    """

    def __init__(self, encoder: str, num_actions: int) -> None:
        super().__init__()
        if encoder == "impala":
            self.encoder = IMPALAEncoder()
            hidden_dim = IMPALAEncoder.OUT_DIM
        elif encoder == "nature":
            self.encoder = NatureEncoder()
            hidden_dim = NatureEncoder.OUT_DIM
        elif encoder == "small":
            self.encoder = SmallCNNEncoder()
            hidden_dim = SmallCNNEncoder.OUT_DIM
        else:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose 'impala', 'nature', or 'small'.")

        self.q_head = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

    def get_q_values(self, obs: Tensor) -> Tensor:
        """Return Q(s, a) for all actions.  obs: (B, H, W, C) uint8  →  (B, num_actions)"""
        return self.q_head(self.encoder(obs))

    def forward(self, obs: Tensor) -> Tensor:
        """Alias for get_q_values — makes nn.Module API consistent."""
        return self.get_q_values(obs)
