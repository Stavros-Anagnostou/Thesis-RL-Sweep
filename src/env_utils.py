"""
Environment creation and wrappers — EnvPool primary, Procgen fallback.

EnvPool provides a C++-based batched environment pool with a thread pool that
achieves ~30-50% throughput improvement over native Procgen vectorisation.
It supports Procgen natively, making it a drop-in replacement.

Key design decisions:
  - **EnvPool is the default backend.**  Sync mode (batch_size == num_envs)
    so every step returns results for all envs simultaneously.
  - **Procgen native is the fallback** when EnvPool is unavailable.
    Set ``env_backend="procgen"`` in config to force the fallback.
  - Both backends expose the same interface: ``.reset() -> (N,H,W,C) uint8``,
    ``.step(actions) -> (obs, reward, done, info)``.
  - Reward normalisation wraps whichever backend is active.  The wrapper's
    gamma MUST match the PPO discount factor.
  - Evaluation envs do NOT use reward normalisation.

EnvPool Procgen task ID format:
  ``{GameCapitalized}{ModeCapitalized}-v0``  e.g. "CoinrunEasy-v0", "BigfishHard-v0"
  (NOT the old "procgen:procgen-coinrun-v0" format)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Any

import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _envpool_available() -> bool:
    """Check whether EnvPool is importable and has Procgen support."""
    try:
        import envpool  # noqa: F401
        all_envs = envpool.list_all_envs()
        # EnvPool registers Procgen as "{Game}{Mode}-v0", e.g. "CoinrunEasy-v0"
        return any("CoinrunEasy" in e for e in all_envs)
    except (ImportError, Exception):
        return False


_ENVPOOL_OK = _envpool_available()

# Map string distribution modes to EnvPool task ID suffixes.
_DIST_MODE_LABEL: dict[str, str] = {
    "easy": "Easy",
    "hard": "Hard",
    "extreme": "Extreme",
    "memory": "Memory",
    "exploration": "Exploration",
}


# ---------------------------------------------------------------------------
# EnvPool-backed vectorised wrapper
# ---------------------------------------------------------------------------

class EnvPoolVecWrapper:
    """
    Wraps an EnvPool Procgen environment to match the VecProcgenWrapper
    interface used by the rest of the codebase.

    Translates:
      - EnvPool gymnasium reset()  -> obs (N, H, W, C) uint8
      - EnvPool gymnasium step()   -> (obs, reward, done, infos)
        where done = terminated | truncated
    """

    def __init__(self, envpool_env: Any) -> None:
        self._env = envpool_env
        self.num_envs: int = envpool_env.config["num_envs"]
        self.observation_space: spaces.Box = envpool_env.observation_space
        self.action_space: spaces.Discrete = envpool_env.action_space

    def reset(self) -> np.ndarray:
        """Reset all envs, return obs (N, H, W, C) uint8."""
        obs, _info = self._env.reset()
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all envs. Returns (obs, reward, done, infos)."""
        obs, reward, terminated, truncated, info = self._env.step(actions)
        done = np.logical_or(terminated, truncated)
        # EnvPool returns info as a dict-of-arrays. Convert to list-of-dicts
        # to match VecProcgenWrapper's interface. Some values may be scalars
        # or non-indexable, so handle those gracefully.
        n = len(reward)
        if isinstance(info, dict):
            infos: list[dict] = []
            for i in range(n):
                d = {}
                for k, v in info.items():
                    try:
                        d[k] = v[i]
                    except (IndexError, TypeError, KeyError):
                        d[k] = v
                infos.append(d)
        else:
            infos = [{}] * n
        return obs, reward.astype(np.float32), done.astype(np.float32), infos

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Procgen-native vectorised wrapper (fallback)
# ---------------------------------------------------------------------------

class VecProcgenWrapper:
    """
    Minimal VecEnv-style wrapper around the native ProcgenEnv.
    Only used when env_backend="procgen" or EnvPool is unavailable.
    """

    def __init__(self, procgen_env: Any) -> None:
        self._env = procgen_env
        self.num_envs: int = procgen_env.num_envs
        self.observation_space: spaces.Box = procgen_env.observation_space["rgb"]
        self.action_space: spaces.Discrete = procgen_env.action_space

    def reset(self) -> np.ndarray:
        obs_dict = self._env.reset()
        return obs_dict["rgb"]

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs_dict, reward, done, info = self._env.step(actions)
        return obs_dict["rgb"], reward, done, info

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Gymnasium-compatible single-env shim (legacy fallback only)
# ---------------------------------------------------------------------------

class ProcgenGymWrapper(gym.Env):
    """Wraps a single-environment view of a ProcgenEnv batch."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, procgen_env: Any) -> None:
        super().__init__()
        self._env = procgen_env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self._env.reset()
        self._last_obs = obs["rgb"]
        return self._last_obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(np.array([action]))
        obs_rgb = obs["rgb"][0]
        self._last_obs = obs_rgb
        return obs_rgb, float(reward[0]), bool(done[0]), False, info[0] if info else {}

    def render(self):
        return self._last_obs

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Running reward normaliser (backend-agnostic)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for computing running mean and variance."""

    def __init__(self, epsilon: float = 1e-8, shape: tuple = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def get_state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def set_state(self, state: dict) -> None:
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class RewardNormWrapper:
    """
    Wraps any vec env and normalises rewards using a running return estimate.
    Uses the same gamma as the PPO agent (critical!).
    """

    def __init__(self, env: Any, gamma: float, epsilon: float = 1e-8) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(env.num_envs, dtype=np.float64)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs

    def reset(self) -> np.ndarray:
        self.returns[:] = 0.0
        return self.env.reset()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs, reward, done, info = self.env.step(actions)
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        reward_norm = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        self.returns[done.astype(bool)] = 0.0
        return obs, reward_norm.astype(np.float32), done, info

    def close(self) -> None:
        self.env.close()

    def get_reward_normalizer_state(self) -> dict:
        return {"ret_rms": self.ret_rms.get_state(), "returns": self.returns.copy()}

    def set_reward_normalizer_state(self, state: dict) -> None:
        self.ret_rms.set_state(state["ret_rms"])
        self.returns = state["returns"].copy()


# ---------------------------------------------------------------------------
# Internal: create raw vectorised env
# ---------------------------------------------------------------------------

def _make_envpool_procgen(
    env_id: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    seed: int,
    num_threads: int | None = None,
) -> EnvPoolVecWrapper:
    """Create a Procgen env via EnvPool and wrap it."""
    import envpool

    # EnvPool task IDs: "{GameCapitalized}{ModeCapitalized}-v0"
    mode_label = _DIST_MODE_LABEL.get(distribution_mode, "Easy")
    task_id = f"{env_id.capitalize()}{mode_label}-v0"

    if num_threads is None:
        num_threads = min(num_envs, os.cpu_count() or num_envs)

    raw_env = envpool.make(
        task_id,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        num_threads=num_threads,
        seed=seed,
        num_levels=num_levels,
        start_level=start_level,
        channel_first=False,          # (H, W, C) to match our networks
    )
    return EnvPoolVecWrapper(raw_env)


def _make_native_procgen(
    env_id: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    seed: int,
) -> VecProcgenWrapper:
    """Create a Procgen env via the native ProcgenEnv (fallback)."""
    try:
        from procgen import ProcgenEnv
    except ImportError as e:
        raise ImportError(
            "Neither EnvPool nor procgen are available.\n"
            "Install EnvPool:  pip install envpool\n"
            "Or install Procgen:  pip install procgen==0.10.7"
        ) from e

    raw_env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        rand_seed=seed,
    )
    return VecProcgenWrapper(raw_env)


def _choose_backend(cfg_backend: str | None) -> str:
    """Resolve the effective backend: 'envpool' or 'procgen'."""
    if cfg_backend == "procgen":
        return "procgen"
    if cfg_backend == "envpool" or cfg_backend is None or cfg_backend == "auto":
        if _ENVPOOL_OK:
            return "envpool"
        import warnings
        warnings.warn(
            "EnvPool is not available -- falling back to native Procgen.\n"
            "Install EnvPool for a ~30-50 % speedup: pip install envpool"
        )
        return "procgen"
    raise ValueError(
        f"Unknown env_backend '{cfg_backend}'. Use 'envpool', 'procgen', or 'auto'."
    )


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_procgen_envs(
    env_id: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    gamma: float,
    normalize_reward: bool,
    seed: int,
    env_backend: str | None = None,
) -> RewardNormWrapper | EnvPoolVecWrapper | VecProcgenWrapper:
    """Create vectorised Procgen training environments."""
    backend = _choose_backend(env_backend)

    if backend == "envpool":
        vec_env = _make_envpool_procgen(
            env_id=env_id, num_envs=num_envs, num_levels=num_levels,
            start_level=start_level, distribution_mode=distribution_mode,
            seed=seed,
        )
    else:
        vec_env = _make_native_procgen(
            env_id=env_id, num_envs=num_envs, num_levels=num_levels,
            start_level=start_level, distribution_mode=distribution_mode,
            seed=seed,
        )

    if normalize_reward:
        return RewardNormWrapper(vec_env, gamma=gamma)
    return vec_env


def make_eval_envs(
    env_id: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    seed: int = 0,
    env_backend: str | None = None,
) -> EnvPoolVecWrapper | VecProcgenWrapper:
    """Create vectorised Procgen evaluation environments (no reward normalisation)."""
    backend = _choose_backend(env_backend)

    if backend == "envpool":
        return _make_envpool_procgen(
            env_id=env_id, num_envs=num_envs, num_levels=num_levels,
            start_level=start_level, distribution_mode=distribution_mode,
            seed=seed,
        )
    else:
        return _make_native_procgen(
            env_id=env_id, num_envs=num_envs, num_levels=num_levels,
            start_level=start_level, distribution_mode=distribution_mode,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# MiniGrid support (not available in EnvPool)
# ---------------------------------------------------------------------------

MINIGRID_ENVS = [
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-ObstructedMaze-1Dl-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
]


def _make_single_minigrid(
    env_id: str,
    seed: int,
    use_rgb: bool = True,
    img_size: int = 64,
) -> gym.Env:
    """Create a single MiniGrid environment with pixel observations."""
    try:
        import minigrid  # noqa: F401
        from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
    except ImportError as e:
        raise ImportError("minigrid is not installed. Run: pip install minigrid") from e

    env = gym.make(env_id)
    env.reset(seed=seed)
    if use_rgb:
        env = RGBImgObsWrapper(env)
    else:
        env = ImgObsWrapper(env)
    env = _MiniGridResizeWrapper(env, img_size=img_size)
    return env


class _MiniGridResizeWrapper(gym.ObservationWrapper):
    """Resizes MiniGrid pixel observations to (img_size, img_size, 3) uint8."""

    def __init__(self, env: gym.Env, img_size: int = 64) -> None:
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8,
        )

    def observation(self, obs):
        import cv2
        img = obs["image"] if isinstance(obs, dict) else obs
        resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


def make_minigrid_envs(
    env_id: str,
    num_envs: int,
    seed: int = 0,
    use_rgb: bool = True,
    img_size: int = 64,
) -> gym.vector.SyncVectorEnv:
    """Create vectorised MiniGrid environments using Gymnasium's SyncVectorEnv."""
    if env_id not in MINIGRID_ENVS:
        import warnings
        warnings.warn(f"env_id '{env_id}' not in standard MINIGRID_ENVS list.")

    def _make_env(rank: int):
        def _init():
            return _make_single_minigrid(env_id, seed=seed + rank, use_rgb=use_rgb, img_size=img_size)
        return _init

    return gym.vector.SyncVectorEnv([_make_env(i) for i in range(num_envs)])
