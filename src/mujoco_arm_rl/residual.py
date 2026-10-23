"""Residual RL for the ball catch: a learned correction on top of the model-based controller.

    action = clip(base(obs) + scale * pi([obs, base(obs)]), -1, 1)

Why: the model-based controller already catches ~100 % in nominal conditions; what it gets
wrong is structured (servo lag on Earth with a low-bounce ball, over-trusting a noisy
estimate). Learning only the correction starts the policy at controller-level performance
instead of zero, keeps exploration around a good behaviour, and bounds the damage a bad
residual can do (``scale``). The policy also sees the base action, so it can learn *when*
to deviate. ``scale = 0`` reproduces the base controller exactly.

The C++ policy node runs the same thing (``policies.cpp``: NetworkController with a residual
export), so a residual policy trained here deploys unchanged.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mujoco_arm_rl.catching import CatchController, CatchParams, load_tuned

BASES = {"catch_tuned": "tuned", "catch_model": "model"}


def make_base(name: str) -> CatchController:
    if name == "catch_tuned":
        tuned = load_tuned()
        if not tuned:
            raise FileNotFoundError("configs/catch_tuned.json missing (run scripts/tune_catch.py)")
        return CatchController(tuned)
    if name == "catch_model":
        return CatchController(CatchParams())
    raise ValueError(f"unknown residual base {name!r}; use {sorted(BASES)}")


class ResidualCatchEnv(gym.Wrapper):
    """Observation: the env's 34 floats + the base action (37). Action: the residual (3)."""

    def __init__(self, env: gym.Env, base: str = "catch_tuned", scale: float = 0.3) -> None:
        super().__init__(env)
        self.base_name, self.scale = base, float(scale)
        self.base = make_base(base)
        n = env.observation_space.shape[0] + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._obs = None
        self._base_action = np.zeros(3)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        self._obs = obs
        self._base_action = self.base(obs, self.env.unwrapped)
        return np.concatenate([obs, self._base_action]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.base.reset()
        return self._augment(obs), info

    def step(self, action):
        res = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a = np.clip(self._base_action + self.scale * res, -1.0, 1.0)
        obs, r, term, trunc, info = self.env.step(a)
        info["residual_norm"] = float(np.linalg.norm(res))
        return self._augment(obs), r, term, trunc, info


class ResidualPolicy:
    """Evaluation-side twin: wraps a trained SB3 model as an ``act(obs, env)`` catch policy."""

    def __init__(self, model, base: str, scale: float) -> None:
        self.model, self.scale = model, float(scale)
        self.base = make_base(base)

    def reset(self) -> None:
        self.base.reset()

    def __call__(self, obs, env):
        b = self.base(obs, env)
        x = np.concatenate([obs, b]).astype(np.float32)
        res = self.model.predict(x, deterministic=True)[0]
        return np.clip(b + self.scale * np.asarray(res, dtype=np.float64), -1.0, 1.0)
