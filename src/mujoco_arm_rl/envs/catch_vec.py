"""Vectorised ball-catch env on top of the C++ simulator (module ``mujoco_arm_cpp_py``).

The Python ``BallCatchEnv`` steps one trial at a time and spends most of its time in Python
code around MuJoCo. ``CatchBatch`` (C++) keeps N simulators and steps them in a thread pool,
so here we only wrap it:

* ``CatchVecEnv``      -- a ``gymnasium.vector.VectorEnv`` (same-step autoreset), and
* ``CatchSB3VecEnv``   -- the same thing with the Stable-Baselines3 ``VecEnv`` interface
                          (``terminal_observation``, ``TimeLimit.truncated``, ``is_success``).

Observations, rewards and outcomes are the same as ``BallCatchEnv`` (see
tests/test_cpp_backend.py). The only difference is the random number generator (C++ uses
``std::mt19937_64``), so a given seed gives other balls than in Python.

With ``residual_scale > 0`` the C++ side also runs the tuned model-based controller and the
policy only outputs a correction (like ``residual.py``, observation 37 values).

The module is optional; build it with
``pip install --no-build-isolation ./ros2/src/mujoco_arm_cpp``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from mujoco_arm_rl.envs.ballcatch import DEFAULT_MODEL_PATH, GRAVITIES, resolve_gravity

TUNED_PATH = Path(__file__).resolve().parents[3] / "configs" / "catch_tuned.json"


def load_cpp():
    """Import the compiled module, with a useful message if it is not installed."""
    try:
        import mujoco_arm_cpp_py
    except ImportError as err:  # pragma: no cover - depends on the local build
        raise ImportError(
            "the C++ backend is not installed. Build it with:\n"
            "    pip install scikit-build-core pybind11\n"
            "    pip install --no-build-isolation ./ros2/src/mujoco_arm_cpp"
        ) from err
    return mujoco_arm_cpp_py


def cpp_available() -> bool:
    try:
        load_cpp()
        return True
    except ImportError:
        return False


def _gravity_list(gravity) -> list[float]:
    if gravity in (None, "random"):
        return list(GRAVITIES.values())
    if isinstance(gravity, (list, tuple)):
        return [resolve_gravity(g) for g in gravity]
    return [resolve_gravity(gravity)]


def _tuned_params() -> dict:
    if not TUNED_PATH.exists():
        raise FileNotFoundError(f"{TUNED_PATH} not found (run scripts/tune_catch.py)")
    return json.loads(TUNED_PATH.read_text())["params"]


def make_batch(n_envs: int = 8, env_kwargs: dict[str, Any] | None = None, seed: int = 0,
               threads: int = 0, residual_scale: float = 0.0, base_params: dict | None = None):
    """Build a ``CatchBatch``. env_kwargs are the same keywords as ``BallCatchEnv``."""
    cpp = load_cpp()
    kwargs = dict(env_kwargs or {})
    gravities = _gravity_list(kwargs.pop("gravity", "random"))
    kwargs.pop("start", None)  # the batch env always starts at the release (training setup)
    if residual_scale > 0 and base_params is None:
        base_params = _tuned_params()
    return cpp.CatchBatch(str(DEFAULT_MODEL_PATH), n_envs=n_envs, env_kwargs=kwargs, gravities=gravities,
                          seed=seed, threads=threads, residual_scale=residual_scale,
                          base_params=base_params or {})


class CatchVecEnv:
    """gymnasium-style vector env. ``step`` returns (obs, reward, terminated, truncated, infos).

    Autoreset is "same step": when an env finishes, the returned obs is already the first obs of
    the next episode and the last one is in ``infos["final_obs"]`` (mask ``infos["_final_obs"]``).
    I did not inherit from gymnasium.vector.VectorEnv on purpose because its autoreset API
    changed between gymnasium 0.29 and 1.x; the attributes below are the ones tools use.
    """

    metadata = {"autoreset_mode": "SameStep"}

    def __init__(self, n_envs: int = 8, env_kwargs: dict[str, Any] | None = None, seed: int = 0,
                 threads: int = 0, residual_scale: float = 0.0) -> None:
        self.batch = make_batch(n_envs, env_kwargs, seed, threads, residual_scale)
        self.num_envs = self.batch.num_envs
        obs_dim = self.batch.obs_dim
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.num_envs, obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.num_envs, 3), dtype=np.float32)
        self.residual_scale = residual_scale

    def reset(self, seed: int | None = None, options=None):
        return self.batch.reset(seed), {}

    def step(self, actions):
        a = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, 3)
        obs, rew, term, trunc, succ, final = self.batch.step(a)
        done = term | trunc
        infos: dict[str, Any] = {}
        if done.any():
            infos = {"final_obs": final, "_final_obs": done, "is_success": succ, "_is_success": done}
        return obs, rew, term, trunc, infos

    def close(self) -> None:
        self.batch = None


def make_sb3_vec_env(n_envs: int = 8, env_kwargs: dict[str, Any] | None = None, seed: int = 0,
                     threads: int = 0, residual_scale: float = 0.0):
    """Stable-Baselines3 ``VecEnv`` around CatchBatch (imported lazily, SB3 is optional)."""
    from stable_baselines3.common.vec_env import VecEnv

    class CatchSB3VecEnv(VecEnv):
        def __init__(self) -> None:
            self.batch = make_batch(n_envs, env_kwargs, seed, threads, residual_scale)
            obs_space = spaces.Box(-np.inf, np.inf, shape=(self.batch.obs_dim,), dtype=np.float32)
            act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            super().__init__(self.batch.num_envs, obs_space, act_space)
            self._actions = None
            self._seed = seed

        # -- the VecEnv API ------------------------------------------------------
        def reset(self):
            return self.batch.reset(self._seed)

        def step_async(self, actions) -> None:
            self._actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, 3)

        def step_wait(self):
            obs, rew, term, trunc, succ, final = self.batch.step(self._actions)
            dones = term | trunc
            infos = [{} for _ in range(self.num_envs)]
            for i in np.flatnonzero(dones):
                infos[i] = {"terminal_observation": final[i].copy(),
                            "TimeLimit.truncated": bool(trunc[i] and not term[i]),
                            "is_success": bool(succ[i]),
                            "outcome": self.batch.last_outcome(int(i))}
            return obs, rew.astype(np.float32), dones, infos

        def close(self) -> None:
            pass

        def seed(self, seed: int | None = None):
            self._seed = 0 if seed is None else int(seed)
            return [self._seed + i for i in range(self.num_envs)]

        # not much to say for these, there are no Python sub-envs to talk to
        def get_attr(self, attr_name, indices=None):
            return [getattr(self, attr_name, None)] * len(self._idx(indices))

        def set_attr(self, attr_name, value, indices=None) -> None:
            setattr(self, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            raise NotImplementedError("CatchSB3VecEnv has no Python sub-environments")

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._idx(indices))

        def _idx(self, indices):
            if indices is None:
                return list(range(self.num_envs))
            return [indices] if isinstance(indices, int) else list(indices)

    return CatchSB3VecEnv()
