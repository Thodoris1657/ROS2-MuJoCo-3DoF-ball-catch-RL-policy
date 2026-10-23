"""Roll-outs and scoring for the ball-catch task, shared by the tuner, benchmark and tests.

A *catch policy* is anything with ``act(obs, env) -> action`` and an optional
``reset()``; ``make_catch_policy(spec)`` builds one from a short string:

    hold                the arm never moves (lower bound)
    chase               go straight for the predicted apex, ignoring the falling ball
    model               model-based controller, default (untuned) parameters
    tuned[:file.json]   model-based controller with per-gravity optimised parameters
    <path>.zip          a Stable-Baselines3 policy trained by scripts/train_catch_sb3.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mujoco_arm_rl.catching import TUNED_PATH, CatchController, CatchParams, load_tuned
from mujoco_arm_rl.envs.ballcatch import GRAVITIES, BallCatchEnv

CATCH_SET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "catch_set_v1.json"

# The "stress" distribution: harder than training/tuning in every randomised quantity.
STRESS_KWARGS: dict[str, Any] = dict(
    restitution=(0.55, 0.90),
    release_push=0.04, release_push_max=0.08,
    obs_pos_noise=0.003, obs_vel_noise=0.05,
)


class _Policy:
    def __init__(self, fn: Callable, reset: Callable | None = None, name: str = "") -> None:
        self.fn, self._reset, self.name = fn, reset, name

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def __call__(self, obs, env):
        return self.fn(obs, env)


def make_catch_policy(spec: str) -> _Policy:
    if spec == "hold":
        return _Policy(lambda obs, env: np.zeros(3), name=spec)
    if spec == "chase":
        ctrl = CatchController(CatchParams(lookahead=0.0, clearance=-1.0))
        return _Policy(ctrl, ctrl.reset, spec)
    if spec == "model":
        ctrl = CatchController(CatchParams())
        return _Policy(ctrl, ctrl.reset, spec)
    if spec.startswith("tuned"):
        path = spec.split(":", 1)[1] if ":" in spec else TUNED_PATH
        tuned = load_tuned(path)
        if not tuned:
            raise FileNotFoundError(f"no tuned parameters at {path} (run scripts/tune_catch.py)")
        ctrl = CatchController(tuned)
        return _Policy(ctrl, ctrl.reset, spec)
    if spec.endswith(".zip"):
        from stable_baselines3 import PPO, SAC, TD3  # noqa: F401

        model = None
        for cls in (SAC, TD3, PPO):
            try:
                model = cls.load(spec, device="cpu")
                break
            except Exception:  # noqa: BLE001 - try the next algorithm
                continue
        if model is None:
            raise ValueError(f"could not load {spec} with SAC/TD3/PPO")
        for folder in (Path(spec).parent, Path(spec).parent.parent):
            cfg = folder / "env_config.json"
            if cfg.exists():
                res = json.loads(cfg.read_text()).get("residual")
                if res:
                    from mujoco_arm_rl.residual import ResidualPolicy

                    rp = ResidualPolicy(model, res["base"], res["scale"])
                    return _Policy(rp, rp.reset, spec)
                break
        return _Policy(lambda obs, env: model.predict(obs, deterministic=True)[0], name=spec)
    raise ValueError(f"unknown catch policy {spec!r}")


@dataclass
class EpisodeResult:
    gravity: str
    success: bool
    outcome: str
    restitution: float
    catch_time_error: float | None = None     # s, latch time - true apex time (negative = before)
    catch_height_error: float | None = None   # m, ball height at latch - true apex height
    impact_speed: float | None = None         # m/s, relative ball-cup speed at latch
    steps: int = 0


def run_episode(env: BallCatchEnv, policy, seed: int, options: dict[str, Any] | None = None,
                frame_cb: Callable | None = None) -> EpisodeResult:
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        steps += 1
        if frame_cb is not None:
            frame_cb(env, info)
        if term or trunc:
            break
    return EpisodeResult(
        gravity=info["gravity_name"], success=bool(info["is_success"]), outcome=str(info["outcome"]),
        restitution=float(info["restitution"]),
        catch_time_error=info.get("catch_time_error"), catch_height_error=info.get("catch_height_error"),
        impact_speed=info.get("catch_rel_speed"), steps=steps)


def outcome_key(outcome: str) -> str:
    return outcome.split("(")[0].split(":")[0].strip()


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    caught = [r for r in results if r.success]
    out: dict[str, Any] = {
        "episodes": n,
        "success": len(caught) / n if n else 0.0,
        "outcomes": dict(Counter(outcome_key(r.outcome) for r in results)),
    }
    if caught:
        out["median_impact_speed"] = float(np.median([r.impact_speed for r in caught]))
        out["median_catch_time_error_ms"] = float(np.median([r.catch_time_error for r in caught]) * 1e3)
        out["p95_abs_height_error_mm"] = float(np.percentile(
            [abs(r.catch_height_error) for r in caught], 95) * 1e3)
    return out


def score(results: list[EpisodeResult], capture_speed: float = 0.5,
          softness_weight: float = 0.01) -> float:
    """Optimiser objective: success rate, tie-broken towards softer catches."""
    s = np.mean([r.success for r in results])
    soft = np.mean([1.0 - (r.impact_speed or capture_speed) / capture_speed if r.success else 0.0
                    for r in results])
    return float(s + softness_weight * soft)


# --------------------------------------------------------------- frozen set


def make_catch_set(n_per_gravity: int = 100, seed: int = 20261023, stress: bool = False) -> dict:
    """Pinned (restitution, drop point, release puff, noise seed) cases for every gravity."""
    env = BallCatchEnv(**(STRESS_KWARGS if stress else {}))
    rng = np.random.default_rng(seed)
    cases = []
    for gname in GRAVITIES:
        for _ in range(n_per_gravity):
            env.reset(seed=int(rng.integers(2 ** 31)), options={"gravity": gname})
            cases.append(dict(
                gravity=gname, seed=int(rng.integers(2 ** 31)),
                restitution=round(env.restitution, 4),
                drop=[round(float(v), 4) for v in env.drop_tip],
                push=[round(float(v), 4) for v in env.release_state["vel"][:2]],
            ))
    return {"seed": seed, "stress": stress, "env_kwargs": STRESS_KWARGS if stress else {},
            "cases": cases}


def load_catch_set(path: str | Path = CATCH_SET_PATH) -> dict:
    return json.loads(Path(path).read_text())


def run_catch_set(policy, catch_set: dict, gravities: list[str] | None = None,
                  env: BallCatchEnv | None = None) -> dict[str, list[EpisodeResult]]:
    env = env or BallCatchEnv(**catch_set.get("env_kwargs", {}))
    out: dict[str, list[EpisodeResult]] = {}
    for case in catch_set["cases"]:
        if gravities and case["gravity"] not in gravities:
            continue
        opts = dict(gravity=case["gravity"], restitution=case["restitution"],
                    drop=case["drop"], push=case["push"])
        out.setdefault(case["gravity"], []).append(run_episode(env, policy, case["seed"], opts))
    return out
