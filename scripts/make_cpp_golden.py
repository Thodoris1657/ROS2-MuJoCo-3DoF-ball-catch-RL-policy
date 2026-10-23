#!/usr/bin/env python
"""Record Python reference roll-outs that the C++ port must reproduce (its parity tests).

Writes ros2/src/mujoco_arm_cpp/test/golden/:
    catch.json   per-step observations, controller actions and outcomes of BallCatchEnv +
                 the tuned CatchController (sensor noise off, every random quantity pinned),
                 three gravities from the release plus one full pick-and-carry trial
    reach.json   the observation-only IK oracle on 10 frozen reach cases (Reach3Env v1)
    mlp.json     an exported SAC network + 20 input/output pairs from SB3's own predict()

    python scripts/make_cpp_golden.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

from mujoco_arm_rl.benchmark import load_eval_set  # noqa: E402
from mujoco_arm_rl.catching import CatchController, load_tuned  # noqa: E402
from mujoco_arm_rl.envs import BallCatchEnv, Reach3Env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ros2" / "src" / "mujoco_arm_cpp" / "test" / "golden"

CATCH_CASES = [
    dict(gravity="earth", restitution=0.70, drop=[0.30, 0.02, 0.47], push=[0.012, -0.008], start="release"),
    dict(gravity="mars", restitution=0.80, drop=[0.27, -0.06, 0.45], push=[-0.015, 0.010], start="release"),
    dict(gravity="moon", restitution=0.75, drop=[0.32, 0.05, 0.49], push=[0.020, 0.015], start="release"),
    dict(gravity="moon", restitution=0.66, drop=[0.29, 0.00, 0.46], push=[0.005, -0.012], start="release"),
    dict(gravity="earth", restitution=0.78, drop=[0.30, 0.00, 0.47], push=[0.0, 0.0], start="pick"),
]


def catch_golden() -> dict:
    env = BallCatchEnv(gravity="earth", obs_pos_noise=0.0, obs_vel_noise=0.0)
    ctrl = CatchController(load_tuned())
    cases = []
    for i, c in enumerate(CATCH_CASES):
        obs, _ = env.reset(seed=i, options=c)
        ctrl.reset()
        rec = dict(options=c, obs=[obs.tolist()], actions=[])
        while True:
            a = ctrl(obs, env)
            rec["actions"].append([float(v) for v in a])
            obs, _, term, trunc, info = env.step(a)
            rec["obs"].append(obs.tolist())
            if term or trunc:
                break
        rec.update(outcome=info["outcome"], success=bool(info["is_success"]),
                   catch_time_error=info.get("catch_time_error"), steps=len(rec["actions"]))
        cases.append(rec)
        print(f"catch {c['gravity']:5s} {c['start']:7s}: {info['outcome']} ({rec['steps']} steps)")
    return {"cases": cases}


def oracle_act(obs: np.ndarray, env: Reach3Env, cache: dict) -> np.ndarray:
    """The observation-only oracle of the ROS bridge (q and q_target recovered from obs)."""
    o = np.asarray(obs, dtype=np.float64)
    q = np.arctan2(o[3:6], o[0:3])
    q_target = q + o[9:12]
    target = o[18:21]
    key = tuple(np.round(target * 1e6).astype(np.int64))
    if cache.get("key") != key:
        cache["goal"] = env.solve_ik(target, near=q)
        cache["key"] = key
    if cache["goal"] is None:
        return np.zeros(3)
    return np.clip((cache["goal"] - q_target) / env.max_step, -1.0, 1.0)


def reach_golden() -> dict:
    env = Reach3Env()
    cases = []
    for c in load_eval_set()[:10]:
        obs, _ = env.reset(options={"qpos": c["qpos"], "target": c["target"]})
        cache: dict = {}
        rec = dict(qpos=c["qpos"], target=c["target"], obs=[obs.tolist()], distance=[])
        for _ in range(env.max_episode_steps):
            obs, _, _, _, info = env.step(oracle_act(obs, env, cache))
            rec["obs"].append(obs.tolist())
            rec["distance"].append(info["distance"])
        cases.append(rec)
    print(f"reach: 10 oracle episodes, final distance median "
          f"{1000 * np.median([r['distance'][-1] for r in cases]):.1f} mm")
    return {"cases": cases}


def mlp_golden() -> dict:
    import subprocess
    import sys

    from stable_baselines3 import SAC

    env = BallCatchEnv()
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]), seed=3, device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "net.zip"
        model.save(zp)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_policy.py"), str(zp),
                        "--out", str(OUT / "mlp_net.json")], check=True)
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(20):
        o = rng.normal(0, 0.7, 34).astype(np.float32)
        pairs.append(dict(obs=o.tolist(), action=model.predict(o, deterministic=True)[0].tolist()))
    return {"network": "mlp_net.json", "pairs": pairs}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catch.json").write_text(json.dumps(catch_golden()))
    (OUT / "reach.json").write_text(json.dumps(reach_golden()))
    (OUT / "mlp.json").write_text(json.dumps(mlp_golden()))
    print(f"golden files -> {OUT}")


if __name__ == "__main__":
    main()
