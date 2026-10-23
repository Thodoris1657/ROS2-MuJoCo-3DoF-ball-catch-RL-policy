#!/usr/bin/env python
"""Train ONE catch policy for Earth, Mars and the Moon with Stable-Baselines3 SAC.

Gravity is drawn per episode and is part of the observation, so the policy has to
learn how timing scales with g. Episodes start at the release (the pick and carry
are scripted and identical for every policy).

    python scripts/train_catch_sb3.py --timesteps 300000 --out runs/catch_sac
    python scripts/benchmark_catch.py --entry "rl=runs/catch_sac/best_model.zip" --entry tuned

Reward (catch phase, 100 Hz): ``-d + (1 - tanh(d / 2 cm))`` towards the point where
the cup meets the ball at its apex (privileged: computed from the true state, used
only for the reward), a penalty for being lower than the falling ball, +20 for a
valid catch, -5 for any failure.

Two policy types:

    --residual 0      pure RL: the network outputs the action
    --residual 0.3    residual RL: action = clip(catch_tuned(obs) + 0.3 * network), see residual.py

``--train-set stress`` (default) trains on the wide distribution (restitution 0.55-0.90,
2x release puff, 3x sensor noise), which contains the nominal one; held-out evaluation
during training uses both.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

from mujoco_arm_rl.catch_eval import STRESS_KWARGS, run_episode  # noqa: E402
from mujoco_arm_rl.envs import CATCH_ENV_ID  # noqa: E402
from mujoco_arm_rl.envs.ballcatch import GRAVITIES, BallCatchEnv  # noqa: E402
from mujoco_arm_rl.policies import save_env_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="runs/catch_sac")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--net", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=30, help="per gravity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--residual", type=float, default=0.0, help="residual scale; 0 = pure RL")
    ap.add_argument("--base", default="catch_tuned", choices=["catch_tuned", "catch_model"])
    ap.add_argument("--train-set", default="stress", choices=["nominal", "stress"])
    ap.add_argument("--ent-coef", default="auto")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(gravity="random", **(STRESS_KWARGS if args.train_set == "stress" else {}))
    save_env_config(out, CATCH_ENV_ID, env_kwargs)
    residual = {"base": args.base, "scale": args.residual} if args.residual > 0 else None
    if residual:   # recorded beside the checkpoint: evaluation, export and ROS read it back
        cfg = json.loads((out / "env_config.json").read_text())
        cfg["residual"] = residual
        (out / "env_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    from mujoco_arm_rl.residual import ResidualCatchEnv, ResidualPolicy

    base_env = BallCatchEnv(**env_kwargs)
    train_env = ResidualCatchEnv(base_env, args.base, args.residual) if residual else base_env
    env = Monitor(train_env, info_keywords=("is_success",))
    eval_envs = {"nominal": BallCatchEnv(gravity="random"),
                 "stress": BallCatchEnv(gravity="random", **STRESS_KWARGS)}
    eval_seeds = {g: [10_000 + 97 * i + j for i in range(args.eval_episodes)]
                  for j, g in enumerate(GRAVITIES)}

    class EvalCatch(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.log = []
            self.t0 = time.time()

        def _on_step(self) -> bool:
            if self.n_calls % args.eval_every:
                return True
            if residual:
                pol = ResidualPolicy(self.model, args.base, args.residual)
            else:
                pol = lambda obs, e: self.model.predict(obs, deterministic=True)[0]  # noqa: E731
            rates = {}
            for set_name, eval_env in eval_envs.items():
                for g, seeds in eval_seeds.items():
                    res = [run_episode(eval_env, pol, s, {"gravity": g}) for s in seeds]
                    rates[f"{set_name}_{g}"] = float(np.mean([r.success for r in res]))
            mean = float(np.mean(list(rates.values())))
            row = dict(steps=self.num_timesteps, minutes=(time.time() - self.t0) / 60, mean=mean, **rates)
            self.log.append(row)
            (out / "eval_log.json").write_text(json.dumps(self.log, indent=2))
            print(f"[eval] {self.num_timesteps:>7d} steps  " + "  ".join(
                f"{g} {r:.0%}" for g, r in rates.items()) + f"  mean {mean:.0%}  "
                f"({row['minutes']:.1f} min)", flush=True)
            if mean >= self.best:
                self.best = mean
                self.model.save(out / "best_model")
            return True

    ent = args.ent_coef if args.ent_coef == "auto" or args.ent_coef.startswith("auto") else float(args.ent_coef)
    model = SAC("MlpPolicy", env, learning_rate=args.lr, gamma=args.gamma, batch_size=args.batch, ent_coef=ent,
                buffer_size=1_000_000, learning_starts=5_000, train_freq=1, gradient_steps=1,
                policy_kwargs=dict(net_arch=args.net), seed=args.seed, verbose=0, device="cpu")
    model.learn(total_timesteps=args.timesteps, callback=EvalCatch(), log_interval=50)
    model.save(out / "final_model")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
