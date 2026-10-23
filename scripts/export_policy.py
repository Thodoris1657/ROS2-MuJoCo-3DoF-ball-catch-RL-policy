#!/usr/bin/env python
"""Export a trained policy network to JSON for the C++ ROS 2 policy node (no Python at run time).

Supports Stable-Baselines3 SAC / TQC / TD3 / PPO checkpoints (.zip) and the repo's TorchRL PPO
(.pt). The deterministic action is exported: SAC/TQC tanh(mu), TD3 the tanh-ed actor, PPO the
clipped mean, TorchRL tanh(loc) with its ObservationNorm replayed. A residual catch policy
(scripts/train_catch_sb3.py --residual) also carries its base controller and scale.

    python scripts/export_policy.py runs/catch_sac/best_model.zip            # -> best_model.json
    python scripts/export_policy.py runs/sb3_v1/best_model.zip --out reach.json --check

``--check`` replays 200 random observations through both the original network and the exported
JSON (with a numpy re-implementation of the C++ forward pass) and fails on any difference > 1e-5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FORMAT = "mujoco_arm_mlp_v1"


def _linear_layers(seq) -> list[dict]:
    """torch Sequential of Linear / ReLU / Tanh -> [{weight, bias, activation}]."""
    import torch.nn as nn

    layers: list[dict] = []
    for mod in seq:
        if isinstance(mod, nn.Linear):
            layers.append({"weight": mod.weight.detach().cpu().double().numpy().tolist(),
                           "bias": mod.bias.detach().cpu().double().numpy().tolist(),
                           "activation": "none"})
        elif isinstance(mod, nn.ReLU):
            layers[-1]["activation"] = "relu"
        elif isinstance(mod, nn.Tanh):
            layers[-1]["activation"] = "tanh"
        elif isinstance(mod, (nn.Identity, nn.Flatten)):
            continue
        else:
            raise ValueError(f"unsupported layer {type(mod).__name__}")
    return layers


def export_sb3(path: Path) -> dict:
    from stable_baselines3 import PPO, SAC, TD3

    classes = [SAC, TD3, PPO]
    try:
        from sb3_contrib import TQC
        classes.insert(0, TQC)
    except ImportError:
        pass
    model, err = None, None
    for cls in classes:
        try:
            model = cls.load(str(path), device="cpu")
            break
        except Exception as e:  # noqa: BLE001 - try the next class
            err = e
    if model is None:
        raise RuntimeError(f"could not load {path}: {err}")
    algo = type(model).__name__
    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])
    if not (np.allclose(model.action_space.low, -1) and np.allclose(model.action_space.high, 1)):
        raise ValueError("only [-1, 1] action spaces are supported")
    pol = model.policy
    if algo in ("SAC", "TQC"):
        layers = _linear_layers(pol.actor.latent_pi) + _linear_layers([pol.actor.mu])
        output = "tanh"
    elif algo == "TD3":
        layers = _linear_layers(pol.actor.mu)
        output = "clip"
    else:  # PPO
        layers = _linear_layers(pol.mlp_extractor.policy_net) + _linear_layers([pol.action_net])
        output = "clip"
    return dict(algo=algo, obs_dim=obs_dim, act_dim=act_dim, layers=layers, output=output,
                obs_norm=None, _model=model)


def export_torchrl(path: Path) -> dict:
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = blob["policy_state_dict"]
    act = "relu" if blob.get("activation", "tanh") == "relu" else "tanh"
    keys = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight")}, key=lambda s: int(s))
    layers = []
    for i, k in enumerate(keys):
        layers.append({"weight": sd[k + ".weight"].double().numpy().tolist(),
                       "bias": sd[k + ".bias"].double().numpy().tolist(),
                       "activation": act if i < len(keys) - 1 else "none"})
    return dict(algo="TorchRL-PPO", obs_dim=int(blob["n_obs"]), act_dim=int(blob["n_act"]),
                layers=layers, output="tanh",
                obs_norm={"loc": blob["obs_loc"].double().numpy().tolist(),
                          "scale": blob["obs_scale"].double().numpy().tolist(),
                          "standard_normal": bool(blob.get("obs_standard_normal", False))},
                _blob=blob)


def forward(spec: dict, obs: np.ndarray) -> np.ndarray:
    """numpy twin of the C++ MlpNetwork::forward."""
    x = np.asarray(obs, dtype=np.float64)
    if spec.get("obs_norm"):
        n = spec["obs_norm"]
        loc, scale = np.asarray(n["loc"]), np.asarray(n["scale"])
        x = (x - loc) / scale if n["standard_normal"] else x * scale + loc
    for L in spec["layers"]:
        x = np.asarray(L["weight"]) @ x + np.asarray(L["bias"])
        if L["activation"] == "relu":
            x = np.maximum(x, 0.0)
        elif L["activation"] == "tanh":
            x = np.tanh(x)
    a = x[: spec["act_dim"]]
    return np.tanh(a) if spec["output"] == "tanh" else np.clip(a, -1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = Path(args.checkpoint)
    spec = export_torchrl(path) if path.suffix == ".pt" else export_sb3(path)

    for folder in (path.parent, path.parent.parent):
        if (folder / "env_config.json").exists():
            cfg = json.loads((folder / "env_config.json").read_text())
            spec["env_id"], spec["env_kwargs"] = cfg.get("env_id"), cfg.get("env_kwargs", {})
            spec["residual"] = cfg.get("residual")
            break

    if args.check:
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(200):
            o = rng.normal(0, 0.5, spec["obs_dim"]).astype(np.float32)
            if "_model" in spec:
                ref = spec["_model"].predict(o, deterministic=True)[0]
            else:
                from mujoco_arm_rl.policies import load_torchrl
                ref = load_torchrl(path)[0](o, None)
            worst = max(worst, float(np.max(np.abs(forward(spec, o) - ref))))
        print(f"check: max |exported - original| = {worst:.2e} over 200 observations")
        if worst > 1e-5:
            raise SystemExit("export check FAILED")

    spec.pop("_model", None)
    spec.pop("_blob", None)
    spec.update(format=FORMAT, source=str(path))
    out = Path(args.out) if args.out else path.with_suffix(".json")
    out.write_text(json.dumps(spec))
    print(f"{spec['algo']} {spec['obs_dim']}->{spec['act_dim']}, {len(spec['layers'])} layers"
          f"{', residual on ' + spec['residual']['base'] if spec.get('residual') else ''} -> {out}")


if __name__ == "__main__":
    main()
