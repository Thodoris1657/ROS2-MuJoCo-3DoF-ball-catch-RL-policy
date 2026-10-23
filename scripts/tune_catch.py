#!/usr/bin/env python
"""Optimise the model-based catch controller's parameters, separately for each gravity.

Cross-entropy method (CEM) over the parameters in ``catching.PARAM_BOUNDS``:
every generation samples ``--pop`` candidates from a Gaussian, scores each on the
same ``--episodes`` randomised trials (common random numbers, so candidates are
compared on identical balls), refits the Gaussian to the best ``--elite``, repeats.
Trials use fresh seeds every generation, and the frozen benchmark seeds are never
used, so the result is not fitted to the benchmark.

    python scripts/tune_catch.py                       # all three gravities -> configs/catch_tuned.json
    python scripts/tune_catch.py --gravity earth --generations 6 --episodes 48
    python scripts/tune_catch.py --backend cpp --pop 24 --generations 16 --episodes 256 --validate 4000

``--backend cpp`` evaluates candidates with the C++ port: in-process through the
``mujoco_arm_cpp_py`` module if it is installed, otherwise with the ``catch_eval_params`` tool
(built by colcon or plain CMake from ros2/src/mujoco_arm_cpp) -- the same physics and controller, verified
step-for-step against Python by its parity tests, ~50x faster, so the optimiser can afford
hundreds of trials per candidate and thousands for validation.

Objective: success rate + 0.01 x mean "softness" (1 - impact speed / 0.5 m/s), so among
equally successful settings the gentler one wins -- but only as a tie-breaker.

Robustness (default): every candidate is scored on half *nominal* trials and half
*stress* trials (wider restitution range, 2x release puff, 3x sensor noise; see
``catch_eval.STRESS_KWARGS``). Tuning on nominal trials alone was tried first and
over-fitted: it bought slightly softer catches with smaller safety margins and lost
15-23 points of success on the stress set. ``--no-robust`` reproduces that.
The untuned defaults always compete in the final validation, so tuning can never
return something worse than the defaults on the validation trials.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

import shutil  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from mujoco_arm_rl.catch_eval import STRESS_KWARGS, run_episode, score, summarise  # noqa: E402
from mujoco_arm_rl.catching import (  # noqa: E402
    PARAM_BOUNDS,
    TUNED_PATH,
    CatchController,
    CatchParams,
)
from mujoco_arm_rl.envs.ballcatch import GRAVITIES, BallCatchEnv  # noqa: E402

NAMES = list(PARAM_BOUNDS)
LO = np.array([PARAM_BOUNDS[k][0] for k in NAMES])
HI = np.array([PARAM_BOUNDS[k][1] for k in NAMES])


def to_params(x: np.ndarray) -> CatchParams:
    x = np.clip(x, LO, HI)
    return CatchParams(**{k: float(v) for k, v in zip(NAMES, x)})


ROBUST = True
BACKEND = "python"
REPO = Path(__file__).resolve().parents[1]


def cpp_binary() -> str:
    for cand in (os.environ.get("CATCH_EVAL_PARAMS", ""),
                 REPO / "ros2/install/mujoco_arm_cpp/lib/mujoco_arm_cpp/catch_eval_params",
                 REPO / "ros2/build/mujoco_arm_cpp/catch_eval_params",
                 REPO / "ros2/src/mujoco_arm_cpp/build/catch_eval_params",
                 shutil.which("catch_eval_params") or ""):
        if cand and Path(cand).exists():
            return str(cand)
    raise SystemExit("catch_eval_params not found: build ros2/src/mujoco_arm_cpp (or set CATCH_EVAL_PARAMS)")


def evaluate_cpp(args) -> tuple[float, dict]:
    x, gravity, seeds, robust = args
    n = len(seeds)
    n_nom, n_str = (n // 2, n - n // 2) if robust else (n, 0)
    try:
        # preferred: call the C++ code in-process through the pybind11 module (no new process
        # per candidate). It releases the GIL, so the ThreadPool workers run in parallel.
        import mujoco_arm_cpp_py as cpp

        from mujoco_arm_rl.envs.ballcatch import DEFAULT_MODEL_PATH, resolve_gravity

        r = cpp.evaluate_params(str(DEFAULT_MODEL_PATH), to_params(x).to_dict(), resolve_gravity(gravity),
                                n_nominal=n_nom, n_stress=n_str, seed0=int(seeds[0]), threads=1)
        summ = {"episodes": r["trials"], "success": r["success"], "outcomes": r["outcomes"]}
        return float(r["success"] + 0.01 * r["softness"]), summ
    except ImportError:
        pass  # module not built: fall back to the command line tool below
    out = subprocess.run([cpp_binary(), str(REPO), gravity, json.dumps(to_params(x).to_dict()),
                          str(int(seeds[0])), str(n_nom), str(n_str)],
                         check=True, capture_output=True, text=True).stdout
    r = json.loads(out)
    summ = {"episodes": r["trials"], "success": r["success"], "outcomes": r["outcomes"]}
    return float(r["success"] + 0.01 * r["softness"]), summ


def evaluate(args) -> tuple[float, dict]:
    x, gravity, seeds, robust = args          # robust passed explicitly: works with spawn too
    ctrl = CatchController(to_params(x))
    if robust:
        half = len(seeds) // 2
        env_n, env_s = BallCatchEnv(gravity=gravity), BallCatchEnv(gravity=gravity, **STRESS_KWARGS)
        res = ([run_episode(env_n, ctrl, int(s)) for s in seeds[:half]]
               + [run_episode(env_s, ctrl, int(s)) for s in seeds[half:]])
    else:
        env = BallCatchEnv(gravity=gravity)
        res = [run_episode(env, ctrl, int(s)) for s in seeds]
    return score(res, softness_weight=0.01), summarise(res)


def tune(gravity: str, generations: int, pop: int, elite: int, episodes: int,
         workers: int, seed: int, validate: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    default = CatchParams()
    mu = np.array([getattr(default, k) for k in NAMES], dtype=float)
    sigma = 0.25 * (HI - LO)
    best = (-np.inf, mu.copy(), {})
    history = []
    Pool = ThreadPoolExecutor if BACKEND == "cpp" else ProcessPoolExecutor
    fn = evaluate_cpp if BACKEND == "cpp" else evaluate
    with Pool(max_workers=workers) as pool:
        for gen in range(generations):
            seeds = rng.integers(10 ** 6, 10 ** 9, size=episodes)
            cands = [mu.copy()] + [np.clip(mu + sigma * rng.normal(size=len(NAMES)), LO, HI)
                                   for _ in range(pop - 1)]
            t0 = time.time()
            scores = list(pool.map(fn, [(c, gravity, seeds, ROBUST) for c in cands]))
            vals = np.array([s for s, _ in scores])
            order = np.argsort(-vals)
            elites = np.array([cands[i] for i in order[:elite]])
            mu = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), 0.02 * (HI - LO))
            if vals[order[0]] > best[0]:
                best = (float(vals[order[0]]), cands[order[0]].copy(), scores[order[0]][1])
            history.append(dict(gen=gen, best=float(vals[order[0]]), mean=float(vals.mean()),
                                incumbent=float(vals[0])))
            print(f"[{gravity}] gen {gen:2d}  best {vals[order[0]]:.3f}  "
                  f"(success {scores[order[0]][1]['success']:.0%})  mean {vals.mean():.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    # final check of the mean vs the best-seen point vs the defaults on a larger fresh set
    seeds = rng.integers(10 ** 6, 10 ** 9, size=validate or episodes * 3)
    final = {}
    cands = {"mean": mu, "best": best[1], "default": np.array([getattr(default, k) for k in NAMES])}
    with Pool(max_workers=workers) as pool:
        outs = list(pool.map(fn, [(x, gravity, seeds, ROBUST) for x in cands.values()]))
    for (tag, x), o in zip(cands.items(), outs):
        final[tag] = (x, *o)
    pick = max(("default", "mean", "best"), key=lambda t: final[t][1])
    x = final[pick][0]
    print(f"[{gravity}] validation ({len(seeds)} trials): "
          + ", ".join(f"{t} {final[t][2]['success']:.1%}" for t in final) + f" -> keep {pick}")
    return dict(params=to_params(x).to_dict(), validation={t: final[t][2] for t in final},
                history=history)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravity", nargs="*", default=list(GRAVITIES), choices=list(GRAVITIES))
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(TUNED_PATH))
    ap.add_argument("--no-robust", action="store_true", help="tune on nominal trials only")
    ap.add_argument("--backend", default="python", choices=["python", "cpp"])
    ap.add_argument("--validate", type=int, default=0, help="validation trials (default 3x --episodes)")
    args = ap.parse_args()
    global ROBUST, BACKEND
    ROBUST = not args.no_robust
    BACKEND = args.backend

    out = Path(args.out)
    blob = json.loads(out.read_text()) if out.exists() else {"params": {}, "details": {}}
    for g in args.gravity:
        res = tune(g, args.generations, args.pop, args.elite, args.episodes, args.workers, args.seed,
                   args.validate)
        blob["params"][g] = res["params"]
        blob["robust"] = ROBUST
        blob["backend"] = BACKEND
        blob.pop("note", None)
        blob.setdefault("details", {})[g] = {"validation": res["validation"], "history": res["history"]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2))
        print(f"[{g}] params -> {out}: {res['params']}")


if __name__ == "__main__":
    main()
