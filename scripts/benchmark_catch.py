#!/usr/bin/env python
"""Score catch policies on the frozen ball-catch benchmark, per gravity.

The benchmark pins every random quantity of each trial (gravity, ball restitution,
drop point, release puff, sensor-noise seed), 100 trials per gravity, so numbers are
comparable across policies and across runs. Two sets ship with the repo:

    benchmarks/catch_set_v1.json         nominal  -- the distribution policies are tuned/trained on
    benchmarks/catch_set_v1_stress.json  stress   -- wider restitution range (0.55-0.90), 2x
                                                     release puff, 3x sensor noise

    python scripts/benchmark_catch.py                                  # hold / chase / model / tuned
    python scripts/benchmark_catch.py --entry tuned --entry "rl=runs/catch_sac/best_model.zip"
    python scripts/benchmark_catch.py --set benchmarks/catch_set_v1_stress.json
    python scripts/benchmark_catch.py --make                           # regenerate both sets

Entries: ``hold``, ``chase``, ``model``, ``tuned[:file]``, a ``.zip`` path; prefix
``name=`` to label one.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

from mujoco_arm_rl.catch_eval import (  # noqa: E402
    CATCH_SET_PATH,
    load_catch_set,
    make_catch_policy,
    make_catch_set,
    run_catch_set,
    summarise,
)
from mujoco_arm_rl.envs.ballcatch import GRAVITIES  # noqa: E402

STRESS_PATH = CATCH_SET_PATH.with_name("catch_set_v1_stress.json")


def _one(job):
    label, spec, set_path, gravity = job
    res = run_catch_set(make_catch_policy(spec), load_catch_set(set_path), [gravity])
    return label, gravity, summarise(res[gravity])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", default=None)
    ap.add_argument("--set", default=str(CATCH_SET_PATH))
    ap.add_argument("--make", action="store_true", help="regenerate the nominal and stress sets")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--json", default="", help="also write the results here")
    args = ap.parse_args()

    if args.make:
        for path, stress in ((CATCH_SET_PATH, False), (STRESS_PATH, True)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(make_catch_set(stress=stress), indent=1))
            print(f"wrote {path}")
        if not args.entry:
            return

    entries = args.entry or ["hold", "chase", "model", "tuned"]
    jobs = []
    for e in entries:
        label, spec = e.split("=", 1) if "=" in e and not e.startswith("tuned:") else (e, e)
        for g in GRAVITIES:
            jobs.append((label, spec, args.set, g))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_one, jobs))

    table: dict[str, dict] = {}
    for label, g, summ in results:
        table.setdefault(label, {})[g] = summ
    name = Path(args.set).stem
    print(f"\nBenchmark {name} ({sum(1 for _ in load_catch_set(args.set)['cases'])} trials)\n")
    print(f"| Policy | {' | '.join(g.capitalize() for g in GRAVITIES)} | Failure modes |")
    print("| --- |" + " --- |" * (len(GRAVITIES) + 1))
    for label, per_g in table.items():
        fails: dict[str, int] = {}
        for s in per_g.values():
            for k, v in s["outcomes"].items():
                if k != "caught":
                    fails[k] = fails.get(k, 0) + v
        cells = []
        for g in GRAVITIES:
            s = per_g[g]
            extra = f" ({s['median_impact_speed']:.2f} m/s)" if "median_impact_speed" in s else ""
            cells.append(f"{s['success']:.0%}{extra}")
        fail_txt = ", ".join(f"{k}: {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])) or "-"
        print(f"| {label} | {' | '.join(cells)} | {fail_txt} |")
    print("\n(success rate; in brackets the median cup-ball impact speed of the catches)")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
