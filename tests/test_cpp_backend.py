"""Tests for the C++ backend (mujoco_arm_cpp_py). Skipped if the module is not built.

Build it with:  pip install --no-build-isolation ./ros2/src/mujoco_arm_cpp

The important ones are the parity tests: the C++ simulator has to give the same observations
and the same reward as BallCatchEnv for the same actions, otherwise training on it would be
training on a different task.
"""

from __future__ import annotations

import numpy as np
import pytest

cpp = pytest.importorskip("mujoco_arm_cpp_py")

from mujoco_arm_rl.catching import CatchController, load_tuned  # noqa: E402
from mujoco_arm_rl.envs.ballcatch import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    BallCatchEnv,
    resolve_gravity,
)
from mujoco_arm_rl.envs.catch_vec import CatchVecEnv, make_batch  # noqa: E402

MODEL = str(DEFAULT_MODEL_PATH)
NO_NOISE = {"obs_pos_noise": 0.0, "obs_vel_noise": 0.0}
CASES = [
    dict(gravity="earth", restitution=0.70, drop=[0.30, 0.02, 0.47], push=[0.012, -0.008]),
    dict(gravity="moon", restitution=0.80, drop=[0.28, -0.05, 0.45], push=[-0.01, 0.02]),
]


def tuned_dict():
    return {g: p.to_dict() for g, p in load_tuned().items()}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["gravity"])
def test_obs_and_reward_match_python(case):
    """Same actions in, same observations and rewards out (noise off, everything pinned)."""
    py_env = BallCatchEnv(gravity=case["gravity"], **NO_NOISE)
    ctrl = CatchController(load_tuned())
    sim = cpp.CatchSim(MODEL, env_kwargs=NO_NOISE)
    obs_py, _ = py_env.reset(seed=0, options=case)
    obs_c = sim.reset(seed=0, gravity=resolve_gravity(case["gravity"]), restitution=case["restitution"],
                      drop=case["drop"], push=case["push"])
    np.testing.assert_allclose(obs_c, obs_py, atol=1e-6)
    rng = np.random.default_rng(0)
    ctrl.reset()
    for k in range(400):
        # controller action plus a bit of noise, so the reward terms for "bad" actions are used too
        a = np.clip(ctrl(obs_py, py_env) + 0.2 * rng.normal(size=3) * (k % 3 == 0), -1, 1)
        obs_py, r_py, term_py, trunc_py, info = py_env.step(a)
        obs_c, r_c, term_c, trunc_c = sim.step(a)
        np.testing.assert_allclose(obs_c, obs_py, atol=1e-5, err_msg=f"step {k}")
        assert r_c == pytest.approx(r_py, abs=1e-6), f"reward at step {k}"
        assert (term_c, trunc_c) == (term_py, trunc_py)
        if term_py or trunc_py:
            break
    assert sim.outcome == info["outcome"]


def test_cpp_controller_matches_python_controller():
    py_env = BallCatchEnv(gravity="mars", **NO_NOISE)
    obs, _ = py_env.reset(seed=1, options=dict(gravity="mars", restitution=0.75, drop=[0.3, 0.0, 0.47], push=[0, 0]))
    sim = cpp.CatchSim(MODEL, env_kwargs=NO_NOISE)   # only used for IK by the C++ controller
    c_ctrl = cpp.CatchController(tuned_dict(), per_gravity=True)
    p_ctrl = CatchController(load_tuned())
    for _ in range(60):
        a_py = p_ctrl(obs, py_env)
        a_c = c_ctrl.act(obs, sim)
        np.testing.assert_allclose(a_c, a_py, atol=1e-6)
        obs, *_ = py_env.step(a_py)


def test_batch_is_deterministic_and_independent_of_threads():
    def run(threads):
        b = make_batch(n_envs=6, env_kwargs={"gravity": "random"}, seed=42, threads=threads)
        b.reset(42)
        rng = np.random.default_rng(3)
        total = 0.0
        for _ in range(150):
            obs, rew, *_ = b.step(rng.uniform(-1, 1, (6, 3)))
            total += float(rew.sum())
        return obs, total
    o1, t1 = run(1)
    o2, t2 = run(2)
    np.testing.assert_array_equal(o1, o2)
    assert t1 == t2


def test_batch_autoreset_and_final_obs():
    env = CatchVecEnv(n_envs=4, env_kwargs={"gravity": "earth"}, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 34)
    finished = 0
    for _ in range(300):
        obs, rew, term, trunc, infos = env.step(np.zeros((4, 3)))   # holding still always misses
        if "final_obs" in infos:
            done = infos["_final_obs"]
            finished += int(done.sum())
            # the returned obs already belongs to the new episode (ball just released, in flight)
            assert np.all(obs[done, 33] == 1.0)
            assert np.all(infos["final_obs"][done, 33] == 0.0)
            assert not infos["is_success"][done].any()
    assert finished >= 4
    assert env.batch.episodes_finished == finished


def test_residual_batch_with_zero_correction_is_the_controller():
    """residual mode + zero network output must catch exactly like the controller alone."""
    seed = 11
    b = make_batch(n_envs=3, env_kwargs={"gravity": "mars"}, seed=seed, residual_scale=0.2)
    obs = b.reset(seed)
    assert obs.shape == (3, 37)
    outcomes = {}
    for _ in range(400):
        obs, rew, term, trunc, succ, final = b.step(np.zeros((3, 3)))
        for i in np.flatnonzero(term | trunc):
            outcomes.setdefault(int(i), b.last_outcome(int(i)))
        if len(outcomes) == 3:
            break
    # the same three trials, controller alone (env i is seeded with seed + i)
    for i in range(3):
        sim = cpp.CatchSim(MODEL, gravities=[resolve_gravity("mars")])
        ctrl = cpp.CatchController(tuned_dict(), per_gravity=True)
        o = sim.reset(seed=seed + i)
        for _ in range(400):
            o, _, term, trunc = sim.step(ctrl.act(o, sim))
            if term or trunc:
                break
        assert sim.outcome == outcomes[i]


def test_evaluate_params_is_reproducible():
    p = tuned_dict()["earth"]
    a = cpp.evaluate_params(MODEL, p, 9.81, n_nominal=20, n_stress=20, seed0=5, threads=1)
    b = cpp.evaluate_params(MODEL, p, 9.81, n_nominal=20, n_stress=20, seed0=5, threads=2)
    assert a == b
    assert a["trials"] == 40 and a["success"] > 0.9


def test_sb3_vec_env_interface():
    pytest.importorskip("stable_baselines3")
    from mujoco_arm_rl.envs.catch_vec import make_sb3_vec_env

    venv = make_sb3_vec_env(n_envs=3, env_kwargs={"gravity": "moon"}, seed=0)
    obs = venv.reset()
    assert obs.shape == (3, 34)
    seen_terminal = False
    for _ in range(300):
        venv.step_async(np.zeros((3, 3), dtype=np.float32))
        obs, rew, dones, infos = venv.step_wait()
        assert rew.dtype == np.float32 and len(infos) == 3
        for i in np.flatnonzero(dones):
            seen_terminal = True
            assert infos[i]["terminal_observation"].shape == (34,)
            assert "is_success" in infos[i] and "outcome" in infos[i]
    assert seen_terminal
