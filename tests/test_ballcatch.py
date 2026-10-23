"""Contract tests for the ball-catch task. Run with: pytest -q (MUJOCO_GL=disable on headless)."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from mujoco_arm_rl.catch_eval import CATCH_SET_PATH, make_catch_policy, run_episode
from mujoco_arm_rl.catching import BallisticFit, CatchController, CatchParams
from mujoco_arm_rl.envs import CATCH_ENV_ID, DEFAULT_MODEL_PATH
from mujoco_arm_rl.envs.ballcatch import (
    BALL_R,
    BALL_REST,
    CONTACT_DIST,
    GRAVITIES,
    BallCatchEnv,
    restitution_solref,
)
from mujoco_arm_rl.envs.ballcatch import DEFAULT_MODEL_PATH as CATCH_MODEL_PATH


@pytest.fixture(scope="module")
def env():
    e = BallCatchEnv()
    yield e
    e.close()


def test_same_arm_as_reach_model(env):
    """The catch scene must not change the robot: tip FK identical to arm3.xml."""
    m0 = mujoco.MjModel.from_xml_path(DEFAULT_MODEL_PATH)
    d0 = mujoco.MjData(m0)
    tip0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, "tip")
    rng = np.random.default_rng(0)
    d = env._scratch
    for _ in range(50):
        q = rng.uniform(env.jnt_lo, env.jnt_hi)
        d0.qpos[:] = q
        mujoco.mj_kinematics(m0, d0)
        d.qpos[:3] = q
        mujoco.mj_kinematics(env.model, d)
        assert np.allclose(d0.site_xpos[tip0], d.site_xpos[env._tip_id], atol=1e-12)
    assert np.allclose(env.torque_limit, m0.actuator_gear[:, 0])


def test_registered():
    import gymnasium as gym

    e = gym.make(CATCH_ENV_ID)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (34,)
    e.close()


@pytest.mark.parametrize("gname", list(GRAVITIES))
def test_restitution_is_gravity_independent(gname):
    """A free drop onto the floor rebounds to ~e^2 h under every gravity."""
    e_set = 0.75
    m = mujoco.MjModel.from_xml_path(CATCH_MODEL_PATH)
    m.opt.gravity[:] = (0, 0, -GRAVITIES[gname])
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
    m.pair_solref[pid] = restitution_solref(e_set)
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    a = m.jnt_qposadr[jid]
    v = m.jnt_dofadr[jid]
    d.qpos[:3] = [0.0, -1.0, 0.0]          # arm pointing away
    d.qpos[a:a + 3] = [0.5, 0.5, BALL_R + 0.4]
    mujoco.mj_forward(m, d)
    state, top = "fall", None
    for _ in range(20000):
        mujoco.mj_step(m, d)
        vz = d.qvel[v + 2]
        if state == "fall" and vz > 0:
            state = "rise"
        elif state == "rise" and vz <= 0:
            top = d.qpos[a + 2] - BALL_R
            break
    e_eff = np.sqrt(top / 0.4)
    assert abs(e_eff - e_set) < 0.05, e_eff


def test_reset_options_pin_everything(env):
    opts = dict(gravity="moon", restitution=0.7, drop=[0.3, 0.0, 0.47], push=[0.01, -0.02])
    o1, i1 = env.reset(seed=3, options=opts)
    assert np.isclose(env.model.opt.gravity[2], -GRAVITIES["moon"])
    assert np.isclose(env.restitution, 0.7)
    assert np.allclose(env.release_state["vel"][:2], [0.01, -0.02])
    o2, _ = env.reset(seed=3, options=opts)
    assert np.allclose(o1, o2)
    assert np.isclose(o1[30], GRAVITIES["moon"] / 9.81)


def test_drop_points_are_reachable_and_clear(env):
    for s in range(30):
        env.reset(seed=s)
        assert env.reachable(env.drop_tip)
        assert env.drop_is_clear(env.drop_tip)


def test_benchmark_set_is_valid(env):
    blob = json.loads(CATCH_SET_PATH.read_text())
    cases = blob["cases"]
    assert len(cases) == 300
    assert {c["gravity"] for c in cases} == set(GRAVITIES)
    for c in cases[::10]:
        assert env.reachable(np.array(c["drop"])) and env.drop_is_clear(np.array(c["drop"]))


def test_hold_policy_never_catches(env):
    hold = make_catch_policy("hold")
    for g in GRAVITIES:
        r = run_episode(env, hold, 1, {"gravity": g})
        assert not r.success
        assert r.outcome.startswith("missed")


def test_fast_contact_knocks_the_ball(env):
    """A ball arriving at the cup faster than the soft-catch limit is not latched."""
    env.reset(seed=0, options=dict(gravity="earth"))
    env.phase, env._grip_armed, env._bounces = "rise", True, 1
    tip = env.ee_pos
    d = env.data
    d.qpos[env._bq:env._bq + 3] = tip - np.array([0.0, 0.0, CONTACT_DIST + 0.003])
    d.qvel[env._bv:env._bv + 3] = [0.0, 0.0, 1.5]
    d.qvel[:3] = 0.0
    mujoco.mj_forward(env.model, d)
    assert not env._try_latch()
    d.qvel[env._bv:env._bv + 3] = [0.0, 0.0, 0.2]
    mujoco.mj_forward(env.model, d)
    assert env._try_latch()


def test_pick_and_carry_script(env):
    """start='pick': the cup latches the ball off the cube and releases it at the drop point."""
    e = BallCatchEnv(start="pick", gravity="mars")
    e.reset(seed=4, options=dict(drop=[0.30, 0.02, 0.47]))
    assert np.allclose(e.ball_pos, BALL_REST, atol=1e-3)
    saw_latch = False
    for _ in range(800):
        _, _, term, trunc, info = e.step(np.zeros(3))
        saw_latch |= e.holding
        if e.phase == "fall":
            break
    assert saw_latch, "never latched the ball on the pedestal"
    assert e.phase == "fall"
    assert np.linalg.norm(e.release_state["pos"][:2] - np.array([0.30, 0.02])) < 0.01
    e.close()


def test_ballistic_fit_recovers_state():
    rng = np.random.default_rng(1)
    g = 3.71
    fit = BallisticFit(g)
    p0, v0 = np.array([0.3, 0.0, 0.5]), np.array([0.02, -0.01, -0.3])
    for k in range(20):
        t = 0.01 * k
        p = p0 + v0 * t - 0.5 * g * t * t * np.array([0, 0, 1])
        v = v0 - g * t * np.array([0, 0, 1])
        fit.add(t, p + rng.normal(0, 0.001, 3), v + rng.normal(0, 0.02, 3))
    p, v = fit.state_at(0.19)
    p_true = p0 + v0 * 0.19 - 0.5 * g * 0.19 ** 2 * np.array([0, 0, 1])
    assert np.linalg.norm(p - p_true) < 0.002
    assert np.linalg.norm(v - (v0 - g * 0.19 * np.array([0, 0, 1]))) < 0.02


@pytest.mark.parametrize("gname", list(GRAVITIES))
def test_model_controller_catches(env, gname):
    ctrl = CatchController(CatchParams())
    wins = sum(run_episode(env, ctrl, s, {"gravity": gname}).success for s in range(8))
    assert wins >= 7, f"{gname}: {wins}/8"


def test_chasing_without_the_follow_constraint_fails_on_the_moon(env):
    """The reason the controller keeps above the falling ball: on the Moon the arm outruns it."""
    chase = make_catch_policy("chase")
    res = [run_episode(env, chase, s, {"gravity": "moon"}) for s in range(6)]
    assert sum(r.success for r in res) <= 2
    assert any("before the bounce" in r.outcome for r in res)


def test_residual_wrapper_with_zero_scale_is_the_base_controller(env):
    """scale = 0 must reproduce the model-based controller exactly (same outcome, same steps)."""
    from mujoco_arm_rl.residual import ResidualCatchEnv

    wrapped = ResidualCatchEnv(BallCatchEnv(gravity="mars"), "catch_model", scale=0.0)
    obs, _ = wrapped.reset(seed=5, options={"gravity": "mars"})
    assert obs.shape == (37,)
    rng = np.random.default_rng(0)
    while True:
        obs, _, term, trunc, info = wrapped.step(rng.uniform(-1, 1, 3))   # ignored at scale 0
        if term or trunc:
            break
    ref = run_episode(BallCatchEnv(gravity="mars"), make_catch_policy("model"), 5, {"gravity": "mars"})
    assert info["outcome"] == ref.outcome


def test_policy_export_matches_the_network(tmp_path):
    """scripts/export_policy.py: the JSON forward pass equals SB3's deterministic predict()."""
    pytest.importorskip("stable_baselines3")
    import subprocess
    import sys
    from pathlib import Path

    from stable_baselines3 import SAC

    model = SAC("MlpPolicy", BallCatchEnv(), policy_kwargs=dict(net_arch=[32, 32]), seed=1, device="cpu")
    model.save(tmp_path / "m.zip")
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_policy.py"
    out = subprocess.run([sys.executable, str(script), str(tmp_path / "m.zip"), "--check"],
                         capture_output=True, text=True, check=True).stdout
    assert "max |exported - original|" in out
    spec = json.loads((tmp_path / "m.json").read_text())
    assert spec["format"] == "mujoco_arm_mlp_v1" and spec["obs_dim"] == 34 and spec["output"] == "tanh"
