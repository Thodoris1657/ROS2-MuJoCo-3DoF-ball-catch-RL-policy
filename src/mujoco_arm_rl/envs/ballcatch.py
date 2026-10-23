"""Pick, drop and catch a bouncing ball with the same 3-DOF arm, under Earth / Mars / Moon gravity.

One trial, in simulated time:

    pick     the arm descends on a ball resting on a pedestal cube; a gimballed
             suction cup at the end-effector latches when it touches the ball slowly
    carry    lift, move to a drop point above the floor, settle
    release  the cup lets go (with a small random lateral "puff" from the release)
    fall     the ball drops straight down and bounces off the floor
    rise     ... rises towards its new, lower apex
    catch    the arm must be waiting at the apex: the cup latches only if the ball
             arrives within 6 mm of it, within 15 mm of its axis, and with a relative
             speed below 0.5 m/s. Anything faster bounces off the arm.

The **success** condition is strict: the catch must happen after the first bounce,
before a second floor contact, near the top of the rebound (|v_z| of the ball below
0.5 m/s at the moment of latching), without the arm ever touching the ball on the
way down, and the ball must then be held for 0.3 s.

Why gravity matters: the apex *height* after a bounce is ``e^2 h`` whatever g is,
but every *time* scales with ``1/sqrt(g)``. On Earth the window in which the ball
is slower than 0.5 m/s is about +-50 ms around the apex; on the Moon the arm can
easily outrun the falling ball and hit it from above.

Episodes start either at the release (``start="release"``, used for training and
the benchmark: the pick / carry phase is deterministic and skipped by placing the
arm at the drop pose with the ball held) or at the very beginning
(``start="pick"``, used for demos and the ROS bridge). In both cases the policy
only acts during the catch; scripted phases ignore the action.

Observation (34,)
    [0:3] cos q        [3:6] sin q          [6:9] qvel / 10      [9:12] servo error
    [12:15] prev action                     [15:18] tip position [18:21] tip velocity
    [21:24] ball position (noisy)           [24:27] ball velocity (noisy)
    [27:30] 10 * (ball - tip)               [30] g / 9.81        [31] bounced  [32] holding
    [33] in flight: 1 from the release until the outcome (the policy is in control)

Action (3,)  normalised joint-target changes in [-1, 1], exactly as in Reach3 v1
(rate-limited to ``max_joint_speed``; a PD servo with gravity feed-forward turns
targets into torque, capped at the same motor limits).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_arm_rl.envs.reach3 import ASSETS_DIR, MAX_REACH, SHOULDER, analytic_ik, wrap_angle

DEFAULT_MODEL_PATH = os.path.join(ASSETS_DIR, "arm3_ballcatch.xml")

GRAVITIES: dict[str, float] = {"earth": 9.81, "mars": 3.71, "moon": 1.62}

BALL_R = 0.033                     # m, tennis-ball size
EE_R = 0.035                       # m, end-effector sphere
CONTACT_DIST = BALL_R + EE_R       # centre distance when the ball touches the cup
HOLD_DIST = CONTACT_DIST + 0.002   # held ball sits 2 mm proud of the cup (no fight with the contact)
PEDESTAL_TOP = 0.12
BALL_REST = np.array([0.0, 0.30, PEDESTAL_TOP + BALL_R])

# ball-floor contact: spring-damper with natural frequency OMEGA (rad/s) and
# constant impedance IMP; the damping sets the restitution (see restitution_solref)
CONTACT_OMEGA = 314.0
CONTACT_IMP = 0.9

ARM_GEOMS = ("base_geom", "link1_geom", "link2_geom", "link3_geom", "ee_geom")


def restitution_solref(e: float, omega: float = CONTACT_OMEGA, imp: float = CONTACT_IMP):
    """MuJoCo direct (negative) solref giving a linear spring-damper with restitution e."""
    e = float(np.clip(e, 0.05, 0.99))
    zeta = -np.log(e) / np.sqrt(np.pi ** 2 + np.log(e) ** 2)
    return -(omega ** 2) / imp, -2.0 * zeta * omega / imp


def resolve_gravity(g: str | float) -> float:
    if isinstance(g, str):
        key = g.lower()
        if key not in GRAVITIES:
            raise ValueError(f"unknown gravity {g!r}; use one of {sorted(GRAVITIES)} or a number")
        return GRAVITIES[key]
    return float(g)


def gravity_name(g: float) -> str:
    for k, v in GRAVITIES.items():
        if abs(v - g) < 1e-6:
            return k
    return f"{g:.2f} m/s^2"


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


@dataclass
class _Segment:
    """One piece of a scripted motion."""
    name: str
    duration: float
    kind: str = "joint"                      # "joint" (min-jerk in q) | "line" (min-jerk in tip space)
    goal: np.ndarray | None = None           # q (joint) or tip point (line)
    grip: str = "keep"                       # "keep" | "arm" (latch on soft contact) | "release"
    stop_on_latch: bool = False
    start: np.ndarray | None = field(default=None, repr=False)
    t0: float = 0.0


class BallCatchEnv(gym.Env):
    """3-DOF arm: pick a ball off a cube, drop it, catch it at the top of the rebound."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        gravity: str | float | Sequence[str | float] = "random",
        start: str = "release",                 # "release" | "pick"
        # --- robot (same servo as Reach3 v1) --------------------------------
        control_hz: float = 100.0,
        max_joint_speed: float = 3.0,
        servo_kp: Sequence[float] = (200.0, 300.0, 150.0),
        servo_damping_ratio: float = 1.0,
        # --- ball / world randomisation --------------------------------------
        restitution: float | Sequence[float] = (0.65, 0.85),
        drop_x: Sequence[float] = (0.24, 0.34),
        drop_y: Sequence[float] = (-0.10, 0.10),
        drop_z: Sequence[float] = (0.42, 0.52),  # tip height at release (forearm-clear)
        release_push: float = 0.02,              # m/s, std of the lateral release puff
        release_push_max: float = 0.04,
        obs_pos_noise: float = 0.001,            # m
        obs_vel_noise: float = 0.02,             # m/s
        # --- grip / success ---------------------------------------------------
        capture_gap: float = 0.006,
        capture_radius: float = 0.015,
        capture_speed: float = 0.5,
        apex_speed: float = 0.5,
        hold_time: float = 0.3,
        return_ball: bool = False,               # start="pick": put the ball back after a catch
        # --- reward (catch phase) --------------------------------------------
        catch_bonus: float = 20.0,
        fail_penalty: float = 5.0,
        action_cost: float = 0.01,
        max_catch_seconds: float = 4.0,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
        camera: str | int = -1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if start not in ("release", "pick"):
            raise ValueError(f"start must be 'release' or 'pick', got {start!r}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MJCF model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        m = self.model

        if isinstance(gravity, (list, tuple)):
            self.gravity_choices = [resolve_gravity(g) for g in gravity]
        elif isinstance(gravity, str) and gravity.lower() == "random":
            self.gravity_choices = list(GRAVITIES.values())
        else:
            self.gravity_choices = [resolve_gravity(gravity)]
        self.gravity = self.gravity_choices[0]
        self.start = start

        ts = m.opt.timestep
        self.frame_skip = max(1, int(round(1.0 / (control_hz * ts))))
        self.max_step = float(max_joint_speed) * self.dt
        self.max_episode_steps = int(round(max_catch_seconds / self.dt))

        if isinstance(restitution, (int, float)):
            self.restitution_range = (float(restitution), float(restitution))
        else:
            self.restitution_range = (float(restitution[0]), float(restitution[1]))
        self.drop_lo = np.array([drop_x[0], drop_y[0], drop_z[0]], dtype=np.float64)
        self.drop_hi = np.array([drop_x[1], drop_y[1], drop_z[1]], dtype=np.float64)
        self.release_push = float(release_push)
        self.release_push_max = float(release_push_max)
        self.obs_pos_noise = float(obs_pos_noise)
        self.obs_vel_noise = float(obs_vel_noise)
        self.capture_gap = float(capture_gap)
        self.capture_radius = float(capture_radius)
        self.capture_speed = float(capture_speed)
        self.apex_speed = float(apex_speed)
        self.hold_time = float(hold_time)
        self.return_ball = bool(return_ball)
        self.catch_bonus = float(catch_bonus)
        self.fail_penalty = float(fail_penalty)
        self.action_cost = float(action_cost)

        # ids
        name2id = mujoco.mj_name2id
        self._tip_id = name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self._ee_body = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee")
        self._ball_body = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_geom = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._floor_geom = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._arm_geoms = np.array([name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ARM_GEOMS])
        self._grip_eq = name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
        self._floor_pair = name2id(m, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
        jid = name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self._bq = int(m.jnt_qposadr[jid])        # ball qpos: [x y z qw qx qy qz]
        self._bv = int(m.jnt_dofadr[jid])         # ball qvel: [vx vy vz wx wy wz]
        self.nq_arm = 3
        self.jnt_lo = m.jnt_range[:3, 0].copy()
        self.jnt_hi = m.jnt_range[:3, 1].copy()
        self.torque_limit = m.actuator_gear[:, 0].copy()
        self._configure_servo(np.asarray(servo_kp, dtype=np.float64), servo_damping_ratio)
        self._jacp = np.zeros((3, m.nv))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(34,), dtype=np.float32)
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.dt)))

        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self.camera = camera
        self._viewer = None
        self._renderer = None

        self._reset_bookkeeping()
        if seed is not None:
            self.reset(seed=seed)

    # --------------------------------------------------------------- set-up

    def _configure_servo(self, kp: np.ndarray, zeta: float) -> None:
        """Same PD servo as Reach3 v1 (critically damped at a mid pose, torque-capped).

        Gravity feed-forward comes from the MJCF (``gravcomp`` on the links routed
        through the actuators), so the servo holds position equally well on Earth,
        Mars and the Moon and the cap still applies to the total torque.
        """
        m, d = self.model, self._scratch
        d.qpos[:3] = [0.0, 1.0, 0.8]
        d.qpos[self._bq:self._bq + 3] = [5.0, 5.0, 5.0]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        try:                                   # MuJoCo >= 3.3: mj_fullM(m, d, dst)
            mujoco.mj_fullM(m, d, M)
        except TypeError:                      # older: mj_fullM(m, dst, qM)
            mujoco.mj_fullM(m, M, d.qM)
        inertia = np.diag(M)[:3]
        kv = 2.0 * zeta * np.sqrt(kp * inertia)
        for i in range(3):
            j = m.actuator_trnid[i, 0]
            m.actuator_gear[i, 0] = 1.0
            m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[i, :] = 0.0
            m.actuator_gainprm[i, 0] = kp[i]
            m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_biasprm[i, :] = 0.0
            m.actuator_biasprm[i, 1] = -kp[i]
            m.actuator_biasprm[i, 2] = -kv[i]
            m.actuator_ctrlrange[i] = m.jnt_range[j]
            m.actuator_ctrllimited[i] = 1
            m.actuator_forcerange[i] = (-self.torque_limit[i], self.torque_limit[i])
            m.actuator_forcelimited[i] = 1
        self.servo_kp, self.servo_kv = kp, kv

    def _reset_bookkeeping(self) -> None:
        self.phase = "idle"
        self.sim_time = 0.0
        self._q_target = np.zeros(3)
        self._prev_action = np.zeros(3)
        self._segments: list[_Segment] = []
        self._seg_i = 0
        self._latched = False
        self._grip_armed = False
        self._grip_offset = np.array([0.0, 0.0, -HOLD_DIST])
        self._grip_offset_goal = np.array([0.0, 0.0, -HOLD_DIST])
        self._released_clear = True
        self._floor_contact = False
        self._bounces = 0
        self._t_release = None
        self._t_bounce = None
        self._t_catch = None
        self._apex_true: dict[str, Any] | None = None
        self._catch_info: dict[str, Any] = {}
        self._outcome: str | None = None      # None while running
        self._success = False
        self._catch_steps = 0
        self.restitution = 0.75
        self.drop_tip = np.zeros(3)
        self._ball_obs_noise = (np.zeros(3), np.zeros(3))

    # ---------------------------------------------------------------- props

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    @property
    def q_target(self) -> np.ndarray:
        return self._q_target.copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_id].copy()

    @property
    def ee_vel(self) -> np.ndarray:
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self._tip_id)
        return self._jacp[:, :3] @ self.data.qvel[:3]

    @property
    def ball_pos(self) -> np.ndarray:
        return self.data.qpos[self._bq:self._bq + 3].copy()

    @property
    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self._bv:self._bv + 3].copy()

    @property
    def holding(self) -> bool:
        return self._latched

    @property
    def catching(self) -> bool:
        """True while the policy is in control (after the release, until the outcome)."""
        return self.phase in ("fall", "rise") and self._outcome is None

    @property
    def target_pos(self) -> np.ndarray:  # the green marker (predicted catch point)
        return self.data.mocap_pos[0].copy()

    # ------------------------------------------------------ kinematics tools

    def config_is_valid(self, q: np.ndarray, margin: float = 0.0) -> bool:
        """Joint limits (with margin) and no contact of the arm with floor / cube / itself."""
        q = np.asarray(q, dtype=np.float64)
        if np.any(q < self.jnt_lo + margin) or np.any(q > self.jnt_hi - margin):
            return False
        d = self._scratch
        d.qpos[:3] = q
        d.qpos[self._bq:self._bq + 3] = [5.0, 5.0, 5.0]   # ball out of the way
        mujoco.mj_forward(self.model, d)
        return d.ncon == 0

    def solve_ik(self, p: np.ndarray, near: np.ndarray | None = None,
                 margin: float = 0.0) -> np.ndarray | None:
        sols = [q for q in analytic_ik(p) if self.config_is_valid(q, margin)]
        if not sols:
            return None
        near = np.zeros(3) if near is None else np.asarray(near, dtype=np.float64)
        cost = [float(np.sum(wrap_angle(q - near) ** 2)) for q in sols]
        return sols[int(np.argmin(cost))]

    def reachable(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=np.float64)
        return float(np.linalg.norm(p - SHOULDER)) <= MAX_REACH and self.solve_ik(p, margin=margin) is not None

    def ball_clearance(self, q: np.ndarray, ball: np.ndarray, skip_ee: bool = False) -> float:
        """Smallest distance (m) between a ball at ``ball`` and the arm in configuration ``q``."""
        m, d = self.model, self._scratch
        d.qpos[:3] = q
        d.qpos[self._bq:self._bq + 3] = ball
        mujoco.mj_kinematics(m, d)
        best = np.inf
        for gid in self._arm_geoms:
            if skip_ee and gid == self._arm_geoms[-1]:
                continue
            best = min(best, mujoco.mj_geomDistance(m, d, self._ball_geom, int(gid), 0.2, None))
        return float(best)

    def drop_is_clear(self, tip: np.ndarray, q: np.ndarray | None = None, margin: float = 0.005) -> bool:
        """The held ball, and the ball falling straight down from there, stay clear of the arm."""
        q = self.solve_ik(tip, near=np.array([0.0, 0.5, 1.0])) if q is None else q
        if q is None:
            return False
        for depth in (HOLD_DIST, HOLD_DIST + 0.02, HOLD_DIST + 0.05, HOLD_DIST + 0.10):
            ball = np.asarray(tip, dtype=np.float64) - np.array([0.0, 0.0, depth])
            if self.ball_clearance(q, ball, skip_ee=True) < margin:
                return False
        return True

    # ------------------------------------------------------------- physics

    def _set_restitution(self, e: float) -> None:
        self.restitution = float(e)
        self.model.pair_solref[self._floor_pair] = restitution_solref(e)

    def _set_gravity(self, g: float) -> None:
        self.gravity = float(g)
        self.model.opt.gravity[:] = (0.0, 0.0, -self.gravity)

    def _grip_on(self, offset_world: np.ndarray) -> None:
        self._latched = True
        self._grip_offset = np.asarray(offset_world, dtype=np.float64).copy()
        self._update_grip_anchor()
        self.data.eq_active[self._grip_eq] = 1

    def _grip_off(self) -> None:
        self._latched = False
        self.data.eq_active[self._grip_eq] = 0

    def _update_grip_anchor(self) -> None:
        """Keep the ball below the cup in the world frame (gimbal), easing towards centre."""
        if not self._latched:
            return
        self._grip_offset += 0.15 * (self._grip_offset_goal - self._grip_offset)
        R = self.data.xmat[self._ee_body].reshape(3, 3)
        self.model.eq_data[self._grip_eq, 0:3] = R.T @ self._grip_offset
        self.model.eq_data[self._grip_eq, 3:6] = 0.0

    def _contacts(self) -> tuple[bool, bool]:
        """(ball touches floor, ball touches the arm) for the current contact set."""
        d = self.data
        n = d.ncon
        if n == 0:
            return False, False
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        mine = (g1 == self._ball_geom) | (g2 == self._ball_geom)
        if not mine.any():
            return False, False
        other = np.where(g1 == self._ball_geom, g2, g1)[mine]
        return bool(np.any(other == self._floor_geom)), bool(np.isin(other, self._arm_geoms).any())

    def _try_latch(self) -> bool:
        """Soft-contact latch: close enough, centred enough, slow enough."""
        tip = self.data.site_xpos[self._tip_id]
        rel = self.ball_pos - tip
        dist = float(np.linalg.norm(rel))
        if dist - CONTACT_DIST > self.capture_gap or rel[2] > 0.0:
            return False
        if float(np.hypot(rel[0], rel[1])) > self.capture_radius:
            return False
        vrel = self.ball_vel - self.ee_vel
        if float(np.linalg.norm(vrel)) > self.capture_speed:
            return False
        self._catch_info = dict(rel_speed=float(np.linalg.norm(vrel)), offset=rel.copy(),
                                ball_vz=float(self.ball_vel[2]), ball_z=float(self.ball_pos[2]))
        self._grip_on(rel)
        return True

    def _substep(self) -> None:
        """One physics step plus the event logic that must run at 1 kHz."""
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.model.opt.timestep
        floor, arm = self._contacts()
        if not self._released_clear:     # 15 ms for the ball to leave the cup after the release
            self._released_clear = self.sim_time - self._t_release > 0.015
            arm = False

        if self.phase == "pick" and self._grip_armed and not self._latched:
            self._try_latch()
        elif self.phase in ("fall", "rise") and self._outcome is None:
            bz = self.ball_vel[2]
            if floor and not self._floor_contact:
                self._bounces += 1
                if self._bounces >= 2:
                    self._finish("missed: second bounce")
                elif self._t_bounce is None:
                    self._t_bounce = self.sim_time
            if self.phase == "fall" and self._bounces >= 1 and not floor and bz > 0.0:
                self.phase = "rise"
                self._grip_armed = True
                z, g = self.ball_pos[2], self.gravity
                self._apex_true = dict(t=self.sim_time + bz / g, z=z + bz * bz / (2 * g),
                                       xy=self.ball_pos[:2] + self.ball_vel[:2] * bz / g)
            if self._outcome is None:
                if self._grip_armed and not self._latched and self._try_latch():
                    self._t_catch = self.sim_time
                    if abs(self._catch_info["ball_vz"]) > self.apex_speed:
                        self._finish("latched outside the apex window")
                    else:
                        self.phase = "hold"
                elif arm and not self._latched:
                    if self._bounces == 0:
                        self._finish("touched the ball before the bounce")
                    else:
                        why = self._why_no_latch()
                        self._finish(f"knocked the ball ({why})")
        elif self.phase == "hold" and self._outcome is None:
            if self.sim_time - self._t_catch >= self.hold_time:
                self._success = True
                self._finish("caught")
        self._floor_contact = floor

    def _why_no_latch(self) -> str:
        tip = self.data.site_xpos[self._tip_id]
        rel = self.ball_pos - tip
        off = float(np.hypot(rel[0], rel[1]))
        vrel = float(np.linalg.norm(self.ball_vel - self.ee_vel))
        if off > self.capture_radius:
            return f"{off * 1000:.0f} mm off-centre"
        if vrel > self.capture_speed:
            return f"impact {vrel:.2f} m/s"
        return "side contact"

    def _finish(self, outcome: str) -> None:
        if self._outcome is None:
            self._outcome = outcome
            if not self._success and self.phase in ("pick", "fall", "rise", "hold"):
                self.phase = "failed"

    # ---------------------------------------------------- scripted motions

    def _plan_pick_and_carry(self) -> None:
        q_home = self.q
        p_ball = self.ball_pos
        pre = p_ball + np.array([0.0, 0.0, CONTACT_DIST + 0.06])
        touch = p_ball + np.array([0.0, 0.0, CONTACT_DIST - 0.004])
        lift = pre + np.array([0.0, 0.0, 0.10])
        q_pre = self.solve_ik(pre, near=q_home)
        q_drop = self.solve_ik(self.drop_tip, near=q_pre)
        if q_pre is None or q_drop is None:
            raise RuntimeError("pick or drop pose unreachable")
        self._segments = [
            _Segment("approach", 1.2, "joint", q_pre),
            _Segment("descend", 1.0, "line", touch, grip="arm", stop_on_latch=True),
            _Segment("lift", 0.8, "line", lift),
            _Segment("carry", 1.4, "joint", q_drop),
            _Segment("settle", 0.5, "joint", q_drop),
        ]
        self._seg_i = 0
        self.phase = "pick"

    def _plan_return(self) -> None:
        q_now = self.q
        above = BALL_REST + np.array([0.0, 0.0, CONTACT_DIST + 0.10])
        place = BALL_REST + np.array([0.0, 0.0, CONTACT_DIST + 0.002])
        q_above = self.solve_ik(above, near=q_now)
        self._segments = [
            _Segment("carry back", 1.6, "joint", q_above),
            _Segment("lower", 1.0, "line", place),
            _Segment("let go", 0.3, "line", place, grip="release"),
            _Segment("retreat", 0.8, "line", above),
        ]
        self._seg_i = 0
        self.phase = "return"

    def _scripted_target(self) -> np.ndarray | None:
        """Servo target for the current scripted segment (None when the script is done)."""
        while self._seg_i < len(self._segments):
            seg = self._segments[self._seg_i]
            if seg.start is None:
                seg.start = self.ee_pos if seg.kind == "line" else self._q_target.copy()
                seg.t0 = self.sim_time
                if seg.grip == "arm":
                    self._grip_armed = True
                elif seg.grip == "release":
                    self._grip_off()
                    self._grip_armed = False
            s = (self.sim_time - seg.t0) / seg.duration
            done = s >= 1.0 or (seg.stop_on_latch and self._latched)
            if done:
                if seg.grip == "arm":
                    self._grip_armed = False
                self._seg_i += 1
                continue
            a = min_jerk(s + self.dt / seg.duration)
            if seg.kind == "joint":
                return seg.start + a * (seg.goal - seg.start)
            p = seg.start + a * (seg.goal - seg.start)
            q = self.solve_ik(p, near=self._q_target)
            return self._q_target if q is None else q
        return None

    # -------------------------------------------------------------- release

    def _release(self) -> None:
        self._grip_off()
        self._grip_armed = False
        push = self.np_random.normal(0.0, self.release_push, size=2)
        n = float(np.linalg.norm(push))
        if n > self.release_push_max:
            push *= self.release_push_max / n
        self.data.qvel[self._bv:self._bv + 2] += push
        self._t_release = self.sim_time
        self._released_clear = False
        self._bounces = 0
        self._floor_contact = False
        self.phase = "fall"
        self.release_state = dict(t=self.sim_time, pos=self.ball_pos, vel=self.ball_vel)

    # ----------------------------------------------------------- sampling

    def _sample_drop(self) -> np.ndarray:
        for _ in range(200):
            p = self.np_random.uniform(self.drop_lo, self.drop_hi)
            if self.reachable(p, margin=0.05) and self.drop_is_clear(p):
                return p
        return np.array([0.30, 0.0, 0.50])  # pragma: no cover

    # ---------------------------------------------------------- observation

    def ideal_catch_point(self) -> np.ndarray:
        """Where the tip should be to meet the ball at its apex (privileged: uses true state).

        After the bounce the apex is exact; before it, the episode's nominal restitution
        is used. This is what the reward measures against and what the marker shows.
        """
        g = self.gravity
        p, v = self.ball_pos, self.ball_vel
        if self._apex_true is not None:
            xy, z = self._apex_true["xy"], self._apex_true["z"]
        else:
            h = p[2] - BALL_R
            v_imp = np.sqrt(max(v[2] * v[2] + 2.0 * g * max(h, 0.0), 0.0))
            t_imp = (v[2] + v_imp) / g
            v_out = self.restitution * v_imp
            xy = p[:2] + v[:2] * (t_imp + v_out / g)
            z = BALL_R + v_out * v_out / (2.0 * g)
        return np.array([xy[0], xy[1], z + CONTACT_DIST + 0.002])

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[:3]
        qd = self.data.qvel[:3]
        tip = self.ee_pos
        bp = self.ball_pos + self.np_random.normal(0.0, self.obs_pos_noise, 3)
        bv = self.ball_vel + self.np_random.normal(0.0, self.obs_vel_noise, 3)
        return np.concatenate([
            np.cos(q), np.sin(q), qd / 10.0, self._q_target - q, self._prev_action,
            tip, self.ee_vel, bp, bv, 10.0 * (bp - tip),
            [self.gravity / 9.81, float(self._bounces > 0), float(self._latched),
             float(self.catching)],
        ]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "phase": self.phase, "gravity": self.gravity, "gravity_name": gravity_name(self.gravity),
            "restitution": self.restitution, "is_success": self._success,
            "outcome": self._outcome, "sim_time": self.sim_time,
            "ball_pos": self.ball_pos, "tip_pos": self.ee_pos, "bounces": self._bounces,
        }
        if self._t_release is not None:
            info["t_release"] = self._t_release
        if self._apex_true is not None:
            info["apex"] = dict(self._apex_true)
        if self._t_catch is not None:
            info["t_catch"] = self._t_catch
            info.update({f"catch_{k}": v for k, v in self._catch_info.items()})
            if self._apex_true is not None:
                info["catch_time_error"] = self._t_catch - self._apex_true["t"]
                info["catch_height_error"] = self._catch_info["ball_z"] - self._apex_true["z"]
        return info

    # --------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Options (all optional; the benchmark pins every one of them):
        ``gravity`` (name or m/s^2), ``restitution``, ``drop`` (tip point at release),
        ``push`` (2,) lateral release velocity, ``start`` ("release" | "pick")."""
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data
        self._reset_bookkeeping()
        mujoco.mj_resetData(m, d)

        g = options.get("gravity")
        self._set_gravity(resolve_gravity(g) if g is not None
                          else self.gravity_choices[int(self.np_random.integers(len(self.gravity_choices)))])
        e = options.get("restitution")
        lo, hi = self.restitution_range
        self._set_restitution(float(e) if e is not None else float(self.np_random.uniform(lo, hi)))
        drop = options.get("drop")
        self.drop_tip = np.asarray(drop, dtype=np.float64) if drop is not None else self._sample_drop()
        if not self.reachable(self.drop_tip):
            raise ValueError(f"drop point {self.drop_tip} is not reachable")
        start = options.get("start", self.start)

        if start == "pick":
            d.qpos[:3] = 0.0
            d.qpos[self._bq:self._bq + 3] = BALL_REST
            d.qpos[self._bq + 3:self._bq + 7] = (1, 0, 0, 0)
            self._q_target = d.qpos[:3].copy()
            d.ctrl[:3] = self._q_target
            mujoco.mj_forward(m, d)
            self._plan_pick_and_carry()
        else:
            q_drop = self.solve_ik(self.drop_tip, near=np.array([0.0, 0.5, 1.0]))
            d.qpos[:3] = q_drop
            mujoco.mj_forward(m, d)
            d.qpos[self._bq:self._bq + 3] = self.ee_pos + np.array([0.0, 0.0, -HOLD_DIST])
            d.qpos[self._bq + 3:self._bq + 7] = (1, 0, 0, 0)
            self._q_target = q_drop.copy()
            d.ctrl[:3] = self._q_target
            mujoco.mj_forward(m, d)
            self._grip_on(np.array([0.0, 0.0, -HOLD_DIST]))
            for _ in range(int(round(0.15 / m.opt.timestep))):   # let the servo settle
                mujoco.mj_step(m, d)
            self.sim_time = 0.0
            self._release()
            push = options.get("push")
            if push is not None:   # replace the random puff by the pinned one
                d.qvel[self._bv:self._bv + 2] = np.asarray(push, dtype=np.float64)
                self.release_state["vel"] = self.ball_vel
        self._pending_push = options.get("push")
        mujoco.mj_forward(m, d)
        self.data.mocap_pos[0] = self.ideal_catch_point() if self.phase == "fall" else self.drop_tip
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        policy_in_control = self.catching
        q = self.data.qpos[:3]

        if policy_in_control:
            tgt = self._q_target + a * self.max_step
        else:
            tgt = self._scripted_target() if self.phase in ("pick", "return") else None
            if tgt is None:
                tgt = self._q_target.copy()
            a = np.zeros(3)
        tgt = np.clip(tgt, q - 0.3, q + 0.3)
        self._q_target = np.clip(tgt, self.jnt_lo, self.jnt_hi)
        self.data.ctrl[:3] = self._q_target

        was_running = self._outcome is None
        self._update_grip_anchor()
        for _ in range(self.frame_skip):
            self._substep()
        if policy_in_control:
            self._catch_steps += 1

        # phase transitions driven by the script
        if self.phase == "pick" and self._seg_i >= len(self._segments):
            if not self._latched:
                self._finish("pick failed")
            else:
                self._release()
                if self._pending_push is not None:
                    self.data.qvel[self._bv:self._bv + 2] = np.asarray(self._pending_push, float)
                    self.release_state["vel"] = self.ball_vel
        if self.phase == "hold" and self._outcome == "caught" and self.start == "pick" and self.return_ball:
            self._plan_return()
        if self.phase == "fall" or self.phase == "rise":
            self.data.mocap_pos[0] = self.ideal_catch_point()

        # reward (only meaningful while the policy acts)
        reward = 0.0
        if policy_in_control:
            err = float(np.linalg.norm(self.ee_pos - self.ideal_catch_point()))
            reward = -err + (1.0 - np.tanh(err / 0.02)) * (0.5 + 0.5 * float(self._bounces > 0))
            if self._bounces == 0:     # stay above the falling ball
                rel = self.ball_pos - self.ee_pos
                if np.hypot(rel[0], rel[1]) < 0.06:
                    reward -= 20.0 * max(0.0, rel[2] + CONTACT_DIST + 0.015)
            reward -= self.action_cost * float(np.sum(a * a))
            if self.phase == "hold":                          # latched inside the apex window
                reward += self.catch_bonus
            elif was_running and self._outcome is not None:   # any failure decided this step
                reward -= self.fail_penalty
        self._prev_action = a

        if not np.isfinite(self.data.qpos).all():
            self._finish("numerical failure")
        script_running = self.phase in ("pick", "return")
        terminated = self._outcome is not None and not script_running
        if self.phase == "return" and self._seg_i >= len(self._segments):
            terminated = True
        truncated = (not terminated) and self._catch_steps >= self.max_episode_steps
        if truncated and self._outcome is None:
            self._outcome = "timeout"
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._info()

    # ---------------------------------------------------------------- render

    def render(self):
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer  # (a bare `import mujoco.viewer` here would
            #                                            make `mujoco` local and break rgb_array)
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False)
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        return None

    @property
    def viewer_is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def wait_for_viewer(self, fps: int = 50) -> None:
        if self._viewer is None:
            return
        try:
            while self._viewer.is_running():
                self._viewer.sync()
                time.sleep(1.0 / max(fps, 1))
        except KeyboardInterrupt:
            pass

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
