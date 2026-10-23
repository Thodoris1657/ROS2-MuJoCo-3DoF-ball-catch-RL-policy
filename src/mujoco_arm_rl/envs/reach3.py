"""Reach-and-hold task for a 3-DOF MuJoCo manipulator.

The environment is written directly against the ``mujoco`` bindings (rather than
``gymnasium.envs.mujoco.MujocoEnv``) so every part of the loop -- low-level
control, observation, reward, goal sampling -- is visible in one file.

Two registered versions share this class:

``MujocoArm3Reach-v1`` (default, recommended)
    * **Position servo control.** The policy outputs *changes* to joint targets
      (rate-limited to ``max_joint_speed``); a stiff, near-critically-damped PD
      servo inside MuJoCo turns them into torque, capped at the motor limits.
      This is how real arms are commanded, and it removes gravity compensation
      from the learning problem entirely.
    * **25 Hz decisions**, 2 s episodes (50 steps). Fewer, more meaningful
      decisions make credit assignment far easier than 100 Hz torque control.
    * **Only reachable goals.** Targets are sampled uniformly by volume in the
      workspace and kept only if the analytic IK finds a collision-free
      solution inside the joint limits.
    * **Shaped reward**: linear distance for the global pull, a ``1 - tanh``
      precision term for the final centimetres, and small action / action-rate
      penalties so the arm settles instead of dithering.

``MujocoArm3Reach-v0`` (legacy)
    The original task: raw torques at 100 Hz, 200-step episodes, a sampler that
    produced ~4 % unreachable targets, and an 18-dim observation. Kept so old
    checkpoints still load and so the ablation in the README is reproducible.

Observation, full (24,)
    [0:3]   cos(q)            [3:6]   sin(q)
    [6:9]   qvel / 10         [9:12]  servo error (q_target - q), zeros in torque mode
    [12:15] previous action   [15:18] end-effector position
    [18:21] target position   [21:24] 10 * (target - end-effector)

Observation, legacy (18,)
    cos(q), sin(q), qvel/10, ee, target, target - ee
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install 'gymnasium>=0.29'") from exc


ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "assets",
)
DEFAULT_MODEL_PATH = os.path.join(ASSETS_DIR, "arm3.xml")

# Arm geometry (metres) -- must match assets/arm3.xml.
SHOULDER = np.array([0.0, 0.0, 0.22])   # joint2 (pitch) anchor
L2, L3 = 0.25, 0.25                      # upper arm, forearm (joint3 -> tip site)
MAX_REACH = L2 + L3                      # 0.50 m from the shoulder

LEGACY_KWARGS = dict(
    control_mode="torque",
    control_hz=100.0,
    episode_seconds=2.0,
    reward_mode="legacy",
    target_sampler="legacy",
    obs_mode="legacy",
)


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def analytic_ik(p: np.ndarray) -> list[np.ndarray]:
    """Every (pan, pitch, elbow) solution placing the tip site at ``p``.

    Pan aligns the arm plane with the target (or points away, with the pitch
    flipped); within that plane it is the textbook two-link problem, with angles
    measured from vertical. Joint limits and collisions are *not* checked here.
    Verified against MuJoCo forward kinematics to 1e-15 m (see tests).
    """
    p = np.asarray(p, dtype=np.float64)
    r_xy = float(np.hypot(p[0], p[1]))
    h = float(p[2] - SHOULDER[2])
    yaw0 = float(np.arctan2(p[1], p[0]))
    out = []
    for yaw, r in ((yaw0, r_xy), (yaw0 + np.pi, -r_xy)):
        cos_elbow = (r * r + h * h - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        if abs(cos_elbow) > 1.0:
            continue
        for sign in (1.0, -1.0):
            q3 = sign * float(np.arccos(cos_elbow))
            q2 = float(np.arctan2(r, h) - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3)))
            out.append(np.array([wrap_angle(yaw), wrap_angle(q2), q3]))
    return out


class Reach3Env(gym.Env):
    """3-joint arm that must bring its end-effector onto a target and hold it."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        # --- control ---------------------------------------------------------
        control_mode: str = "position",        # "position" | "torque"
        control_hz: float = 25.0,
        episode_seconds: float = 2.0,
        max_joint_speed: float = 3.0,          # rad/s, position mode rate limit
        servo_kp: Sequence[float] = (200.0, 300.0, 150.0),
        servo_damping_ratio: float = 1.0,
        # --- task ------------------------------------------------------------
        success_threshold: float = 0.05,
        target_sampler: str = "reachable",     # "reachable" | "legacy"
        fixed_target: Sequence[float] | None = None,
        start_noise: float = 0.3,              # rad, around the upright home pose
        # --- reward ----------------------------------------------------------
        reward_mode: str = "shaped",           # "shaped" | "legacy"
        precision_scale: float = 0.05,         # m, width of the 1 - tanh(d/scale) term
        precision_weight: float = 1.0,
        action_cost: float = 0.02,
        action_rate_cost: float = 0.02,
        # --- observation / rendering ----------------------------------------
        obs_mode: str = "full",                # "full" (24) | "legacy" (18)
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
        seed: int | None = None,
        # legacy aliases (v0 checkpoints and older scripts pass these)
        frame_skip: int | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        for name, val, allowed in (
            ("control_mode", control_mode, ("position", "torque")),
            ("target_sampler", target_sampler, ("reachable", "legacy")),
            ("reward_mode", reward_mode, ("shaped", "legacy")),
            ("obs_mode", obs_mode, ("full", "legacy")),
        ):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MJCF model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)  # for IK / collision checks

        self.control_mode = control_mode
        self.reward_mode = reward_mode
        self.target_sampler = target_sampler
        self.obs_mode = obs_mode

        # timing: derive frame_skip from the control rate
        ts = self.model.opt.timestep
        self.frame_skip = int(frame_skip) if frame_skip else max(1, int(round(1.0 / (control_hz * ts))))
        self.max_episode_steps = (
            int(max_episode_steps) if max_episode_steps
            else max(1, int(round(episode_seconds / (ts * self.frame_skip))))
        )

        self.success_threshold = float(success_threshold)
        self.start_noise = float(start_noise)
        self.max_step = float(max_joint_speed) * self.dt
        self.precision_scale = float(precision_scale)
        self.precision_weight = float(precision_weight)
        self.action_cost = float(action_cost)
        self.action_rate_cost = float(action_rate_cost)

        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        assert self.nu == 3, f"expected a 3-actuator model, got {self.nu}"
        self.jnt_lo = self.model.jnt_range[:, 0].copy()
        self.jnt_hi = self.model.jnt_range[:, 1].copy()
        self._tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")

        self.torque_limit = self.model.actuator_gear[:, 0].copy()
        if control_mode == "position":
            self._configure_servo(np.asarray(servo_kp, dtype=np.float64), servo_damping_ratio)

        self.fixed_target = None if fixed_target is None else np.asarray(fixed_target, np.float64)
        if self.fixed_target is not None:
            if self.fixed_target.shape != (3,):
                raise ValueError(f"fixed_target must be 3 numbers, got {self.fixed_target!r}")
            ok, why = self.check_target_reachable(self.fixed_target)
            if not ok:
                pretty = tuple(round(float(v), 3) for v in self.fixed_target)
                raise ValueError(f"fixed_target {pretty} is unreachable: {why}")

        obs_dim = 24 if obs_mode == "full" else 18
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.dt)))

        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self._viewer = None
        self._renderer = None

        self._q_target = np.zeros(self.nu)
        self._prev_action = np.zeros(self.nu)
        self._elapsed_steps = 0
        self._reached_at: int | None = None
        if seed is not None:
            self.reset(seed=seed)

    # --------------------------------------------------------------- set-up

    def _configure_servo(self, kp: np.ndarray, zeta: float) -> None:
        """Turn the MJCF torque motors into PD position servos, in place.

        force = kp * (ctrl - q) - kv * qdot, clipped to the original motor
        torque (the MJCF ``gear``). kv is chosen for damping ratio ``zeta`` using
        the joint-space inertia at a mid-workspace pose, and the implicitfast
        integrator treats that velocity term implicitly, so stiff gains stay stable.
        """
        m = self.model
        d = self._scratch
        d.qpos[:] = [0.0, 1.0, 0.8]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        try:                                   # MuJoCo >= 3.3: mj_fullM(m, d, dst)
            mujoco.mj_fullM(m, d, M)
        except TypeError:                      # older: mj_fullM(m, dst, qM)
            mujoco.mj_fullM(m, M, d.qM)
        inertia = np.diag(M)
        kv = 2.0 * zeta * np.sqrt(kp * inertia)

        for i in range(self.nu):
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

    # ---------------------------------------------------------------- props

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_id].copy()

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    @property
    def q_target(self) -> np.ndarray:
        return self._q_target.copy()

    # ------------------------------------------------------ kinematics tools

    def config_is_valid(self, q: np.ndarray, margin: float = 0.0) -> bool:
        """Inside joint limits (with margin) and free of floor / self contact."""
        q = np.asarray(q, dtype=np.float64)
        if np.any(q < self.jnt_lo + margin) or np.any(q > self.jnt_hi - margin):
            return False
        self._scratch.qpos[:] = q
        mujoco.mj_forward(self.model, self._scratch)
        return self._scratch.ncon == 0

    def solve_ik(self, p: np.ndarray, near: np.ndarray | None = None,
                 margin: float = 0.0) -> np.ndarray | None:
        """Best valid IK solution for tip position ``p`` (closest to ``near``)."""
        sols = [q for q in analytic_ik(p) if self.config_is_valid(q, margin)]
        if not sols:
            return None
        if near is None:
            near = np.zeros(self.nq)
        cost = [float(np.sum(wrap_angle(q - near) ** 2)) for q in sols]
        return sols[int(np.argmin(cost))]

    def check_target_reachable(self, point: np.ndarray) -> tuple[bool, str]:
        point = np.asarray(point, dtype=np.float64)
        dist = float(np.linalg.norm(point - SHOULDER))
        if dist > MAX_REACH:
            return False, (f"{dist:.3f} m from the shoulder at {tuple(SHOULDER)}, "
                           f"but the arm only reaches {MAX_REACH:.2f} m")
        if self.solve_ik(point) is None:
            return False, ("no collision-free joint configuration within the joint "
                           "limits puts the tip there (too close to the shoulder or "
                           "to the floor/base?)")
        return True, f"{dist:.3f} m from the shoulder (max {MAX_REACH:.2f} m)"

    # ------------------------------------------------------------- sampling

    def _sample_target(self) -> np.ndarray:
        if self.fixed_target is not None:
            return self.fixed_target.copy()
        if self.target_sampler == "legacy":
            return self._sample_target_legacy()
        # Uniform by volume in a shell around the shoulder, rejection-tested by IK
        # with a small joint-limit margin so goals are comfortably reachable.
        r_lo, r_hi = 0.16, 0.47
        for _ in range(500):
            v = self.np_random.normal(size=3)
            v /= np.linalg.norm(v) + 1e-12
            r = self.np_random.uniform(r_lo ** 3, r_hi ** 3) ** (1.0 / 3.0)
            p = SHOULDER + v * r
            if p[2] < 0.08:
                continue
            if self.solve_ik(p, margin=0.05) is not None:
                return p
        return SHOULDER + np.array([0.3, 0.0, 0.0])  # pragma: no cover

    def _sample_target_legacy(self) -> np.ndarray:
        """Original v0 sampler. ~4 % of its targets are unreachable -- see README."""
        centre = np.array([0.0, 0.0, 0.10])
        for _ in range(100):
            v = self.np_random.normal(size=3)
            v /= np.linalg.norm(v) + 1e-9
            p = centre + v * self.np_random.uniform(0.18, 0.45)
            if p[2] >= 0.10:
                return p
        return centre + np.array([0.18, 0.0, 0.0])  # pragma: no cover

    def _sample_start(self) -> np.ndarray:
        for _ in range(100):
            q = self.np_random.uniform(-self.start_noise, self.start_noise, size=self.nq)
            if self.config_is_valid(q):
                return q
        return np.zeros(self.nq)  # pragma: no cover

    # ----------------------------------------------------------- observation

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[: self.nq]
        qd = self.data.qvel[: self.nv]
        ee, tgt = self.ee_pos, self.target_pos
        if self.obs_mode == "legacy":
            parts = [np.cos(q), np.sin(q), qd / 10.0, ee, tgt, tgt - ee]
        else:
            servo_err = (self._q_target - q) if self.control_mode == "position" else np.zeros(self.nu)
            parts = [np.cos(q), np.sin(q), qd / 10.0, servo_err, self._prev_action,
                     ee, tgt, 10.0 * (tgt - ee)]
        return np.concatenate(parts).astype(np.float32)

    # --------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Options (all optional, used by the benchmark for reproducibility):
        ``target`` (3,) goal position, ``qpos`` (3,) start configuration."""
        super().reset(seed=seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)
        q0 = options.get("qpos")
        self.data.qpos[: self.nq] = self._sample_start() if q0 is None else np.asarray(q0, float)
        if self.reward_mode == "legacy" and q0 is None:
            self.data.qvel[: self.nv] = self.np_random.uniform(-0.05, 0.05, size=self.nv)
        tgt = options.get("target")
        self.data.mocap_pos[0] = self._sample_target() if tgt is None else np.asarray(tgt, float)

        self._q_target = self.data.qpos[: self.nq].copy()
        if self.control_mode == "position":
            self.data.ctrl[: self.nu] = self._q_target
        self._prev_action = np.zeros(self.nu)
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        self._reached_at = None
        info = {"distance": float(np.linalg.norm(self.target_pos - self.ee_pos))}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.nu), -1.0, 1.0)

        if self.control_mode == "position":
            q = self.data.qpos[: self.nq]
            tgt = self._q_target + a * self.max_step
            tgt = np.clip(tgt, q - 0.3, q + 0.3)            # no wind-up against contact
            self._q_target = np.clip(tgt, self.jnt_lo, self.jnt_hi)
            self.data.ctrl[: self.nu] = self._q_target
        else:
            self.data.ctrl[: self.nu] = a

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        d = float(np.linalg.norm(self.target_pos - self.ee_pos))
        success = d < self.success_threshold
        self._elapsed_steps += 1
        if success and self._reached_at is None:
            self._reached_at = self._elapsed_steps

        if self.reward_mode == "legacy":
            reward = (-d + (1.0 if success else 0.0)
                      - 0.01 * float(np.sum(a * a))
                      - 0.001 * float(np.sum(np.square(self.data.qvel))))
        else:
            reward = (-d
                      + self.precision_weight * (1.0 - np.tanh(d / self.precision_scale))
                      - self.action_cost * float(np.sum(a * a))
                      - self.action_rate_cost * float(np.sum(np.square(a - self._prev_action))))
        self._prev_action = a

        terminated = not np.isfinite(self.data.qpos).all()
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = {
            "distance": d,
            "is_success": bool(success),
            "is_success_2cm": bool(d < 0.02),
            "time_to_reach": None if self._reached_at is None else self._reached_at * self.dt,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ---------------------------------------------------------------- render

    def render(self):
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer  # (a bare `import mujoco.viewer` here would
            #                                            make `mujoco` local and break rgb_array)
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False
                )
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self._renderer.update_scene(self.data, camera=-1)
            return self._renderer.render()
        return None

    @property
    def viewer_is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def wait_for_viewer(self, fps: int = 50) -> None:
        """Hold the viewer window open until the user closes it."""
        if self._viewer is None:
            return
        print("Viewer is open -- close the window (or press Ctrl+C) to exit.")
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
