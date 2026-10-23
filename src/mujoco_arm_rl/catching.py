"""Model-based catch controller for ``BallCatchEnv`` (the reference the RL policy is compared to).

It works from the 34-float observation only -- noisy ball position / velocity, the
gravity value and the "bounced" flag -- plus the arm's own kinematics (IK), so it
could run as a separate process exactly like the learned policy.

Pipeline, every control step (100 Hz):

1. **Estimate** the ball's ballistic state. Since the last event (release or bounce)
   the ball obeys ``p(t) = p0 + v0 t - g t^2 / 2 z``; a weighted least-squares fit of
   (p0, v0) to all noisy position *and* velocity samples of the current flight
   averages the sensor noise away (g is known: it is an input, Earth/Mars/Moon).
2. **Predict** the apex of the rebound.
   * before the bounce: time to impact, impact speed, ``v_out = e_prior * v_in``
     with a prior restitution (the true one is unknown, 0.65-0.85);
   * after the bounce: from the fitted upward velocity (exact up to noise).
3. **Plan** the cup position: directly above the apex, ``catch_offset`` higher
   than touching, so the ball arrives with ~zero speed.
4. **Stay above the falling ball**: the tip may not go lower than where the
   ball will be ``lookahead`` seconds from now plus ``clearance`` -- this is what
   stops the arm from hitting the ball from above on the Moon.
5. **Soften** (optional): near the apex, feed the ball's remaining vertical velocity
   forward into the cup target (``vz_match``), reducing the impact speed.
6. **Track**: IK on the planned tip point, then the same rate-limited joint-target
   action every policy in this repo uses.

The free parameters live in ``CatchParams``; ``scripts/tune_catch.py`` optimises them
per gravity with the cross-entropy method on randomised trials.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_arm_rl.envs.ballcatch import BALL_R, CONTACT_DIST, GRAVITIES, gravity_name


@dataclass
class CatchParams:
    e_prior: float = 0.75        # restitution assumed until the rebound is observed
    catch_offset: float = 0.002  # m, cup height above "touching the ball at its apex"
    lookahead: float = 0.05      # s, how far ahead the falling ball is projected
    clearance: float = 0.02      # m, minimum gap kept above the falling ball
    vz_match: float = 0.0        # 0..1, fraction of the ball's v_z fed into the cup target
    overshoot: float = 0.0       # 0..1, aim past the goal by this fraction of the tip error (servo lag)
    liftoff: float = 0.008       # m, rebound samples are trusted once the ball is this far off the floor

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# Bounds used by the optimiser (and to clip loaded files).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "e_prior": (0.55, 0.95),
    "catch_offset": (-0.008, 0.008),
    "lookahead": (0.0, 0.15),
    "clearance": (0.0, 0.05),
    "vz_match": (0.0, 1.0),
    "overshoot": (0.0, 1.0),
    "liftoff": (0.0, 0.012),
}

TUNED_PATH = Path(__file__).resolve().parents[2] / "configs" / "catch_tuned.json"


def load_tuned(path: str | Path = TUNED_PATH) -> dict[str, CatchParams]:
    """{gravity name: params} from the optimiser's output (missing file -> defaults)."""
    path = Path(path)
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    out = {}
    for name, p in blob.get("params", {}).items():
        out[name] = CatchParams(**{k: v for k, v in p.items() if k in CatchParams.names()})
    return out


class BallisticFit:
    """Least-squares (p0, v0) of a ballistic flight from noisy position + velocity samples."""

    def __init__(self, g: float, pos_std: float = 0.001, vel_std: float = 0.02) -> None:
        self.g = g
        self.wp = 1.0 / pos_std
        self.wv = 1.0 / vel_std
        self.t: list[float] = []
        self.p: list[np.ndarray] = []
        self.v: list[np.ndarray] = []

    def clear(self) -> None:
        self.t.clear()
        self.p.clear()
        self.v.clear()

    def add(self, t: float, p: np.ndarray, v: np.ndarray) -> None:
        self.t.append(t)
        self.p.append(np.asarray(p, float))
        self.v.append(np.asarray(v, float))

    def __len__(self) -> int:
        return len(self.t)

    def state_at(self, t_now: float) -> tuple[np.ndarray, np.ndarray]:
        """Fitted (position, velocity) at ``t_now``; gravity removed analytically."""
        t = np.asarray(self.t) - t_now                     # <= 0
        P = np.asarray(self.p)
        V = np.asarray(self.v)
        gz = np.array([0.0, 0.0, self.g])
        # remove the known gravity term: p + g t^2/2 z = p0 + v0 t ;  v + g t z = v0
        Pc = P + 0.5 * np.outer(t * t, gz)
        Vc = V + np.outer(t, gz)
        # per axis: unknowns (p0, v0); rows: wp*[1, t] -> wp*Pc ; wv*[0, 1] -> wv*Vc
        n = len(t)
        A = np.zeros((2 * n, 2))
        A[:n, 0] = self.wp
        A[:n, 1] = self.wp * t
        A[n:, 1] = self.wv
        B = np.vstack([self.wp * Pc, self.wv * Vc])         # (2n, 3)
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        return sol[0], sol[1]


class CatchController:
    """Observation -> action. Call ``reset()`` at every episode start."""

    def __init__(self, params: CatchParams | dict[str, CatchParams] | None = None,
                 pos_std: float = 0.001, vel_std: float = 0.02) -> None:
        if params is None:
            params = load_tuned() or CatchParams()
        self.params_by_g = params if isinstance(params, dict) else None
        self.default = params if isinstance(params, CatchParams) else CatchParams()
        self.pos_std, self.vel_std = pos_std, vel_std
        self.reset()

    def params_for(self, g: float) -> CatchParams:
        if self.params_by_g:
            name = gravity_name(g)
            if name in self.params_by_g:
                return self.params_by_g[name]
            # nearest tuned gravity
            near = min(self.params_by_g, key=lambda k: abs(GRAVITIES.get(k, 1e9) - g))
            return self.params_by_g[near]
        return self.default

    def reset(self) -> None:
        self.t = 0.0
        self.fit: BallisticFit | None = None
        self.bounced = False
        self.last_plan: dict[str, Any] = {}

    # ------------------------------------------------------------ prediction

    def predict(self, p: np.ndarray, v: np.ndarray, g: float, e: float) -> dict[str, Any]:
        """Apex (time-from-now, point) of the coming rebound, from the ball state now."""
        if not self.bounced:
            h = p[2] - BALL_R
            v_in = float(np.sqrt(max(v[2] * v[2] + 2.0 * g * max(h, 0.0), 0.0)))
            t_imp = (v[2] + v_in) / g
            v_out = e * v_in
            t_apex = t_imp + v_out / g
            z_apex = BALL_R + v_out * v_out / (2.0 * g)
        else:
            t_imp = 0.0
            vz = max(float(v[2]), 0.0)
            t_apex = vz / g
            z_apex = float(p[2]) + vz * vz / (2.0 * g)
        xy = p[:2] + v[:2] * t_apex
        return dict(t_apex=t_apex, t_impact=t_imp, apex=np.array([xy[0], xy[1], z_apex]))

    # ----------------------------------------------------------------- act

    def __call__(self, obs: np.ndarray, env) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        g = float(obs[30]) * 9.81
        prm = self.params_for(g)
        tip = obs[15:18]
        bp, bv = obs[21:24], obs[24:27]
        bounced, holding = obs[31] > 0.5, obs[32] > 0.5
        if obs.shape[0] > 33 and obs[33] < 0.5 and not holding:
            # not in flight (scripted pick / carry): forget everything, wait for the release
            self.reset()
            return np.zeros(3)
        if self.fit is None:
            self.fit = BallisticFit(g, max(self.pos_std, 1e-4), max(self.vel_std, 1e-3))
        if bounced and not self.bounced:
            self.bounced = True
            self.fit.clear()
        if holding:
            self.t += env.dt
            return np.zeros(3)

        # a ball that is suddenly far from where the current flight puts it means a new
        # trial started without a "not in flight" observation in between: start over
        if len(self.fit) >= 2 and not self.bounced:
            p_pred, _ = self.fit.state_at(self.t)
            if float(np.linalg.norm(p_pred - bp)) > 0.05:
                self.reset()
                self.fit = BallisticFit(g, max(self.pos_std, 1e-4), max(self.vel_std, 1e-3))
        elif self.bounced and not bounced:          # the bounce flag went back to 0: new trial
            self.reset()
            self.fit = BallisticFit(g, max(self.pos_std, 1e-4), max(self.vel_std, 1e-3))

        # ballistic samples only: skip the few ms the ball is squashed on the floor
        if not (self.bounced and (bp[2] < BALL_R + prm.liftoff or bv[2] < -0.05 and len(self.fit) == 0)):
            self.fit.add(self.t, bp, bv)
        if len(self.fit) >= 2:
            p_hat, v_hat = self.fit.state_at(self.t)
        else:
            p_hat, v_hat = bp, bv

        plan = self.predict(p_hat, v_hat, g, prm.e_prior)
        goal = plan["apex"] + np.array([0.0, 0.0, CONTACT_DIST + prm.catch_offset])

        # soften: ride along with the ball's remaining vertical motion near the apex
        if self.bounced and prm.vz_match > 0.0:
            goal[2] += prm.vz_match * max(v_hat[2], 0.0) * min(plan["t_apex"], 0.1)

        # never get below the falling ball (projected lookahead seconds ahead), and no
        # sideways motion while the ball is still right under the cup
        if not self.bounced:
            if tip[2] - bp[2] - CONTACT_DIST < prm.clearance:
                goal[:2] = tip[:2]
            tl = prm.lookahead
            z_future = p_hat[2] + v_hat[2] * tl - 0.5 * g * tl * tl
            floor_z = max(z_future, BALL_R) + CONTACT_DIST + prm.clearance
            goal[2] = max(goal[2], min(floor_z, p_hat[2] + CONTACT_DIST + prm.clearance))

        # servo-lag compensation: aim a little past the goal while far from it
        err = goal - tip
        n = float(np.linalg.norm(err))
        target = goal + prm.overshoot * err * min(1.0, 0.03 / max(n, 1e-9))

        # the commanded joint targets, recovered from the observation (q + servo error)
        q = np.arctan2(obs[3:6], obs[0:3])
        q_cmd = q + obs[9:12]
        q_goal = env.solve_ik(target, near=q_cmd)
        if q_goal is None:
            q_goal = env.solve_ik(goal, near=q_cmd)
        self.last_plan = dict(plan, goal=goal, target=target)
        self.t += env.dt
        if q_goal is None:
            return np.zeros(3)
        # scale the whole joint step (not per joint) so the tip keeps its direction
        step = (q_goal - q_cmd) / env.max_step
        return step / max(1.0, float(np.max(np.abs(step))))
