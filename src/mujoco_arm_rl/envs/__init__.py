"""Environment package. Importing it registers the Gymnasium environment IDs.

MujocoArm3Reach-v1   position-servo control at 25 Hz, reachable goals, shaped reward
MujocoArm3Reach-v0   the original torque-control task, kept for old checkpoints/ablations
MujocoArm3BallCatch-v0  pick a ball off a cube, drop it, catch it at the top of the
                        rebound -- Earth / Mars / Moon gravity (see ballcatch.py)

The env truncates its own episodes, so no TimeLimit wrapper is registered: that
keeps ``episode_seconds`` / ``control_hz`` overrides from fighting a fixed limit.
"""

from gymnasium.envs.registration import register, registry

from mujoco_arm_rl.envs.ballcatch import GRAVITIES, BallCatchEnv
from mujoco_arm_rl.envs.reach3 import (
    DEFAULT_MODEL_PATH,
    LEGACY_KWARGS,
    Reach3Env,
    analytic_ik,
)

ENV_ID = "MujocoArm3Reach-v1"
LEGACY_ENV_ID = "MujocoArm3Reach-v0"
CATCH_ENV_ID = "MujocoArm3BallCatch-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="mujoco_arm_rl.envs.reach3:Reach3Env")
if LEGACY_ENV_ID not in registry:
    register(id=LEGACY_ENV_ID, entry_point="mujoco_arm_rl.envs.reach3:Reach3Env",
             kwargs=dict(LEGACY_KWARGS))
if CATCH_ENV_ID not in registry:
    register(id=CATCH_ENV_ID, entry_point="mujoco_arm_rl.envs.ballcatch:BallCatchEnv")

__all__ = ["Reach3Env", "ENV_ID", "LEGACY_ENV_ID", "LEGACY_KWARGS",
           "DEFAULT_MODEL_PATH", "analytic_ik", "BallCatchEnv", "CATCH_ENV_ID", "GRAVITIES"]
