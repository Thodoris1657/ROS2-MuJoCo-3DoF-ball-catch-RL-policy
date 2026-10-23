# ROS 2 integration (C++)

The MuJoCo arm, its controllers and its trained RL policies as ordinary **C++ ROS 2 nodes**
(rclcpp, Jazzy): simulator, controller, visualisation and our code are separate processes
that talk over topics, services and actions — the way they would on a real robot. Two tasks:

- **reach** — `MujocoArm3Reach-v1`: put the tip on a target and hold it;
- **ball catch** — pick a ball off a cube, drop it, catch it at the top of the rebound under
  Earth / Mars / Moon gravity (main README, *Ball catch*).

Python stays where it is best — training, tuning, analysis. Everything that runs *on the
robot side* is C++: the simulators are ports onto the MuJoCo C API, the controllers are ports
of the Python ones, and RL policies run from a JSON export with a small built-in MLP runtime
(no Python, no PyTorch at run time).

```
                         /arm/catch/start (StartCatch srv)   ┌──────────────┐  action:
   ┌──────────────────┐◄──────────────────────────────────── │ catch_server │  /arm/catch_ball
   │ catch_sim_node   │   /arm/action                         └──────▲───────┘  (CatchBall)
   │ (CatchSim, C++)  │◄──────────────────────┐                      │ catch_status
   │                  │ /arm/observation      │   ┌──────────────┐   │
   │                  ├───────────────────────┼──►│ policy_node  │ catch_tuned | catch_model |
   └──┬───┬───┬───────┘ /arm/catch_status ────┘   │ (C++)        │ catch_hold | policy.json
      │   │   └─► /arm/markers, /tf world→ball ─► RViz           │ (pure or residual RL)
      │   └─────► /arm/ball, /arm/ee_pose         └──────────────┘
      └─────────► /arm/joint_states ─► robot_state_publisher ─► /tf
```

The reach side is the same picture with `sim_node`, `reach_server` (`ReachTarget`),
`eval_client`, `demo_targets` and the `oracle` policy.

| Package | Contents |
| --- | --- |
| `mujoco_arm_interfaces` | `ArmObservation`, `ArmStatus`, `CatchStatus`; `ResetEpisode.srv`, `StartCatch.srv`; `ReachTarget.action`, `CatchBall.action` |
| `mujoco_arm_cpp` | ROS-free core library (`arm_model`, `catch_sim`, `catch_controller`, `reach_sim`, `policies`, `catch_core`), 9 nodes, 2 tools (`catch_bench`, `catch_eval_params`), launch files, URDF, RViz configs, GoogleTest suite |

---

## Build (Ubuntu 24.04 + ROS 2 Jazzy, e.g. WSL2)

Keep the repo in the Linux filesystem (`~/`), not under `/mnt/c/` — builds are much faster.

```bash
cd ~/mujoco-arm-rl                          # this repo
bash ros2/scripts/setup_wsl.sh              # once; add --sb3 for training / policy export
source ros2/scripts/env.sh
cd ros2 && colcon build && cd ..
source ros2/scripts/env.sh                  # again, to pick up install/
```

**MuJoCo comes from the `mujoco` pip wheel** of the venv (`env.sh` exports `MUJOCO_PYTHON`):
CMake asks that Python where the wheel lives and links its `libmujoco.so`, so the C++ nodes
run *exactly* the physics library the Python training used. To use a MuJoCo release tarball
instead: `colcon build --cmake-args -DMUJOCO_INCLUDE_DIR=... -DMUJOCO_LIBRARY=.../libmujoco.so`.

The core library and its tests also build **without ROS**:

```bash
cmake -S ros2/src/mujoco_arm_cpp -B build_cpp && cmake --build build_cpp -j && ./build_cpp/test_core
```

---

## Ball catch

```bash
ros2 launch mujoco_arm_cpp catch.launch.py demo:=true             # RViz; cycles Earth -> Mars -> Moon
ros2 launch mujoco_arm_cpp catch.launch.py gravity:=moon rtf:=0.5   # slow motion
ros2 launch mujoco_arm_cpp catch.launch.py policy:=$PWD/runs/catch_residual/best_model.json

# one trial, with feedback
ros2 action send_goal /arm/catch_ball mujoco_arm_interfaces/action/CatchBall \
    "{gravity: mars, start_from_pick: true}" --feedback

# pin everything and start straight from the release
ros2 service call /arm/catch/start mujoco_arm_interfaces/srv/StartCatch \
    "{gravity: moon, restitution: 0.8, use_drop: true, drop: {x: 0.3, y: 0.0, z: 0.47}, seed: 1}"

ros2 param set /arm/catch_sim_node gravity mars      # default for the next trials
ros2 param set /arm/policy_node policy catch_hold     # watch it miss
ros2 topic echo /arm/catch_status --field outcome

# frozen benchmark through ROS (lock-step, headless) -- 300 trials in ~20 s
ros2 launch mujoco_arm_cpp catch.launch.py eval:=true rtf:=100 rviz:=false
ros2 launch mujoco_arm_cpp catch.launch.py eval:=true rtf:=100 rviz:=false \
    set:=$PWD/benchmarks/catch_set_v1_stress.json
```

| Argument (`catch.launch.py`) | Default | |
| --- | --- | --- |
| `gravity` | `earth` | `earth`, `mars`, `moon` or a number (m/s²) — default for trials that do not name one |
| `policy` | `catch_tuned` | `catch_tuned`, `catch_model`, `catch_hold`, or an exported `.json` network |
| `mode` | `realtime` | `realtime` or `lockstep` (`eval:=true` forces lockstep) |
| `rtf` | `1.0` | real-time factor (`0.25` = slow motion) |
| `demo` | `false` | send `CatchBall` goals forever, cycling the three gravities |
| `auto_start` | `false` | start one full trial at start-up |
| `env_kwargs` | `{}` | simulator settings (JSON, BallCatchEnv keywords); `eval:=true` takes them from the set |
| `rviz` / `rsp` | `true` / `true` | RViz / robot_state_publisher |
| `eval`, `set`, `per_gravity` | `false`, nominal set, `100` | replay a frozen benchmark through ROS, then exit |

| | Name | Type | |
| --- | --- | --- | --- |
| in | `action` | `std_msgs/Float32MultiArray` | 3 normalised joint-target deltas |
| out | `observation` | `ArmObservation` | the 34 floats the catch policies use (`data[33]` = ball in flight) |
| out | `catch_status` | `CatchStatus` | phase, gravity, restitution, ball state, outcome, catch time / height error, impact speed |
| out | `ball`, `ee_pose`, `joint_states`, `markers`, `/tf` | | ball centre, cup pose, joints, RViz markers, `world → ball` |
| srv | `catch/start` | `StartCatch` | new trial; gravity, restitution, drop point, release puff and seed can be pinned |
| srv | `pause` | `std_srvs/SetBool` | |
| action | `catch_ball` | `CatchBall` | goal: gravity (+ optional pins); feedback: phase, ball height, bounces; result: outcome, catch errors |

## Reach

```bash
ros2 launch mujoco_arm_cpp arm_rl.launch.py demo:=true             # IK oracle + RViz
ros2 launch mujoco_arm_cpp arm_rl.launch.py policy:=$PWD/runs/sb3_v1/best_model.json
ros2 action send_goal /arm/reach_target mujoco_arm_interfaces/action/ReachTarget \
    "{target: {x: 0.3, y: 0.1, z: 0.35}}" --feedback
ros2 launch mujoco_arm_cpp arm_rl.launch.py eval:=true rtf:=10 rviz:=false   # frozen reach benchmark
```

Topics, services and the `ReachTarget` action are unchanged from the Python bridge:
`action`, `joint_command`, `target` in; `joint_states`, `observation` (24 floats), `status`,
`ee_pose`, `markers`, `/tf world→target` out; `reset`, `pause`; `reach_target`.
The C++ port implements the v1 task (position servo, 25 Hz); the legacy v0 torque task is not
ported.

## RL policies in C++

```bash
python scripts/export_policy.py runs/catch_residual/best_model.zip --check   # -> best_model.json
ros2 param set /arm/policy_node policy $PWD/runs/catch_residual/best_model.json
```

The exporter writes the deterministic actor (SB3 SAC / TQC / TD3 / PPO, or the repo's TorchRL
PPO with its observation normalisation) as plain weights; `--check` replays 200 observations
through the original network and the export and fails above 1e-5. A **residual** policy
carries its base controller and scale; the node runs `clip(base(obs) + scale * net([obs,
base(obs)]))`, exactly as in training (`src/mujoco_arm_rl/residual.py`).

---

## Design notes

**Lock-step without off-by-one.** In `lockstep` the simulator steps only when a fresh action
arrives and re-publishes the observation while it waits. `policy_node` stamps every action
with the `seq` of the observation it answers (`layout.dim[0].label = "seq"`,
`layout.data_offset = seq`), and the simulator drops a stamped action that does not answer
its *current* observation. Without this, a late reply to the last observation of the previous
trial was taken as the first action of the next one — shifting the whole trial by 10 ms and
changing 1 % of Earth outcomes. (Unstamped actions, e.g. from the CLI, are still accepted.)

**Realtime safety.** In `realtime` an action older than `action_timeout` (0.25 s) becomes
"hold" (zero), so a crashed controller cannot send the arm running.

**The C++ is the Python, checked.** `test_parity.cpp` replays roll-outs recorded by
`scripts/make_cpp_golden.py` — five catch trials (three gravities from the release, one full
pick-and-carry), ten reach episodes with the oracle, and an exported SAC network — and
requires every observation and action to agree within 1e-5, the same outcome, and the same
catch timing to 1e-9 s. Only the random-number generator differs (`std::mt19937_64` vs
NumPy), so randomised trials agree statistically, pinned ones exactly.

**Frozen benchmark through ROS = offline C++, exactly.** With the seq stamping, the 300 nominal
and 300 stress trials replayed through `catch_sim_node` ⇄ `policy_node` over DDS give the
identical per-gravity success counts as `catch_bench` (no ROS) on the same seeds.

## Tests

```bash
cd ros2 && colcon test --packages-select mujoco_arm_cpp && colcon test-result --verbose
# or, ROS-free:   ./build_cpp/test_core
```

`test_core` (16 tests): IK vs MuJoCo FK, numpy-compatible angle wrapping, gravity-independent
restitution, observation layout and pinned resets, drop-point clearance, hold/chase baselines,
the controller under all three gravities, a full pick-to-catch trial, the policy factory,
env-kwargs parsing, reach bookkeeping, and the three parity tests above.

## Troubleshooting

- **`MuJoCo not found`** at configure time — the active Python has no `mujoco`:
  `source ros2/scripts/env.sh` (venv) or `pip install mujoco`, or set `MUJOCO_PYTHON`.
- **`error while loading shared libraries: libmujoco.so`** — the RPATH points at the wheel
  you built against; rebuild after moving/recreating the venv.
- **The arm does not move** — nothing is driving it: `ros2 node list` should show
  `policy_node`. In `lockstep` the simulator waits for an action forever.
- **`policy ... .zip: export it first`** — the C++ node runs exported networks:
  `python scripts/export_policy.py <checkpoint>.zip`.
- **No RViz window** — WSLg needs Windows 11 or a recent Windows 10 with WSL ≥ 1.0.
