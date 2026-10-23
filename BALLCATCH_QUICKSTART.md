# Ball catch — what to type

Everything below is run from this folder, inside WSL2 Ubuntu (where your ROS 2 Jazzy lives).
Copy the folder into the Linux filesystem first (much faster than /mnt/c):

```bash
cp -r "/mnt/c/Users/theod/Desktop/MuJoCo/mujoco-arm-rl-ballcatch" ~/mujoco-arm-rl-ballcatch
cd ~/mujoco-arm-rl-ballcatch
```

## 1. Python setup (once)

```bash
python3 -m venv --system-site-packages ~/venvs/mujoco-ros      # same venv the ROS bridge uses
source ~/venvs/mujoco-ros/bin/activate
pip install "numpy<2" mujoco gymnasium pyyaml pytest pillow matplotlib
pip install stable-baselines3 torch --extra-index-url https://download.pytorch.org/whl/cpu   # RL + running the shipped residual .zip
pip install -e . --no-deps
```

## 2. Check it works (~1 min)

```bash
MUJOCO_GL=disable pytest -q                         # 34 tests should pass
```

## 3. Watch it (MuJoCo window)

```bash
python - <<'PY'
from mujoco_arm_rl.envs import BallCatchEnv
from mujoco_arm_rl.catch_eval import make_catch_policy, run_episode
pol = make_catch_policy("tuned")
for g in ("earth", "mars", "moon"):
    env = BallCatchEnv(gravity=g, start="pick", render_mode="human")
    r = run_episode(env, pol, seed=1)
    print(g, r.outcome); env.close()
PY
```

## 4. Benchmark (per gravity, ~2-5 min each)

```bash
python scripts/benchmark_catch.py                                          # hold / chase / model / tuned
python scripts/benchmark_catch.py --set benchmarks/catch_set_v1_stress.json
```

## 5. Videos + plot

```bash
MUJOCO_GL=egl python scripts/render_catch.py --policy tuned --out media    # needs ffmpeg: sudo apt install ffmpeg
```
-> `media/catch_all_gravities.mp4`, `media/catch_{earth,mars,moon}.mp4`, `media/catch_heights.png`

## 6. Optimise the controller (optional; the shipped configs/catch_tuned.json is the result)

```bash
python scripts/tune_catch.py --backend cpp --pop 24 --generations 16 --episodes 256 --validate 4000   # ~30 min, needs step 8 built
python scripts/tune_catch.py --gravity moon          # Python-only backend (slower, smaller search)
```

## 7. RL (optional, long on CPU)

```bash
python scripts/train_catch_sb3.py --residual 0.2 --timesteps 150000 --out runs/catch_residual   # ~1 h
python scripts/train_catch_sb3.py --timesteps 400000 --out runs/catch_sac                        # pure RL, ~3 h
python scripts/benchmark_catch.py --entry tuned --entry "rl=runs/catch_residual/best_model.zip"
python scripts/export_policy.py runs/catch_residual/best_model.zip --check   # -> best_model.json for C++
```
(The trained residual policy ships in `runs/catch_residual/`.)

## 8. ROS 2 (C++)

```bash
bash ros2/scripts/setup_wsl.sh                 # once (apt: nlohmann-json3-dev, libgtest-dev, ...)
source ros2/scripts/env.sh
cd ros2 && colcon build && cd ..
source ros2/scripts/env.sh

ros2 launch mujoco_arm_cpp catch.launch.py demo:=true                    # RViz, cycles Earth -> Mars -> Moon
ros2 launch mujoco_arm_cpp catch.launch.py gravity:=moon rtf:=0.5        # slow motion
ros2 launch mujoco_arm_cpp catch.launch.py policy:=$PWD/runs/catch_residual/best_model.json
```
In a second terminal (after `source ros2/scripts/env.sh`):
```bash
ros2 action send_goal /arm/catch_ball mujoco_arm_interfaces/action/CatchBall "{gravity: mars, start_from_pick: true}" --feedback
ros2 topic echo /arm/catch_status --field outcome
ros2 param set /arm/policy_node policy catch_hold          # watch it miss
```
Benchmarks through ROS (must match step 4):
```bash
ros2 launch mujoco_arm_cpp catch.launch.py eval:=true rtf:=100 rviz:=false
ros2 launch mujoco_arm_cpp catch.launch.py eval:=true rtf:=100 rviz:=false set:=$PWD/benchmarks/catch_set_v1_stress.json
ros2 launch mujoco_arm_cpp arm_rl.launch.py eval:=true rtf:=50 rviz:=false          # reach task, IK oracle
```
C++ tests: `cd ros2 && colcon test --packages-select mujoco_arm_cpp && colcon test-result --verbose`
Fast C++ benchmark, no ROS: `ros2/install/mujoco_arm_cpp/lib/mujoco_arm_cpp/catch_bench . catch_tuned`
