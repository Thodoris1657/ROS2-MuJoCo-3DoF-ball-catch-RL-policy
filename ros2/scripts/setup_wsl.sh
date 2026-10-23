#!/usr/bin/env bash
# One-time setup on Ubuntu 24.04 (WSL2 or native) with ROS 2 Jazzy already installed.
#
#   bash ros2/scripts/setup_wsl.sh            # C++ bridge + Python env (MuJoCo, gymnasium)
#   bash ros2/scripts/setup_wsl.sh --sb3      # also stable-baselines3 + torch (training, export)
#
# What it does: installs the ROS / C++ packages the bridge needs (the nodes are C++ on the
# MuJoCo C API), creates a Python venv that can still see ROS (--system-site-packages),
# installs MuJoCo into it -- the C++ build links the libmujoco.so shipped in that wheel, so
# the nodes run exactly the physics the Python training used -- and installs this repo's
# `mujoco_arm_rl` package in editable mode (training, tuning, benchmarks, policy export).
set -euo pipefail

ROS_SETUP=/opt/ros/jazzy/setup.bash
VENV="${MUJOCO_ROS_VENV:-$HOME/venvs/mujoco-ros}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -f "$ROS_SETUP" ] || { echo "ROS 2 Jazzy not found at $ROS_SETUP -- install ros-jazzy-desktop first."; exit 1; }

sudo apt-get update
sudo apt-get install -y python3-venv ros-dev-tools build-essential cmake \
    nlohmann-json3-dev libgtest-dev \
    ros-jazzy-rclcpp ros-jazzy-rclcpp-action ros-jazzy-tf2-ros \
    ros-jazzy-robot-state-publisher ros-jazzy-rviz2

python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# numpy<2: ROS 2 Jazzy's own packages are built against numpy 1.x; do not shadow it.
python -m pip install "numpy<2" mujoco gymnasium pyyaml
if [ "${1:-}" = "--sb3" ]; then
    python -m pip install stable-baselines3 sb3-contrib torch --extra-index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e "$REPO" --no-deps

echo
echo "Done. Next:  source $REPO/ros2/scripts/env.sh && cd $REPO/ros2 && colcon build && source $REPO/ros2/scripts/env.sh"
