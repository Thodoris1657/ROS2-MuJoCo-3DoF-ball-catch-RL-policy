#!/usr/bin/env bash
# Source this in every new terminal:   source ros2/scripts/env.sh
# (sourced, not executed -- so no `set -e` here)
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VENV="${MUJOCO_ROS_VENV:-$HOME/venvs/mujoco-ros}"
source /opt/ros/jazzy/setup.bash
[ -f "$_VENV/bin/activate" ] && source "$_VENV/bin/activate"
[ -f "$_HERE/../install/setup.bash" ] && source "$_HERE/../install/setup.bash"
export MUJOCO_ARM_RL_ROOT="$(cd "$_HERE/../.." && pwd)"
# the C++ build links the MuJoCo shipped in this Python's `mujoco` wheel (same physics as training)
export MUJOCO_PYTHON="$(command -v python3)"
echo "ROS 2 $ROS_DISTRO ready; repo: $MUJOCO_ARM_RL_ROOT"
