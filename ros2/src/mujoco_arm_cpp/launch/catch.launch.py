"""Launch the ball-catch simulator (Earth / Mars / Moon) with a catch policy, RViz and the
CatchBall action server.

    ros2 launch mujoco_arm_cpp catch.launch.py                         # tuned controller, Earth
    ros2 launch mujoco_arm_cpp catch.launch.py gravity:=moon demo:=true   # demo cycles all three
    ros2 launch mujoco_arm_cpp catch.launch.py policy:=$HOME/mujoco-arm-rl/runs/catch_residual/best_model.json
    ros2 launch mujoco_arm_cpp catch.launch.py eval:=true rtf:=50 rviz:=false   # frozen benchmark through ROS

Then, e.g.:
    ros2 action send_goal /arm/catch_ball mujoco_arm_interfaces/action/CatchBall "{gravity: mars, start_from_pick: true}" --feedback
"""

import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NS = "arm"


def _bool(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).strip().lower() in ("true", "1", "yes")


def _setup(context, *_):
    share = Path(get_package_share_directory("mujoco_arm_cpp"))
    params = str(share / "config" / "catch_sim.yaml")
    urdf = (share / "urdf" / "arm3.urdf").read_text()

    evaluating = _bool(context, "eval")
    mode = "lockstep" if evaluating else LaunchConfiguration("mode").perform(context)
    policy = LaunchConfiguration("policy").perform(context)
    bench = LaunchConfiguration("set").perform(context) or str(share / "benchmarks" / "catch_set_v1.json")
    env_kwargs = LaunchConfiguration("env_kwargs").perform(context)
    if evaluating:   # the benchmark file pins the simulator settings it was generated with
        env_kwargs = json.dumps(json.loads(Path(bench).read_text()).get("env_kwargs", {}))

    sim = Node(
        package="mujoco_arm_cpp", executable="catch_sim_node", namespace=NS, name="catch_sim_node",
        output="screen",
        parameters=[params, {
            "mode": mode, "gravity": LaunchConfiguration("gravity").perform(context),
            "real_time_factor": float(LaunchConfiguration("rtf").perform(context)),
            "seed": int(LaunchConfiguration("seed").perform(context)),
            "auto_start": _bool(context, "auto_start") and not evaluating,
            "env_kwargs": env_kwargs,
        }])
    policy_node = Node(
        package="mujoco_arm_cpp", executable="policy_node", namespace=NS, name="policy_node",
        output="screen", parameters=[params, {"policy": policy, "catch_env_kwargs": env_kwargs}])
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", namespace=NS,
        name="robot_state_publisher", output="screen", parameters=[{"robot_description": urdf}])
    actions = [sim, policy_node] + ([rsp] if _bool(context, "rsp") else [])

    if evaluating:
        client = Node(
            package="mujoco_arm_cpp", executable="catch_eval_client", namespace=NS,
            name="catch_eval_client", output="screen",
            parameters=[params, {"per_gravity": int(LaunchConfiguration("per_gravity").perform(context)),
                                 "set": bench}])
        actions += [client, RegisterEventHandler(OnProcessExit(
            target_action=client, on_exit=[EmitEvent(event=Shutdown(reason="evaluation finished"))]))]
        return actions

    actions.append(Node(
        package="mujoco_arm_cpp", executable="catch_server", namespace=NS, name="catch_server",
        output="screen"))
    if _bool(context, "demo"):
        actions.append(Node(
            package="mujoco_arm_cpp", executable="catch_demo", namespace=NS, name="catch_demo",
            output="screen", parameters=[params]))
    if _bool(context, "rviz"):
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", str(share / "rviz" / "catch.rviz")]))
    return actions


def generate_launch_description() -> LaunchDescription:
    def arg(name, default, doc):
        return DeclareLaunchArgument(name, default_value=default, description=doc)

    return LaunchDescription([
        arg("gravity", "earth", "earth | mars | moon | a number in m/s^2 (default for new trials)"),
        arg("policy", "catch_tuned", "catch_tuned | catch_model | catch_hold | exported .json network"),
        arg("mode", "realtime", "realtime | lockstep (eval:=true forces lockstep)"),
        arg("rtf", "1.0", "real-time factor (0.25 = slow motion)"),
        arg("rviz", "true", "start RViz"),
        arg("rsp", "true", "start robot_state_publisher (arm TF for RViz)"),
        arg("set", "", "eval:=true -- benchmark file (default: share/benchmarks/catch_set_v1.json)"),
        arg("env_kwargs", "{}", "JSON for the simulator, e.g. '{\"obs_pos_noise\": 0.003}' (eval: from the set)"),
        arg("seed", "-1", "simulator RNG seed (-1 = random)"),
        arg("demo", "false", "send CatchBall goals forever, cycling Earth -> Mars -> Moon"),
        arg("auto_start", "false", "start one full trial at start-up"),
        arg("eval", "false", "run the frozen catch benchmark through ROS, then exit"),
        arg("per_gravity", "100", "benchmark trials per gravity when eval:=true"),
        OpaqueFunction(function=_setup),
    ])
