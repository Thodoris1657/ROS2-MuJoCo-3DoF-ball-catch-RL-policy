"""Launch the MuJoCo arm with an RL policy, RViz and the ReachTarget action server.

    ros2 launch mujoco_arm_cpp arm_rl.launch.py                       # IK oracle + RViz
    ros2 launch mujoco_arm_cpp arm_rl.launch.py policy:=/path/to/runs/sb3_v1/best_model.json
    ros2 launch mujoco_arm_cpp arm_rl.launch.py demo:=true             # new random target every 4 s
    ros2 launch mujoco_arm_cpp arm_rl.launch.py eval:=true rtf:=10 policy:=/path/to/best_model.json

(.json = the checkpoint exported by scripts/export_policy.py; the C++ node runs it directly)

``eval:=true`` runs the frozen 100-case benchmark through ROS (lock-step, headless
RViz off) and shuts everything down when it is done.
"""

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


def _env_config_for(policy: str, explicit: str) -> str:
    """env_config.json beside a checkpoint (or one folder up), like the repo's policies.py."""
    if explicit:
        return explicit
    path = Path(policy).expanduser()
    if path.suffix in (".zip", ".pt", ".json") and path.exists():
        for folder in (path.parent, path.parent.parent):
            if (folder / "env_config.json").exists():
                return str(folder / "env_config.json")
    return ""


def _setup(context, *_):
    share = Path(get_package_share_directory("mujoco_arm_cpp"))
    params = str(share / "config" / "arm_sim.yaml")
    urdf = (share / "urdf" / "arm3.urdf").read_text()

    policy = LaunchConfiguration("policy").perform(context)
    evaluating = _bool(context, "eval")
    mode = "lockstep" if evaluating else LaunchConfiguration("mode").perform(context)
    env_config = _env_config_for(policy, LaunchConfiguration("env_config").perform(context))
    rtf = float(LaunchConfiguration("rtf").perform(context))

    sim = Node(
        package="mujoco_arm_cpp", executable="sim_node", namespace=NS, name="sim_node",
        output="screen",
        parameters=[params, {
            "mode": mode, "episodic": evaluating, "real_time_factor": rtf,
            "env_config": env_config,
            "seed": int(LaunchConfiguration("seed").perform(context)),
        }])
    policy_node = Node(
        package="mujoco_arm_cpp", executable="policy_node", namespace=NS, name="policy_node",
        output="screen", parameters=[params, {"policy": policy}])
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", namespace=NS,
        name="robot_state_publisher", output="screen",
        parameters=[{"robot_description": urdf}])

    actions = [sim, policy_node] + ([rsp] if _bool(context, "rsp") else [])
    if evaluating:
        client = Node(
            package="mujoco_arm_cpp", executable="eval_client", namespace=NS, name="eval_client",
            output="screen",
            parameters=[params, {"episodes": int(LaunchConfiguration("episodes").perform(context)),
                                 "control_hz": 25.0}])
        actions += [client, RegisterEventHandler(OnProcessExit(
            target_action=client, on_exit=[EmitEvent(event=Shutdown(reason="evaluation finished"))]))]
        return actions

    if _bool(context, "demo"):
        actions.append(Node(
            package="mujoco_arm_cpp", executable="demo_targets", namespace=NS, name="demo_targets",
            output="screen"))
    actions.append(Node(
        package="mujoco_arm_cpp", executable="reach_server", namespace=NS, name="reach_server",
        output="screen", parameters=[params]))
    if _bool(context, "rviz"):
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", str(share / "rviz" / "arm.rviz")]))
    return actions


def generate_launch_description() -> LaunchDescription:
    def arg(name, default, doc):
        return DeclareLaunchArgument(name, default_value=default, description=doc)

    return LaunchDescription([
        arg("policy", "oracle", "oracle | random | exported .json network (scripts/export_policy.py)"),
        arg("mode", "realtime", "realtime | lockstep (eval:=true forces lockstep)"),
        arg("rtf", "1.0", "real-time factor of the simulation (10 = ten times faster)"),
        arg("env_config", "", "env_config.json; default: the one beside the checkpoint, if any"),
        arg("rviz", "true", "start RViz"),
        arg("rsp", "true", "start robot_state_publisher (arm TF for RViz)"),
        arg("seed", "-1", "simulator RNG seed (-1 = random)"),
        arg("demo", "false", "publish a new random reachable target every few seconds"),
        arg("eval", "false", "run the frozen 100-case benchmark through ROS, then exit"),
        arg("episodes", "100", "number of benchmark episodes when eval:=true"),
        OpaqueFunction(function=_setup),
    ])
