cmake_minimum_required(VERSION 3.8)
project(mujoco_arm_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmObservation.msg"
  "msg/ArmStatus.msg"
  "msg/CatchStatus.msg"
  "srv/ResetEpisode.srv"
  "srv/StartCatch.srv"
  "action/ReachTarget.action"
  "action/CatchBall.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
