// Small helpers shared by the C++ ROS 2 nodes.
#pragma once

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>

#include <string>

#include "mujoco_arm_cpp/arm_model.hpp"
#include "mujoco_arm_cpp/policies.hpp"

namespace mujoco_arm::ros {

constexpr const char* kFrame = "world";

/// <install>/share/mujoco_arm_cpp
std::string share_dir();
/// `value` if non-empty, else share_dir()/default_relative.
std::string or_share(const std::string& value, const std::string& default_relative);
/// MJCF models, tuned parameters -- overridable by parameters of the same name.
PolicyPaths policy_paths(rclcpp::Node& node);

geometry_msgs::msg::Point to_point(const Vec3& v);
geometry_msgs::msg::Vector3 to_vector3(const Vec3& v);
rclcpp::QoS reliable(size_t depth = 10);

}  // namespace mujoco_arm::ros
