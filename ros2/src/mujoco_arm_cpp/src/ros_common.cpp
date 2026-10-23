#include "mujoco_arm_cpp/ros_common.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "mujoco_arm_cpp/catch_core.hpp"

namespace mujoco_arm::ros {

std::string share_dir() { return ament_index_cpp::get_package_share_directory(PKG_NAME); }

std::string or_share(const std::string& value, const std::string& rel) {
  return value.empty() ? share_dir() + "/" + rel : value;
}

PolicyPaths policy_paths(rclcpp::Node& node) {
  auto get = [&](const std::string& name) {
    if (!node.has_parameter(name)) node.declare_parameter<std::string>(name, "");
    return node.get_parameter(name).as_string();
  };
  PolicyPaths p;
  p.catch_model = or_share(get("catch_model_path"), "assets/arm3_ballcatch.xml");
  p.reach_model = or_share(get("reach_model_path"), "assets/arm3.xml");
  p.catch_tuned = or_share(get("catch_tuned_path"), "config/catch_tuned.json");
  // the simulator's env_kwargs, so the controllers use the same rate limit / control rate
  p.catch_config = catch_config_from_json(get("catch_env_kwargs").empty() ? "{}" : get("catch_env_kwargs"));
  p.reach_config = reach_config_from_json(get("reach_env_kwargs").empty() ? "{}" : get("reach_env_kwargs"));
  return p;
}

geometry_msgs::msg::Point to_point(const Vec3& v) {
  geometry_msgs::msg::Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

geometry_msgs::msg::Vector3 to_vector3(const Vec3& v) {
  geometry_msgs::msg::Vector3 p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

rclcpp::QoS reliable(size_t depth) { return rclcpp::QoS(depth).reliable(); }

}  // namespace mujoco_arm::ros
