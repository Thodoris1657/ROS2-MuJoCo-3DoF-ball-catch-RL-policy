// demo_targets -- publish a new random *reachable* target every few seconds (C++).
#include <chrono>
#include <memory>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

namespace mros = mujoco_arm::ros;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("demo_targets");
  const double period = node->declare_parameter<double>("period", 4.0);
  const auto seed = node->declare_parameter<int64_t>("seed", 0);
  mujoco_arm::ReachSim sim(mros::or_share(node->declare_parameter<std::string>("model_path", ""), "assets/arm3.xml"));
  sim.reset(static_cast<uint64_t>(seed));
  auto pub = node->create_publisher<geometry_msgs::msg::PointStamped>("target", mros::reliable());
  auto timer = node->create_wall_timer(std::chrono::duration<double>(period), [&] {
    const auto p = sim.sample_target();
    geometry_msgs::msg::PointStamped m;
    m.header.frame_id = mros::kFrame;
    m.header.stamp = node->now();
    m.point = mros::to_point(p);
    pub->publish(m);
    RCLCPP_INFO(node->get_logger(), "new target (%.2f, %.2f, %.2f)", p[0], p[1], p[2]);
  });
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
