// catch_demo -- send CatchBall goals forever, cycling Earth -> Mars -> Moon (C++).
//
//   ros2 run mujoco_arm_cpp catch_demo --ros-args -r __ns:=/arm -p gravities:="[earth, moon]"
#include <chrono>
#include <memory>
#include <thread>

#include <mujoco_arm_interfaces/action/catch_ball.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using namespace std::chrono_literals;
using CatchBall = mujoco_arm_interfaces::action::CatchBall;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("catch_demo");
  const auto gravities = node->declare_parameter<std::vector<std::string>>(
      "gravities", std::vector<std::string>{"earth", "mars", "moon"});
  const bool from_pick = node->declare_parameter<bool>("start_from_pick", true);
  const double pause = node->declare_parameter<double>("pause", 1.0);
  const int64_t trials = node->declare_parameter<int64_t>("trials", 0);
  auto client = rclcpp_action::create_client<CatchBall>(node, "catch_ball");
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread spin([&] { exec.spin(); });

  int rc = 0;
  if (!client->wait_for_action_server(20s)) {
    RCLCPP_ERROR(node->get_logger(), "catch_ball action server not available");
    rc = 1;
  }
  int wins = 0;
  for (int64_t k = 0; rc == 0 && rclcpp::ok() && (trials <= 0 || k < trials); ++k) {
    CatchBall::Goal goal;
    goal.gravity = gravities[static_cast<size_t>(k) % gravities.size()];
    goal.start_from_pick = from_pick;
    auto gh_future = client->async_send_goal(goal);
    if (gh_future.wait_for(10s) != std::future_status::ready || !gh_future.get()) {
      RCLCPP_WARN(node->get_logger(), "%s: goal rejected", goal.gravity.c_str());
      std::this_thread::sleep_for(1s);
      continue;
    }
    auto res_future = client->async_get_result(gh_future.get());
    while (rclcpp::ok() && res_future.wait_for(100ms) != std::future_status::ready) {
    }
    if (!rclcpp::ok()) break;
    const auto res = res_future.get().result;
    wins += res->success;
    if (res->success)
      RCLCPP_INFO(node->get_logger(), "[%ld] caught on %s (e=%.2f): %+.0f ms from the apex, %+.1f mm, impact %.2f m/s   (%d/%ld)",
                  static_cast<long>(k + 1), res->gravity_name.c_str(), res->restitution, 1000.0 * res->catch_time_error,
                  1000.0 * res->catch_height_error, res->impact_speed, wins, static_cast<long>(k + 1));
    else
      RCLCPP_INFO(node->get_logger(), "[%ld] %s: %s   (%d/%ld)", static_cast<long>(k + 1), res->gravity_name.c_str(),
                  res->outcome.c_str(), wins, static_cast<long>(k + 1));
    std::this_thread::sleep_for(std::chrono::duration<double>(pause));
  }
  exec.cancel();
  spin.join();
  rclcpp::shutdown();
  return rc;
}
