// policy_node -- runs a controller / exported RL network as a ROS 2 node (C++).
//
//   in   observation  mujoco_arm_interfaces/ArmObservation
//   out  action       std_msgs/Float32MultiArray   (normalised joint-target deltas)
//
// Answers once per observation `seq` (the simulator repeats an observation while it waits in
// lock-step) and stamps the answer with that seq (layout.dim[0].label = "seq",
// layout.data_offset = seq mod 2^32) so the simulator can drop replies to stale observations. Parameter `policy`, changeable at run time:
//   oracle | random | catch_tuned | catch_model | catch_hold | /path/to/policy.json
// (.json = a network exported by scripts/export_policy.py; residual catch policies included)
//   ros2 param set /arm/policy_node policy catch_hold
#include <chrono>
#include <memory>
#include <mutex>

#include <mujoco_arm_interfaces/msg/arm_observation.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
namespace mros = mujoco_arm::ros;

class PolicyNode : public rclcpp::Node {
 public:
  PolicyNode() : Node("policy_node") {
    const auto spec = declare_parameter<std::string>("policy", "oracle");
    seed_ = static_cast<uint64_t>(declare_parameter<int64_t>("seed", 0));
    paths_ = mros::policy_paths(*this);
    controller_ = mujoco_arm::make_controller(spec, paths_, seed_);
    RCLCPP_INFO(get_logger(), "policy: %s", controller_->description().c_str());

    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("action", mros::reliable());
    sub_ = create_subscription<mujoco_arm_interfaces::msg::ArmObservation>(
        "observation", mros::reliable(),
        [this](mujoco_arm_interfaces::msg::ArmObservation::ConstSharedPtr m) { on_obs(*m); });
    param_cb_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& ps) {
      rcl_interfaces::msg::SetParametersResult res;
      res.successful = true;
      for (const auto& p : ps) {
        if (p.get_name() != "policy") continue;
        try {
          auto next = mujoco_arm::make_controller(p.as_string(), paths_, seed_);
          std::lock_guard<std::mutex> lk(mu_);
          controller_ = std::move(next);
          last_seq_.reset();
          RCLCPP_INFO(get_logger(), "policy switched to: %s", controller_->description().c_str());
        } catch (const std::exception& e) {
          res.successful = false;
          res.reason = e.what();
        }
      }
      return res;
    });
    timer_ = create_wall_timer(10s, [this] { report(); });
  }

 private:
  void on_obs(const mujoco_arm_interfaces::msg::ArmObservation& m) {
    std::lock_guard<std::mutex> lk(mu_);
    if (last_seq_ && *last_seq_ == m.seq) return;  // the simulator repeated an observation
    last_seq_ = m.seq;
    const auto t0 = std::chrono::steady_clock::now();
    mujoco_arm::Vec3 a{0, 0, 0};
    try {
      a = controller_->act(std::vector<float>(m.data.begin(), m.data.end()));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "policy failed: %s", e.what());
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n_;
    total_ += dt;
    max_ = std::max(max_, dt);
    std_msgs::msg::Float32MultiArray out;
    for (double v : a) out.data.push_back(static_cast<float>(std::clamp(v, -1.0, 1.0)));
    // stamp the answer with the observation it answers: in lock-step the simulator only accepts
    // an action for its *current* observation (a late reply from before a reset is dropped)
    out.layout.dim.resize(1);
    out.layout.dim[0].label = "seq";
    out.layout.dim[0].size = 3;
    out.layout.dim[0].stride = 3;
    out.layout.data_offset = static_cast<uint32_t>(m.seq & 0xffffffffu);
    pub_->publish(out);
  }

  void report() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!n_) return;
    RCLCPP_INFO(get_logger(), "%zu inferences, mean %.3f ms, max %.3f ms", n_, 1000.0 * total_ / n_, 1000.0 * max_);
    n_ = 0;
    total_ = max_ = 0.0;
  }

  std::mutex mu_;
  mujoco_arm::PolicyPaths paths_;
  uint64_t seed_ = 0;
  std::unique_ptr<mujoco_arm::Controller> controller_;
  std::optional<uint64_t> last_seq_;
  size_t n_ = 0;
  double total_ = 0.0, max_ = 0.0;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::Subscription<mujoco_arm_interfaces::msg::ArmObservation>::SharedPtr sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PolicyNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("policy_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
