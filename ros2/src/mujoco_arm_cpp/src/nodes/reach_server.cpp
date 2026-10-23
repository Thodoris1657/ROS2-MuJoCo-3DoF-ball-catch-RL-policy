// reach_server -- ReachTarget action server (C++): "move the tip to (x, y, z) and hold it".
//
// Publishes the goal on `target`, then follows `status`. Something must drive the arm
// (policy_node). Limits are in simulated time; a wall watchdog aborts if the sim goes silent.
//   ros2 action send_goal /arm/reach_target mujoco_arm_interfaces/action/ReachTarget
//       "{target: {x: 0.3, y: 0.1, z: 0.35}}" --feedback
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <mujoco_arm_interfaces/action/reach_target.hpp>
#include <mujoco_arm_interfaces/msg/arm_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
using Reach = mujoco_arm_interfaces::action::ReachTarget;
using GoalHandle = rclcpp_action::ServerGoalHandle<Reach>;
using Status = mujoco_arm_interfaces::msg::ArmStatus;
namespace mros = mujoco_arm::ros;

class ReachServer : public rclcpp::Node {
 public:
  ReachServer() : Node("reach_server") {
    kin_ = std::make_unique<mujoco_arm::ReachSim>(
        mros::or_share(declare_parameter<std::string>("model_path", ""), "assets/arm3.xml"));
    group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = group_;
    pub_ = create_publisher<geometry_msgs::msg::PointStamped>("target", mros::reliable());
    sub_ = create_subscription<Status>("status", mros::reliable(50),
                                       [this](Status::ConstSharedPtr st) { on_status(*st); }, opts);
    server_ = rclcpp_action::create_server<Reach>(
        this, "reach_target",
        [this](const rclcpp_action::GoalUUID&, std::shared_ptr<const Reach::Goal> g) {
          std::lock_guard<std::mutex> lk(mu_);
          const auto [ok, why] = kin_->check_target_reachable({g->target.x, g->target.y, g->target.z});
          if (!ok) {
            RCLCPP_WARN(get_logger(), "goal rejected: %s", why.c_str());
            return rclcpp_action::GoalResponse::REJECT;
          }
          if (busy_) {
            RCLCPP_WARN(get_logger(), "goal rejected: another goal is running");
            return rclcpp_action::GoalResponse::REJECT;
          }
          busy_ = true;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle>) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](const std::shared_ptr<GoalHandle> h) { std::thread([this, h] { execute(h); }).detach(); },
        rcl_action_server_get_default_options(), group_);
    RCLCPP_INFO(get_logger(), "action server ready: reach_target");
  }

 private:
  struct Track {
    double want[3] = {0, 0, 0}, tol = 0.02, hold = 0.5, timeout = 10.0;
    std::optional<double> t0, inside_since;
    std::optional<uint64_t> last_seq;
    double time_to_reach = -1, distance = -1, elapsed = 0, time_inside = 0;
    std::optional<std::string> outcome;
    std::chrono::steady_clock::time_point last_wall = std::chrono::steady_clock::now();
  };

  void on_status(const Status& st) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!tr_) return;
    auto& t = *tr_;
    t.last_wall = std::chrono::steady_clock::now();
    if (!t.t0) {
      if (std::abs(st.target.x - t.want[0]) > 1e-6 || std::abs(st.target.y - t.want[1]) > 1e-6 ||
          std::abs(st.target.z - t.want[2]) > 1e-6)
        return;
      t.t0 = st.sim_time;
      t.last_seq = st.seq;
      t.distance = st.distance;
      return;
    }
    if ((t.last_seq && st.seq == *t.last_seq) || t.outcome) return;
    t.last_seq = st.seq;
    const double elapsed = st.sim_time - *t.t0;
    if (st.distance < t.tol) {
      if (!t.inside_since) t.inside_since = st.sim_time;
      if (t.time_to_reach < 0) t.time_to_reach = elapsed;
    } else {
      t.inside_since.reset();
    }
    t.distance = st.distance;
    t.elapsed = elapsed;
    t.time_inside = t.inside_since ? st.sim_time - *t.inside_since : 0.0;
    if (t.time_inside >= t.hold) t.outcome = "success";
    else if (elapsed > t.timeout) t.outcome = "timeout";
  }

  void execute(const std::shared_ptr<GoalHandle> h) {
    run(h);
    std::lock_guard<std::mutex> lk(mu_);
    busy_ = false;
    tr_.reset();
  }

  void run(const std::shared_ptr<GoalHandle>& h) {
    const auto g = h->get_goal();
    auto res = std::make_shared<Reach::Result>();
    res->time_to_reach = res->final_distance = -1.0f;
    const double tol = g->tolerance > 0 ? g->tolerance : 0.02;
    const double hold = g->hold_time > 0 ? g->hold_time : 0.5;
    const double timeout = g->timeout > 0 ? g->timeout : 10.0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      tr_ = Track{};
      tr_->want[0] = g->target.x;
      tr_->want[1] = g->target.y;
      tr_->want[2] = g->target.z;
      tr_->tol = tol;
      tr_->hold = hold;
      tr_->timeout = timeout;
    }
    geometry_msgs::msg::PointStamped msg;
    msg.header.frame_id = mros::kFrame;
    msg.header.stamp = now();
    msg.point = g->target;
    pub_->publish(msg);
    auto fb = std::make_shared<Reach::Feedback>();
    char buf[128];
    while (rclcpp::ok()) {
      Track t;
      {
        std::lock_guard<std::mutex> lk(mu_);
        t = *tr_;
      }
      res->time_to_reach = static_cast<float>(t.time_to_reach);
      res->final_distance = static_cast<float>(t.distance);
      const double silent = std::chrono::duration<double>(std::chrono::steady_clock::now() - t.last_wall).count();
      if (h->is_canceling()) {
        res->success = false;
        res->message = "canceled";
        h->canceled(res);
        return;
      }
      if (!t.t0) {
        if (silent > 5.0) {
          res->success = false;
          res->message = "simulator did not accept the target (is sim_node running?)";
          h->abort(res);
          return;
        }
      } else {
        fb->distance = static_cast<float>(t.distance);
        fb->elapsed = static_cast<float>(t.elapsed);
        fb->time_inside = static_cast<float>(t.time_inside);
        h->publish_feedback(fb);
        if (t.outcome && *t.outcome == "success") {
          res->success = true;
          std::snprintf(buf, sizeof(buf), "held within %.1f cm for %.2f s", tol * 100, hold);
          res->message = buf;
          h->succeed(res);
          return;
        }
        if (t.outcome && *t.outcome == "timeout") {
          res->success = false;
          std::snprintf(buf, sizeof(buf), "timed out after %.1f s of sim time", timeout);
          res->message = buf;
          h->abort(res);
          return;
        }
        if (silent > 5.0) {
          res->success = false;
          res->message = "simulator stopped publishing status";
          h->abort(res);
          return;
        }
      }
      std::this_thread::sleep_for(20ms);
    }
  }

  std::mutex mu_;
  bool busy_ = false;
  std::optional<Track> tr_;
  std::unique_ptr<mujoco_arm::ReachSim> kin_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_;
  rclcpp::Subscription<Status>::SharedPtr sub_;
  rclcpp_action::Server<Reach>::SharedPtr server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ReachServer>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
