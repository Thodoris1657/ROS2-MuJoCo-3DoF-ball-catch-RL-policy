// catch_server -- CatchBall action server (C++): "pick the ball, drop it, catch it".
//
// Starts a trial through the simulator's catch/start service, then follows catch_status until
// the outcome. Something must drive the arm during the flight (policy_node with a catch policy).
// Limits are in simulated time; a wall-clock watchdog aborts if the simulator goes silent.
//
//   ros2 action send_goal /arm/catch_ball mujoco_arm_interfaces/action/CatchBall
//       "{gravity: moon, start_from_pick: true}" --feedback
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>

#include <mujoco_arm_interfaces/action/catch_ball.hpp>
#include <mujoco_arm_interfaces/msg/catch_status.hpp>
#include <mujoco_arm_interfaces/srv/start_catch.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
using CatchBall = mujoco_arm_interfaces::action::CatchBall;
using GoalHandle = rclcpp_action::ServerGoalHandle<CatchBall>;
using StatusMsg = mujoco_arm_interfaces::msg::CatchStatus;
using StartSrv = mujoco_arm_interfaces::srv::StartCatch;
namespace mros = mujoco_arm::ros;

constexpr double kDefaultTimeout = 20.0;
constexpr double kWatchdog = 5.0;

class CatchServer : public rclcpp::Node {
 public:
  CatchServer() : Node("catch_server") {
    group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = group_;
    sub_ = create_subscription<StatusMsg>(
        "catch_status", mros::reliable(50), [this](StatusMsg::ConstSharedPtr st) { on_status(*st); }, opts);
    client_ = create_client<StartSrv>("catch/start", rclcpp::ServicesQoS(), group_);
    server_ = rclcpp_action::create_server<CatchBall>(
        this, "catch_ball",
        [this](const rclcpp_action::GoalUUID&, std::shared_ptr<const CatchBall::Goal>) {
          std::lock_guard<std::mutex> lk(mu_);
          if (busy_) {
            RCLCPP_WARN(get_logger(), "goal rejected: another catch is running");
            return rclcpp_action::GoalResponse::REJECT;
          }
          busy_ = true;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle>) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](const std::shared_ptr<GoalHandle> h) { std::thread([this, h] { execute(h); }).detach(); },
        rcl_action_server_get_default_options(), group_);
    RCLCPP_INFO(get_logger(), "action server ready: catch_ball");
  }

 private:
  struct Track {
    std::optional<uint32_t> trial;
    std::optional<double> t0;
    std::optional<StatusMsg> status;
    std::optional<std::string> outcome;
    double timeout = kDefaultTimeout;
    std::chrono::steady_clock::time_point last_wall = std::chrono::steady_clock::now();
  };

  void on_status(const StatusMsg& st) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!track_ || !track_->trial || st.trial != *track_->trial) return;
    track_->last_wall = std::chrono::steady_clock::now();
    if (!track_->t0) track_->t0 = st.sim_time;
    track_->status = st;
    if (st.done && !track_->outcome) track_->outcome = st.outcome;
    else if (!track_->outcome && st.sim_time - *track_->t0 > track_->timeout) track_->outcome = "timeout";
  }

  static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  void execute(const std::shared_ptr<GoalHandle> h) {
    auto res = std::make_shared<CatchBall::Result>();
    res->catch_time_error = res->catch_height_error = res->impact_speed = NAN;
    run(h, *res);
    std::lock_guard<std::mutex> lk(mu_);
    busy_ = false;
    track_.reset();
  }

  void run(const std::shared_ptr<GoalHandle>& h, CatchBall::Result& res) {
    const auto goal = h->get_goal();
    auto result = std::make_shared<CatchBall::Result>();
    {
      std::lock_guard<std::mutex> lk(mu_);
      track_ = Track{};
      track_->timeout = goal->timeout > 0 ? goal->timeout : kDefaultTimeout;
    }
    auto abort = [&](const std::string& why) {
      res.outcome = why;
      *result = res;
      h->abort(result);
    };
    if (!client_->wait_for_service(std::chrono::duration<double>(kWatchdog)))
      return abort("catch/start service not available (is catch_sim_node running?)");
    auto req = std::make_shared<StartSrv::Request>();
    req->gravity = goal->gravity;
    req->restitution = goal->restitution;
    req->use_drop = goal->use_drop;
    req->drop = goal->drop;
    req->use_push = false;
    req->start_from_pick = goal->start_from_pick;
    req->seed = -1;
    auto fut = client_->async_send_request(req);
    if (fut.wait_for(std::chrono::duration<double>(kWatchdog)) != std::future_status::ready)
      return abort("catch/start did not answer");
    const auto started = fut.get();
    if (!started->success) return abort("rejected: " + started->message);
    {
      std::lock_guard<std::mutex> lk(mu_);
      track_->trial = started->trial;
      track_->last_wall = std::chrono::steady_clock::now();
    }
    res.trial = started->trial;
    res.gravity_name = started->gravity_name;
    res.restitution = started->restitution;

    auto fb = std::make_shared<CatchBall::Feedback>();
    while (rclcpp::ok()) {
      Track tr;
      {
        std::lock_guard<std::mutex> lk(mu_);
        tr = *track_;
      }
      if (h->is_canceling()) {
        res.outcome = "canceled";
        *result = res;
        h->canceled(result);
        return;
      }
      if (tr.status) {
        fb->phase = tr.status->phase;
        fb->ball_height = static_cast<float>(tr.status->ball.z);
        fb->bounces = tr.status->bounces;
        fb->time_since_release = tr.status->time_since_release;
        h->publish_feedback(fb);
      }
      if (tr.outcome) {
        res.outcome = *tr.outcome;
        if (tr.status) {
          res.catch_time_error = tr.status->catch_time_error;
          res.catch_height_error = tr.status->catch_height_error;
          res.impact_speed = tr.status->impact_speed;
          res.success = tr.status->success;
        }
        *result = res;
        if (res.success) h->succeed(result);
        else h->abort(result);
        RCLCPP_INFO(get_logger(), "%s: %s", res.gravity_name.c_str(), res.outcome.c_str());
        return;
      }
      if (since(tr.last_wall) > kWatchdog) return abort("simulator stopped publishing catch_status");
      std::this_thread::sleep_for(10ms);
    }
  }

  std::mutex mu_;
  bool busy_ = false;
  std::optional<Track> track_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Subscription<StatusMsg>::SharedPtr sub_;
  rclcpp::Client<StartSrv>::SharedPtr client_;
  rclcpp_action::Server<CatchBall>::SharedPtr server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CatchServer>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
