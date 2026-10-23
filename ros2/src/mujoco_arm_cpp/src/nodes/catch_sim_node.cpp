// catch_sim_node -- the ball-catch simulator (Earth / Mars / Moon) as a ROS 2 node (C++).
//
// Topics (relative to the node namespace, /arm in the launch file):
//   in   action        std_msgs/Float32MultiArray            normalised joint-target deltas in [-1, 1]
//   out  joint_states  sensor_msgs/JointState
//   out  observation   mujoco_arm_interfaces/ArmObservation  the 34 floats a catch policy sees
//   out  catch_status  mujoco_arm_interfaces/CatchStatus     phase, ball state, outcome, catch errors
//   out  ee_pose       geometry_msgs/PoseStamped             cup (tip) pose
//   out  ball          geometry_msgs/PointStamped            ball centre
//   out  markers       visualization_msgs/MarkerArray        ball, cube, catch point, status text
//   tf   world -> ball
// Services: catch/start (mujoco_arm_interfaces/StartCatch), pause (std_srvs/SetBool).
// Parameter `gravity` (earth | mars | moon | number): default for the next trial, settable live.
//
// mode = realtime: a wall timer steps the physics; an action older than action_timeout becomes
//                  "hold" (zero), so a dead controller cannot make the arm run away.
// mode = lockstep: the physics steps only on a fresh action and the observation is re-published
//                  while waiting, so the loop is exactly the offline benchmark's.
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mujoco_arm_interfaces/msg/arm_observation.hpp>
#include <mujoco_arm_interfaces/msg/catch_status.hpp>
#include <mujoco_arm_interfaces/srv/start_catch.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "mujoco_arm_cpp/catch_core.hpp"
#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
using mujoco_arm::Vec3;
namespace mros = mujoco_arm::ros;
using ObsMsg = mujoco_arm_interfaces::msg::ArmObservation;
using StatusMsg = mujoco_arm_interfaces::msg::CatchStatus;
using StartSrv = mujoco_arm_interfaces::srv::StartCatch;

class CatchSimNode : public rclcpp::Node {
 public:
  CatchSimNode() : Node("catch_sim_node") {
    const auto gravity = declare_parameter<std::string>("gravity", "earth");
    const auto env_kwargs = declare_parameter<std::string>("env_kwargs", "{}");
    const auto model = declare_parameter<std::string>("model_path", "");
    mode_ = declare_parameter<std::string>("mode", "realtime");
    const double rtf = std::max(declare_parameter<double>("real_time_factor", 1.0), 0.01);
    action_timeout_ = declare_parameter<double>("action_timeout", 0.25);
    const auto seed = declare_parameter<int64_t>("seed", -1);
    markers_ = declare_parameter<bool>("publish_markers", true);
    const bool auto_start = declare_parameter<bool>("auto_start", false);
    const bool from_pick = declare_parameter<bool>("start_from_pick", true);
    const double post_roll = declare_parameter<double>("post_roll", 1.0);
    if (mode_ != "realtime" && mode_ != "lockstep")
      throw std::invalid_argument("mode must be 'realtime' or 'lockstep'");

    mujoco_arm::CatchConfig cfg = mujoco_arm::catch_config_from_json(env_kwargs);
    core_ = std::make_unique<mujoco_arm::CatchTrialRunner>(
        mros::or_share(model, "assets/arm3_ballcatch.xml"), cfg, gravity,
        seed >= 0 ? std::optional<uint64_t>(seed) : std::nullopt, post_roll);

    pub_joint_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    pub_obs_ = create_publisher<ObsMsg>("observation", mros::reliable());
    pub_status_ = create_publisher<StatusMsg>("catch_status", mros::reliable());
    pub_ee_ = create_publisher<geometry_msgs::msg::PoseStamped>("ee_pose", 10);
    pub_ball_ = create_publisher<geometry_msgs::msg::PointStamped>("ball", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("markers", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    sub_action_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "action", mros::reliable(), [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) { on_action(*m); });
    srv_start_ = create_service<StartSrv>(
        "catch/start", [this](const std::shared_ptr<StartSrv::Request> q, std::shared_ptr<StartSrv::Response> r) {
          on_start(*q, *r);
        });
    srv_pause_ = create_service<std_srvs::srv::SetBool>(
        "pause", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q,
                        std::shared_ptr<std_srvs::srv::SetBool::Response> r) {
          std::lock_guard<std::mutex> lk(mu_);
          paused_ = q->data;
          r->success = true;
          r->message = paused_ ? "paused" : "running";
        });
    param_cb_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& ps) {
      rcl_interfaces::msg::SetParametersResult res;
      res.successful = true;
      for (const auto& p : ps) {
        if (p.get_name() != "gravity") continue;
        try {
          mujoco_arm::resolve_gravity(p.as_string());
        } catch (const std::exception& e) {
          res.successful = false;
          res.reason = e.what();
          return res;
        }
        std::lock_guard<std::mutex> lk(mu_);
        core_->default_gravity = p.as_string();
        RCLCPP_INFO(get_logger(), "gravity for the next trial: %s", p.as_string().c_str());
      }
      return res;
    });

    const auto period = std::chrono::duration<double>(core_->sim().dt() / rtf);
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { tick(); });
    if (auto_start) {
      mujoco_arm::CatchTrialRunner::StartRequest req;
      req.from_pick = from_pick;
      core_->start(req);
    }
    RCLCPP_INFO(get_logger(), "ball-catch sim up: mode=%s, dt=%.0f ms, gravity=%s, real_time_factor=%g, obs_dim=%d",
                mode_.c_str(), core_->sim().dt() * 1000.0, gravity.c_str(), rtf, mujoco_arm::kCatchObsDim);
  }

 private:
  void on_action(const std_msgs::msg::Float32MultiArray& m) {
    if (m.data.size() != 3) {
      RCLCPP_WARN(get_logger(), "action needs 3 values, got %zu; ignored", m.data.size());
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    // an action stamped with an observation seq (policy_node does this) must answer the current
    // observation; in lock-step a late reply to an older one would shift the episode by a step
    if (mode_ == "lockstep" && !m.layout.dim.empty() && m.layout.dim[0].label == "seq" &&
        m.layout.data_offset != static_cast<uint32_t>(core_->seq() & 0xffffffffu))
      return;
    for (int i = 0; i < 3; ++i) action_[i] = std::clamp(static_cast<double>(m.data[i]), -1.0, 1.0);
    action_time_ = now_s();
    fresh_ = true;
  }

  void on_start(const StartSrv::Request& q, StartSrv::Response& r) {
    std::lock_guard<std::mutex> lk(mu_);
    mujoco_arm::CatchTrialRunner::StartRequest req;
    req.gravity = q.gravity;
    req.restitution = q.restitution;
    if (q.use_drop) req.drop = Vec3{q.drop.x, q.drop.y, q.drop.z};
    if (q.use_push) req.push = std::array<double, 2>{q.push[0], q.push[1]};
    req.from_pick = q.start_from_pick;
    if (q.seed >= 0) req.seed = static_cast<uint64_t>(q.seed);
    try {
      core_->start(req);
    } catch (const std::exception& e) {
      r.success = false;
      r.message = e.what();
      return;
    }
    action_ = {0, 0, 0};
    fresh_ = false;
    publish();
    const auto& sim = core_->sim();
    r.success = true;
    r.message = "ok";
    r.trial = core_->trial();
    r.gravity_name = mujoco_arm::gravity_name(sim.gravity());
    r.gravity = static_cast<float>(sim.gravity());
    r.restitution = static_cast<float>(sim.restitution());
    r.drop = mros::to_point(sim.drop_tip());
  }

  std::optional<Vec3> next_action() {
    if (mode_ == "lockstep") {
      if (!fresh_) return std::nullopt;
      fresh_ = false;
      return action_;
    }
    fresh_ = false;
    if (now_s() - action_time_ > action_timeout_) return Vec3{0, 0, 0};  // stale: hold
    return action_;
  }

  void tick() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!paused_ && !core_->idle()) {
      if (auto a = next_action()) core_->step(*a);
    }
    publish();
  }

  double now_s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void publish() {
    auto& sim = core_->sim();
    const auto stamp = now();
    const auto& o = core_->obs();

    ObsMsg obs;
    obs.header.stamp = stamp;
    obs.header.frame_id = mros::kFrame;
    obs.seq = core_->seq();
    obs.data.assign(o.begin(), o.end());
    pub_obs_->publish(obs);

    StatusMsg st;
    st.header = obs.header;
    st.seq = core_->seq();
    st.trial = core_->trial();
    st.phase = core_->phase();
    st.gravity_name = mujoco_arm::gravity_name(sim.gravity());
    st.gravity = static_cast<float>(sim.gravity());
    st.restitution = static_cast<float>(sim.restitution());
    st.sim_time = sim.sim_time();
    st.time_since_release = sim.t_release() ? static_cast<float>(sim.sim_time() - *sim.t_release()) : -1.0f;
    st.bounces = static_cast<uint32_t>(sim.bounces());
    st.holding = sim.holding();
    st.done = sim.outcome().has_value();
    st.success = sim.success();
    st.outcome = sim.outcome().value_or("");
    st.ball = mros::to_point(sim.ball_pos());
    st.ball_velocity = mros::to_vector3(sim.ball_vel());
    st.ee = mros::to_point(sim.ee_pos());
    st.catch_point = mros::to_point(sim.catch_point());
    st.catch_time_error = static_cast<float>(sim.catch_time_error());
    st.catch_height_error = static_cast<float>(sim.catch_height_error());
    st.impact_speed = sim.catch_info() ? static_cast<float>(sim.catch_info()->rel_speed) : NAN;
    pub_status_->publish(st);

    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"joint1", "joint2", "joint3"};
    const Vec3 q = sim.q(), qd = sim.qd(), tau = sim.tau();
    js.position.assign(q.begin(), q.end());
    js.velocity.assign(qd.begin(), qd.end());
    js.effort.assign(tau.begin(), tau.end());
    pub_joint_->publish(js);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = obs.header;
    pose.pose.position = mros::to_point(sim.ee_pos());
    const auto qx = sim.ee_quat_xyzw();
    pose.pose.orientation.x = qx[0];
    pose.pose.orientation.y = qx[1];
    pose.pose.orientation.z = qx[2];
    pose.pose.orientation.w = qx[3];
    pub_ee_->publish(pose);

    geometry_msgs::msg::PointStamped ball;
    ball.header = obs.header;
    ball.point = st.ball;
    pub_ball_->publish(ball);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = obs.header;
    tf.child_frame_id = "ball";
    tf.transform.translation.x = st.ball.x;
    tf.transform.translation.y = st.ball.y;
    tf.transform.translation.z = st.ball.z;
    tf.transform.rotation.w = 1.0;
    tf_->sendTransform(tf);

    if (markers_) pub_markers_->publish(make_markers(st));
  }

  visualization_msgs::msg::MarkerArray make_markers(const StatusMsg& st) {
    using visualization_msgs::msg::Marker;
    auto base = [&](int id, int type) {
      Marker m;
      m.header = st.header;
      m.ns = "catch";
      m.id = id;
      m.type = type;
      m.action = Marker::ADD;
      m.pose.orientation.w = 1.0;
      return m;
    };
    auto ball = base(0, Marker::SPHERE);
    ball.pose.position = st.ball;
    ball.scale.x = ball.scale.y = ball.scale.z = 2 * mujoco_arm::kBallR;
    ball.color.r = 0.95f; ball.color.g = 0.55f; ball.color.b = 0.10f; ball.color.a = 1.0f;

    auto cube = base(1, Marker::CUBE);
    cube.pose.position.y = 0.30;
    cube.pose.position.z = mujoco_arm::kPedestalTop / 2;
    cube.scale.x = cube.scale.y = cube.scale.z = mujoco_arm::kPedestalTop;
    cube.color.r = 0.25f; cube.color.g = 0.35f; cube.color.b = 0.75f; cube.color.a = 1.0f;

    auto aim = base(2, Marker::SPHERE);
    aim.pose.position = st.catch_point;
    aim.scale.x = aim.scale.y = aim.scale.z = 0.024;
    aim.color.r = 0.15f; aim.color.g = 0.85f; aim.color.b = 0.25f;
    aim.color.a = (st.phase == "fall" || st.phase == "rise") ? 0.8f : 0.0f;

    auto text = base(3, Marker::TEXT_VIEW_FACING);
    text.pose.position.z = 0.85;
    text.scale.z = 0.045;
    const bool good = st.success || !st.done;
    text.color.r = 1.0f; text.color.g = good ? 1.0f : 0.4f; text.color.b = good ? 1.0f : 0.3f; text.color.a = 1.0f;
    char buf[256];
    std::string tail = st.done ? st.outcome : st.phase;
    if (st.done && st.success && std::isfinite(st.impact_speed)) {
      std::snprintf(buf, sizeof(buf), "  (%.2f m/s, %+.0f ms)", st.impact_speed, 1000.0 * st.catch_time_error);
      tail += buf;
    }
    std::snprintf(buf, sizeof(buf), "%s  g=%.2f  e=%.2f  |  ", st.gravity_name.c_str(), st.gravity, st.restitution);
    text.text = buf + tail;

    visualization_msgs::msg::MarkerArray arr;
    arr.markers = {ball, cube, aim, text};
    return arr;
  }

  std::mutex mu_;
  std::unique_ptr<mujoco_arm::CatchTrialRunner> core_;
  std::string mode_;
  double action_timeout_ = 0.25, action_time_ = -1e9;
  bool markers_ = true, paused_ = false, fresh_ = false;
  Vec3 action_{0, 0, 0};
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_;
  rclcpp::Publisher<ObsMsg>::SharedPtr pub_obs_;
  rclcpp::Publisher<StatusMsg>::SharedPtr pub_status_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_ee_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_ball_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_action_;
  rclcpp::Service<StartSrv>::SharedPtr srv_start_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_pause_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CatchSimNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("catch_sim_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
