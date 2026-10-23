// sim_node -- the MuJoCo arm reach task (MujocoArm3Reach-v1) as a ROS 2 node (C++).
//
// Topics (relative to the node namespace, /arm in the launch file):
//   in   action         std_msgs/Float32MultiArray     normalised joint-target deltas in [-1, 1]
//   in   joint_command  sensor_msgs/JointState         absolute joint goals [rad]
//   in   target         geometry_msgs/PointStamped     move the goal (RViz "Publish Point")
//   out  joint_states   sensor_msgs/JointState
//   out  observation    mujoco_arm_interfaces/ArmObservation   the 24 floats the policy sees
//   out  status         mujoco_arm_interfaces/ArmStatus        per-step task status
//   out  ee_pose        geometry_msgs/PoseStamped
//   out  markers        visualization_msgs/MarkerArray
//   tf   world -> target
// Services: reset (mujoco_arm_interfaces/ResetEpisode), pause (std_srvs/SetBool).
// mode: realtime (wall timer, stale action -> hold) | lockstep (step only on a fresh command).
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mujoco_arm_interfaces/msg/arm_observation.hpp>
#include <mujoco_arm_interfaces/msg/arm_status.hpp>
#include <mujoco_arm_interfaces/srv/reset_episode.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "mujoco_arm_cpp/catch_core.hpp"
#include "mujoco_arm_cpp/ros_common.hpp"

using mujoco_arm::Vec3;
namespace mros = mujoco_arm::ros;
using ResetSrv = mujoco_arm_interfaces::srv::ResetEpisode;

/// env_config.json written by the trainers: {"env_id": ..., "env_kwargs": {...}}
std::string kwargs_from_env_config(const std::string& path) {
  if (path.empty()) return "{}";
  std::string p = path;
  std::ifstream test(p);
  if (!test) p = path + "/env_config.json";
  std::ifstream f(p);
  if (!f) throw std::runtime_error("env_config not found: " + path);
  const auto j = nlohmann::json::parse(f);
  const auto id = j.value("env_id", std::string("MujocoArm3Reach-v1"));
  if (id != "MujocoArm3Reach-v1")
    throw std::runtime_error("sim_node runs MujocoArm3Reach-v1; env_config says " + id);
  return j.value("env_kwargs", nlohmann::json::object()).dump();
}

class SimNode : public rclcpp::Node {
 public:
  SimNode() : Node("sim_node") {
    const auto model = declare_parameter<std::string>("model_path", "");
    const auto env_config = declare_parameter<std::string>("env_config", "");
    const auto env_kwargs = declare_parameter<std::string>("env_kwargs", "{}");
    mode_ = declare_parameter<std::string>("mode", "realtime");
    const double rtf = std::max(declare_parameter<double>("real_time_factor", 1.0), 0.01);
    action_timeout_ = declare_parameter<double>("action_timeout", 0.25);
    episodic_ = declare_parameter<bool>("episodic", false);
    const auto seed = declare_parameter<int64_t>("seed", -1);
    markers_ = declare_parameter<bool>("publish_markers", true);
    paused_ = declare_parameter<bool>("start_paused", false);
    if (mode_ != "realtime" && mode_ != "lockstep")
      throw std::invalid_argument("mode must be 'realtime' or 'lockstep'");

    auto cfg = mujoco_arm::reach_config_from_json(kwargs_from_env_config(env_config));
    cfg = mujoco_arm::reach_config_from_json(env_kwargs, cfg);
    core_ = std::make_unique<mujoco_arm::ReachSimCore>(
        mros::or_share(model, "assets/arm3.xml"), cfg,
        seed >= 0 ? std::optional<uint64_t>(seed) : std::nullopt);
    snap_ = core_->snapshot();

    pub_joint_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    pub_obs_ = create_publisher<mujoco_arm_interfaces::msg::ArmObservation>("observation", mros::reliable());
    pub_status_ = create_publisher<mujoco_arm_interfaces::msg::ArmStatus>("status", mros::reliable());
    pub_ee_ = create_publisher<geometry_msgs::msg::PoseStamped>("ee_pose", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("markers", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    sub_action_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "action", mros::reliable(), [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) { on_action(*m); });
    sub_joint_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_command", mros::reliable(), [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { on_joint(*m); });
    sub_target_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "target", mros::reliable(), [this](geometry_msgs::msg::PointStamped::ConstSharedPtr m) { on_target(*m); });
    srv_reset_ = create_service<ResetSrv>(
        "reset", [this](const std::shared_ptr<ResetSrv::Request> q, std::shared_ptr<ResetSrv::Response> r) {
          on_reset(*q, *r);
        });
    srv_pause_ = create_service<std_srvs::srv::SetBool>(
        "pause", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q,
                        std::shared_ptr<std_srvs::srv::SetBool::Response> r) {
          std::lock_guard<std::mutex> lk(mu_);
          paused_ = q->data;
          r->success = true;
          r->message = paused_ ? "paused" : "running";
        });
    const auto period = std::chrono::duration<double>(core_->dt() / rtf);
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "MuJoCo arm up: mode=%s, dt=%.0f ms, real_time_factor=%g, obs_dim=%d, episode=%d steps",
                mode_.c_str(), core_->dt() * 1000.0, rtf, mujoco_arm::kReachObsDim,
                core_->sim().max_episode_steps());
  }

 private:
  double now_s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void on_action(const std_msgs::msg::Float32MultiArray& m) {
    if (m.data.size() != 3) {
      RCLCPP_WARN(get_logger(), "action needs 3 values, got %zu; ignored", m.data.size());
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    // an action stamped with an observation seq (policy_node does this) must answer the current
    // observation; in lock-step a late reply to an older one would shift the episode by a step
    if (mode_ == "lockstep" && !m.layout.dim.empty() && m.layout.dim[0].label == "seq" &&
        m.layout.data_offset != static_cast<uint32_t>(snap_.seq & 0xffffffffu))
      return;
    for (int i = 0; i < 3; ++i) action_[i] = std::clamp(static_cast<double>(m.data[i]), -1.0, 1.0);
    action_time_ = now_s();
    fresh_ = true;
    from_goal_ = false;
  }

  void on_joint(const sensor_msgs::msg::JointState& m) {
    static const char* names[3] = {"joint1", "joint2", "joint3"};
    Vec3 goal{};
    if (!m.name.empty()) {
      for (int i = 0; i < 3; ++i) {
        const auto it = std::find(m.name.begin(), m.name.end(), names[i]);
        if (it == m.name.end() || static_cast<size_t>(it - m.name.begin()) >= m.position.size()) {
          RCLCPP_WARN(get_logger(), "joint_command must name joint1, joint2, joint3; ignored");
          return;
        }
        goal[i] = m.position[static_cast<size_t>(it - m.name.begin())];
      }
    } else if (m.position.size() == 3) {
      for (int i = 0; i < 3; ++i) goal[i] = m.position[i];
    } else {
      RCLCPP_WARN(get_logger(), "joint_command needs 3 positions, got %zu; ignored", m.position.size());
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    const auto& s = core_->sim();
    for (int i = 0; i < 3; ++i) goal[i] = std::clamp(goal[i], s.joint_lo()[i], s.joint_hi()[i]);
    q_goal_ = goal;
    action_time_ = now_s();
    fresh_ = true;
    from_goal_ = true;
  }

  void on_target(const geometry_msgs::msg::PointStamped& m) {
    if (!m.header.frame_id.empty() && m.header.frame_id != mros::kFrame) {
      RCLCPP_WARN(get_logger(), "target must be in frame 'world', got '%s'; ignored", m.header.frame_id.c_str());
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    try {
      snap_ = core_->set_target({m.point.x, m.point.y, m.point.z});
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "target rejected: %s", e.what());
      return;
    }
    publish(true);
  }

  void on_reset(const ResetSrv::Request& q, ResetSrv::Response& r) {
    std::lock_guard<std::mutex> lk(mu_);
    try {
      snap_ = core_->reset(q.use_target ? std::optional<Vec3>(Vec3{q.target.x, q.target.y, q.target.z}) : std::nullopt,
                           q.use_qpos ? std::optional<Vec3>(Vec3{q.qpos[0], q.qpos[1], q.qpos[2]}) : std::nullopt,
                           q.seed >= 0 ? std::optional<uint64_t>(q.seed) : std::nullopt);
    } catch (const std::exception& e) {
      r.success = false;
      r.message = e.what();
      return;
    }
    action_ = {0, 0, 0};
    fresh_ = false;
    q_goal_.reset();
    from_goal_ = false;
    publish(true);
    r.success = true;
    r.message = "ok";
    r.episode = snap_.episode;
    r.target = mros::to_point(snap_.target);
    r.distance = static_cast<float>(snap_.distance);
  }

  std::optional<Vec3> next_action() {
    if (from_goal_ && q_goal_) {
      if (mode_ == "lockstep" && !fresh_) return std::nullopt;
      fresh_ = false;
      return core_->sim().action_from_joint_goal(*q_goal_);
    }
    if (mode_ == "lockstep") {
      if (!fresh_) return std::nullopt;
      fresh_ = false;
      return action_;
    }
    fresh_ = false;
    if (now_s() - action_time_ > action_timeout_) return Vec3{0, 0, 0};
    return action_;
  }

  void tick() {
    std::lock_guard<std::mutex> lk(mu_);
    bool stepped = false;
    if (!paused_ && !(episodic_ && snap_.episode_done)) {
      if (auto a = next_action()) {
        snap_ = core_->step(*a);
        stepped = true;
      }
    }
    publish(stepped);
  }

  void publish(bool new_status) {
    const auto& s = snap_;
    const auto stamp = now();
    mujoco_arm_interfaces::msg::ArmObservation obs;
    obs.header.stamp = stamp;
    obs.header.frame_id = mros::kFrame;
    obs.seq = s.seq;
    obs.data.assign(s.obs.begin(), s.obs.end());
    pub_obs_->publish(obs);
    if (new_status) {
      mujoco_arm_interfaces::msg::ArmStatus st;
      st.header = obs.header;
      st.seq = s.seq;
      st.episode = s.episode;
      st.episode_step = s.episode_step;
      st.episode_steps_max = s.episode_steps_max;
      st.episode_done = s.episode_done;
      st.sim_time = s.sim_time;
      st.distance = static_cast<float>(s.distance);
      st.reward = static_cast<float>(s.reward);
      st.success = s.success;
      st.success_2cm = s.success_2cm;
      st.time_to_reach = static_cast<float>(s.time_to_reach);
      st.ee = mros::to_point(s.ee);
      st.target = mros::to_point(s.target);
      pub_status_->publish(st);
    }
    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"joint1", "joint2", "joint3"};
    js.position.assign(s.q.begin(), s.q.end());
    js.velocity.assign(s.qd.begin(), s.qd.end());
    js.effort.assign(s.tau.begin(), s.tau.end());
    pub_joint_->publish(js);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = obs.header;
    pose.pose.position = mros::to_point(s.ee);
    pose.pose.orientation.x = s.ee_quat_xyzw[0];
    pose.pose.orientation.y = s.ee_quat_xyzw[1];
    pose.pose.orientation.z = s.ee_quat_xyzw[2];
    pose.pose.orientation.w = s.ee_quat_xyzw[3];
    pub_ee_->publish(pose);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = obs.header;
    tf.child_frame_id = "target";
    tf.transform.translation.x = s.target[0];
    tf.transform.translation.y = s.target[1];
    tf.transform.translation.z = s.target[2];
    tf.transform.rotation.w = 1.0;
    tf_->sendTransform(tf);

    if (!markers_) return;
    using visualization_msgs::msg::Marker;
    auto base = [&](int id, int type) {
      Marker m;
      m.header = obs.header;
      m.ns = "arm";
      m.id = id;
      m.type = type;
      m.action = Marker::ADD;
      m.pose.orientation.w = 1.0;
      return m;
    };
    auto goal = base(0, Marker::SPHERE);
    goal.pose.position = mros::to_point(s.target);
    goal.scale.x = goal.scale.y = goal.scale.z = 0.10;
    goal.color.r = 0.15f; goal.color.g = 0.80f; goal.color.b = 0.25f; goal.color.a = 0.35f;
    auto line = base(1, Marker::LINE_LIST);
    line.scale.x = 0.006;
    line.points = {mros::to_point(s.ee), mros::to_point(s.target)};
    line.color.r = s.success ? 0.15f : 0.95f; line.color.g = s.success ? 0.85f : 0.30f;
    line.color.b = s.success ? 0.25f : 0.20f; line.color.a = 0.9f;
    auto text = base(2, Marker::TEXT_VIEW_FACING);
    text.pose.position = mros::to_point({s.target[0], s.target[1], s.target[2] + 0.12});
    text.scale.z = 0.04;
    text.color.r = text.color.g = text.color.b = text.color.a = 1.0f;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f cm", s.distance * 100.0);
    text.text = buf;
    visualization_msgs::msg::MarkerArray arr;
    arr.markers = {goal, line, text};
    pub_markers_->publish(arr);
  }

  std::mutex mu_;
  std::unique_ptr<mujoco_arm::ReachSimCore> core_;
  mujoco_arm::ReachSnapshot snap_{};
  std::string mode_;
  double action_timeout_ = 0.25, action_time_ = -1e9;
  bool episodic_ = false, markers_ = true, paused_ = false, fresh_ = false, from_goal_ = false;
  Vec3 action_{0, 0, 0};
  std::optional<Vec3> q_goal_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_;
  rclcpp::Publisher<mujoco_arm_interfaces::msg::ArmObservation>::SharedPtr pub_obs_;
  rclcpp::Publisher<mujoco_arm_interfaces::msg::ArmStatus>::SharedPtr pub_status_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_ee_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_action_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_target_;
  rclcpp::Service<ResetSrv>::SharedPtr srv_reset_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_pause_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SimNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("sim_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
