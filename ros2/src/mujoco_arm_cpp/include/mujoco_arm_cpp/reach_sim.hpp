// Reach-and-hold task in C++ (ROS-free): port of MujocoArm3Reach-v1 (reach3.py, position
// servo, 25 Hz, full 24-float observation, shaped reward) plus the ArmSimCore bookkeeping
// of the Python ROS bridge (continuous mode, observation sequence numbers, moving targets).
// The legacy v0 torque task is not ported.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "mujoco_arm_cpp/arm_model.hpp"

namespace mujoco_arm {

constexpr int kReachObsDim = 24;
using ReachObs = std::array<float, kReachObsDim>;

struct ReachConfig {
  double control_hz = 25.0;
  double episode_seconds = 2.0;
  double max_joint_speed = 3.0;
  ServoGains servo{};
  double success_threshold = 0.05;
  double start_noise = 0.3;
  double precision_scale = 0.05;
  double precision_weight = 1.0;
  double action_cost = 0.02;
  double action_rate_cost = 0.02;
  std::optional<Vec3> fixed_target;
};

class ReachSim {
 public:
  explicit ReachSim(const std::string& model_path, ReachConfig cfg = {});
  ReachSim(const ReachSim&) = delete;
  ReachSim& operator=(const ReachSim&) = delete;

  ReachObs reset(std::optional<uint64_t> seed = std::nullopt,
                 const std::optional<Vec3>& target = std::nullopt,
                 const std::optional<Vec3>& qpos = std::nullopt);
  struct Step { ReachObs obs{}; double reward = 0; double distance = 0; bool success = false;
                bool terminated = false; bool truncated = false; };
  Step step(const Vec3& action);
  ReachObs observe() const;

  /// Move the goal without touching the arm (throws std::invalid_argument if unreachable).
  void set_target(const Vec3& p);
  /// (ok, human-readable reason)
  std::pair<bool, std::string> check_target_reachable(const Vec3& p);
  Vec3 sample_target();
  Vec3 sample_start();

  bool config_is_valid(const Vec3& q, double margin = 0.0);
  std::optional<Vec3> solve_ik(const Vec3& p, const Vec3& near = {0, 0, 0}, double margin = 0.0);
  /// Normalised action that drives the servo target towards q_goal at full speed.
  Vec3 action_from_joint_goal(const Vec3& q_goal) const;

  Vec3 q() const;
  Vec3 qd() const;
  Vec3 tau() const;
  const Vec3& q_target() const { return q_target_; }
  Vec3 ee_pos() const;
  std::array<double, 4> ee_quat_xyzw() const;
  Vec3 target_pos() const;
  double dt() const { return m_->opt.timestep * frame_skip_; }
  double max_step() const { return cfg_.max_joint_speed * dt(); }
  int max_episode_steps() const { return max_episode_steps_; }
  int elapsed_steps() const { return elapsed_; }
  const Vec3& joint_lo() const { return lo_; }
  const Vec3& joint_hi() const { return hi_; }
  const ReachConfig& config() const { return cfg_; }

 private:
  ReachConfig cfg_;
  ModelPtr m_;
  DataPtr d_, scratch_;
  int frame_skip_ = 20, max_episode_steps_ = 50, tip_id_ = 0;
  Vec3 lo_{}, hi_{}, q_target_{}, prev_action_{};
  int elapsed_ = 0;
  std::mt19937_64 rng_;
};

/// Snapshot + bookkeeping for a long-running node (port of mujoco_arm_ros/core.py).
struct ReachSnapshot {
  uint64_t seq;
  uint32_t episode, episode_step, episode_steps_max;
  bool episode_done;
  double sim_time;
  Vec3 q, qd, tau, q_target, ee, target;
  std::array<double, 4> ee_quat_xyzw;
  ReachObs obs;
  double reward, distance;
  bool success, success_2cm;
  double time_to_reach;  // -1 = not yet
};

class ReachSimCore {
 public:
  explicit ReachSimCore(const std::string& model_path, ReachConfig cfg = {},
                        std::optional<uint64_t> seed = std::nullopt);
  ReachSnapshot snapshot() const;
  ReachSnapshot reset(const std::optional<Vec3>& target = std::nullopt,
                      const std::optional<Vec3>& qpos = std::nullopt,
                      std::optional<uint64_t> seed = std::nullopt);
  ReachSnapshot set_target(const Vec3& p);
  ReachSnapshot step(const Vec3& action);
  ReachSim& sim() { return sim_; }
  double dt() const { return sim_.dt(); }

 private:
  ReachSim sim_;
  uint64_t seq_ = 0;
  uint32_t episode_ = 0;
  int ep_step_ = 0;
  std::optional<int> reached_at_;
  double reward_ = 0.0, sim_time_ = 0.0;
  ReachObs obs_{};
};

}  // namespace mujoco_arm
