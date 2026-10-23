// Ball-catch simulator in C++ (ROS-free): a faithful port of mujoco_arm_rl/envs/ballcatch.py.
//
// Pick a ball off a cube, drop it, catch it at the top of the rebound, under Earth / Mars /
// Moon gravity. Same MJCF (assets/arm3_ballcatch.xml), same servo, same 1 kHz event logic,
// same latch rule and the same 34-float observation as the Python env, so a policy trained
// in Python runs unchanged here. test_catch_parity.cpp replays Python roll-outs and requires
// the trajectories to agree (the only intended difference is the random-number generator).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "mujoco_arm_cpp/arm_model.hpp"

namespace mujoco_arm {

constexpr double kBallR = 0.033;
constexpr double kEeR = 0.035;
constexpr double kContactDist = kBallR + kEeR;
constexpr double kHoldDist = kContactDist + 0.002;
constexpr double kPedestalTop = 0.12;
constexpr double kContactOmega = 314.0;
constexpr double kContactImp = 0.9;
inline const Vec3 kBallRest{0.0, 0.30, kPedestalTop + kBallR};

constexpr int kCatchObsDim = 34;
using CatchObs = std::array<float, kCatchObsDim>;

/// Earth / Mars / Moon, m/s^2.
double resolve_gravity(const std::string& name_or_number);
std::string gravity_name(double g);
const std::vector<std::pair<std::string, double>>& gravity_table();

/// Direct (negative) MuJoCo solref for a spring-damper contact with restitution e.
std::array<double, 2> restitution_solref(double e);

struct CatchConfig {
  double control_hz = 100.0;
  double max_joint_speed = 3.0;
  ServoGains servo{};
  std::array<double, 2> restitution{0.65, 0.85};
  Vec3 drop_lo{0.24, -0.10, 0.42};
  Vec3 drop_hi{0.34, 0.10, 0.52};
  double release_push = 0.02;
  double release_push_max = 0.04;
  double obs_pos_noise = 0.001;
  double obs_vel_noise = 0.02;
  double capture_gap = 0.006;
  double capture_radius = 0.015;
  double capture_speed = 0.5;
  double apex_speed = 0.5;
  double hold_time = 0.3;
  bool return_ball = false;
  double max_catch_seconds = 4.0;
  std::vector<double> gravity_choices{9.81};
  std::string start = "release";  // "release" | "pick"
};

struct CatchResetOptions {
  std::optional<double> gravity;
  std::optional<double> restitution;
  std::optional<Vec3> drop;
  std::optional<std::array<double, 2>> push;
  std::optional<std::string> start;
};

struct CatchStep {
  CatchObs obs{};
  bool terminated = false;
  bool truncated = false;
};

class CatchSim {
 public:
  explicit CatchSim(const std::string& model_path, CatchConfig cfg = {});
  CatchSim(const CatchSim&) = delete;
  CatchSim& operator=(const CatchSim&) = delete;

  CatchObs reset(std::optional<uint64_t> seed = std::nullopt, const CatchResetOptions& opt = {});
  CatchStep step(const Vec3& action);
  CatchObs observe();  // draws fresh sensor noise, like Python's _get_obs

  // ---- state ---------------------------------------------------------------
  const std::string& phase() const { return phase_; }
  const std::optional<std::string>& outcome() const { return outcome_; }
  bool success() const { return success_; }
  bool holding() const { return latched_; }
  bool catching() const { return (phase_ == "fall" || phase_ == "rise") && !outcome_; }
  int bounces() const { return bounces_; }
  double gravity() const { return gravity_; }
  double restitution() const { return restitution_; }
  double sim_time() const { return sim_time_; }
  std::optional<double> t_release() const { return t_release_; }
  const Vec3& drop_tip() const { return drop_tip_; }
  const Vec3& release_velocity() const { return release_vel_; }

  Vec3 q() const;
  Vec3 qd() const;
  Vec3 tau() const;
  const Vec3& q_target() const { return q_target_; }
  Vec3 ee_pos() const;
  Vec3 ee_vel();
  std::array<double, 4> ee_quat_xyzw() const;
  Vec3 ball_pos() const;
  Vec3 ball_vel() const;
  Vec3 catch_point() const;  // the green marker (mocap)

  /// Present once the ball has taken off from the bounce: true apex time / height / xy.
  struct Apex { double t, z, x, y; };
  const std::optional<Apex>& apex() const { return apex_; }
  struct CatchInfo { double t, rel_speed, ball_vz, ball_z; Vec3 offset; };
  const std::optional<CatchInfo>& catch_info() const { return catch_info_; }
  double catch_time_error() const;    // NaN if no catch
  double catch_height_error() const;  // NaN if no catch

  // ---- kinematics tools ------------------------------------------------------
  bool config_is_valid(const Vec3& q, double margin = 0.0);
  std::optional<Vec3> solve_ik(const Vec3& p, const Vec3& near = {0, 0, 0}, double margin = 0.0);
  bool reachable(const Vec3& p, double margin = 0.0);
  double ball_clearance(const Vec3& q, const Vec3& ball, bool skip_ee);
  bool drop_is_clear(const Vec3& tip, double margin = 0.005);

  double dt() const { return m_->opt.timestep * frame_skip_; }
  double max_step() const { return cfg_.max_joint_speed * dt(); }
  int max_episode_steps() const { return max_episode_steps_; }
  const CatchConfig& config() const { return cfg_; }
  const mjModel* model() const { return m_.get(); }
  const mjData* data() const { return d_.get(); }

 private:
  struct Segment {
    std::string name;
    double duration;
    bool line;                 // false = joint-space min-jerk, true = tip-space line
    Vec3 goal;
    std::string grip = "keep"; // keep | arm | release
    bool stop_on_latch = false;
    bool started = false;
    Vec3 start{};
    double t0 = 0.0;
  };

  void reset_bookkeeping();
  void set_gravity(double g);
  void set_restitution(double e);
  void grip_on(const Vec3& offset_world);
  void grip_off();
  void update_grip_anchor();
  void contacts(bool* floor, bool* arm) const;
  bool try_latch();
  void substep();
  std::string why_no_latch();
  void finish(const std::string& outcome);
  void plan_pick_and_carry();
  void plan_return();
  std::optional<Vec3> scripted_target();
  void release();
  Vec3 sample_drop();
  Vec3 ideal_catch_point() const;
  void set_marker(const Vec3& p);
  double normal(double sigma);

  CatchConfig cfg_;
  ModelPtr m_;
  DataPtr d_, scratch_;
  int frame_skip_ = 10;
  int max_episode_steps_ = 400;
  Vec3 jnt_lo_{}, jnt_hi_{};
  int tip_id_, ee_body_, ball_geom_, floor_geom_, grip_eq_, floor_pair_, bq_, bv_;
  std::array<int, 5> arm_geoms_{};
  std::vector<mjtNum> jacp_;
  std::mt19937_64 rng_;

  // episode state
  std::string phase_ = "idle";
  std::string start_mode_ = "release";
  double sim_time_ = 0.0;
  Vec3 q_target_{}, prev_action_{};
  std::vector<Segment> segments_;
  size_t seg_i_ = 0;
  bool latched_ = false, grip_armed_ = false, released_clear_ = true, floor_contact_ = false;
  Vec3 grip_offset_{0, 0, -kHoldDist}, grip_offset_goal_{0, 0, -kHoldDist};
  int bounces_ = 0;
  std::optional<double> t_release_, t_bounce_, t_catch_;
  std::optional<Apex> apex_;
  std::optional<CatchInfo> catch_info_;
  std::optional<std::string> outcome_;
  bool success_ = false;
  int catch_steps_ = 0;
  double gravity_ = 9.81, restitution_ = 0.75;
  Vec3 drop_tip_{}, release_vel_{};
  std::optional<std::array<double, 2>> pending_push_;
};

}  // namespace mujoco_arm
