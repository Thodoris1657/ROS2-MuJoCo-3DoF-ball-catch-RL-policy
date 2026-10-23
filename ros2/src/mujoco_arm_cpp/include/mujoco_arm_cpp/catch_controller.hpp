// Model-based ball-catch controller (ROS-free): port of mujoco_arm_rl/catching.py.
//
// Works from the 34-float observation only, plus IK from a never-stepped CatchSim:
// ballistic least-squares estimate -> rebound-apex prediction -> cup plan above the apex ->
// "stay above the falling ball" constraint -> servo-lag lead -> IK -> rate-limited joint step.
// Per-gravity parameters come from configs/catch_tuned.json (scripts/tune_catch.py).
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "mujoco_arm_cpp/catch_sim.hpp"

namespace mujoco_arm {

struct CatchParams {
  double e_prior = 0.75;
  double catch_offset = 0.002;
  double lookahead = 0.05;
  double clearance = 0.02;
  double vz_match = 0.0;
  double overshoot = 0.0;
  double liftoff = 0.008;  // m: rebound samples are trusted once the ball is this far off the floor
};

/// {gravity name: params} from catch_tuned.json; empty map if the file is missing.
std::map<std::string, CatchParams> load_tuned_params(const std::string& path);

/// Weighted least-squares (p0, v0) of a ballistic flight with known gravity.
class BallisticFit {
 public:
  BallisticFit(double g, double pos_std, double vel_std);
  void clear();
  void add(double t, const Vec3& p, const Vec3& v);
  size_t size() const { return t_.size(); }
  /// Fitted position and velocity at t_now.
  void state_at(double t_now, Vec3* p, Vec3* v) const;

 private:
  double g_, wp2_, wv2_;
  std::vector<double> t_;
  std::vector<Vec3> p_, v_;
};

class CatchController {
 public:
  explicit CatchController(CatchParams params = {}, double pos_std = 0.001, double vel_std = 0.02);
  explicit CatchController(std::map<std::string, CatchParams> by_gravity, double pos_std = 0.001,
                           double vel_std = 0.02);

  void reset();
  /// obs: the 34 floats of CatchSim / BallCatchEnv. kin: any CatchSim on the same model
  /// (used for IK and the rate limit only; it is not stepped).
  Vec3 act(const float* obs, size_t n, CatchSim& kin);
  const CatchParams& params_for(double g) const;

  /// Last plan, for logging: predicted apex, cup goal and the aimed target.
  struct Plan { double t_apex = 0; Vec3 apex{}, goal{}, target{}; };
  const Plan& last_plan() const { return plan_; }

 private:
  CatchParams default_;
  std::map<std::string, CatchParams> by_g_;
  double pos_std_, vel_std_;
  double t_ = 0.0;
  std::optional<BallisticFit> fit_;
  bool bounced_ = false;
  Plan plan_;
};

}  // namespace mujoco_arm
