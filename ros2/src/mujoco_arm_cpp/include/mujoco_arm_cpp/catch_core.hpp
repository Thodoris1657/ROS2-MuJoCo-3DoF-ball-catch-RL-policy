// Trial runner for a long-running catch simulator node (ROS-free): port of catch_core.py.
//
// start() begins a trial (full pick-carry-drop or straight from the release, with gravity,
// restitution, drop point and release puff optionally pinned); step() advances one control
// step; after the outcome the arm holds for `post_roll` seconds, then the runner goes idle and
// stops stepping until the next start().
#pragma once

#include <optional>
#include <string>

#include "mujoco_arm_cpp/catch_sim.hpp"
#include "mujoco_arm_cpp/reach_sim.hpp"

namespace mujoco_arm {

/// BallCatchEnv-style kwargs (JSON object text) -> config. Unknown keys throw.
CatchConfig catch_config_from_json(const std::string& json_text, CatchConfig base = {});
/// Reach3Env-style kwargs (JSON object text) -> config. Unknown keys throw.
ReachConfig reach_config_from_json(const std::string& json_text, ReachConfig base = {});

class CatchTrialRunner {
 public:
  CatchTrialRunner(const std::string& model_path, CatchConfig cfg, std::string default_gravity,
                   std::optional<uint64_t> seed, double post_roll);

  struct StartRequest {
    std::string gravity;                 // "" = default
    double restitution = 0.0;            // <= 0 = random
    std::optional<Vec3> drop;
    std::optional<std::array<double, 2>> push;
    bool from_pick = true;
    std::optional<uint64_t> seed;
  };
  /// Throws std::invalid_argument for an unknown gravity or a bad drop point.
  void start(const StartRequest& req);
  void step(const Vec3& action);

  bool idle() const { return idle_; }
  uint64_t seq() const { return seq_; }
  uint32_t trial() const { return trial_; }
  std::string phase() const;
  const CatchObs& obs() const { return obs_; }
  CatchSim& sim() { return sim_; }
  const CatchSim& sim() const { return sim_; }
  std::string default_gravity;

 private:
  CatchSim sim_;
  double post_roll_;
  uint64_t seq_ = 0;
  uint32_t trial_ = 0;
  bool idle_ = true;
  std::optional<double> done_at_;
  CatchObs obs_{};
};

}  // namespace mujoco_arm
