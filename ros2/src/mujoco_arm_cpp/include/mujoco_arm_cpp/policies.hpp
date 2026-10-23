// Policies behind one interface (ROS-free): observation in, normalised action out.
//
//   oracle         reach: analytic-IK reference controller, from the observation only
//   random         uniform random actions
//   catch_tuned    ball catch: model-based controller, per-gravity tuned parameters
//   catch_model    ball catch: model-based controller, default parameters
//   catch_hold     ball catch: never moves (baseline)
//   <file>.json    a network exported by scripts/export_policy.py (SB3 SAC/TD3/PPO or the
//                  repo's TorchRL PPO), optionally a *residual* policy on top of catch_tuned
//
// The ROS policy node never sees simulator internals -- exactly like a real controller.
#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mujoco_arm_cpp/catch_controller.hpp"
#include "mujoco_arm_cpp/catch_sim.hpp"
#include "mujoco_arm_cpp/reach_sim.hpp"

namespace mujoco_arm {

class Controller {
 public:
  virtual ~Controller() = default;
  virtual Vec3 act(const std::vector<float>& obs) = 0;
  virtual void reset() {}
  virtual std::string description() const = 0;
};

/// Dense MLP loaded from the JSON written by scripts/export_policy.py.
class MlpNetwork {
 public:
  explicit MlpNetwork(const std::string& json_path);
  /// Raw network output after the export's output transform (tanh / clip / mean-of-gaussian).
  std::vector<double> forward(const std::vector<float>& obs) const;
  size_t obs_dim() const { return obs_dim_; }
  size_t act_dim() const { return act_dim_; }
  const std::string& algo() const { return algo_; }

  struct Residual { std::string base; double scale = 0.0; };
  const std::optional<Residual>& residual() const { return residual_; }

 private:
  struct Layer {
    size_t in, out;
    std::vector<double> w;  // row-major out x in
    std::vector<double> b;
    std::string activation;  // relu | tanh | none
  };
  std::vector<Layer> layers_;
  size_t obs_dim_ = 0, act_dim_ = 0;
  std::string output_ = "tanh", algo_;
  bool norm_ = false, standard_normal_ = false;
  std::vector<double> loc_, scale_;
  std::optional<Residual> residual_;
};

struct PolicyPaths {
  std::string catch_model;   // assets/arm3_ballcatch.xml (IK for the catch controllers)
  std::string reach_model;   // assets/arm3.xml (IK for the oracle)
  std::string catch_tuned;   // configs/catch_tuned.json
  CatchConfig catch_config;  // must match the simulator's (rate limit, control rate)
  ReachConfig reach_config;
};

/// Build a controller from a spec string (see the header comment). Throws on bad specs.
std::unique_ptr<Controller> make_controller(const std::string& spec, const PolicyPaths& paths,
                                            uint64_t seed = 0);

}  // namespace mujoco_arm
