// Many catch simulators stepped together (for RL training and for the tuner).
//
// CatchBatch keeps N independent CatchSim objects (each one loads its own mjModel, so they
// can run in different threads without sharing anything) and steps them in a small thread
// pool. When an episode ends the env is reset right away in the same step() call, and the
// last observation of the old episode is kept in final_obs(). This is the "same-step"
// autoreset that Stable-Baselines3 expects.
//
// Optional residual mode: every env also runs the model-based CatchController, its action is
// appended to the observation (34 -> 37 values) and the action from Python is used as a
// correction:  a = clip(base + scale * residual).  Same thing as residual.py, but in C++.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mujoco_arm_cpp/catch_controller.hpp"
#include "mujoco_arm_cpp/catch_sim.hpp"

namespace mujoco_arm {

/// Very small fixed-size thread pool. parallel_for(n, f) calls f(0..n-1) and waits.
/// The calling thread also takes work, so threads = 1 means "no extra threads".
class WorkerPool {
 public:
  explicit WorkerPool(int threads);
  ~WorkerPool();
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  void parallel_for(int count, const std::function<void(int)>& fn);
  int size() const { return static_cast<int>(workers_.size()) + 1; }

 private:
  void worker_loop();
  void drain();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable start_cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  int running_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

struct BatchConfig {
  CatchConfig sim;          // randomisation, noise, gravity choices ...
  int n_envs = 8;
  int threads = 0;          // 0 = std::thread::hardware_concurrency()
  uint64_t seed = 0;
  bool residual = false;    // see the header comment
  double residual_scale = 0.2;
  std::map<std::string, CatchParams> base_params;  // empty = default CatchParams
};

class CatchBatch {
 public:
  CatchBatch(const std::string& model_path, const BatchConfig& cfg);

  int num_envs() const { return static_cast<int>(envs_.size()); }
  int obs_dim() const { return cfg_.residual ? kCatchObsDim + 3 : kCatchObsDim; }
  int threads() const { return pool_.size(); }

  /// Reset all envs. With a seed, env i is re-seeded with seed + i.
  void reset(std::optional<uint64_t> seed = std::nullopt);
  /// actions: num_envs x 3 values (row major). Finished envs are reset inside this call.
  void step(const double* actions);

  // Buffers of the last reset()/step() (row major, num_envs rows).
  const std::vector<float>& obs() const { return obs_; }
  const std::vector<float>& final_obs() const { return final_obs_; }  // valid where done
  const std::vector<double>& rewards() const { return reward_; }
  const std::vector<uint8_t>& terminated() const { return terminated_; }
  const std::vector<uint8_t>& truncated() const { return truncated_; }
  const std::vector<uint8_t>& success() const { return success_; }  // of the episode that just ended
  const std::vector<double>& gravity() const { return gravity_; }    // of the episode that just ended
  const std::string& last_outcome(int i) const { return outcome_[static_cast<size_t>(i)]; }
  uint64_t episodes_finished() const { return episodes_; }

 private:
  struct Env {
    std::unique_ptr<CatchSim> sim;
    std::unique_ptr<CatchController> base;
    CatchObs obs{};
    Vec3 base_action{0, 0, 0};
  };
  void write_obs(int i, float* dst);
  void compute_base(int i);

  BatchConfig cfg_;
  std::vector<Env> envs_;
  WorkerPool pool_;
  std::vector<float> obs_, final_obs_;
  std::vector<double> reward_, gravity_;
  std::vector<uint8_t> terminated_, truncated_, success_;
  std::vector<std::string> outcome_;
  std::atomic<uint64_t> episodes_{0};
  const double* actions_ = nullptr;
};

// ------------------------------------------------------------------ tuner helper

struct EvalOptions {
  double gravity = 9.81;
  int n_nominal = 128;
  int n_stress = 128;
  uint64_t seed0 = 1;
  int threads = 0;
  double max_joint_speed = 3.0;
  bool oracle_e = false;     // give the controller the true restitution (upper bound test)
  bool per_trial = false;
};

struct TrialRecord {
  bool stress;
  double restitution, drop_z;
  bool success;
  std::string outcome;
};

struct EvalResult {
  int trials = 0, caught = 0;
  double success = 0.0, softness = 0.0;   // softness = mean of (1 - impact / 0.5 m/s) over all trials
  std::map<std::string, int> outcomes;
  std::vector<TrialRecord> per_trial;
};

/// Run n_nominal + n_stress randomised trials with fixed controller parameters. Trial i uses
/// seed seed0 + i (stress trials: seed0 + 1000003 + i), so the result does not depend on the
/// number of threads. Used by tools/catch_eval_params.cpp and by the Python tuner.
EvalResult evaluate_params(const std::string& model_path, const CatchParams& params, const EvalOptions& opt);

/// The stress randomisation used everywhere (same as catch_eval.STRESS_KWARGS in Python).
CatchConfig stress_config(CatchConfig base);

}  // namespace mujoco_arm
