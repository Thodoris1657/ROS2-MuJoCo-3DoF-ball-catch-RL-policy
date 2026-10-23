#include "mujoco_arm_cpp/batch.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mujoco_arm_cpp/catch_core.hpp"

namespace mujoco_arm {

// ---------------------------------------------------------------- WorkerPool

WorkerPool::WorkerPool(int threads) {
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { worker_loop(); });
}

WorkerPool::~WorkerPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  start_cv_.notify_all();
  for (auto& w : workers_) w.join();
}

void WorkerPool::drain() {
  // take indices until the job is finished
  for (int i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1)) (*job_)(i);
}

void WorkerPool::worker_loop() {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      start_cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    drain();
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (--running_ == 0) done_cv_.notify_one();
    }
  }
}

void WorkerPool::parallel_for(int count, const std::function<void(int)>& fn) {
  if (count <= 0) return;
  if (workers_.empty() || count == 1) {
    for (int i = 0; i < count; ++i) fn(i);
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    job_ = &fn;
    count_ = count;
    next_.store(0);
    running_ = static_cast<int>(workers_.size());
    ++generation_;
  }
  start_cv_.notify_all();
  drain();  // the calling thread helps too
  std::unique_lock<std::mutex> lk(mu_);
  done_cv_.wait(lk, [&] { return running_ == 0; });
  job_ = nullptr;
}

// ---------------------------------------------------------------- CatchBatch

CatchBatch::CatchBatch(const std::string& model_path, const BatchConfig& cfg)
    : cfg_(cfg), pool_(cfg.threads) {
  if (cfg_.n_envs <= 0) throw std::invalid_argument("n_envs must be > 0");
  envs_.resize(static_cast<size_t>(cfg_.n_envs));
  // building the models is the slow part (XML parsing), so do it in parallel as well
  pool_.parallel_for(cfg_.n_envs, [&](int i) {
    Env& e = envs_[static_cast<size_t>(i)];
    e.sim = std::make_unique<CatchSim>(model_path, cfg_.sim);
    if (cfg_.residual) {
      e.base = cfg_.base_params.empty() ? std::make_unique<CatchController>(CatchParams{})
                                        : std::make_unique<CatchController>(cfg_.base_params);
    }
  });
  const size_t n = envs_.size();
  obs_.assign(n * obs_dim(), 0.0f);
  final_obs_.assign(n * obs_dim(), 0.0f);
  reward_.assign(n, 0.0);
  gravity_.assign(n, 0.0);
  terminated_.assign(n, 0);
  truncated_.assign(n, 0);
  success_.assign(n, 0);
  outcome_.assign(n, "");
  reset(cfg_.seed);
}

void CatchBatch::compute_base(int i) {
  Env& e = envs_[static_cast<size_t>(i)];
  if (!e.base) return;
  // the env's own simulator is used for the IK of its controller (same model, same thread)
  e.base_action = e.base->act(e.obs.data(), e.obs.size(), *e.sim);
}

void CatchBatch::write_obs(int i, float* dst) {
  const Env& e = envs_[static_cast<size_t>(i)];
  std::copy(e.obs.begin(), e.obs.end(), dst);
  if (cfg_.residual)
    for (int k = 0; k < 3; ++k) dst[kCatchObsDim + k] = static_cast<float>(e.base_action[k]);
}

void CatchBatch::reset(std::optional<uint64_t> seed) {
  pool_.parallel_for(num_envs(), [&](int i) {
    Env& e = envs_[static_cast<size_t>(i)];
    e.obs = e.sim->reset(seed ? std::optional<uint64_t>(*seed + static_cast<uint64_t>(i)) : std::nullopt);
    if (e.base) e.base->reset();
    compute_base(i);
    write_obs(i, &obs_[static_cast<size_t>(i) * obs_dim()]);
    terminated_[i] = truncated_[i] = success_[i] = 0;
    reward_[i] = 0.0;
  });
}

void CatchBatch::step(const double* actions) {
  actions_ = actions;
  pool_.parallel_for(num_envs(), [&](int i) {
    Env& e = envs_[static_cast<size_t>(i)];
    const double* a = actions_ + 3 * static_cast<size_t>(i);
    Vec3 act{a[0], a[1], a[2]};
    if (e.base) {
      for (int k = 0; k < 3; ++k)
        act[k] = std::clamp(e.base_action[k] + cfg_.residual_scale * std::clamp(a[k], -1.0, 1.0), -1.0, 1.0);
    }
    const CatchStep r = e.sim->step(act);
    e.obs = r.obs;
    reward_[i] = r.reward;
    terminated_[i] = r.terminated;
    truncated_[i] = r.truncated;
    success_[i] = 0;
    const size_t off = static_cast<size_t>(i) * obs_dim();
    if (r.terminated || r.truncated) {
      // keep the last observation (SB3 needs it for bootstrapping on truncation)
      compute_base(i);
      write_obs(i, &final_obs_[off]);
      success_[i] = e.sim->success();
      gravity_[i] = e.sim->gravity();
      outcome_[static_cast<size_t>(i)] = e.sim->outcome().value_or("running");
      ++episodes_;
      // and start the next episode right away (RNG stream of this env just continues)
      e.obs = e.sim->reset();
      if (e.base) e.base->reset();
    }
    compute_base(i);
    write_obs(i, &obs_[off]);
  });
  actions_ = nullptr;
}

// ---------------------------------------------------------------- evaluate_params

CatchConfig stress_config(CatchConfig base) {
  return catch_config_from_json(
      R"({"restitution": [0.55, 0.90], "release_push": 0.04, "release_push_max": 0.08,
          "obs_pos_noise": 0.003, "obs_vel_noise": 0.05})",
      base);
}

EvalResult evaluate_params(const std::string& model_path, const CatchParams& params, const EvalOptions& opt) {
  CatchConfig nominal;
  nominal.gravity_choices = {opt.gravity};
  nominal.max_joint_speed = opt.max_joint_speed;
  const CatchConfig stress = stress_config(nominal);
  const int total = opt.n_nominal + opt.n_stress;

  struct Out { bool success = false; double soft = 0.0; std::string outcome; double e = 0, drop_z = 0; };
  std::vector<Out> outs(static_cast<size_t>(std::max(total, 0)));

  WorkerPool pool(opt.threads);
  const int chunks = pool.size();
  // every chunk owns its own two simulators; trial i always uses the same seed
  pool.parallel_for(chunks, [&](int c) {
    CatchSim sims[2] = {CatchSim(model_path, nominal), CatchSim(model_path, stress)};
    for (int i = c; i < total; i += chunks) {
      const bool is_stress = i >= opt.n_nominal;
      const int j = is_stress ? i - opt.n_nominal : i;
      CatchSim& sim = sims[is_stress ? 1 : 0];
      CatchObs obs = sim.reset(opt.seed0 + static_cast<uint64_t>(j) + (is_stress ? 1000003ull : 0ull));
      CatchParams p = params;
      if (opt.oracle_e) p.e_prior = sim.restitution();
      CatchController ctrl(p);
      for (int k = 0; k < 2000; ++k) {
        const auto r = sim.step(ctrl.act(obs.data(), obs.size(), sim));
        obs = r.obs;
        if (r.terminated || r.truncated) break;
      }
      Out& o = outs[static_cast<size_t>(i)];
      o.success = sim.success();
      o.soft = o.success ? 1.0 - std::min(sim.catch_info()->rel_speed / 0.5, 1.0) : 0.0;
      o.outcome = sim.outcome().value_or("running");
      o.e = sim.restitution();
      o.drop_z = sim.drop_tip()[2];
    }
  });

  EvalResult res;
  res.trials = total;
  double soft = 0.0;
  for (int i = 0; i < total; ++i) {
    const Out& o = outs[static_cast<size_t>(i)];
    res.caught += o.success;
    soft += o.soft;
    ++res.outcomes[o.outcome.substr(0, o.outcome.find(" ("))];
    if (opt.per_trial) res.per_trial.push_back({i >= opt.n_nominal, o.e, o.drop_z, o.success, o.outcome});
  }
  res.success = total ? static_cast<double>(res.caught) / total : 0.0;
  res.softness = total ? soft / total : 0.0;
  return res;
}

}  // namespace mujoco_arm
