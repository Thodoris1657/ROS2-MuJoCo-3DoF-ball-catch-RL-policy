#include "mujoco_arm_cpp/policies.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mujoco_arm {

namespace {

double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

class RandomController : public Controller {
 public:
  explicit RandomController(uint64_t seed) : rng_(seed) {}
  Vec3 act(const std::vector<float>&) override {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    return {u(rng_), u(rng_), u(rng_)};
  }
  std::string description() const override { return "random actions"; }

 private:
  std::mt19937_64 rng_;
};

/// Reach: IK on the target from the observation, rate-limited step towards it.
class OracleController : public Controller {
 public:
  OracleController(const std::string& model, const ReachConfig& cfg) : kin_(model, cfg) {}
  void reset() override { have_key_ = false; }
  Vec3 act(const std::vector<float>& o) override {
    if (o.size() < static_cast<size_t>(kReachObsDim))
      throw std::invalid_argument("the IK oracle needs the 24-float reach observation");
    Vec3 q, q_target, target;
    for (int i = 0; i < 3; ++i) {
      q[i] = std::atan2(static_cast<double>(o[3 + i]), static_cast<double>(o[i]));
      q_target[i] = q[i] + o[9 + i];
      target[i] = o[18 + i];
    }
    std::array<long long, 3> key;
    for (int i = 0; i < 3; ++i) key[i] = std::llround(target[i] * 1e6);
    if (!have_key_ || key != key_) {
      goal_ = kin_.solve_ik(target, q);
      key_ = key;
      have_key_ = true;
    }
    if (!goal_) return {0, 0, 0};
    Vec3 a;
    for (int i = 0; i < 3; ++i) a[i] = clip(((*goal_)[i] - q_target[i]) / kin_.max_step(), -1.0, 1.0);
    return a;
  }
  std::string description() const override { return "analytic IK oracle"; }

 private:
  ReachSim kin_;  // never stepped: kinematics, collision checks, rate limit
  bool have_key_ = false;
  std::array<long long, 3> key_{};
  std::optional<Vec3> goal_;
};

class CatchModelController : public Controller {
 public:
  CatchModelController(const std::string& model, const CatchConfig& cfg,
                       std::map<std::string, CatchParams> tuned, std::string desc)
      : kin_(model, cfg), ctrl_(tuned.empty() ? CatchController(CatchParams{}) : CatchController(tuned)),
        desc_(std::move(desc)) {}
  void reset() override { ctrl_.reset(); }
  Vec3 act(const std::vector<float>& o) override { return ctrl_.act(o.data(), o.size(), kin_); }
  std::string description() const override { return desc_; }

 private:
  CatchSim kin_;
  CatchController ctrl_;
  std::string desc_;
};

class HoldController : public Controller {
 public:
  Vec3 act(const std::vector<float>&) override { return {0, 0, 0}; }
  std::string description() const override { return "catch baseline: hold still"; }
};

class NetworkController : public Controller {
 public:
  NetworkController(const std::string& path, std::unique_ptr<Controller> base)
      : net_(path), base_(std::move(base)), path_(path) {}
  void reset() override {
    if (base_) base_->reset();
  }
  Vec3 act(const std::vector<float>& o) override {
    if (!net_.residual()) {
      if (o.size() != net_.obs_dim())
        throw std::invalid_argument("observation has " + std::to_string(o.size()) +
                                    " values, the network expects " + std::to_string(net_.obs_dim()));
      const auto y = net_.forward(o);
      return {y[0], y[1], y[2]};
    }
    // residual: action = clip(base(obs) + scale * net([obs, base(obs)]))
    const Vec3 b = base_->act(o);
    std::vector<float> x(o);
    for (double v : b) x.push_back(static_cast<float>(v));
    if (x.size() != net_.obs_dim())
      throw std::invalid_argument("residual input has " + std::to_string(x.size()) +
                                  " values, the network expects " + std::to_string(net_.obs_dim()));
    const auto y = net_.forward(x);
    // the base already returns zeros outside the flight; keep the residual quiet there too
    const bool flying = o.size() > 33 ? o[33] > 0.5f : true;
    Vec3 a;
    for (int i = 0; i < 3; ++i)
      a[i] = clip(b[i] + (flying ? net_.residual()->scale * y[i] : 0.0), -1.0, 1.0);
    return a;
  }
  std::string description() const override {
    return std::string(net_.residual() ? "residual " : "") + net_.algo() + " network " + path_;
  }

 private:
  MlpNetwork net_;
  std::unique_ptr<Controller> base_;
  std::string path_;
};

}  // namespace

// ------------------------------------------------------------------ MLP

MlpNetwork::MlpNetwork(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("policy file not found: " + path);
  const nlohmann::json j = nlohmann::json::parse(f);
  if (j.value("format", "") != "mujoco_arm_mlp_v1")
    throw std::runtime_error(path + ": not a mujoco_arm_mlp_v1 export (use scripts/export_policy.py)");
  obs_dim_ = j.at("obs_dim").get<size_t>();
  act_dim_ = j.at("act_dim").get<size_t>();
  output_ = j.value("output", "tanh");
  algo_ = j.value("algo", "mlp");
  if (j.contains("obs_norm") && !j["obs_norm"].is_null()) {
    norm_ = true;
    loc_ = j["obs_norm"].at("loc").get<std::vector<double>>();
    scale_ = j["obs_norm"].at("scale").get<std::vector<double>>();
    standard_normal_ = j["obs_norm"].value("standard_normal", false);
  }
  for (const auto& L : j.at("layers")) {
    Layer layer;
    const auto w = L.at("weight").get<std::vector<std::vector<double>>>();
    layer.out = w.size();
    layer.in = w.empty() ? 0 : w[0].size();
    for (const auto& row : w) layer.w.insert(layer.w.end(), row.begin(), row.end());
    layer.b = L.at("bias").get<std::vector<double>>();
    layer.activation = L.value("activation", "none");
    if (layer.b.size() != layer.out) throw std::runtime_error(path + ": bias/weight size mismatch");
    layers_.push_back(std::move(layer));
  }
  if (layers_.empty() || layers_.front().in != obs_dim_)
    throw std::runtime_error(path + ": first layer does not match obs_dim");
  if (j.contains("residual") && !j["residual"].is_null())
    residual_ = Residual{j["residual"].value("base", "catch_tuned"), j["residual"].value("scale", 0.3)};
}

std::vector<double> MlpNetwork::forward(const std::vector<float>& obs) const {
  std::vector<double> x(obs.begin(), obs.end());
  if (norm_) {
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = standard_normal_ ? (x[i] - loc_[i]) / scale_[i] : x[i] * scale_[i] + loc_[i];
  }
  for (const Layer& L : layers_) {
    std::vector<double> y(L.out);
    for (size_t r = 0; r < L.out; ++r) {
      double s = L.b[r];
      const double* w = &L.w[r * L.in];
      for (size_t c = 0; c < L.in; ++c) s += w[c] * x[c];
      if (L.activation == "relu") s = std::max(s, 0.0);
      else if (L.activation == "tanh") s = std::tanh(s);
      y[r] = s;
    }
    x.swap(y);
  }
  std::vector<double> a(x.begin(), x.begin() + static_cast<long>(std::min(act_dim_, x.size())));
  for (double& v : a) {
    if (output_ == "tanh") v = std::tanh(v);
    else v = clip(v, -1.0, 1.0);  // "clip" (PPO / TD3 already tanh-ed)
  }
  return a;
}

// ------------------------------------------------------------- factory

std::unique_ptr<Controller> make_controller(const std::string& spec, const PolicyPaths& paths,
                                            uint64_t seed) {
  if (spec == "oracle") return std::make_unique<OracleController>(paths.reach_model, paths.reach_config);
  if (spec == "random") return std::make_unique<RandomController>(seed);
  if (spec == "catch_hold") return std::make_unique<HoldController>();
  if (spec == "catch_model")
    return std::make_unique<CatchModelController>(paths.catch_model, paths.catch_config,
                                                  std::map<std::string, CatchParams>{},
                                                  "model-based catch controller (default parameters)");
  if (spec == "catch_tuned") {
    auto tuned = load_tuned_params(paths.catch_tuned);
    if (tuned.empty()) throw std::runtime_error("no tuned parameters at " + paths.catch_tuned);
    return std::make_unique<CatchModelController>(paths.catch_model, paths.catch_config, tuned,
                                                  "model-based catch controller (tuned per gravity)");
  }
  if (ends_with(spec, ".json")) {
    MlpNetwork probe(spec);  // validates the file, tells us whether it is residual
    std::unique_ptr<Controller> base;
    if (probe.residual()) base = make_controller(probe.residual()->base, paths, seed);
    return std::make_unique<NetworkController>(spec, std::move(base));
  }
  if (ends_with(spec, ".zip") || ends_with(spec, ".pt"))
    throw std::invalid_argument(spec + ": export it first:  python scripts/export_policy.py " + spec);
  throw std::invalid_argument("unknown policy '" + spec + "'");
}

}  // namespace mujoco_arm
