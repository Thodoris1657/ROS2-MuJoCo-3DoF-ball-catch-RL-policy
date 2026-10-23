#include "mujoco_arm_cpp/catch_controller.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mujoco_arm {

std::map<std::string, CatchParams> load_tuned_params(const std::string& path) {
  std::map<std::string, CatchParams> out;
  std::ifstream f(path);
  if (!f) return out;
  const nlohmann::json blob = nlohmann::json::parse(f);
  if (!blob.contains("params")) return out;
  for (const auto& [name, p] : blob["params"].items()) {
    CatchParams c;
    c.e_prior = p.value("e_prior", c.e_prior);
    c.catch_offset = p.value("catch_offset", c.catch_offset);
    c.lookahead = p.value("lookahead", c.lookahead);
    c.clearance = p.value("clearance", c.clearance);
    c.vz_match = p.value("vz_match", c.vz_match);
    c.overshoot = p.value("overshoot", c.overshoot);
    c.liftoff = p.value("liftoff", c.liftoff);
    out[name] = c;
  }
  return out;
}

// ------------------------------------------------------------------ fit

BallisticFit::BallisticFit(double g, double pos_std, double vel_std)
    : g_(g), wp2_(1.0 / (pos_std * pos_std)), wv2_(1.0 / (vel_std * vel_std)) {}

void BallisticFit::clear() {
  t_.clear();
  p_.clear();
  v_.clear();
}

void BallisticFit::add(double t, const Vec3& p, const Vec3& v) {
  t_.push_back(t);
  p_.push_back(p);
  v_.push_back(v);
}

void BallisticFit::state_at(double t_now, Vec3* p0, Vec3* v0) const {
  // Rows (per axis): wp [1, t] (p0, v0) = wp Pc ; wv [0, 1] (p0, v0) = wv Vc, with the known
  // gravity removed: Pc = p + g t^2 / 2 z, Vc = v + g t z. Solved via the 2x2 normal equations.
  const size_t n = t_.size();
  double st = 0.0, stt = 0.0;
  for (double ti : t_) {
    const double t = ti - t_now;
    st += t;
    stt += t * t;
  }
  const double a11 = wp2_ * static_cast<double>(n);
  const double a12 = wp2_ * st;
  const double a22 = wp2_ * stt + wv2_ * static_cast<double>(n);
  const double det = a11 * a22 - a12 * a12;
  for (int ax = 0; ax < 3; ++ax) {
    double sp = 0.0, stp = 0.0, sv = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double t = t_[i] - t_now;
      const double pc = p_[i][ax] + (ax == 2 ? 0.5 * g_ * t * t : 0.0);
      const double vc = v_[i][ax] + (ax == 2 ? g_ * t : 0.0);
      sp += pc;
      stp += t * pc;
      sv += vc;
    }
    const double b1 = wp2_ * sp;
    const double b2 = wp2_ * stp + wv2_ * sv;
    (*p0)[ax] = (a22 * b1 - a12 * b2) / det;
    (*v0)[ax] = (a11 * b2 - a12 * b1) / det;
  }
}

// ----------------------------------------------------------- controller

CatchController::CatchController(CatchParams params, double pos_std, double vel_std)
    : default_(params), pos_std_(pos_std), vel_std_(vel_std) {
  reset();
}

CatchController::CatchController(std::map<std::string, CatchParams> by_gravity, double pos_std,
                                 double vel_std)
    : by_g_(std::move(by_gravity)), pos_std_(pos_std), vel_std_(vel_std) {
  reset();
}

void CatchController::reset() {
  t_ = 0.0;
  fit_.reset();
  bounced_ = false;
  plan_ = Plan{};
}

const CatchParams& CatchController::params_for(double g) const {
  if (by_g_.empty()) return default_;
  const auto it = by_g_.find(gravity_name(g));
  if (it != by_g_.end()) return it->second;
  // nearest tuned gravity
  const CatchParams* best = &by_g_.begin()->second;
  double best_d = std::numeric_limits<double>::infinity();
  for (const auto& [name, prm] : by_g_) {
    double gv = 1e9;
    for (const auto& [n2, v] : gravity_table())
      if (n2 == name) gv = v;
    if (std::abs(gv - g) < best_d) {
      best_d = std::abs(gv - g);
      best = &prm;
    }
  }
  return *best;
}

Vec3 CatchController::act(const float* obs_f, size_t n, CatchSim& kin) {
  if (n < 33) throw std::invalid_argument("catch controller needs the 34-float catch observation");
  double obs[kCatchObsDim] = {0};
  for (size_t i = 0; i < std::min(n, static_cast<size_t>(kCatchObsDim)); ++i) obs[i] = obs_f[i];
  const double g = obs[30] * 9.81;
  const CatchParams& prm = params_for(g);
  const Vec3 tip{obs[15], obs[16], obs[17]};
  const Vec3 bp{obs[21], obs[22], obs[23]};
  const Vec3 bv{obs[24], obs[25], obs[26]};
  const bool bounced = obs[31] > 0.5, holding = obs[32] > 0.5;
  const double pstd = std::max(pos_std_, 1e-4), vstd = std::max(vel_std_, 1e-3);

  if (n > 33 && obs[33] < 0.5 && !holding) {  // not in flight: wait for the release
    reset();
    return {0, 0, 0};
  }
  if (!fit_) fit_.emplace(g, pstd, vstd);
  if (bounced && !bounced_) {
    bounced_ = true;
    fit_->clear();
  }
  if (holding) {
    t_ += kin.dt();
    return {0, 0, 0};
  }

  // a ball far from where the current flight puts it = a new trial without a pause: start over
  if (fit_->size() >= 2 && !bounced_) {
    Vec3 pp, vv;
    fit_->state_at(t_, &pp, &vv);
    const double d = std::sqrt((pp[0] - bp[0]) * (pp[0] - bp[0]) + (pp[1] - bp[1]) * (pp[1] - bp[1]) +
                               (pp[2] - bp[2]) * (pp[2] - bp[2]));
    if (d > 0.05) {
      reset();
      fit_.emplace(g, pstd, vstd);
    }
  } else if (bounced_ && !bounced) {  // the bounce flag went back to 0: new trial
    reset();
    fit_.emplace(g, pstd, vstd);
  }

  // ballistic samples only: skip the few ms the ball is squashed on the floor
  const bool skip = bounced_ && (bp[2] < kBallR + prm.liftoff || (bv[2] < -0.05 && fit_->size() == 0));
  if (!skip) fit_->add(t_, bp, bv);
  Vec3 p_hat = bp, v_hat = bv;
  if (fit_->size() >= 2) fit_->state_at(t_, &p_hat, &v_hat);

  // predict the rebound apex
  double t_apex, z_apex;
  if (!bounced_) {
    const double h = p_hat[2] - kBallR;
    const double v_in = std::sqrt(std::max(v_hat[2] * v_hat[2] + 2.0 * g * std::max(h, 0.0), 0.0));
    const double t_imp = (v_hat[2] + v_in) / g;
    const double v_out = prm.e_prior * v_in;
    t_apex = t_imp + v_out / g;
    z_apex = kBallR + v_out * v_out / (2.0 * g);
  } else {
    const double vz = std::max(v_hat[2], 0.0);
    t_apex = vz / g;
    z_apex = p_hat[2] + vz * vz / (2.0 * g);
  }
  Vec3 goal{p_hat[0] + v_hat[0] * t_apex, p_hat[1] + v_hat[1] * t_apex,
            z_apex + kContactDist + prm.catch_offset};
  plan_.t_apex = t_apex;
  plan_.apex = {goal[0], goal[1], z_apex};

  if (bounced_ && prm.vz_match > 0.0)
    goal[2] += prm.vz_match * std::max(v_hat[2], 0.0) * std::min(t_apex, 0.1);

  if (!bounced_) {
    if (tip[2] - bp[2] - kContactDist < prm.clearance) {
      goal[0] = tip[0];
      goal[1] = tip[1];
    }
    const double tl = prm.lookahead;
    const double z_future = p_hat[2] + v_hat[2] * tl - 0.5 * g * tl * tl;
    const double floor_z = std::max(z_future, kBallR) + kContactDist + prm.clearance;
    goal[2] = std::max(goal[2], std::min(floor_z, p_hat[2] + kContactDist + prm.clearance));
  }

  // servo-lag lead
  const Vec3 err{goal[0] - tip[0], goal[1] - tip[1], goal[2] - tip[2]};
  const double en = std::sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2]);
  const double k = prm.overshoot * std::min(1.0, 0.03 / std::max(en, 1e-9));
  const Vec3 target{goal[0] + k * err[0], goal[1] + k * err[1], goal[2] + k * err[2]};

  Vec3 q_cmd;
  for (int i = 0; i < 3; ++i) q_cmd[i] = std::atan2(obs[3 + i], obs[i]) + obs[9 + i];
  auto q_goal = kin.solve_ik(target, q_cmd);
  if (!q_goal) q_goal = kin.solve_ik(goal, q_cmd);
  plan_.goal = goal;
  plan_.target = target;
  t_ += kin.dt();
  if (!q_goal) return {0, 0, 0};
  Vec3 step;
  double mx = 1.0;
  for (int i = 0; i < 3; ++i) {
    step[i] = ((*q_goal)[i] - q_cmd[i]) / kin.max_step();
    mx = std::max(mx, std::abs(step[i]));
  }
  for (int i = 0; i < 3; ++i) step[i] /= mx;
  return step;
}

}  // namespace mujoco_arm
