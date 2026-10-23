#include "mujoco_arm_cpp/reach_sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace mujoco_arm {

namespace {
double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }
}  // namespace

ReachSim::ReachSim(const std::string& model_path, ReachConfig cfg) : cfg_(std::move(cfg)) {
  m_ = load_model(model_path);
  d_.reset(mj_makeData(m_.get()));
  scratch_.reset(mj_makeData(m_.get()));
  if (m_->nu != 3 || m_->nq != 3) throw std::runtime_error("expected the 3-joint arm model");
  const double ts = m_->opt.timestep;
  frame_skip_ = std::max(1, static_cast<int>(std::lround(1.0 / (cfg_.control_hz * ts))));
  max_episode_steps_ = std::max(1, static_cast<int>(std::lround(cfg_.episode_seconds / (ts * frame_skip_))));
  tip_id_ = require_id(m_.get(), mjOBJ_SITE, "tip");
  joint_limits(m_.get(), &lo_, &hi_);
  scratch_->qpos[0] = 0.0;
  scratch_->qpos[1] = 1.0;
  scratch_->qpos[2] = 0.8;
  configure_servo(m_.get(), scratch_.get(), cfg_.servo);
  if (cfg_.fixed_target) {
    const auto [ok, why] = check_target_reachable(*cfg_.fixed_target);
    if (!ok) throw std::invalid_argument("fixed_target is unreachable: " + why);
  }
  rng_.seed(0);
}

Vec3 ReachSim::q() const { return {d_->qpos[0], d_->qpos[1], d_->qpos[2]}; }
Vec3 ReachSim::qd() const { return {d_->qvel[0], d_->qvel[1], d_->qvel[2]}; }
Vec3 ReachSim::tau() const { return {d_->qfrc_actuator[0], d_->qfrc_actuator[1], d_->qfrc_actuator[2]}; }
Vec3 ReachSim::ee_pos() const {
  const mjtNum* p = d_->site_xpos + 3 * tip_id_;
  return {p[0], p[1], p[2]};
}
std::array<double, 4> ReachSim::ee_quat_xyzw() const {
  mjtNum w[4];
  mju_mat2Quat(w, d_->site_xmat + 9 * tip_id_);
  return {w[1], w[2], w[3], w[0]};
}
Vec3 ReachSim::target_pos() const { return {d_->mocap_pos[0], d_->mocap_pos[1], d_->mocap_pos[2]}; }

bool ReachSim::config_is_valid(const Vec3& q, double margin) {
  for (int i = 0; i < 3; ++i)
    if (q[i] < lo_[i] + margin || q[i] > hi_[i] - margin) return false;
  for (int i = 0; i < 3; ++i) scratch_->qpos[i] = q[i];
  mj_forward(m_.get(), scratch_.get());
  return scratch_->ncon == 0;
}

std::optional<Vec3> ReachSim::solve_ik(const Vec3& p, const Vec3& near, double margin) {
  std::optional<Vec3> best;
  double best_cost = std::numeric_limits<double>::infinity();
  for (const Vec3& q : analytic_ik(p)) {
    if (!config_is_valid(q, margin)) continue;
    double c = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double w = wrap_angle(q[i] - near[i]);
      c += w * w;
    }
    if (c < best_cost) {
      best_cost = c;
      best = q;
    }
  }
  return best;
}

std::pair<bool, std::string> ReachSim::check_target_reachable(const Vec3& p) {
  const double dist = distance_to_shoulder(p);
  char buf[256];
  if (dist > kMaxReach) {
    std::snprintf(buf, sizeof(buf),
                  "%.3f m from the shoulder at (0.0, 0.0, 0.22), but the arm only reaches %.2f m",
                  dist, kMaxReach);
    return {false, buf};
  }
  if (!solve_ik(p))
    return {false, "no collision-free joint configuration within the joint limits puts the tip "
                   "there (too close to the shoulder or to the floor/base?)"};
  std::snprintf(buf, sizeof(buf), "%.3f m from the shoulder (max %.2f m)", dist, kMaxReach);
  return {true, buf};
}

Vec3 ReachSim::sample_target() {
  if (cfg_.fixed_target) return *cfg_.fixed_target;
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.16 * 0.16 * 0.16, 0.47 * 0.47 * 0.47);
  for (int k = 0; k < 500; ++k) {
    Vec3 v{n(rng_), n(rng_), n(rng_)};
    const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-12;
    const double r = std::cbrt(u(rng_));
    const Vec3 p{v[0] / len * r, v[1] / len * r, kShoulderZ + v[2] / len * r};
    if (p[2] < 0.08) continue;
    if (solve_ik(p, {0, 0, 0}, 0.05)) return p;
  }
  return {0.3, 0.0, kShoulderZ};
}

Vec3 ReachSim::sample_start() {
  std::uniform_real_distribution<double> u(-cfg_.start_noise, cfg_.start_noise);
  for (int k = 0; k < 100; ++k) {
    Vec3 q{u(rng_), u(rng_), u(rng_)};
    if (config_is_valid(q)) return q;
  }
  return {0, 0, 0};
}

ReachObs ReachSim::observe() const {
  ReachObs o{};
  const Vec3 qq = q(), qv = qd(), ee = ee_pos(), tg = target_pos();
  for (int i = 0; i < 3; ++i) {
    o[i] = static_cast<float>(std::cos(qq[i]));
    o[3 + i] = static_cast<float>(std::sin(qq[i]));
    o[6 + i] = static_cast<float>(qv[i] / 10.0);
    o[9 + i] = static_cast<float>(q_target_[i] - qq[i]);
    o[12 + i] = static_cast<float>(prev_action_[i]);
    o[15 + i] = static_cast<float>(ee[i]);
    o[18 + i] = static_cast<float>(tg[i]);
    o[21 + i] = static_cast<float>(10.0 * (tg[i] - ee[i]));
  }
  return o;
}

ReachObs ReachSim::reset(std::optional<uint64_t> seed, const std::optional<Vec3>& target,
                         const std::optional<Vec3>& qpos) {
  if (seed) rng_.seed(*seed);
  mj_resetData(m_.get(), d_.get());
  const Vec3 q0 = qpos ? *qpos : sample_start();
  for (int i = 0; i < 3; ++i) d_->qpos[i] = q0[i];
  const Vec3 tg = target ? *target : sample_target();
  for (int i = 0; i < 3; ++i) d_->mocap_pos[i] = tg[i];
  q_target_ = q0;
  for (int i = 0; i < 3; ++i) d_->ctrl[i] = q_target_[i];
  prev_action_ = {0, 0, 0};
  mj_forward(m_.get(), d_.get());
  elapsed_ = 0;
  return observe();
}

void ReachSim::set_target(const Vec3& p) {
  const auto [ok, why] = check_target_reachable(p);
  if (!ok) throw std::invalid_argument("unreachable target: " + why);
  for (int i = 0; i < 3; ++i) d_->mocap_pos[i] = p[i];
  mj_forward(m_.get(), d_.get());
}

Vec3 ReachSim::action_from_joint_goal(const Vec3& q_goal) const {
  Vec3 a;
  for (int i = 0; i < 3; ++i) a[i] = clip((q_goal[i] - q_target_[i]) / max_step(), -1.0, 1.0);
  return a;
}

ReachSim::Step ReachSim::step(const Vec3& action) {
  Vec3 a;
  for (int i = 0; i < 3; ++i) a[i] = clip(std::isfinite(action[i]) ? action[i] : 0.0, -1.0, 1.0);
  const Vec3 qq = q();
  for (int i = 0; i < 3; ++i) {
    double t = q_target_[i] + a[i] * max_step();
    t = clip(t, qq[i] - 0.3, qq[i] + 0.3);
    q_target_[i] = clip(t, lo_[i], hi_[i]);
    d_->ctrl[i] = q_target_[i];
  }
  for (int k = 0; k < frame_skip_; ++k) mj_step(m_.get(), d_.get());
  mj_rnePostConstraint(m_.get(), d_.get());

  const Vec3 ee = ee_pos(), tg = target_pos();
  const double dist = std::sqrt((tg[0] - ee[0]) * (tg[0] - ee[0]) + (tg[1] - ee[1]) * (tg[1] - ee[1]) +
                                (tg[2] - ee[2]) * (tg[2] - ee[2]));
  ++elapsed_;
  double aa = 0.0, da = 0.0;
  for (int i = 0; i < 3; ++i) {
    aa += a[i] * a[i];
    da += (a[i] - prev_action_[i]) * (a[i] - prev_action_[i]);
  }
  Step out;
  out.reward = -dist + cfg_.precision_weight * (1.0 - std::tanh(dist / cfg_.precision_scale)) -
               cfg_.action_cost * aa - cfg_.action_rate_cost * da;
  prev_action_ = a;
  out.distance = dist;
  out.success = dist < cfg_.success_threshold;
  for (int i = 0; i < m_->nq; ++i)
    if (!std::isfinite(d_->qpos[i])) out.terminated = true;
  out.truncated = elapsed_ >= max_episode_steps_;
  out.obs = observe();
  return out;
}

// ------------------------------------------------------------ node core

ReachSimCore::ReachSimCore(const std::string& model_path, ReachConfig cfg, std::optional<uint64_t> seed)
    : sim_(model_path, std::move(cfg)) {
  reset(std::nullopt, std::nullopt, seed);
  episode_ = 0;  // the start-up reset is episode 0
}

ReachSnapshot ReachSimCore::snapshot() const {
  const Vec3 ee = sim_.ee_pos(), tg = sim_.target_pos();
  const double dist = std::sqrt((tg[0] - ee[0]) * (tg[0] - ee[0]) + (tg[1] - ee[1]) * (tg[1] - ee[1]) +
                                (tg[2] - ee[2]) * (tg[2] - ee[2]));
  ReachSnapshot s;
  s.seq = seq_;
  s.episode = episode_;
  s.episode_step = static_cast<uint32_t>(ep_step_);
  s.episode_steps_max = static_cast<uint32_t>(sim_.max_episode_steps());
  s.episode_done = ep_step_ >= sim_.max_episode_steps();
  s.sim_time = sim_time_;
  s.q = sim_.q();
  s.qd = sim_.qd();
  s.tau = sim_.tau();
  s.q_target = sim_.q_target();
  s.ee = ee;
  s.target = tg;
  s.ee_quat_xyzw = sim_.ee_quat_xyzw();
  s.obs = obs_;
  s.reward = reward_;
  s.distance = dist;
  s.success = dist < sim_.config().success_threshold;
  s.success_2cm = dist < 0.02;
  s.time_to_reach = reached_at_ ? *reached_at_ * sim_.dt() : -1.0;
  return s;
}

ReachSnapshot ReachSimCore::reset(const std::optional<Vec3>& target, const std::optional<Vec3>& qpos,
                                  std::optional<uint64_t> seed) {
  if (target) {
    const auto [ok, why] = sim_.check_target_reachable(*target);
    if (!ok) throw std::invalid_argument("unreachable target: " + why);
  }
  if (qpos && !sim_.config_is_valid(*qpos))
    throw std::invalid_argument("start pose is outside the joint limits or in collision");
  obs_ = sim_.reset(seed, target, qpos);
  ++episode_;
  ep_step_ = 0;
  reached_at_.reset();
  reward_ = 0.0;
  ++seq_;
  return snapshot();
}

ReachSnapshot ReachSimCore::set_target(const Vec3& p) {
  sim_.set_target(p);
  obs_ = sim_.observe();
  ep_step_ = 0;
  reached_at_.reset();
  reward_ = 0.0;
  ++seq_;
  return snapshot();
}

ReachSnapshot ReachSimCore::step(const Vec3& action) {
  const auto r = sim_.step(action);
  obs_ = r.obs;
  ++ep_step_;
  sim_time_ += sim_.dt();
  reward_ = r.reward;
  if (!reached_at_ && r.success) reached_at_ = ep_step_;
  ++seq_;
  if (r.terminated) return reset();
  return snapshot();
}

}  // namespace mujoco_arm
