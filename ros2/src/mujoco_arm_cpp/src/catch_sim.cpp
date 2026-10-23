#include "mujoco_arm_cpp/catch_sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace mujoco_arm {

namespace {

double norm3(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

double min_jerk(double s) {
  s = clip(s, 0.0, 1.0);
  return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
}

constexpr const char* kArmGeoms[5] = {"base_geom", "link1_geom", "link2_geom", "link3_geom",
                                      "ee_geom"};

}  // namespace

// ------------------------------------------------------------------- gravity

const std::vector<std::pair<std::string, double>>& gravity_table() {
  static const std::vector<std::pair<std::string, double>> table{
      {"earth", 9.81}, {"mars", 3.71}, {"moon", 1.62}};
  return table;
}

double resolve_gravity(const std::string& s) {
  std::string key;
  for (char c : s) key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  for (const auto& [name, g] : gravity_table())
    if (name == key) return g;
  try {
    size_t used = 0;
    const double g = std::stod(s, &used);
    if (used == s.size() && g > 0.0) return g;
  } catch (const std::exception&) {
  }
  throw std::invalid_argument("unknown gravity '" + s + "'; use earth, mars, moon or a number");
}

std::string gravity_name(double g) {
  for (const auto& [name, v] : gravity_table())
    if (std::abs(v - g) < 1e-6) return name;
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f m/s^2", g);
  return buf;
}

std::array<double, 2> restitution_solref(double e) {
  e = clip(e, 0.05, 0.99);
  const double le = std::log(e);
  const double zeta = -le / std::sqrt(kPi * kPi + le * le);
  return {-(kContactOmega * kContactOmega) / kContactImp, -2.0 * zeta * kContactOmega / kContactImp};
}

// ------------------------------------------------------------- construction

CatchSim::CatchSim(const std::string& model_path, CatchConfig cfg) : cfg_(std::move(cfg)) {
  if (cfg_.start != "release" && cfg_.start != "pick")
    throw std::invalid_argument("start must be 'release' or 'pick'");
  if (cfg_.gravity_choices.empty()) cfg_.gravity_choices = {9.81};
  m_ = load_model(model_path);
  d_.reset(mj_makeData(m_.get()));
  scratch_.reset(mj_makeData(m_.get()));
  mjModel* m = m_.get();

  frame_skip_ = std::max(1, static_cast<int>(std::lround(1.0 / (cfg_.control_hz * m->opt.timestep))));
  max_episode_steps_ = static_cast<int>(std::lround(cfg_.max_catch_seconds / dt()));

  tip_id_ = require_id(m, mjOBJ_SITE, "tip");
  ee_body_ = require_id(m, mjOBJ_BODY, "ee");
  ball_geom_ = require_id(m, mjOBJ_GEOM, "ball_geom");
  floor_geom_ = require_id(m, mjOBJ_GEOM, "floor");
  for (int i = 0; i < 5; ++i) arm_geoms_[i] = require_id(m, mjOBJ_GEOM, kArmGeoms[i]);
  grip_eq_ = require_id(m, mjOBJ_EQUALITY, "grip");
  floor_pair_ = require_id(m, mjOBJ_PAIR, "ball_floor");
  const int jid = require_id(m, mjOBJ_JOINT, "ball_free");
  bq_ = m->jnt_qposadr[jid];
  bv_ = m->jnt_dofadr[jid];
  joint_limits(m, &jnt_lo_, &jnt_hi_);

  // servo, with the inertia taken at the same pose as the Python env
  mjData* s = scratch_.get();
  s->qpos[0] = 0.0;
  s->qpos[1] = 1.0;
  s->qpos[2] = 0.8;
  for (int k = 0; k < 3; ++k) s->qpos[bq_ + k] = 5.0;
  configure_servo(m, s, cfg_.servo);
  jacp_.assign(3 * static_cast<size_t>(m->nv), 0.0);
  rng_.seed(0);
  reset_bookkeeping();
}

void CatchSim::reset_bookkeeping() {
  phase_ = "idle";
  sim_time_ = 0.0;
  q_target_ = {0, 0, 0};
  prev_action_ = {0, 0, 0};
  segments_.clear();
  seg_i_ = 0;
  latched_ = grip_armed_ = floor_contact_ = false;
  released_clear_ = true;
  grip_offset_ = {0, 0, -kHoldDist};
  grip_offset_goal_ = {0, 0, -kHoldDist};
  bounces_ = 0;
  t_release_.reset();
  t_bounce_.reset();
  t_catch_.reset();
  apex_.reset();
  catch_info_.reset();
  outcome_.reset();
  success_ = false;
  catch_steps_ = 0;
  restitution_ = 0.75;
  drop_tip_ = {0, 0, 0};
  release_vel_ = {0, 0, 0};
}

// ------------------------------------------------------------------ state

Vec3 CatchSim::q() const { return {d_->qpos[0], d_->qpos[1], d_->qpos[2]}; }
Vec3 CatchSim::qd() const { return {d_->qvel[0], d_->qvel[1], d_->qvel[2]}; }
Vec3 CatchSim::tau() const {
  return {d_->qfrc_actuator[0], d_->qfrc_actuator[1], d_->qfrc_actuator[2]};
}
Vec3 CatchSim::ee_pos() const {
  const mjtNum* p = d_->site_xpos + 3 * tip_id_;
  return {p[0], p[1], p[2]};
}
Vec3 CatchSim::ee_vel() {
  mj_jacSite(m_.get(), d_.get(), jacp_.data(), nullptr, tip_id_);
  Vec3 v{0, 0, 0};
  const int nv = m_->nv;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) v[r] += jacp_[static_cast<size_t>(r) * nv + c] * d_->qvel[c];
  return v;
}
std::array<double, 4> CatchSim::ee_quat_xyzw() const {
  mjtNum wxyz[4];
  mju_mat2Quat(wxyz, d_->site_xmat + 9 * tip_id_);
  return {wxyz[1], wxyz[2], wxyz[3], wxyz[0]};
}
Vec3 CatchSim::ball_pos() const { return {d_->qpos[bq_], d_->qpos[bq_ + 1], d_->qpos[bq_ + 2]}; }
Vec3 CatchSim::ball_vel() const { return {d_->qvel[bv_], d_->qvel[bv_ + 1], d_->qvel[bv_ + 2]}; }
Vec3 CatchSim::catch_point() const { return {d_->mocap_pos[0], d_->mocap_pos[1], d_->mocap_pos[2]}; }

double CatchSim::catch_time_error() const {
  if (!catch_info_ || !apex_) return std::numeric_limits<double>::quiet_NaN();
  return catch_info_->t - apex_->t;
}
double CatchSim::catch_height_error() const {
  if (!catch_info_ || !apex_) return std::numeric_limits<double>::quiet_NaN();
  return catch_info_->ball_z - apex_->z;
}

// --------------------------------------------------------- kinematics tools

bool CatchSim::config_is_valid(const Vec3& q, double margin) {
  for (int i = 0; i < 3; ++i)
    if (q[i] < jnt_lo_[i] + margin || q[i] > jnt_hi_[i] - margin) return false;
  mjData* s = scratch_.get();
  for (int i = 0; i < 3; ++i) s->qpos[i] = q[i];
  for (int k = 0; k < 3; ++k) s->qpos[bq_ + k] = 5.0;
  mj_forward(m_.get(), s);
  return s->ncon == 0;
}

std::optional<Vec3> CatchSim::solve_ik(const Vec3& p, const Vec3& near, double margin) {
  std::optional<Vec3> best;
  double best_cost = std::numeric_limits<double>::infinity();
  for (const Vec3& q : analytic_ik(p)) {
    if (!config_is_valid(q, margin)) continue;
    double c = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double w = wrap_angle(q[i] - near[i]);
      c += w * w;
    }
    if (c < best_cost) {  // strict: first minimum wins, like np.argmin
      best_cost = c;
      best = q;
    }
  }
  return best;
}

bool CatchSim::reachable(const Vec3& p, double margin) {
  return distance_to_shoulder(p) <= kMaxReach && solve_ik(p, {0, 0, 0}, margin).has_value();
}

double CatchSim::ball_clearance(const Vec3& q, const Vec3& ball, bool skip_ee) {
  mjData* s = scratch_.get();
  for (int i = 0; i < 3; ++i) s->qpos[i] = q[i];
  for (int k = 0; k < 3; ++k) s->qpos[bq_ + k] = ball[k];
  mj_kinematics(m_.get(), s);
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 5; ++i) {
    if (skip_ee && i == 4) continue;
    best = std::min(best, static_cast<double>(
                              mj_geomDistance(m_.get(), s, ball_geom_, arm_geoms_[i], 0.2, nullptr)));
  }
  return best;
}

bool CatchSim::drop_is_clear(const Vec3& tip, double margin) {
  const auto q = solve_ik(tip, {0.0, 0.5, 1.0});
  if (!q) return false;
  for (double depth : {kHoldDist, kHoldDist + 0.02, kHoldDist + 0.05, kHoldDist + 0.10}) {
    const Vec3 ball{tip[0], tip[1], tip[2] - depth};
    if (ball_clearance(*q, ball, true) < margin) return false;
  }
  return true;
}

// ------------------------------------------------------------------ physics

void CatchSim::set_gravity(double g) {
  gravity_ = g;
  m_->opt.gravity[0] = 0.0;
  m_->opt.gravity[1] = 0.0;
  m_->opt.gravity[2] = -g;
}

void CatchSim::set_restitution(double e) {
  restitution_ = e;
  const auto ref = restitution_solref(e);
  m_->pair_solref[mjNREF * floor_pair_] = ref[0];
  m_->pair_solref[mjNREF * floor_pair_ + 1] = ref[1];
}

void CatchSim::grip_on(const Vec3& offset_world) {
  latched_ = true;
  grip_offset_ = offset_world;
  update_grip_anchor();
  d_->eq_active[grip_eq_] = 1;
}

void CatchSim::grip_off() {
  latched_ = false;
  d_->eq_active[grip_eq_] = 0;
}

void CatchSim::update_grip_anchor() {
  if (!latched_) return;
  for (int i = 0; i < 3; ++i) grip_offset_[i] += 0.15 * (grip_offset_goal_[i] - grip_offset_[i]);
  const mjtNum* R = d_->xmat + 9 * ee_body_;  // row-major
  mjtNum* eq = m_->eq_data + mjNEQDATA * grip_eq_;
  for (int i = 0; i < 3; ++i)
    eq[i] = R[0 * 3 + i] * grip_offset_[0] + R[1 * 3 + i] * grip_offset_[1] + R[2 * 3 + i] * grip_offset_[2];
  eq[3] = eq[4] = eq[5] = 0.0;
}

void CatchSim::contacts(bool* floor, bool* arm) const {
  *floor = *arm = false;
  for (int k = 0; k < d_->ncon; ++k) {
    const mjContact& c = d_->contact[k];
    int other;
    if (c.geom[0] == ball_geom_) other = c.geom[1];
    else if (c.geom[1] == ball_geom_) other = c.geom[0];
    else continue;
    if (other == floor_geom_) *floor = true;
    for (int g : arm_geoms_)
      if (other == g) *arm = true;
  }
}

bool CatchSim::try_latch() {
  const Vec3 tip = ee_pos();
  const Vec3 bp = ball_pos();
  const Vec3 rel = sub(bp, tip);
  const double dist = norm3(rel);
  if (dist - kContactDist > cfg_.capture_gap || rel[2] > 0.0) return false;
  if (std::hypot(rel[0], rel[1]) > cfg_.capture_radius) return false;
  const Vec3 vrel = sub(ball_vel(), ee_vel());
  const double speed = norm3(vrel);
  if (speed > cfg_.capture_speed) return false;
  catch_info_ = CatchInfo{sim_time_, speed, ball_vel()[2], bp[2], rel};
  grip_on(rel);
  return true;
}

void CatchSim::substep() {
  mj_step(m_.get(), d_.get());
  sim_time_ += m_->opt.timestep;
  bool floor = false, arm = false;
  contacts(&floor, &arm);
  if (!released_clear_) {  // 15 ms for the ball to leave the cup after the release
    released_clear_ = sim_time_ - t_release_.value_or(0.0) > 0.015;
    arm = false;
  }

  if (phase_ == "pick" && grip_armed_ && !latched_) {
    try_latch();
  } else if ((phase_ == "fall" || phase_ == "rise") && !outcome_) {
    const double bz = ball_vel()[2];
    if (floor && !floor_contact_) {
      ++bounces_;
      if (bounces_ >= 2) finish("missed: second bounce");
      else if (!t_bounce_) t_bounce_ = sim_time_;
    }
    if (phase_ == "fall" && bounces_ >= 1 && !floor && bz > 0.0) {
      phase_ = "rise";
      grip_armed_ = true;
      const Vec3 bp = ball_pos(), bv = ball_vel();
      const double g = gravity_;
      apex_ = Apex{sim_time_ + bz / g, bp[2] + bz * bz / (2.0 * g), bp[0] + bv[0] * bz / g,
                   bp[1] + bv[1] * bz / g};
    }
    if (!outcome_) {
      if (grip_armed_ && !latched_ && try_latch()) {
        t_catch_ = sim_time_;
        if (std::abs(catch_info_->ball_vz) > cfg_.apex_speed) finish("latched outside the apex window");
        else phase_ = "hold";
      } else if (arm && !latched_) {
        if (bounces_ == 0) finish("touched the ball before the bounce");
        else finish("knocked the ball (" + why_no_latch() + ")");
      }
    }
  } else if (phase_ == "hold" && !outcome_) {
    if (sim_time_ - *t_catch_ >= cfg_.hold_time) {
      success_ = true;
      finish("caught");
    }
  }
  floor_contact_ = floor;
}

std::string CatchSim::why_no_latch() {
  const Vec3 rel = sub(ball_pos(), ee_pos());
  const double off = std::hypot(rel[0], rel[1]);
  const double vrel = norm3(sub(ball_vel(), ee_vel()));
  char buf[64];
  if (off > cfg_.capture_radius) {
    std::snprintf(buf, sizeof(buf), "%.0f mm off-centre", off * 1000.0);
    return buf;
  }
  if (vrel > cfg_.capture_speed) {
    std::snprintf(buf, sizeof(buf), "impact %.2f m/s", vrel);
    return buf;
  }
  return "side contact";
}

void CatchSim::finish(const std::string& outcome) {
  if (outcome_) return;
  outcome_ = outcome;
  if (!success_ && (phase_ == "pick" || phase_ == "fall" || phase_ == "rise" || phase_ == "hold"))
    phase_ = "failed";
}

// --------------------------------------------------------- scripted motions

void CatchSim::plan_pick_and_carry() {
  const Vec3 q_home = q();
  const Vec3 pb = ball_pos();
  const Vec3 pre{pb[0], pb[1], pb[2] + kContactDist + 0.06};
  const Vec3 touch{pb[0], pb[1], pb[2] + kContactDist - 0.004};
  const Vec3 lift{pre[0], pre[1], pre[2] + 0.10};
  const auto q_pre = solve_ik(pre, q_home);
  if (!q_pre) throw std::runtime_error("pick pose unreachable");
  const auto q_drop = solve_ik(drop_tip_, *q_pre);
  if (!q_drop) throw std::runtime_error("drop pose unreachable");
  segments_ = {
      {"approach", 1.2, false, *q_pre},
      {"descend", 1.0, true, touch, "arm", true},
      {"lift", 0.8, true, lift},
      {"carry", 1.4, false, *q_drop},
      {"settle", 0.5, false, *q_drop},
  };
  seg_i_ = 0;
  phase_ = "pick";
}

void CatchSim::plan_return() {
  const Vec3 q_now = q();
  const Vec3 above{kBallRest[0], kBallRest[1], kBallRest[2] + kContactDist + 0.10};
  const Vec3 place{kBallRest[0], kBallRest[1], kBallRest[2] + kContactDist + 0.002};
  const auto q_above = solve_ik(above, q_now);
  if (!q_above) return;
  segments_ = {
      {"carry back", 1.6, false, *q_above},
      {"lower", 1.0, true, place},
      {"let go", 0.3, true, place, "release"},
      {"retreat", 0.8, true, above},
  };
  seg_i_ = 0;
  phase_ = "return";
}

std::optional<Vec3> CatchSim::scripted_target() {
  while (seg_i_ < segments_.size()) {
    Segment& seg = segments_[seg_i_];
    if (!seg.started) {
      seg.started = true;
      seg.start = seg.line ? ee_pos() : q_target_;
      seg.t0 = sim_time_;
      if (seg.grip == "arm") grip_armed_ = true;
      else if (seg.grip == "release") {
        grip_off();
        grip_armed_ = false;
      }
    }
    const double s = (sim_time_ - seg.t0) / seg.duration;
    if (s >= 1.0 || (seg.stop_on_latch && latched_)) {
      if (seg.grip == "arm") grip_armed_ = false;
      ++seg_i_;
      continue;
    }
    const double a = min_jerk(s + dt() / seg.duration);
    Vec3 x;
    for (int i = 0; i < 3; ++i) x[i] = seg.start[i] + a * (seg.goal[i] - seg.start[i]);
    if (!seg.line) return x;
    const auto qs = solve_ik(x, q_target_);
    return qs ? *qs : q_target_;
  }
  return std::nullopt;
}

double CatchSim::normal(double sigma) {
  std::normal_distribution<double> n(0.0, 1.0);
  return sigma * n(rng_);
}

void CatchSim::release() {
  grip_off();
  grip_armed_ = false;
  double px = normal(cfg_.release_push), py = normal(cfg_.release_push);
  const double n = std::hypot(px, py);
  if (n > cfg_.release_push_max) {
    px *= cfg_.release_push_max / n;
    py *= cfg_.release_push_max / n;
  }
  d_->qvel[bv_] += px;
  d_->qvel[bv_ + 1] += py;
  t_release_ = sim_time_;
  released_clear_ = false;
  bounces_ = 0;
  floor_contact_ = false;
  phase_ = "fall";
  release_vel_ = ball_vel();
}

Vec3 CatchSim::sample_drop() {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int k = 0; k < 200; ++k) {
    Vec3 p;
    for (int i = 0; i < 3; ++i) p[i] = cfg_.drop_lo[i] + (cfg_.drop_hi[i] - cfg_.drop_lo[i]) * u(rng_);
    if (reachable(p, 0.05) && drop_is_clear(p)) return p;
  }
  return {0.30, 0.0, 0.50};
}

Vec3 CatchSim::ideal_catch_point() const {
  const double g = gravity_;
  double x, y, z;
  if (apex_) {
    x = apex_->x;
    y = apex_->y;
    z = apex_->z;
  } else {
    const Vec3 p = ball_pos(), v = ball_vel();
    const double h = p[2] - kBallR;
    const double v_imp = std::sqrt(std::max(v[2] * v[2] + 2.0 * g * std::max(h, 0.0), 0.0));
    const double t_imp = (v[2] + v_imp) / g;
    const double v_out = restitution_ * v_imp;
    x = p[0] + v[0] * (t_imp + v_out / g);
    y = p[1] + v[1] * (t_imp + v_out / g);
    z = kBallR + v_out * v_out / (2.0 * g);
  }
  return {x, y, z + kContactDist + 0.002};
}

void CatchSim::set_marker(const Vec3& p) {
  for (int i = 0; i < 3; ++i) d_->mocap_pos[i] = p[i];
}

// ------------------------------------------------------------- observation

CatchObs CatchSim::observe() {
  CatchObs o{};
  const Vec3 qq = q(), qv = qd(), tip = ee_pos(), tv = ee_vel();
  Vec3 bp = ball_pos(), bvel = ball_vel();
  for (int i = 0; i < 3; ++i) bp[i] += normal(cfg_.obs_pos_noise);
  for (int i = 0; i < 3; ++i) bvel[i] += normal(cfg_.obs_vel_noise);
  for (int i = 0; i < 3; ++i) {
    o[i] = static_cast<float>(std::cos(qq[i]));
    o[3 + i] = static_cast<float>(std::sin(qq[i]));
    o[6 + i] = static_cast<float>(qv[i] / 10.0);
    o[9 + i] = static_cast<float>(q_target_[i] - qq[i]);
    o[12 + i] = static_cast<float>(prev_action_[i]);
    o[15 + i] = static_cast<float>(tip[i]);
    o[18 + i] = static_cast<float>(tv[i]);
    o[21 + i] = static_cast<float>(bp[i]);
    o[24 + i] = static_cast<float>(bvel[i]);
    o[27 + i] = static_cast<float>(10.0 * (bp[i] - tip[i]));
  }
  o[30] = static_cast<float>(gravity_ / 9.81);
  o[31] = bounces_ > 0 ? 1.0f : 0.0f;
  o[32] = latched_ ? 1.0f : 0.0f;
  o[33] = catching() ? 1.0f : 0.0f;
  return o;
}

// -------------------------------------------------------------------- API

CatchObs CatchSim::reset(std::optional<uint64_t> seed, const CatchResetOptions& opt) {
  if (seed) rng_.seed(*seed);
  mjModel* m = m_.get();
  mjData* d = d_.get();
  reset_bookkeeping();
  mj_resetData(m, d);

  if (opt.gravity) {
    set_gravity(*opt.gravity);
  } else {
    std::uniform_int_distribution<size_t> pick(0, cfg_.gravity_choices.size() - 1);
    set_gravity(cfg_.gravity_choices[pick(rng_)]);
  }
  if (opt.restitution) {
    set_restitution(*opt.restitution);
  } else {
    std::uniform_real_distribution<double> u(cfg_.restitution[0], cfg_.restitution[1]);
    set_restitution(cfg_.restitution[0] == cfg_.restitution[1] ? cfg_.restitution[0] : u(rng_));
  }
  drop_tip_ = opt.drop ? *opt.drop : sample_drop();
  if (!reachable(drop_tip_)) throw std::invalid_argument("drop point is not reachable");
  start_mode_ = opt.start.value_or(cfg_.start);

  if (start_mode_ == "pick") {
    for (int i = 0; i < 3; ++i) d->qpos[i] = 0.0;
    for (int k = 0; k < 3; ++k) d->qpos[bq_ + k] = kBallRest[k];
    d->qpos[bq_ + 3] = 1.0;
    d->qpos[bq_ + 4] = d->qpos[bq_ + 5] = d->qpos[bq_ + 6] = 0.0;
    q_target_ = {0, 0, 0};
    for (int i = 0; i < 3; ++i) d->ctrl[i] = 0.0;
    mj_forward(m, d);
    plan_pick_and_carry();
  } else {
    const auto q_drop = solve_ik(drop_tip_, {0.0, 0.5, 1.0});
    if (!q_drop) throw std::invalid_argument("drop point has no valid IK solution");
    for (int i = 0; i < 3; ++i) d->qpos[i] = (*q_drop)[i];
    mj_forward(m, d);
    const Vec3 tip = ee_pos();
    d->qpos[bq_] = tip[0];
    d->qpos[bq_ + 1] = tip[1];
    d->qpos[bq_ + 2] = tip[2] - kHoldDist;
    d->qpos[bq_ + 3] = 1.0;
    d->qpos[bq_ + 4] = d->qpos[bq_ + 5] = d->qpos[bq_ + 6] = 0.0;
    q_target_ = *q_drop;
    for (int i = 0; i < 3; ++i) d->ctrl[i] = q_target_[i];
    mj_forward(m, d);
    grip_on({0.0, 0.0, -kHoldDist});
    const int settle = static_cast<int>(std::lround(0.15 / m->opt.timestep));
    for (int k = 0; k < settle; ++k) mj_step(m, d);
    sim_time_ = 0.0;
    release();
    if (opt.push) {
      d->qvel[bv_] = (*opt.push)[0];
      d->qvel[bv_ + 1] = (*opt.push)[1];
      release_vel_ = ball_vel();
    }
  }
  pending_push_ = opt.push;
  mj_forward(m, d);
  set_marker(phase_ == "fall" ? ideal_catch_point() : drop_tip_);
  return observe();
}

CatchStep CatchSim::step(const Vec3& action) {
  Vec3 a;
  for (int i = 0; i < 3; ++i) a[i] = clip(std::isfinite(action[i]) ? action[i] : 0.0, -1.0, 1.0);
  const bool policy = catching();
  const Vec3 qq = q();
  Vec3 tgt;
  if (policy) {
    for (int i = 0; i < 3; ++i) tgt[i] = q_target_[i] + a[i] * max_step();
  } else {
    std::optional<Vec3> s;
    if (phase_ == "pick" || phase_ == "return") s = scripted_target();
    tgt = s ? *s : q_target_;
    a = {0, 0, 0};
  }
  for (int i = 0; i < 3; ++i) {
    tgt[i] = clip(tgt[i], qq[i] - 0.3, qq[i] + 0.3);
    q_target_[i] = clip(tgt[i], jnt_lo_[i], jnt_hi_[i]);
    d_->ctrl[i] = q_target_[i];
  }

  update_grip_anchor();
  for (int k = 0; k < frame_skip_; ++k) substep();
  if (policy) ++catch_steps_;

  if (phase_ == "pick" && seg_i_ >= segments_.size()) {
    if (!latched_) {
      finish("pick failed");
    } else {
      release();
      if (pending_push_) {
        d_->qvel[bv_] = (*pending_push_)[0];
        d_->qvel[bv_ + 1] = (*pending_push_)[1];
        release_vel_ = ball_vel();
      }
    }
  }
  if (phase_ == "hold" && outcome_ && *outcome_ == "caught" && start_mode_ == "pick" && cfg_.return_ball)
    plan_return();
  if (phase_ == "fall" || phase_ == "rise") set_marker(ideal_catch_point());
  prev_action_ = a;

  for (int i = 0; i < m_->nq; ++i)
    if (!std::isfinite(d_->qpos[i])) {
      finish("numerical failure");
      break;
    }
  const bool script_running = phase_ == "pick" || phase_ == "return";
  CatchStep out;
  out.terminated = outcome_.has_value() && !script_running;
  if (phase_ == "return" && seg_i_ >= segments_.size()) out.terminated = true;
  out.truncated = !out.terminated && catch_steps_ >= max_episode_steps_;
  if (out.truncated && !outcome_) outcome_ = "timeout";
  out.obs = observe();
  return out;
}

}  // namespace mujoco_arm
