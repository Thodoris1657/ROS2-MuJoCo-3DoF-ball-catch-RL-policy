#include "mujoco_arm_cpp/catch_core.hpp"

#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mujoco_arm {

namespace {
Vec3 vec3(const nlohmann::json& v) { return {v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>()}; }
nlohmann::json parse_object(const std::string& text) {
  const auto j = nlohmann::json::parse(text.empty() ? "{}" : text);
  if (!j.is_object()) throw std::invalid_argument("env_kwargs must be a JSON object");
  return j;
}
}  // namespace

CatchConfig catch_config_from_json(const std::string& text, CatchConfig c) {
  const nlohmann::json kwargs = parse_object(text);  // named: items() must not outlive it
  for (const auto& [k, v] : kwargs.items()) {
    if (k == "restitution") {
      if (v.is_number()) c.restitution = {v.get<double>(), v.get<double>()};
      else c.restitution = {v.at(0).get<double>(), v.at(1).get<double>()};
    } else if (k == "drop_x" || k == "drop_y" || k == "drop_z") {
      const int i = k == "drop_x" ? 0 : k == "drop_y" ? 1 : 2;
      c.drop_lo[i] = v.at(0).get<double>();
      c.drop_hi[i] = v.at(1).get<double>();
    } else if (k == "servo_kp") c.servo.kp = vec3(v);
    else if (k == "servo_damping_ratio") c.servo.damping_ratio = v.get<double>();
    else if (k == "control_hz") c.control_hz = v.get<double>();
    else if (k == "max_joint_speed") c.max_joint_speed = v.get<double>();
    else if (k == "release_push") c.release_push = v.get<double>();
    else if (k == "release_push_max") c.release_push_max = v.get<double>();
    else if (k == "obs_pos_noise") c.obs_pos_noise = v.get<double>();
    else if (k == "obs_vel_noise") c.obs_vel_noise = v.get<double>();
    else if (k == "capture_gap") c.capture_gap = v.get<double>();
    else if (k == "capture_radius") c.capture_radius = v.get<double>();
    else if (k == "capture_speed") c.capture_speed = v.get<double>();
    else if (k == "apex_speed") c.apex_speed = v.get<double>();
    else if (k == "hold_time") c.hold_time = v.get<double>();
    else if (k == "return_ball") c.return_ball = v.get<bool>();
    else if (k == "max_catch_seconds") c.max_catch_seconds = v.get<double>();
    else if (k == "catch_bonus") c.catch_bonus = v.get<double>();
    else if (k == "fail_penalty") c.fail_penalty = v.get<double>();
    else if (k == "action_cost") c.action_cost = v.get<double>();
    else if (k == "gravity" || k == "start" || k == "render_mode") continue;  // node parameters
    else throw std::invalid_argument("unknown catch env kwarg '" + k + "'");
  }
  return c;
}

ReachConfig reach_config_from_json(const std::string& text, ReachConfig c) {
  const nlohmann::json kwargs = parse_object(text);  // named: items() must not outlive it
  for (const auto& [k, v] : kwargs.items()) {
    if (k == "control_hz") c.control_hz = v.get<double>();
    else if (k == "episode_seconds") c.episode_seconds = v.get<double>();
    else if (k == "max_joint_speed") c.max_joint_speed = v.get<double>();
    else if (k == "servo_kp") c.servo.kp = vec3(v);
    else if (k == "servo_damping_ratio") c.servo.damping_ratio = v.get<double>();
    else if (k == "success_threshold") c.success_threshold = v.get<double>();
    else if (k == "start_noise") c.start_noise = v.get<double>();
    else if (k == "precision_scale") c.precision_scale = v.get<double>();
    else if (k == "precision_weight") c.precision_weight = v.get<double>();
    else if (k == "action_cost") c.action_cost = v.get<double>();
    else if (k == "action_rate_cost") c.action_rate_cost = v.get<double>();
    else if (k == "fixed_target") c.fixed_target = v.is_null() ? std::nullopt : std::optional<Vec3>(vec3(v));
    else if (k == "control_mode" || k == "reward_mode" || k == "target_sampler" || k == "obs_mode") {
      const std::string s = v.get<std::string>();
      if (s != "position" && s != "shaped" && s != "reachable" && s != "full")
        throw std::invalid_argument("the C++ port implements the v1 task only (" + k + "=" + s + ")");
    } else if (k == "render_mode" || k == "seed") continue;
    else throw std::invalid_argument("unknown reach env kwarg '" + k + "'");
  }
  return c;
}

CatchTrialRunner::CatchTrialRunner(const std::string& model_path, CatchConfig cfg, std::string g,
                                   std::optional<uint64_t> seed, double post_roll)
    : default_gravity(std::move(g)), sim_(model_path, std::move(cfg)), post_roll_(post_roll) {
  CatchResetOptions o;
  o.gravity = resolve_gravity(default_gravity);
  o.start = "pick";
  obs_ = sim_.reset(seed, o);  // an idle state to publish before the first trial
}

std::string CatchTrialRunner::phase() const {
  return (idle_ && trial_ == 0) ? "idle" : sim_.phase();
}

void CatchTrialRunner::start(const StartRequest& r) {
  CatchResetOptions o;
  o.gravity = resolve_gravity(r.gravity.empty() ? default_gravity : r.gravity);
  o.start = r.from_pick ? "pick" : "release";
  if (r.restitution > 0.0) o.restitution = r.restitution;
  if (r.drop) {
    if (!sim_.reachable(*r.drop)) throw std::invalid_argument("drop point is unreachable");
    if (!sim_.drop_is_clear(*r.drop))
      throw std::invalid_argument("drop point is too high: the held ball would touch the forearm "
                                  "(keep the tip below ~0.52 m)");
    o.drop = r.drop;
  }
  o.push = r.push;
  obs_ = sim_.reset(r.seed, o);
  ++trial_;
  ++seq_;
  idle_ = false;
  done_at_.reset();
}

void CatchTrialRunner::step(const Vec3& action) {
  if (idle_) return;
  const auto r = sim_.step(action);
  obs_ = r.obs;
  ++seq_;
  if ((r.terminated || r.truncated) && !done_at_) done_at_ = sim_.sim_time();
  if (done_at_ && sim_.sim_time() - *done_at_ >= post_roll_) idle_ = true;
}

}  // namespace mujoco_arm
