// catch_eval_params -- fast evaluation of catch-controller parameters for the optimiser.
//
//   catch_eval_params <repo_root> <gravity> <params-json> <seed0> <n_nominal> <n_stress> [options]
//
// Runs n_nominal trials with the nominal randomisation and n_stress with the stress one
// (seeds seed0, seed0+1, ...) and prints one JSON line: success rate, mean softness
// (1 - impact / 0.5 m/s over catches) and the outcome counts.
// Options:
//   --trials               also print one record per trial
//   --oracle-e             the controller is told the true restitution (upper bound for any estimator)
//   --max-joint-speed X    servo-target rate limit, 3 rad/s by default
//   --threads N            worker threads (default: all cores)
// The work itself is done by mujoco_arm::evaluate_params (batch.cpp); the Python tuner calls the
// same function through the mujoco_arm_cpp_py module.
#include <cstdio>
#include <cstring>

#include <nlohmann/json.hpp>

#include "mujoco_arm_cpp/batch.hpp"

using namespace mujoco_arm;
using nlohmann::json;

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: %s <repo> <gravity> <params-json> <seed0> <n_nominal> <n_stress> [options]\n", argv[0]);
    return 2;
  }
  const std::string root = argv[1];
  EvalOptions opt;
  opt.gravity = resolve_gravity(argv[2]);
  const json pj = json::parse(argv[3]);
  CatchParams p;
  p.e_prior = pj.value("e_prior", p.e_prior);
  p.catch_offset = pj.value("catch_offset", p.catch_offset);
  p.lookahead = pj.value("lookahead", p.lookahead);
  p.clearance = pj.value("clearance", p.clearance);
  p.vz_match = pj.value("vz_match", p.vz_match);
  p.overshoot = pj.value("overshoot", p.overshoot);
  p.liftoff = pj.value("liftoff", p.liftoff);
  opt.seed0 = std::stoull(argv[4]);
  opt.n_nominal = std::stoi(argv[5]);
  opt.n_stress = std::stoi(argv[6]);
  for (int a = 7; a < argc; ++a) {
    if (std::strcmp(argv[a], "--trials") == 0) opt.per_trial = true;
    else if (std::strcmp(argv[a], "--oracle-e") == 0) opt.oracle_e = true;
    else if (std::strcmp(argv[a], "--max-joint-speed") == 0 && a + 1 < argc) opt.max_joint_speed = std::stod(argv[++a]);
    else if (std::strcmp(argv[a], "--threads") == 0 && a + 1 < argc) opt.threads = std::stoi(argv[++a]);
  }

  const EvalResult r = evaluate_params(root + "/assets/arm3_ballcatch.xml", p, opt);
  json out{{"trials", r.trials}, {"success", r.success}, {"softness", r.softness}, {"outcomes", r.outcomes}};
  if (opt.per_trial) {
    json t = json::array();
    for (const auto& x : r.per_trial)
      t.push_back({{"set", x.stress ? "stress" : "nominal"}, {"e", x.restitution}, {"drop_z", x.drop_z},
                   {"success", x.success}, {"outcome", x.outcome}});
    out["per_trial"] = t;
  }
  std::printf("%s\n", out.dump().c_str());
  return 0;
}
