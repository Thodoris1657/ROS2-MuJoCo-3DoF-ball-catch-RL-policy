// catch_eval_params -- fast evaluation of catch-controller parameters for the optimiser.
//
//   catch_eval_params <repo_root> <gravity> <params-json> <seed0> <n_nominal> <n_stress> [--trials]
//
// Runs n_nominal trials with the nominal randomisation and n_stress with the stress one
// (seeds seed0, seed0+1, ...; every random quantity drawn by the C++ simulator), and prints one
// JSON line: success rate, mean softness (1 - impact / 0.5 m/s over catches), outcome counts,
// and with --trials one record per trial. scripts/tune_catch.py --backend cpp drives it.
// Ablations: --oracle-e (the controller is told the true restitution: an upper bound for any
// estimator), --max-joint-speed X (the servo-target rate limit, 3 rad/s by default).
#include <cstdio>
#include <cstring>
#include <map>

#include <nlohmann/json.hpp>

#include "mujoco_arm_cpp/catch_controller.hpp"
#include "mujoco_arm_cpp/catch_core.hpp"

using namespace mujoco_arm;
using nlohmann::json;

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: %s <repo> <gravity> <params-json> <seed0> <n_nominal> <n_stress> [--trials]\n", argv[0]);
    return 2;
  }
  const std::string root = argv[1];
  const double g = resolve_gravity(argv[2]);
  const json pj = json::parse(argv[3]);
  CatchParams p;
  p.e_prior = pj.value("e_prior", p.e_prior);
  p.catch_offset = pj.value("catch_offset", p.catch_offset);
  p.lookahead = pj.value("lookahead", p.lookahead);
  p.clearance = pj.value("clearance", p.clearance);
  p.vz_match = pj.value("vz_match", p.vz_match);
  p.overshoot = pj.value("overshoot", p.overshoot);
  p.liftoff = pj.value("liftoff", p.liftoff);
  const uint64_t seed0 = std::stoull(argv[4]);
  const int n_nom = std::stoi(argv[5]), n_str = std::stoi(argv[6]);
  bool per_trial = false, oracle_e = false;
  double max_speed = 3.0;
  for (int a = 7; a < argc; ++a) {
    if (std::strcmp(argv[a], "--trials") == 0) per_trial = true;
    else if (std::strcmp(argv[a], "--oracle-e") == 0) oracle_e = true;  // privileged: true restitution
    else if (std::strcmp(argv[a], "--max-joint-speed") == 0 && a + 1 < argc) max_speed = std::stod(argv[++a]);
  }

  CatchConfig nominal;
  nominal.gravity_choices = {g};
  nominal.max_joint_speed = max_speed;
  CatchConfig stress = catch_config_from_json(
      R"({"restitution": [0.55, 0.90], "release_push": 0.04, "release_push_max": 0.08,
          "obs_pos_noise": 0.003, "obs_vel_noise": 0.05})", nominal);
  const std::string xml = root + "/assets/arm3_ballcatch.xml";
  CatchSim sims[2] = {CatchSim(xml, nominal), CatchSim(xml, stress)};
  CatchSim kin(xml, nominal);
  CatchController ctrl(p);
  int caught = 0, n = 0;
  double soft = 0.0;
  std::map<std::string, int> outcomes;
  json trials = json::array();
  for (int set = 0; set < 2; ++set) {
    for (int i = 0; i < (set ? n_str : n_nom); ++i) {
      CatchSim& sim = sims[set];
      CatchObs obs = sim.reset(seed0 + static_cast<uint64_t>(i) + (set ? 1000003ull : 0ull));
      if (oracle_e) {
        CatchParams q = p;
        q.e_prior = sim.restitution();
        ctrl = CatchController(q);
      }
      ctrl.reset();
      for (int k = 0; k < 2000; ++k) {
        const auto r = sim.step(ctrl.act(obs.data(), obs.size(), kin));
        obs = r.obs;
        if (r.terminated || r.truncated) break;
      }
      ++n;
      const std::string out = sim.outcome().value_or("running");
      ++outcomes[out.substr(0, out.find(" ("))];
      if (sim.success()) {
        ++caught;
        soft += 1.0 - std::min(sim.catch_info()->rel_speed / 0.5, 1.0);
      }
      if (per_trial)
        trials.push_back({{"set", set ? "stress" : "nominal"}, {"e", sim.restitution()},
                          {"drop_z", sim.drop_tip()[2]}, {"success", sim.success()}, {"outcome", out}});
    }
  }
  json res{{"trials", n}, {"success", n ? static_cast<double>(caught) / n : 0.0},
           {"softness", n ? soft / n : 0.0}, {"outcomes", outcomes}};
  if (per_trial) res["per_trial"] = trials;
  std::printf("%s\n", res.dump().c_str());
  return 0;
}
