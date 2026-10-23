// catch_bench -- the frozen ball-catch benchmark in pure C++ (no ROS, no Python).
//
//   catch_bench <repo_root> [policy=catch_tuned] [set=benchmarks/catch_set_v1.json]
//
// Same pinned trials as scripts/benchmark_catch.py. The sensor-noise draws use C++'s RNG, so
// individual noisy trials can differ from Python; the success rates agree statistically.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>

#include "mujoco_arm_cpp/catch_core.hpp"
#include "mujoco_arm_cpp/policies.hpp"

using namespace mujoco_arm;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <repo_root> [policy] [set.json]\n", argv[0]);
    return 2;
  }
  const std::string root = argv[1];
  const std::string spec = argc > 2 ? argv[2] : "catch_tuned";
  const std::string set = argc > 3 ? argv[3] : root + "/benchmarks/catch_set_v1.json";
  std::ifstream f(set);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", set.c_str());
    return 2;
  }
  const auto blob = nlohmann::json::parse(f);
  const CatchConfig cfg = catch_config_from_json(blob.value("env_kwargs", nlohmann::json::object()).dump());
  CatchSim sim(root + "/assets/arm3_ballcatch.xml", cfg);
  PolicyPaths paths{root + "/assets/arm3_ballcatch.xml", root + "/assets/arm3.xml",
                    root + "/configs/catch_tuned.json", cfg, {}};
  auto policy = make_controller(spec, paths);
  std::map<std::string, std::pair<int, int>> score;
  std::map<std::string, int> fails;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto& c : blob["cases"]) {
    CatchResetOptions o;
    o.gravity = resolve_gravity(c["gravity"].get<std::string>());
    o.restitution = c["restitution"].get<double>();
    o.drop = Vec3{c["drop"][0].get<double>(), c["drop"][1].get<double>(), c["drop"][2].get<double>()};
    o.push = std::array<double, 2>{c["push"][0].get<double>(), c["push"][1].get<double>()};
    policy->reset();
    CatchObs obs = sim.reset(c["seed"].get<uint64_t>(), o);
    for (int k = 0; k < 2000; ++k) {
      const auto r = sim.step(policy->act({obs.begin(), obs.end()}));
      obs = r.obs;
      if (r.terminated || r.truncated) break;
    }
    auto& s = score[c["gravity"].get<std::string>()];
    ++s.second;
    if (sim.success()) ++s.first;
    else {
      const auto out = sim.outcome().value_or("running");
      ++fails[out.substr(0, out.find(" ("))];
    }
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("policy %s on %s (%.1f s)\n", policy->description().c_str(), set.c_str(), secs);
  for (const auto& [g, s] : score) std::printf("  %-6s %5.1f %%  (%d/%d)\n", g.c_str(), 100.0 * s.first / s.second, s.first, s.second);
  for (const auto& [k, n] : fails) std::printf("  failure: %s x%d\n", k.c_str(), n);
  return 0;
}
