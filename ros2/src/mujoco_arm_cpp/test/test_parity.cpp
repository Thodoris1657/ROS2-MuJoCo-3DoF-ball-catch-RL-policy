// C++ vs Python parity: replays roll-outs recorded by scripts/make_cpp_golden.py.
//
// The C++ simulator, controller, oracle and MLP must reproduce the Python ones: same
// observations at every step, same actions, same outcome and catch timing. This is what
// makes "trained / tuned in Python, deployed in the C++ ROS 2 nodes" trustworthy.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

#include "mujoco_arm_cpp/catch_controller.hpp"
#include "mujoco_arm_cpp/policies.hpp"
#include "mujoco_arm_cpp/reach_sim.hpp"

using namespace mujoco_arm;
using nlohmann::json;

namespace {
const std::string kRoot = REPO_ROOT;
const std::string kGolden = GOLDEN_DIR;

json load(const std::string& name) {
  std::ifstream f(kGolden + "/" + name);
  if (!f) throw std::runtime_error("missing golden file " + name + " (python scripts/make_cpp_golden.py)");
  return json::parse(f);
}
}  // namespace

TEST(Parity, CatchSimAndControllerMatchPython) {
  const json g = load("catch.json");
  const auto tuned = load_tuned_params(kRoot + "/configs/catch_tuned.json");
  CatchConfig cfg;
  cfg.obs_pos_noise = 0.0;
  cfg.obs_vel_noise = 0.0;
  CatchSim sim(kRoot + "/assets/arm3_ballcatch.xml", cfg);
  CatchSim kin(kRoot + "/assets/arm3_ballcatch.xml");
  for (const auto& c : g["cases"]) {
    const auto& o = c["options"];
    CatchResetOptions opt;
    opt.gravity = resolve_gravity(o["gravity"].get<std::string>());
    opt.restitution = o["restitution"].get<double>();
    const auto drop = o["drop"].get<std::vector<double>>();
    opt.drop = Vec3{drop[0], drop[1], drop[2]};
    const auto push = o["push"].get<std::vector<double>>();
    opt.push = std::array<double, 2>{push[0], push[1]};
    opt.start = o["start"].get<std::string>();
    const std::string tag = o["gravity"].get<std::string>() + "/" + *opt.start;

    CatchController ctrl(tuned);
    CatchObs obs = sim.reset(0, opt);
    const auto& ref_obs = c["obs"];
    const auto& ref_act = c["actions"];
    double worst_obs = 0.0, worst_act = 0.0;
    size_t k = 0;
    for (; k < ref_act.size(); ++k) {
      for (int i = 0; i < kCatchObsDim; ++i)
        worst_obs = std::max(worst_obs, std::abs(obs[i] - ref_obs[k][i].get<double>()));
      const Vec3 a = ctrl.act(obs.data(), obs.size(), kin);
      for (int i = 0; i < 3; ++i) worst_act = std::max(worst_act, std::abs(a[i] - ref_act[k][i].get<double>()));
      const auto r = sim.step(a);
      obs = r.obs;
      if (r.terminated || r.truncated) break;
    }
    EXPECT_EQ(k + 1, c["steps"].get<size_t>()) << tag;
    EXPECT_LT(worst_obs, 1e-5) << tag;
    EXPECT_LT(worst_act, 1e-5) << tag;
    EXPECT_EQ(sim.outcome().value_or(""), c["outcome"].get<std::string>()) << tag;
    if (!c["catch_time_error"].is_null()) {
      EXPECT_NEAR(sim.catch_time_error(), c["catch_time_error"].get<double>(), 1e-9) << tag;
    }
  }
}

TEST(Parity, ReachOracleMatchesPython) {
  const json g = load("reach.json");
  ReachSim sim(kRoot + "/assets/arm3.xml");
  PolicyPaths paths{kRoot + "/assets/arm3_ballcatch.xml", kRoot + "/assets/arm3.xml", "", {}, {}};
  auto oracle = make_controller("oracle", paths);
  for (const auto& c : g["cases"]) {
    const auto q = c["qpos"].get<std::vector<double>>();
    const auto t = c["target"].get<std::vector<double>>();
    oracle->reset();
    ReachObs obs = sim.reset(std::nullopt, Vec3{t[0], t[1], t[2]}, Vec3{q[0], q[1], q[2]});
    double worst = 0.0;
    const auto& dist = c["distance"];
    for (size_t k = 0; k < dist.size(); ++k) {
      const auto r = sim.step(oracle->act({obs.begin(), obs.end()}));
      obs = r.obs;
      worst = std::max(worst, std::abs(r.distance - dist[k].get<double>()));
    }
    EXPECT_LT(worst, 1e-6);
  }
}

TEST(Parity, ExportedNetworkMatchesStableBaselines) {
  const json g = load("mlp.json");
  MlpNetwork net(kGolden + "/" + g["network"].get<std::string>());
  EXPECT_EQ(net.obs_dim(), 34u);
  for (const auto& p : g["pairs"]) {
    const auto o = p["obs"].get<std::vector<float>>();
    const auto ref = p["action"].get<std::vector<double>>();
    const auto y = net.forward(o);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(y[i], ref[i], 2e-6);
  }
}
