// Contract tests for the ROS-free C++ core (no ROS needed).
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>

#include "mujoco_arm_cpp/catch_controller.hpp"
#include "mujoco_arm_cpp/catch_core.hpp"
#include "mujoco_arm_cpp/catch_sim.hpp"
#include "mujoco_arm_cpp/policies.hpp"
#include "mujoco_arm_cpp/reach_sim.hpp"

using namespace mujoco_arm;

namespace {
const std::string kRoot = REPO_ROOT;
const std::string kCatchXml = kRoot + "/assets/arm3_ballcatch.xml";
const std::string kReachXml = kRoot + "/assets/arm3.xml";
const std::string kTuned = kRoot + "/configs/catch_tuned.json";

CatchConfig cfg_for(const std::string& g) {
  CatchConfig c;
  c.gravity_choices = {resolve_gravity(g)};
  return c;
}

/// Roll one catch episode with any action function; returns the sim's outcome.
template <class F>
std::string run(CatchSim& sim, uint64_t seed, F&& act, CatchResetOptions opt = {}) {
  CatchObs o = sim.reset(seed, opt);
  for (int k = 0; k < 5000; ++k) {
    const std::vector<float> v(o.begin(), o.end());
    const auto r = sim.step(act(v));
    o = r.obs;
    if (r.terminated || r.truncated) break;
  }
  return sim.outcome().value_or("running");
}
}  // namespace

TEST(ArmModel, WrapAngleMatchesNumpy) {
  EXPECT_NEAR(wrap_angle(3.5), 3.5 - 2 * kPi, 1e-12);
  EXPECT_NEAR(wrap_angle(-3.5), -3.5 + 2 * kPi, 1e-12);
  EXPECT_NEAR(wrap_angle(kPi), -kPi, 1e-12);  // numpy: (pi + pi) % 2pi - pi = -pi
}

TEST(ArmModel, AnalyticIkMatchesMujocoForwardKinematics) {
  auto m = load_model(kReachXml);
  DataPtr d(mj_makeData(m.get()));
  const int tip = require_id(m.get(), mjOBJ_SITE, "tip");
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-0.45, 0.45), uz(0.05, 0.65);
  int checked = 0;
  for (int k = 0; k < 200; ++k) {
    const Vec3 p{u(rng), u(rng), uz(rng)};
    for (const Vec3& q : analytic_ik(p)) {
      for (int i = 0; i < 3; ++i) d->qpos[i] = q[i];
      mj_kinematics(m.get(), d.get());
      for (int i = 0; i < 3; ++i) EXPECT_NEAR(d->site_xpos[3 * tip + i], p[i], 1e-12);
      ++checked;
    }
  }
  EXPECT_GT(checked, 100);
}

TEST(CatchSim, RestitutionIsGravityIndependent) {
  for (const auto& [name, g] : gravity_table()) {
    auto m = load_model(kCatchXml);
    m->opt.gravity[2] = -g;
    const int pid = require_id(m.get(), mjOBJ_PAIR, "ball_floor");
    const auto ref = restitution_solref(0.75);
    m->pair_solref[2 * pid] = ref[0];
    m->pair_solref[2 * pid + 1] = ref[1];
    DataPtr d(mj_makeData(m.get()));
    const int jid = require_id(m.get(), mjOBJ_JOINT, "ball_free");
    const int a = m->jnt_qposadr[jid], v = m->jnt_dofadr[jid];
    d->qpos[1] = -1.0;
    d->qpos[a] = 0.5;
    d->qpos[a + 1] = 0.5;
    d->qpos[a + 2] = kBallR + 0.4;
    mj_forward(m.get(), d.get());
    bool rising = false;
    double top = -1;
    for (int k = 0; k < 20000; ++k) {
      mj_step(m.get(), d.get());
      const double vz = d->qvel[v + 2];
      if (!rising && vz > 0) rising = true;
      else if (rising && vz <= 0) {
        top = d->qpos[a + 2] - kBallR;
        break;
      }
    }
    EXPECT_NEAR(std::sqrt(top / 0.4), 0.75, 0.05) << name;
  }
}

TEST(CatchSim, ObservationLayoutAndPinnedReset) {
  CatchSim sim(kCatchXml, cfg_for("earth"));
  CatchResetOptions o;
  o.gravity = resolve_gravity("moon");
  o.restitution = 0.7;
  o.drop = Vec3{0.3, 0.0, 0.47};
  o.push = std::array<double, 2>{0.01, -0.02};
  const auto a = sim.reset(3, o);
  const auto b = sim.reset(3, o);
  for (int i = 0; i < kCatchObsDim; ++i) EXPECT_EQ(a[i], b[i]);
  EXPECT_NEAR(a[30], 1.62 / 9.81, 1e-6);
  EXPECT_EQ(a[33], 1.0f);  // in flight right after the release
  EXPECT_NEAR(sim.release_velocity()[0], 0.01, 1e-12);
  EXPECT_EQ(sim.phase(), "fall");
}

TEST(CatchSim, DropPointsAreReachableAndClear) {
  CatchSim sim(kCatchXml);
  for (uint64_t s = 0; s < 20; ++s) {
    sim.reset(s);
    EXPECT_TRUE(sim.reachable(sim.drop_tip()));
    EXPECT_TRUE(sim.drop_is_clear(sim.drop_tip()));
  }
  EXPECT_FALSE(sim.drop_is_clear({0.30, 0.0, 0.57}));  // the forearm would touch the ball
}

TEST(CatchSim, HoldingStillAlwaysMisses) {
  for (const auto& [name, g] : gravity_table()) {
    CatchSim sim(kCatchXml, cfg_for(name));
    const auto out = run(sim, 1, [](const std::vector<float>&) { return Vec3{0, 0, 0}; });
    EXPECT_EQ(out.rfind("missed", 0), 0u) << name << ": " << out;
    EXPECT_FALSE(sim.success());
  }
}

TEST(CatchController, CatchesUnderAllGravities) {
  const auto tuned = load_tuned_params(kTuned);
  ASSERT_FALSE(tuned.empty());
  for (const auto& [name, g] : gravity_table()) {
    CatchSim sim(kCatchXml, cfg_for(name));
    CatchSim kin(kCatchXml);
    CatchController ctrl(tuned);
    int wins = 0;
    for (uint64_t s = 0; s < 10; ++s) {
      ctrl.reset();
      run(sim, s, [&](const std::vector<float>& o) { return ctrl.act(o.data(), o.size(), kin); });
      wins += sim.success();
    }
    EXPECT_GE(wins, 9) << name;
  }
}

TEST(CatchController, ChasingWithoutStayingAboveFailsOnTheMoon) {
  CatchParams p;
  p.lookahead = 0.0;
  p.clearance = -1.0;
  CatchSim sim(kCatchXml, cfg_for("moon"));
  CatchSim kin(kCatchXml);
  CatchController chase(p);
  int touched = 0;
  for (uint64_t s = 0; s < 6; ++s) {
    chase.reset();
    const auto out = run(sim, s, [&](const std::vector<float>& o) { return chase.act(o.data(), o.size(), kin); });
    touched += out.find("before the bounce") != std::string::npos;
  }
  EXPECT_GE(touched, 4);
}

TEST(CatchController, BallisticFitRecoversTheState) {
  std::mt19937 rng(1);
  std::normal_distribution<double> np(0, 0.001), nv(0, 0.02);
  const double g = 3.71;
  BallisticFit fit(g, 0.001, 0.02);
  const Vec3 p0{0.3, 0.0, 0.5}, v0{0.02, -0.01, -0.3};
  for (int k = 0; k < 20; ++k) {
    const double t = 0.01 * k;
    fit.add(t, {p0[0] + v0[0] * t + np(rng), p0[1] + v0[1] * t + np(rng), p0[2] + v0[2] * t - 0.5 * g * t * t + np(rng)},
            {v0[0] + nv(rng), v0[1] + nv(rng), v0[2] - g * t + nv(rng)});
  }
  Vec3 p, v;
  fit.state_at(0.19, &p, &v);
  EXPECT_NEAR(p[2], p0[2] + v0[2] * 0.19 - 0.5 * g * 0.19 * 0.19, 0.002);
  EXPECT_NEAR(v[2], v0[2] - g * 0.19, 0.02);
}

TEST(CatchSim, FullTrialFromThePedestal) {
  CatchConfig c = cfg_for("mars");
  c.start = "pick";
  CatchSim sim(kCatchXml, c);
  CatchSim kin(kCatchXml);
  CatchController ctrl(load_tuned_params(kTuned));
  std::vector<std::string> phases;
  CatchResetOptions o;
  o.drop = Vec3{0.30, 0.0, 0.47};
  o.restitution = 0.75;
  CatchObs obs = sim.reset(4, o);
  for (int k = 0; k < 2000; ++k) {
    const auto r = sim.step(ctrl.act(obs.data(), obs.size(), kin));
    obs = r.obs;
    if (phases.empty() || phases.back() != sim.phase()) phases.push_back(sim.phase());
    if (r.terminated) break;
  }
  ASSERT_GE(phases.size(), 3u);
  EXPECT_EQ(phases[0], "pick");
  EXPECT_EQ(phases[1], "fall");
  EXPECT_TRUE(sim.success()) << sim.outcome().value_or("");
}

TEST(Policies, FactoryAndErrors) {
  PolicyPaths paths{kCatchXml, kReachXml, kTuned, {}, {}};
  EXPECT_EQ(make_controller("catch_hold", paths)->description(), "catch baseline: hold still");
  EXPECT_NE(make_controller("catch_tuned", paths)->description().find("tuned"), std::string::npos);
  EXPECT_NE(make_controller("oracle", paths)->description().find("oracle"), std::string::npos);
  EXPECT_THROW(make_controller("nonsense", paths), std::invalid_argument);
  EXPECT_THROW(make_controller("/no/such/policy.zip", paths), std::invalid_argument);
  EXPECT_THROW(make_controller("/no/such/policy.json", paths), std::runtime_error);
}

TEST(ReachSim, OracleReachesAndCoreBookkeeping) {
  ReachSimCore core(kReachXml, {}, 0);
  PolicyPaths paths{kCatchXml, kReachXml, kTuned, {}, {}};
  auto oracle = make_controller("oracle", paths);
  auto s = core.reset(Vec3{0.3, 0.1, 0.35}, Vec3{0, 0, 0}, 1);
  EXPECT_EQ(s.episode, 1u);
  for (int k = 0; k < 50; ++k) s = core.step(oracle->act({s.obs.begin(), s.obs.end()}));
  EXPECT_TRUE(s.episode_done);
  EXPECT_LT(s.distance, 0.02);
  EXPECT_GT(s.time_to_reach, 0.0);
  EXPECT_THROW(core.reset(Vec3{3.0, 0.0, 0.3}), std::invalid_argument);
  const auto seq = core.snapshot().seq;
  core.set_target({0.2, 0.0, 0.3});
  EXPECT_EQ(core.snapshot().seq, seq + 1);
  EXPECT_EQ(core.snapshot().episode_step, 0u);
}

TEST(Config, KwargsParsing) {
  const auto c = catch_config_from_json(R"({"restitution": [0.55, 0.9], "obs_pos_noise": 0.003, "release_push": 0.04})");
  EXPECT_DOUBLE_EQ(c.restitution[0], 0.55);
  EXPECT_DOUBLE_EQ(c.restitution[1], 0.9);
  EXPECT_DOUBLE_EQ(c.obs_pos_noise, 0.003);
  EXPECT_THROW(catch_config_from_json(R"({"no_such_key": 1})"), std::invalid_argument);
  const auto r = reach_config_from_json(R"({"max_joint_speed": 6.0, "fixed_target": [0.3, 0.0, 0.35]})");
  EXPECT_DOUBLE_EQ(r.max_joint_speed, 6.0);
  ASSERT_TRUE(r.fixed_target.has_value());
  EXPECT_THROW(reach_config_from_json(R"({"control_mode": "torque"})"), std::invalid_argument);
}
