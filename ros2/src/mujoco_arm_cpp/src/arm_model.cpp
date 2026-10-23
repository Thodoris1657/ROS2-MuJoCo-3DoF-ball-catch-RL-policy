#include "mujoco_arm_cpp/arm_model.hpp"

#include <cmath>
#include <stdexcept>

namespace mujoco_arm {

double wrap_angle(double a) {
  const double two_pi = 2.0 * kPi;
  double r = std::fmod(a + kPi, two_pi);
  if (r < 0.0) r += two_pi;  // numpy's % takes the sign of the divisor
  return r - kPi;
}

std::vector<Vec3> analytic_ik(const Vec3& p) {
  const double r_xy = std::hypot(p[0], p[1]);
  const double h = p[2] - kShoulderZ;
  const double yaw0 = std::atan2(p[1], p[0]);
  std::vector<Vec3> out;
  const std::array<std::pair<double, double>, 2> planes{{{yaw0, r_xy}, {yaw0 + kPi, -r_xy}}};
  for (const auto& [yaw, r] : planes) {
    const double cos_elbow = (r * r + h * h - kL2 * kL2 - kL3 * kL3) / (2.0 * kL2 * kL3);
    if (std::abs(cos_elbow) > 1.0) continue;
    for (double sign : {1.0, -1.0}) {
      const double q3 = sign * std::acos(cos_elbow);
      const double q2 = std::atan2(r, h) - std::atan2(kL3 * std::sin(q3), kL2 + kL3 * std::cos(q3));
      out.push_back({wrap_angle(yaw), wrap_angle(q2), q3});
    }
  }
  return out;
}

double distance_to_shoulder(const Vec3& p) {
  return std::sqrt(p[0] * p[0] + p[1] * p[1] + (p[2] - kShoulderZ) * (p[2] - kShoulderZ));
}

ModelPtr load_model(const std::string& path) {
  char err[1024] = {0};
  mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("cannot load MJCF '" + path + "': " + err);
  return ModelPtr(m);
}

Vec3 configure_servo(mjModel* m, mjData* d, const ServoGains& g, Vec3* kv_out) {
  mj_forward(m, d);
  std::vector<mjtNum> M(static_cast<size_t>(m->nv) * m->nv, 0.0);
  mj_fullM(m, d, M.data());
  Vec3 limit{}, kv{};
  for (int i = 0; i < 3; ++i) {
    limit[i] = m->actuator_gear[6 * i];
    const double inertia = M[static_cast<size_t>(i) * m->nv + i];
    kv[i] = 2.0 * g.damping_ratio * std::sqrt(g.kp[i] * inertia);
  }
  for (int i = 0; i < 3; ++i) {
    const int j = m->actuator_trnid[2 * i];
    m->actuator_gear[6 * i] = 1.0;
    m->actuator_gaintype[i] = mjGAIN_FIXED;
    for (int k = 0; k < mjNGAIN; ++k) m->actuator_gainprm[mjNGAIN * i + k] = 0.0;
    m->actuator_gainprm[mjNGAIN * i] = g.kp[i];
    m->actuator_biastype[i] = mjBIAS_AFFINE;
    for (int k = 0; k < mjNBIAS; ++k) m->actuator_biasprm[mjNBIAS * i + k] = 0.0;
    m->actuator_biasprm[mjNBIAS * i + 1] = -g.kp[i];
    m->actuator_biasprm[mjNBIAS * i + 2] = -kv[i];
    m->actuator_ctrlrange[2 * i] = m->jnt_range[2 * j];
    m->actuator_ctrlrange[2 * i + 1] = m->jnt_range[2 * j + 1];
    m->actuator_ctrllimited[i] = 1;
    m->actuator_forcerange[2 * i] = -limit[i];
    m->actuator_forcerange[2 * i + 1] = limit[i];
    m->actuator_forcelimited[i] = 1;
  }
  if (kv_out) *kv_out = kv;
  return limit;
}

void joint_limits(const mjModel* m, Vec3* lo, Vec3* hi) {
  for (int i = 0; i < 3; ++i) {
    (*lo)[i] = m->jnt_range[2 * i];
    (*hi)[i] = m->jnt_range[2 * i + 1];
  }
}

int require_id(const mjModel* m, mjtObj type, const char* name) {
  const int id = mj_name2id(m, type, name);
  if (id < 0) throw std::runtime_error(std::string("MJCF has no object named '") + name + "'");
  return id;
}

}  // namespace mujoco_arm
