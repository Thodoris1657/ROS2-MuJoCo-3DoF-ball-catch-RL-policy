// Shared kinematics / servo helpers for the 3-DOF MuJoCo arm (ROS-free).
//
// Line-for-line ports of mujoco_arm_rl/envs/reach3.py: analytic IK, angle wrapping,
// and the PD position servo that turns the MJCF torque motors into position actuators.
// Kept identical on purpose -- the parity tests compare against the Python env.
#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mujoco_arm {

using Vec3 = std::array<double, 3>;

constexpr double kShoulderZ = 0.22;   // joint2 (pitch) anchor, (0, 0, 0.22)
constexpr double kL2 = 0.25;          // upper arm
constexpr double kL3 = 0.25;          // forearm (joint3 -> tip site)
constexpr double kMaxReach = kL2 + kL3;
constexpr double kPi = 3.14159265358979323846;

/// (a + pi) mod 2 pi - pi, with numpy's sign convention for the modulo.
double wrap_angle(double a);

/// Every (pan, pitch, elbow) solution placing the tip at p (limits/collisions NOT checked).
std::vector<Vec3> analytic_ik(const Vec3& p);

double distance_to_shoulder(const Vec3& p);

/// RAII owners for MuJoCo objects.
struct ModelDeleter { void operator()(mjModel* m) const { mj_deleteModel(m); } };
struct DataDeleter { void operator()(mjData* d) const { mj_deleteData(d); } };
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

/// Load an MJCF file, throwing std::runtime_error with MuJoCo's message on failure.
ModelPtr load_model(const std::string& path);

struct ServoGains {
  Vec3 kp{200.0, 300.0, 150.0};
  double damping_ratio = 1.0;
};

/// Convert actuators 0..2 into PD position servos in place:
///   force = kp (ctrl - q) - kv qdot, clipped to the original motor torque (the MJCF gear).
/// kv is chosen for `damping_ratio` from the joint-space inertia of `scratch` (which the
/// caller has posed). Returns the torque limits (the original gears); writes kv.
Vec3 configure_servo(mjModel* m, mjData* scratch, const ServoGains& gains, Vec3* kv_out = nullptr);

/// Joint limits of the first three joints.
void joint_limits(const mjModel* m, Vec3* lo, Vec3* hi);

/// Index of a named object or throw.
int require_id(const mjModel* m, mjtObj type, const char* name);

}  // namespace mujoco_arm
