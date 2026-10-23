// Python bindings for the C++ catch simulator (module name: mujoco_arm_cpp_py).
//
// Why: the C++ port is checked to behave exactly like the Python env (test_parity.cpp), but it
// is much faster. With these bindings Python code can use it directly: the RL trainer gets a
// vectorised env (CatchBatch) and the tuner can evaluate parameters without starting a process
// for every candidate.
//
// Build / install (needs the venv that has `mujoco`, see ros2/src/mujoco_arm_cpp/pyproject.toml):
//   pip install --no-build-isolation ./ros2/src/mujoco_arm_cpp
//
// Note on the GIL: everything that runs physics releases it, so the worker threads of
// CatchBatch really run in parallel and Python threads are not blocked.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "mujoco_arm_cpp/batch.hpp"
#include "mujoco_arm_cpp/catch_core.hpp"
#include "mujoco_arm_cpp/policies.hpp"

namespace py = pybind11;
using namespace mujoco_arm;

namespace {

// gymnasium-style kwargs (a plain dict) -> CatchConfig. Goes through JSON so the parsing is the
// same code as the ROS nodes use (and unknown keys raise an error there).
CatchConfig config_from(const py::dict& kwargs, const std::vector<double>& gravities) {
  py::module_ json = py::module_::import("json");
  py::dict copy;
  for (auto item : kwargs) {
    const std::string k = py::str(item.first);
    if (k == "gravity" || k == "start" || k == "render_mode") continue;  // handled elsewhere
    copy[item.first] = item.second;
  }
  CatchConfig c = catch_config_from_json(json.attr("dumps")(copy).cast<std::string>());
  if (!gravities.empty()) c.gravity_choices = gravities;
  return c;
}

CatchParams params_from(const py::dict& d) {
  CatchParams p;
  auto get = [&](const char* k, double& v) {
    if (d.contains(k)) v = d[k].cast<double>();
  };
  get("e_prior", p.e_prior);
  get("catch_offset", p.catch_offset);
  get("lookahead", p.lookahead);
  get("clearance", p.clearance);
  get("vz_match", p.vz_match);
  get("overshoot", p.overshoot);
  get("liftoff", p.liftoff);
  return p;
}

// {"earth": {...}, "mars": {...}} -> map
std::map<std::string, CatchParams> params_map(const py::dict& d) {
  std::map<std::string, CatchParams> out;
  for (auto item : d) out[py::str(item.first)] = params_from(item.second.cast<py::dict>());
  return out;
}

py::array_t<float> to_numpy(const float* data, size_t rows, size_t cols) {
  py::array_t<float> a(rows == 1 ? std::vector<size_t>{cols} : std::vector<size_t>{rows, cols});
  std::copy(data, data + rows * cols, a.mutable_data());
  return a;
}

py::array_t<double> vec3(const Vec3& v) {
  py::array_t<double> a(3);
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

Vec3 to_vec3(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
  if (a.size() != 3) throw std::invalid_argument("expected 3 values");
  return {a.data()[0], a.data()[1], a.data()[2]};
}

py::object opt_double(double v) { return std::isfinite(v) ? py::object(py::float_(v)) : py::object(py::none()); }

}  // namespace

PYBIND11_MODULE(mujoco_arm_cpp_py, m) {
  m.doc() = "C++ ball-catch simulator, controller and batch env (port of mujoco_arm_rl)";
  m.attr("OBS_DIM") = kCatchObsDim;
  m.def("resolve_gravity", &resolve_gravity);
  m.def("gravity_name", &gravity_name);

  // ------------------------------------------------------------- single sim
  py::class_<CatchSim>(m, "CatchSim")
      .def(py::init([](const std::string& model, const py::dict& kwargs, const std::vector<double>& gravities) {
             return std::make_unique<CatchSim>(model, config_from(kwargs, gravities));
           }),
           py::arg("model_path"), py::arg("env_kwargs") = py::dict(), py::arg("gravities") = std::vector<double>{})
      .def(
          "reset",
          [](CatchSim& s, std::optional<uint64_t> seed, std::optional<double> gravity, std::optional<double> e,
             std::optional<std::array<double, 3>> drop, std::optional<std::array<double, 2>> push,
             std::optional<std::string> start) {
            CatchResetOptions o;
            o.gravity = gravity;
            o.restitution = e;
            if (drop) o.drop = Vec3{(*drop)[0], (*drop)[1], (*drop)[2]};
            o.push = push;
            o.start = start;
            CatchObs obs;
            {
              py::gil_scoped_release nogil;
              obs = s.reset(seed, o);
            }
            return to_numpy(obs.data(), 1, obs.size());
          },
          py::arg("seed") = py::none(), py::arg("gravity") = py::none(), py::arg("restitution") = py::none(),
          py::arg("drop") = py::none(), py::arg("push") = py::none(), py::arg("start") = py::none())
      .def(
          "step",
          [](CatchSim& s, const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
            const Vec3 act = to_vec3(a);
            CatchStep r;
            {
              py::gil_scoped_release nogil;
              r = s.step(act);
            }
            return py::make_tuple(to_numpy(r.obs.data(), 1, r.obs.size()), r.reward, r.terminated, r.truncated);
          },
          py::arg("action"))
      .def("solve_ik",
           [](CatchSim& s, const py::array_t<double>& p, const py::array_t<double>& near) -> py::object {
             const auto q = s.solve_ik(to_vec3(p), to_vec3(near));
             return q ? py::object(vec3(*q)) : py::object(py::none());
           })
      .def_property_readonly("phase", &CatchSim::phase)
      .def_property_readonly("outcome", [](const CatchSim& s) { return s.outcome(); })
      .def_property_readonly("success", &CatchSim::success)
      .def_property_readonly("holding", &CatchSim::holding)
      .def_property_readonly("catching", &CatchSim::catching)
      .def_property_readonly("bounces", &CatchSim::bounces)
      .def_property_readonly("gravity", &CatchSim::gravity)
      .def_property_readonly("restitution", &CatchSim::restitution)
      .def_property_readonly("sim_time", &CatchSim::sim_time)
      .def_property_readonly("dt", &CatchSim::dt)
      .def_property_readonly("ee_pos", [](const CatchSim& s) { return vec3(s.ee_pos()); })
      .def_property_readonly("ball_pos", [](const CatchSim& s) { return vec3(s.ball_pos()); })
      .def_property_readonly("ball_vel", [](const CatchSim& s) { return vec3(s.ball_vel()); })
      .def_property_readonly("drop_tip", [](const CatchSim& s) { return vec3(s.drop_tip()); })
      .def_property_readonly("catch_time_error", [](const CatchSim& s) { return opt_double(s.catch_time_error()); })
      .def_property_readonly("impact_speed", [](const CatchSim& s) -> py::object {
        return s.catch_info() ? py::object(py::float_(s.catch_info()->rel_speed)) : py::object(py::none());
      });

  // ------------------------------------------------------------- controller
  py::class_<CatchController>(m, "CatchController")
      .def(py::init([](const py::dict& params, bool per_gravity) {
             return per_gravity ? std::make_unique<CatchController>(params_map(params))
                                : std::make_unique<CatchController>(params_from(params));
           }),
           py::arg("params") = py::dict(), py::arg("per_gravity") = false,
           "params: one dict of CatchParams fields, or {gravity name: dict} with per_gravity=True")
      .def("reset", &CatchController::reset)
      .def(
          "act",
          [](CatchController& c, const py::array_t<float, py::array::c_style | py::array::forcecast>& obs,
             CatchSim& kin) { return vec3(c.act(obs.data(), static_cast<size_t>(obs.size()), kin)); },
          py::arg("obs"), py::arg("kin"));

  py::class_<MlpNetwork>(m, "MlpNetwork")
      .def(py::init<const std::string&>())
      .def("forward",
           [](const MlpNetwork& n, const py::array_t<float, py::array::c_style | py::array::forcecast>& obs) {
             std::vector<float> x(obs.data(), obs.data() + obs.size());
             const auto y = n.forward(x);
             return py::array_t<double>(y.size(), y.data());
           })
      .def_property_readonly("obs_dim", &MlpNetwork::obs_dim)
      .def_property_readonly("act_dim", &MlpNetwork::act_dim);

  // ------------------------------------------------------------- batch env
  py::class_<CatchBatch>(m, "CatchBatch")
      .def(py::init([](const std::string& model, int n_envs, const py::dict& kwargs, const std::vector<double>& gravities,
                       uint64_t seed, int threads, double residual_scale, const py::dict& base_params) {
             BatchConfig b;
             b.sim = config_from(kwargs, gravities);
             b.n_envs = n_envs;
             b.seed = seed;
             b.threads = threads;
             b.residual = residual_scale > 0.0;
             b.residual_scale = residual_scale;
             b.base_params = params_map(base_params);
             py::gil_scoped_release nogil;
             return std::make_unique<CatchBatch>(model, b);
           }),
           py::arg("model_path"), py::arg("n_envs") = 8, py::arg("env_kwargs") = py::dict(),
           py::arg("gravities") = std::vector<double>{}, py::arg("seed") = 0, py::arg("threads") = 0,
           py::arg("residual_scale") = 0.0, py::arg("base_params") = py::dict())
      .def_property_readonly("num_envs", &CatchBatch::num_envs)
      .def_property_readonly("obs_dim", &CatchBatch::obs_dim)
      .def_property_readonly("threads", &CatchBatch::threads)
      .def_property_readonly("episodes_finished", &CatchBatch::episodes_finished)
      .def(
          "reset",
          [](CatchBatch& b, std::optional<uint64_t> seed) {
            {
              py::gil_scoped_release nogil;
              b.reset(seed);
            }
            return to_numpy(b.obs().data(), static_cast<size_t>(b.num_envs()), static_cast<size_t>(b.obs_dim()));
          },
          py::arg("seed") = py::none())
      .def(
          "step",
          [](CatchBatch& b, const py::array_t<double, py::array::c_style | py::array::forcecast>& actions) {
            if (actions.ndim() != 2 || actions.shape(0) != b.num_envs() || actions.shape(1) != 3)
              throw std::invalid_argument("actions must have shape (num_envs, 3)");
            {
              py::gil_scoped_release nogil;
              b.step(actions.data());
            }
            const auto n = static_cast<size_t>(b.num_envs());
            const auto d = static_cast<size_t>(b.obs_dim());
            py::array_t<double> rew(n, b.rewards().data());
            py::array_t<bool> term(n), trunc(n), succ(n);
            for (size_t i = 0; i < n; ++i) {
              term.mutable_data()[i] = b.terminated()[i];
              trunc.mutable_data()[i] = b.truncated()[i];
              succ.mutable_data()[i] = b.success()[i];
            }
            py::array_t<float> obs({n, d}), fin({n, d});
            std::copy(b.obs().begin(), b.obs().end(), obs.mutable_data());
            std::copy(b.final_obs().begin(), b.final_obs().end(), fin.mutable_data());
            return py::make_tuple(obs, rew, term, trunc, succ, fin);
          },
          py::arg("actions"),
          "returns (obs, reward, terminated, truncated, success, final_obs); finished envs are already reset")
      .def("last_outcome", &CatchBatch::last_outcome)
      .def("last_gravity", [](const CatchBatch& b, int i) { return b.gravity()[static_cast<size_t>(i)]; });

  // ------------------------------------------------------------- tuner helper
  m.def(
      "evaluate_params",
      [](const std::string& model, const py::dict& params, double gravity, int n_nominal, int n_stress,
         uint64_t seed0, int threads, double max_joint_speed, bool oracle_e) {
        EvalOptions o;
        o.gravity = gravity;
        o.n_nominal = n_nominal;
        o.n_stress = n_stress;
        o.seed0 = seed0;
        o.threads = threads;
        o.max_joint_speed = max_joint_speed;
        o.oracle_e = oracle_e;
        const CatchParams p = params_from(params);
        EvalResult r;
        {
          py::gil_scoped_release nogil;
          r = evaluate_params(model, p, o);
        }
        py::dict out;
        out["trials"] = r.trials;
        out["success"] = r.success;
        out["softness"] = r.softness;
        out["outcomes"] = r.outcomes;
        return out;
      },
      py::arg("model_path"), py::arg("params"), py::arg("gravity"), py::arg("n_nominal") = 128,
      py::arg("n_stress") = 128, py::arg("seed0") = 1, py::arg("threads") = 0, py::arg("max_joint_speed") = 3.0,
      py::arg("oracle_e") = false,
      "same as tools/catch_eval_params: success rate, softness and outcome counts for one parameter set");
}
