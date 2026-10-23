// catch_eval_client -- run the frozen ball-catch benchmark *through ROS* (C++).
//
// Replays benchmarks/catch_set_v1.json (100 pinned trials per gravity) via catch/start and
// reads the outcomes from catch_status. Run the simulator in lock-step with a policy node
// (catch.launch.py eval:=true does this). Prints one RESULT line of JSON and exits.
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <mujoco_arm_interfaces/msg/catch_status.hpp>
#include <mujoco_arm_interfaces/srv/start_catch.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
using StatusMsg = mujoco_arm_interfaces::msg::CatchStatus;
using StartSrv = mujoco_arm_interfaces::srv::StartCatch;
namespace mros = mujoco_arm::ros;
using nlohmann::json;

class CatchEvalClient : public rclcpp::Node {
 public:
  CatchEvalClient() : Node("catch_eval_client") {
    set_ = mros::or_share(declare_parameter<std::string>("set", ""), "benchmarks/catch_set_v1.json");
    per_ = declare_parameter<int64_t>("per_gravity", 100);
    out_ = declare_parameter<std::string>("out", "");
    limit_ = declare_parameter<double>("timeout_s", 60.0);
    sub_ = create_subscription<StatusMsg>("catch_status", mros::reliable(50), [this](StatusMsg::ConstSharedPtr m) {
      std::lock_guard<std::mutex> lk(mu_);
      last_ = *m;
    });
    client_ = create_client<StartSrv>("catch/start");
  }

  int run() {
    std::ifstream f(set_);
    if (!f) {
      RCLCPP_ERROR(get_logger(), "cannot read %s", set_.c_str());
      return 2;
    }
    const json blob = json::parse(f);
    std::map<std::string, int> taken;
    std::vector<json> cases;
    for (const auto& c : blob["cases"]) {
      const auto g = c["gravity"].get<std::string>();
      if (taken[g] < per_) {
        ++taken[g];
        cases.push_back(c);
      }
    }
    if (!client_->wait_for_service(15s)) {
      RCLCPP_ERROR(get_logger(), "catch/start not available -- is catch_sim_node running?");
      return 2;
    }
    struct Row { std::string gravity, outcome; bool success; double impact; };
    std::vector<Row> rows;
    for (size_t i = 0; i < cases.size() && rclcpp::ok(); ++i) {
      const auto& c = cases[i];
      auto req = std::make_shared<StartSrv::Request>();
      req->gravity = c["gravity"].get<std::string>();
      req->restitution = c["restitution"].get<float>();
      req->use_drop = true;
      req->drop.x = c["drop"][0].get<double>();
      req->drop.y = c["drop"][1].get<double>();
      req->drop.z = c["drop"][2].get<double>();
      req->use_push = true;
      req->push = {c["push"][0].get<double>(), c["push"][1].get<double>()};
      req->start_from_pick = false;
      req->seed = c["seed"].get<int64_t>();
      auto fut = client_->async_send_request(req);
      if (fut.wait_for(10s) != std::future_status::ready) {
        RCLCPP_ERROR(get_logger(), "case %zu: catch/start timed out", i);
        return 3;
      }
      const auto started = fut.get();
      if (!started->success) {
        RCLCPP_ERROR(get_logger(), "case %zu: %s", i, started->message.c_str());
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::optional<StatusMsg> done;
      while (rclcpp::ok() && !done) {
        {
          std::lock_guard<std::mutex> lk(mu_);
          if (last_ && last_->trial == started->trial && last_->done) done = last_;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit_) break;
        std::this_thread::sleep_for(1ms);
      }
      if (!done) {
        RCLCPP_ERROR(get_logger(), "case %zu: no outcome within %.0f s (policy_node running? lockstep?)", i, limit_);
        return 3;
      }
      rows.push_back({req->gravity, done->outcome, done->success, done->impact_speed});
      if ((i + 1) % 50 == 0) RCLCPP_INFO(get_logger(), "%zu/%zu trials", i + 1, cases.size());
    }
    std::map<std::string, int> trials, caught;
    std::map<std::string, std::map<std::string, int>> failures;
    for (const auto& r : rows) {
      ++trials[r.gravity];
      if (r.success) ++caught[r.gravity];
      else ++failures[r.gravity][r.outcome.substr(0, r.outcome.find(" ("))];
    }
    json summary = json::object();
    for (const auto& [g, n] : trials) {
      summary[g] = {{"trials", n}, {"caught", caught[g]}, {"failures", failures[g]}};
    }
    for (auto& [g, s] : summary.items()) {
      s["success"] = s["caught"].get<double>() / s["trials"].get<double>();
      RCLCPP_INFO(get_logger(), "%-6s %5.1f %% of %d", g.c_str(), 100.0 * s["success"].get<double>(),
                  s["trials"].get<int>());
    }
    RCLCPP_INFO(get_logger(), "RESULT %s", summary.dump().c_str());
    if (!out_.empty()) {
      json rj = json::array();
      for (const auto& r : rows)
        rj.push_back({{"gravity", r.gravity}, {"success", r.success}, {"outcome", r.outcome},
                      {"impact_speed", std::isfinite(r.impact) ? json(r.impact) : json(nullptr)}});
      std::ofstream(out_) << json{{"summary", summary}, {"trials", rj}}.dump(1);
    }
    return 0;
  }

 private:
  std::string set_, out_;
  int64_t per_ = 100;
  double limit_ = 60.0;
  std::mutex mu_;
  std::optional<StatusMsg> last_;
  rclcpp::Subscription<StatusMsg>::SharedPtr sub_;
  rclcpp::Client<StartSrv>::SharedPtr client_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CatchEvalClient>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  std::thread spin([&] { exec.spin(); });
  const int rc = node->run();
  exec.cancel();
  spin.join();
  rclcpp::shutdown();
  return rc;
}
