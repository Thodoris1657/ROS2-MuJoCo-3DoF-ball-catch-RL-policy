// eval_client -- score whatever drives the arm over ROS on the frozen reach set (C++).
//
// Replays benchmarks/eval_set_v1.json via the reset service; run the simulator with
// mode:=lockstep episodic:=true (arm_rl.launch.py eval:=true). Same metrics as
// scripts/benchmark.py: success @5 / @2 cm, median final distance, time to reach, hold.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

#include <mujoco_arm_interfaces/msg/arm_status.hpp>
#include <mujoco_arm_interfaces/srv/reset_episode.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_arm_cpp/ros_common.hpp"

using namespace std::chrono_literals;
using Status = mujoco_arm_interfaces::msg::ArmStatus;
using ResetSrv = mujoco_arm_interfaces::srv::ResetEpisode;
namespace mros = mujoco_arm::ros;
using nlohmann::json;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("eval_client");
  const auto episodes = node->declare_parameter<int64_t>("episodes", 100);
  const double hz = node->declare_parameter<double>("control_hz", 25.0);
  const auto out = node->declare_parameter<std::string>("out", "");
  const double limit = node->declare_parameter<double>("timeout_s", 60.0);
  const auto set = mros::or_share(node->declare_parameter<std::string>("set", ""), "benchmarks/eval_set_v1.json");
  std::mutex mu;
  std::vector<Status> status;
  auto sub = node->create_subscription<Status>("status", mros::reliable(200), [&](Status::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lk(mu);
    status.push_back(*m);
  });
  auto client = node->create_client<ResetSrv>("reset");
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  std::thread spin([&] { exec.spin(); });
  int rc = 0;
  std::ifstream f(set);
  const json cases = json::parse(f)["cases"];
  if (!client->wait_for_service(15s)) {
    RCLCPP_ERROR(node->get_logger(), "reset service not available -- is sim_node running?");
    rc = 2;
  }
  const double dt = 1.0 / hz;
  std::vector<double> final_d, ttr, holdv;
  int s5 = 0, s2 = 0;
  for (int64_t i = 0; rc == 0 && i < std::min<int64_t>(episodes, cases.size()); ++i) {
    const auto& c = cases[i];
    auto req = std::make_shared<ResetSrv::Request>();
    req->use_target = req->use_qpos = true;
    req->seed = i;
    req->target.x = c["target"][0];
    req->target.y = c["target"][1];
    req->target.z = c["target"][2];
    for (int k = 0; k < 3; ++k) req->qpos[k] = c["qpos"][k];
    {
      std::lock_guard<std::mutex> lk(mu);
      status.clear();
    }
    auto fut = client->async_send_request(req);
    if (fut.wait_for(10s) != std::future_status::ready) {
      RCLCPP_ERROR(node->get_logger(), "episode %ld: reset timed out", static_cast<long>(i));
      continue;
    }
    const auto resp = fut.get();  // a std::future can be read once
    if (!resp->success) {
      RCLCPP_ERROR(node->get_logger(), "episode %ld: reset failed: %s", static_cast<long>(i), resp->message.c_str());
      continue;
    }
    const uint32_t ep = resp->episode;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> d;
    while (rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> lk(mu);
        const auto done = std::find_if(status.begin(), status.end(),
                                       [&](const Status& m) { return m.episode == ep && m.episode_done; });
        if (done != status.end()) {
          for (const auto& m : status)
            if (m.episode == ep && m.episode_step >= 1 && m.episode_step <= done->episode_step) d.push_back(m.distance);
          break;
        }
      }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit) {
        RCLCPP_ERROR(node->get_logger(), "episode %ld did not finish (policy_node? lockstep + episodic?)",
                     static_cast<long>(i));
        rc = 3;
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    if (rc || d.empty()) break;
    const int hold_steps = std::max(1, static_cast<int>(std::lround(0.5 / dt)));
    final_d.push_back(d.back());
    s5 += d.back() < 0.05;
    s2 += d.back() < 0.02;
    const auto first = std::find_if(d.begin(), d.end(), [](double x) { return x < 0.05; });
    ttr.push_back(first == d.end() ? NAN : (std::distance(d.begin(), first) + 1) * dt);
    int in = 0;
    for (size_t k = d.size() - std::min<size_t>(hold_steps, d.size()); k < d.size(); ++k) in += d[k] < 0.05;
    holdv.push_back(static_cast<double>(in) / hold_steps);
  }
  if (rc == 0 && !final_d.empty()) {
    const double n = static_cast<double>(final_d.size());
    std::vector<double> sorted = final_d;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted.size() % 2 ? sorted[sorted.size() / 2]
                                         : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    double tsum = 0;
    int tn = 0;
    for (double t : ttr)
      if (std::isfinite(t)) tsum += t, ++tn;
    const json summary{{"episodes", final_d.size()}, {"success_5cm", s5 / n}, {"success_2cm", s2 / n},
                       {"median_final_dist_mm", 1000.0 * med}, {"mean_time_to_reach_s", tn ? tsum / tn : NAN},
                       {"hold", std::accumulate(holdv.begin(), holdv.end(), 0.0) / n}};
    RCLCPP_INFO(node->get_logger(), "RESULT %s", summary.dump().c_str());
    if (!out.empty()) std::ofstream(out) << json{{"summary", summary}}.dump(1);
  }
  exec.cancel();
  spin.join();
  rclcpp::shutdown();
  return rc;
}
