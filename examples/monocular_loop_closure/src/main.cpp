// Copyright (c) Sleipnir contributors

// Closes the loop on a monocular visual odometry trajectory whose scale
// drifted.
//
// A single camera can't observe absolute scale, so visual odometry estimates
// each keyframe's map at a scale that drifts over time. Here the drift is 3%
// per keyframe. When the camera returns to its starting point, recognizing the
// place gives a loop closure measurement between the last and first keyframes.
//
// A pose graph in SE(3) can only spread the loop closure's error over
// positions and rotations, which bends the trajectory. A pose graph in Sim(3)
// gives each keyframe a scale as well, so the scale drift is spread over the
// trajectory and its shape is recovered (Strasdat et al., "Scale Drift-Aware
// Large Scale Monocular SLAM", RSS 2010).
//
// The camera also has an IMU, which measures the direction of gravity in each
// keyframe's frame. That's a constraint on each keyframe's rotation that the
// solver takes directly: the measured up direction, rotated into the world
// frame, must be within the IMU's accuracy of the world's up axis.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/similarity_twist3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/problem.hpp>

// Squared norm of a column vector.
slp::Real<double> squared_norm(const slp::RealMatrix<double>& v) {
  return (v.T() * v)[0];
}

// Draws standard normal samples with the same algorithm as NumPy's
// RandomState.standard_normal(), so the C++ and Python examples see the same
// noise for the same seed.
class StandardNormal {
 public:
  explicit StandardNormal(std::uint32_t seed) : m_engine{seed} {}

  double operator()() {
    if (m_has_cached) {
      m_has_cached = false;
      return m_cached;
    }

    // Marsaglia polar method
    double x1;
    double x2;
    double r_sq;
    do {
      x1 = 2.0 * uniform() - 1.0;
      x2 = 2.0 * uniform() - 1.0;
      r_sq = x1 * x1 + x2 * x2;
    } while (r_sq >= 1.0 || r_sq == 0.0);

    double f = std::sqrt(-2.0 * std::log(r_sq) / r_sq);
    m_cached = f * x1;
    m_has_cached = true;
    return f * x2;
  }

 private:
  std::mt19937 m_engine;
  bool m_has_cached = false;
  double m_cached = 0.0;

  // Uniform sample in [0, 1) with 53 random bits
  double uniform() {
    double a = m_engine() >> 5;
    double b = m_engine() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }
};

// A relative transform measured between keyframes i and j, expressed in
// keyframe i's frame.
template <typename Transform>
struct Edge {
  int i;
  int j;
  Transform measurement;
};

int main() {
  // Number of keyframes. Keyframe N − 1 is back at keyframe 0's place.
  constexpr int N = 25;

  // Scale drift per keyframe
  constexpr double drift = 1.03;

  // Accuracy of the IMU's gravity direction
  constexpr double imu_accuracy = std::numbers::pi / 180.0;

  // The world's up axis
  const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();

  // The true trajectory is a 5 m radius loop of road with hills 0.5 m high. The
  // camera faces along the road and pitches with its slope. Keyframe frames
  // are x forward, y left, and z up. world2keyframe maps a point in a
  // keyframe's frame to the world frame. Its scale is how many meters one unit
  // of that keyframe's map is, which drifts by 3% per keyframe.
  std::vector<slp::Similarity3<double>> true_world2keyframe;
  for (int i = 0; i < N; ++i) {
    double θ = 2.0 * std::numbers::pi * i / (N - 1);
    Eigen::Vector3d translation{5.0 * std::sin(θ), 5.0 * (1.0 - std::cos(θ)),
                                0.5 * std::sin(2.0 * θ)};
    // Yaw θ, then pitch the nose up by the road's slope angle φ
    double φ = std::atan(std::cos(2.0 * θ) / 5.0);
    Eigen::Quaterniond rotation{std::cos(θ / 2.0) * std::cos(φ / 2.0),
                                std::sin(θ / 2.0) * std::sin(φ / 2.0),
                                -std::cos(θ / 2.0) * std::sin(φ / 2.0),
                                std::sin(θ / 2.0) * std::cos(φ / 2.0)};
    true_world2keyframe.emplace_back(translation, rotation, std::pow(drift, i));
  }

  // Visual odometry measures the relative position and rotation between
  // consecutive keyframes in the earlier keyframe's map units. It assumes the
  // scale stays the same, so it measures a relative scale of 1.
  std::vector<Edge<slp::Similarity3<double>>> sim3_edges;
  for (int i = 0; i + 1 < N; ++i) {
    auto relative =
        true_world2keyframe[i + 1].relative_to(true_world2keyframe[i]);
    sim3_edges.emplace_back(i, i + 1,
                            slp::Similarity3<double>{relative.translation(),
                                                     relative.rotation(), 0.0});
  }

  // The loop closure aligns the last keyframe's map with the first keyframe's
  // map, which also measures their scale ratio.
  sim3_edges.emplace_back(
      N - 1, 0, true_world2keyframe[0].relative_to(true_world2keyframe[N - 1]));

  // Each measurement has Gaussian noise in its own frame, with standard
  // deviations of 2 cm, 0.01 rad, and 1% scale. The seed is fixed so the
  // results are repeatable.
  constexpr double σ_linear = 0.02;
  constexpr double σ_angular = 0.01;
  constexpr double σ_log_scale = 0.01;
  StandardNormal standard_normal{42};
  for (auto& edge : sim3_edges) {
    Eigen::Vector3d linear{σ_linear * standard_normal(),
                           σ_linear * standard_normal(),
                           σ_linear * standard_normal()};
    Eigen::Vector3d angular{σ_angular * standard_normal(),
                            σ_angular * standard_normal(),
                            σ_angular * standard_normal()};
    slp::SimilarityTwist3<double> noise{slp::RealMatrix<double>{linear},
                                        slp::RealMatrix<double>{angular},
                                        σ_log_scale * standard_normal()};
    edge.measurement = edge.measurement * noise.exp();
  }

  // The IMU measures the world's up axis in each keyframe's frame, with noise
  // of 0.2° per axis before normalization
  std::vector<Eigen::Vector3d> measured_up{Eigen::Vector3d::Zero()};
  for (int i = 1; i < N; ++i) {
    Eigen::Vector3d noise{0.0035 * standard_normal(),
                          0.0035 * standard_normal(),
                          0.0035 * standard_normal()};
    Eigen::Vector3d true_up =
        true_world2keyframe[i].rotation().value().conjugate() * up;
    measured_up.push_back((true_up + noise).normalized());
  }

  // The SE(3) pose graph uses the same measurements without the scale
  std::vector<Edge<slp::Pose3<double>>> se3_edges;
  for (const auto& [i, j, measurement] : sim3_edges) {
    se3_edges.emplace_back(
        i, j,
        slp::Pose3<double>{measurement.translation(), measurement.rotation()});
  }

  // Chains the odometry for the initial guess. The first keyframe defines the
  // world frame.
  std::vector<slp::Similarity3<double>> odometry{true_world2keyframe[0]};
  for (int i = 0; i + 1 < N; ++i) {
    odometry.push_back(odometry[i] * sim3_edges[i].measurement);
  }

  // Root mean square position error of a trajectory against the truth
  auto rms_position_error = [&](auto&& position) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      sum += (position(i) - true_world2keyframe[i].translation().value())
                 .squaredNorm();
    }
    return std::sqrt(sum / N);
  };

  // Largest angle between the world's up axis in a trajectory's keyframe
  // frames and in the true keyframe frames
  auto max_tilt_error_degrees = [&](auto&& rotation) {
    double max = 0.0;
    for (int i = 0; i < N; ++i) {
      Eigen::Vector3d true_up =
          true_world2keyframe[i].rotation().value().conjugate() * up;
      double cos_error =
          std::min((rotation(i).conjugate() * up).dot(true_up), 1.0);
      max = std::max(max, std::acos(cos_error) * 180.0 / std::numbers::pi);
    }
    return max;
  };

  std::println("Odometry RMS position error: {:.4f} m",
               rms_position_error(
                   [&](int i) { return odometry[i].translation().value(); }));
  std::println("Odometry max tilt error: {:.2f}°",
               max_tilt_error_degrees(
                   [&](int i) { return odometry[i].rotation().value(); }));

  // SE(3) pose graph
  {
    slp::Problem<double> problem;

    // The first keyframe is fixed because it defines the world frame
    std::vector<slp::Pose3<double>> world2keyframe{
        slp::Pose3<double>{odometry[0].translation(), odometry[0].rotation()}};
    for (int i = 1; i < N; ++i) {
      auto& pose =
          world2keyframe.emplace_back(problem.pose3_decision_variable());
      pose.translation().set_value(odometry[i].translation().value());
      pose.rotation().set_value(odometry[i].rotation().value());
    }

    // The error of each edge is the twist from the measurement to the
    // estimated relative pose. Each component is divided by its noise's
    // standard deviation, so the cost is the measurements' negative
    // log-likelihood up to a constant.
    slp::Real J = 0.0;
    for (const auto& [i, j, measurement] : se3_edges) {
      auto error = world2keyframe[j]
                       .relative_to(world2keyframe[i])
                       .relative_to(measurement)
                       .log();
      J += squared_norm(error.linear()) / (σ_linear * σ_linear) +
           squared_norm(error.angular()) / (σ_angular * σ_angular);
    }
    problem.minimize(J);

    // The IMU's measured up direction, rotated into the world frame, is within
    // the IMU's accuracy of the world's up axis
    for (int i = 1; i < N; ++i) {
      auto keyframe_up = slp::UnitVector<double>{measured_up[i]}.rotate_by(
          world2keyframe[i].rotation());
      problem.subject_to(keyframe_up.dot(up) >= std::cos(imu_accuracy));
    }

    problem.solve({.diagnostics = true});

    std::println("SE(3) RMS position error: {:.4f} m",
                 rms_position_error([&](int i) {
                   return world2keyframe[i].translation().value();
                 }));
    std::println("SE(3) max tilt error: {:.2f}°",
                 max_tilt_error_degrees([&](int i) {
                   return world2keyframe[i].rotation().value();
                 }));
  }

  // Sim(3) pose graph
  {
    slp::Problem<double> problem;

    // The first keyframe is fixed because it defines the world frame and its
    // scale
    std::vector<slp::Similarity3<double>> world2keyframe{odometry[0]};
    for (int i = 1; i < N; ++i) {
      auto& similarity =
          world2keyframe.emplace_back(problem.similarity3_decision_variable());
      similarity.translation().set_value(odometry[i].translation().value());
      similarity.rotation().set_value(odometry[i].rotation().value());
      similarity.log_scale().set_value(odometry[i].log_scale().value());
    }

    // The error of each edge is the similarity twist from the measurement to
    // the estimated relative similarity transform. Each component is divided by
    // its noise's standard deviation, so the cost is the measurements' negative
    // log-likelihood up to a constant.
    slp::Real J = 0.0;
    for (const auto& [i, j, measurement] : sim3_edges) {
      auto error = world2keyframe[j]
                       .relative_to(world2keyframe[i])
                       .relative_to(measurement)
                       .log();
      J += squared_norm(error.linear()) / (σ_linear * σ_linear) +
           squared_norm(error.angular()) / (σ_angular * σ_angular) +
           error.log_scale() * error.log_scale() / (σ_log_scale * σ_log_scale);
    }
    problem.minimize(J);

    // The IMU's measured up direction, rotated into the world frame, is within
    // the IMU's accuracy of the world's up axis
    for (int i = 1; i < N; ++i) {
      auto keyframe_up = slp::UnitVector<double>{measured_up[i]}.rotate_by(
          world2keyframe[i].rotation());
      problem.subject_to(keyframe_up.dot(up) >= std::cos(imu_accuracy));
    }

    problem.solve({.diagnostics = true});

    std::println("Sim(3) RMS position error: {:.4f} m",
                 rms_position_error([&](int i) {
                   return world2keyframe[i].translation().value();
                 }));
    std::println("Sim(3) max tilt error: {:.2f}°",
                 max_tilt_error_degrees([&](int i) {
                   return world2keyframe[i].rotation().value();
                 }));
    std::println("Sim(3) last keyframe scale: {:.4f} (true {:.4f})",
                 world2keyframe[N - 1].scale().value(), std::pow(drift, N - 1));
  }
}
