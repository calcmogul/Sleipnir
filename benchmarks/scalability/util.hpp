// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <concepts>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

#include <casadi/casadi.hpp>
#include <fmt/base.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/function_ref.hpp>

/// Converts std::chrono::duration to a number of milliseconds rounded to three
/// decimals.
template <typename Rep, typename Period = std::ratio<1>>
constexpr double to_ms(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1e3;
}

/// Runs scalability benchmarks for CasADi and Sleipnir versions of an
/// optimization problem, records the setup time and solve time for each, then
/// writes them to scalability-results.csv.
///
/// The scale of the problem is iteratively increased by increasing the number
/// of timesteps within the time horizon.
///
/// @tparam Problem The optimization problem's type (casadi::Opti or
///     slp::Problem).
/// @param filename Results CSV filename.
/// @param diagnostics Whether to enable diagnostic prints.
/// @param T The time horizon of the optimization problem.
/// @param sample_sizes_to_test List of sample sizes for which to record
///     results.
/// @param min_power The minimum power of 10 for the number of samples in the
///     problem.
/// @param max_power The maximum power of 10 for the number of samples in the
///     problem.
/// @param setup A function that takes a time horizon and number of samples and
///     returns an optimization problem instance.
template <typename Problem>
int run_benchmarks_and_log(
    std::string_view filename, bool diagnostics,
    std::chrono::duration<double> T, std::span<int> sample_sizes_to_test,
    slp::function_ref<Problem(std::chrono::duration<double> dt, int N)> setup) {
  std::ofstream results{std::string{filename}};
  if (!results.is_open()) {
    return 1;
  }

  results << "Samples," << "Setup time (ms),Solve time (ms)\n";
  std::flush(results);

  for (int N : sample_sizes_to_test) {
    auto dt = T / N;

    fmt::print(stderr, "N = {}...", N);

    // Record setup time
    auto setup_start_time = std::chrono::steady_clock::now();
    auto problem = setup(dt, N);
    auto setup_end_time = std::chrono::steady_clock::now();

    // Record solve time
    bool success = true;
    auto solve_start_time = std::chrono::steady_clock::now();
    if constexpr (std::same_as<Problem, casadi::Opti>) {
      if (diagnostics) {
        problem.solver("ipopt");
      } else {
        problem.solver("ipopt", {{"print_time", 0}},
                       {{"print_level", 0}, {"sb", "yes"}});
      }
      problem.solve();
    } else {
      success = problem.solve({.diagnostics = diagnostics}) ==
                slp::ExitStatus::SUCCESS;
    }
    auto solve_end_time = std::chrono::steady_clock::now();

    if (success) {
      results << N << ',' << to_ms(setup_end_time - setup_start_time) << ','
              << to_ms(solve_end_time - solve_start_time) << '\n';
      std::flush(results);
    } else {
      fmt::print(stderr, " FAIL");
    }

    fmt::println(stderr, " done.");
  }

  return 0;
}
