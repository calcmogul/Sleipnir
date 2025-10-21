// Copyright (c) Sleipnir contributors

#include <chrono>
#include <print>
#include <vector>

#include "casadi.hpp"
#include "cmdline_args.hpp"
#include "sleipnir.hpp"
#include "util.hpp"

int main(int argc, char* argv[]) {
  using namespace std::chrono_literals;

  CmdlineArgs args{argv, argc};

  bool run_casadi = args.contains("--casadi");
  bool run_sleipnir = args.contains("--sleipnir");
  if (!run_casadi && !run_sleipnir) {
    run_casadi = true;
    run_sleipnir = true;
  }
  bool diagnostics = args.contains("--enable-diagnostics");

  constexpr std::chrono::duration<double> T = 5s;

  std::vector<int> sample_sizes_to_test;
  for (int N = 100; N < 1000; N += 100) {
    sample_sizes_to_test.emplace_back(N);
  }
  for (int N = 1000; N < 5000; N += 1000) {
    sample_sizes_to_test.emplace_back(N);
  }
  sample_sizes_to_test.emplace_back(5000);

  std::println("Solving flywheel problem from N = {} to N = {}.",
               sample_sizes_to_test.front(), sample_sizes_to_test.back());
  if (run_casadi) {
    run_benchmarks_and_log<casadi::Opti>(
        "flywheel-scalability-results-casadi.csv", diagnostics, T,
        sample_sizes_to_test, &flywheel_casadi);
  }
  if (run_sleipnir) {
    run_benchmarks_and_log<slp::Problem<double>>(
        "flywheel-scalability-results-sleipnir.csv", diagnostics, T,
        sample_sizes_to_test, &flywheel_sleipnir);
  }
}
