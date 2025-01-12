// Copyright (c) Sleipnir contributors

#include <chrono>
#include <vector>

#include <fmt/base.h>

#include "CasADi.hpp"
#include "CmdlineArguments.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main(int argc, char* argv[]) {
  using namespace std::chrono_literals;

  CmdlineArgs args{argv, argc};

  bool runCasadi = args.Contains("--casadi");
  bool runSleipnir = args.Contains("--sleipnir");
  if (!runCasadi && !runSleipnir) {
    runCasadi = true;
    runSleipnir = true;
  }
  bool diagnostics = args.Contains("--enable-diagnostics");

  constexpr std::chrono::duration<double> T = 5s;

  std::vector<int> sampleSizesToTest;
  for (int N = 100; N < 300; N += 50) {
    sampleSizesToTest.emplace_back(N);
  }
  sampleSizesToTest.emplace_back(300);

  fmt::println("Solving cart-pole problem from N = {} to N = {}.",
               sampleSizesToTest.front(), sampleSizesToTest.back());
  if (runCasadi) {
    RunBenchmarksAndLog<casadi::Opti>(
        "cart-pole-scalability-results-casadi.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleCasADi);
  }
  if (runSleipnir) {
    RunBenchmarksAndLog<sleipnir::OptimizationProblem>(
        "cart-pole-scalability-results-sleipnir.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleSleipnir);
  }
}
