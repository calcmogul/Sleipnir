// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>

#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "util/ToMs.hpp"

namespace sleipnir {

/**
 * Iteration mode.
 */
enum class IterationMode : uint8_t {
  /// Normal iteration.
  kNormal,
  /// Second-order correction iteration.
  kSecondOrderCorrection,
  /// Feasibility restoration iteration.
  kFeasibilityRestoration
};

/**
 * Prints diagnostics for the current iteration.
 *
 * @param iterations Number of iterations.
 * @param mode Which mode the iteration was in.
 * @param time The iteration duration.
 * @param error The error.
 * @param cost The cost.
 * @param infeasibility The infeasibility.
 * @param δ The Hessian regularization factor.
 * @param primal_α The primal step size.
 * @param dual_α The dual step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void PrintIterationDiagnostics(int iterations, IterationMode mode,
                               const std::chrono::duration<Rep, Period>& time,
                               double error, double cost, double infeasibility,
                               double δ, double primal_α, double dual_α) {
  if (iterations % 20 == 0) {
    sleipnir::println(
        "{:^4}   {:^9}  {:^13}  {:^13}  {:^13}  {:^5}  {:^8}  {:^8}  {:^6}",
        "iter", "time (ms)", "error", "cost", "infeasibility", "reg",
        "primal α", "dual α", "bktrks");
    sleipnir::println("{:=^96}", "");
  }

  constexpr const char* kIterationModes[] = {" ", "s", "r"};
  sleipnir::print("{:4}{}  {:9.3f}  {:13e}  {:13e}  {:13e}  ", iterations,
                  kIterationModes[static_cast<uint8_t>(mode)], ToMs(time),
                  error, cost, infeasibility);

  // Print regularization
  if (δ == 0.0) {
    sleipnir::print(" 0   ");
  } else {
    int exponent = std::log10(δ);

    if (exponent == 0) {
      sleipnir::print(" 1   ");
    } else if (exponent == 1) {
      sleipnir::print("10   ");
    } else {
      // Gather regularization exponent digits
      int n = std::abs(exponent);
      small_vector<int> digits;
      do {
        digits.emplace_back(n % 10);
        n /= 10;
      } while (n > 0);

      std::string reg = "10";

      // Append regularization exponent
      if (exponent < 0) {
        reg += "⁻";
      }
      constexpr const char* strs[] = {"⁰", "¹", "²", "³", "⁴",
                                      "⁵", "⁶", "⁷", "⁸", "⁹"};
      for (const auto& digit : digits | std::views::reverse) {
        reg += strs[digit];
      }

      sleipnir::print("{:<5}", reg);
    }
  }

  // Print step sizes and number of backtracks
  sleipnir::println("  {:.2e}  {:.2e}  {:6d}", primal_α, dual_α,
                    static_cast<int>(-std::log2(primal_α)));
}

/**
 * Renders histogram of the given normalized value.
 *
 * @tparam Width Width of the histogram in characters.
 * @param value Normalized value from 0 to 1.
 */
template <int Width>
  requires(Width > 0)
inline std::string Histogram(double value) {
  value = std::clamp(value, 0.0, 1.0);

  double ipart;
  int fpart = static_cast<int>(std::modf(value * Width, &ipart) * 8);

  constexpr const char* strs[] = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
  std::string hist;

  int index = 0;
  while (index < ipart) {
    hist += strs[8];
    ++index;
  }
  if (fpart > 0) {
    hist += strs[fpart];
    ++index;
  }
  while (index < Width) {
    hist += strs[0];
    ++index;
  }

  return hist;
}

/**
 * Prints total time.
 *
 * @param iterations Number of iterations.
 * @param totalSetupProfiler Total setup profiler.
 * @param totalSolveProfiler Total solve profiler.
 */
inline void PrintTotalTime(int iterations,
                           const SetupProfiler& totalSetupProfiler,
                           const SolveProfiler& totalSolveProfiler) {
  auto setupDuration = ToMs(totalSetupProfiler.Duration());
  auto solveDuration = ToMs(totalSolveProfiler.TotalDuration());
  sleipnir::println("\nTime: {:.3f} ms", setupDuration + solveDuration);
  sleipnir::println("  ↳ setup: {:.3f} ms", setupDuration);
  sleipnir::println("  ↳ solve: {:.3f} ms ({} iterations)", solveDuration,
                    iterations);
}

/**
 * Prints setup diagnostics.
 *
 * @param setupProfilers Setup profilers.
 */
inline void PrintSetupDiagnostics(
    const small_vector<SetupProfiler>& setupProfilers) {
  // Setup heading
  sleipnir::println("\n{:^23}  {:^9}  {:^8}  {:^8}", "trace", "time (ms)",
                    "time (%)", "histogram");
  sleipnir::println("{:=^56}", "");

  constexpr auto setupFormat = "{:<23}  {:>9.3f}  {:>8.2f}  ░{}░";

  for (auto& profiler : setupProfilers) {
    double norm =
        ToMs(profiler.Duration()) / ToMs(setupProfilers[0].Duration());
    sleipnir::println(setupFormat, profiler.name, ToMs(profiler.Duration()),
                      norm * 100.0, Histogram<8>(norm));
  }
}

/**
 * Prints solve diagnostics.
 *
 * @param solveProfilers Solve profilers.
 */
inline void PrintSolveDiagnostics(
    const small_vector<SolveProfiler>& solveProfilers) {
  // Solve heading
  sleipnir::println("\n{:^27}  {:^9}  {:^5}  {:^10}  {:^9}  {:^8}", "trace",
                    "each (ms)", "calls", "total (ms)", "total (%)",
                    "histogram");
  sleipnir::println("{:=^80}", "");

  constexpr auto solveFormat =
      "{:<27}  {:>9.3f}  {:>5}  {:>10.3f}  {:>9.2f}  ░{}░";

  for (auto& profiler : solveProfilers) {
    double norm = ToMs(profiler.TotalDuration()) /
                  ToMs(solveProfilers[0].TotalDuration());
    sleipnir::println(solveFormat, profiler.name,
                      ToMs(profiler.AverageDuration()), profiler.NumSolves(),
                      ToMs(profiler.TotalDuration()), norm * 100.0,
                      Histogram<8>(norm));
  }

  sleipnir::println("");
}

}  // namespace sleipnir
