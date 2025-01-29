// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <type_traits>

#include <wpi/SmallVector.h>

#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"

namespace sleipnir {

/**
 * Iteration type.
 */
enum class IterationType : uint8_t {
  /// Normal iteration.
  kNormal,
  /// Accepted second-order correction iteration.
  kAcceptedSOC,
  /// Rejected second-order correction iteration.
  kRejectedSOC
};

/**
 * Converts an enumeration to its underlying type.
 */
template <typename T>
[[nodiscard]]
constexpr std::underlying_type_t<T> to_underlying(T value) noexcept {
  return static_cast<std::underlying_type_t<T>>(value);
}

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
constexpr double ToMs(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1e3;
}

/**
 * Prints diagnostics for the current iteration.
 *
 * @param iterations Number of iterations.
 * @param type The iteration's type.
 * @param time The iteration duration.
 * @param error The error.
 * @param cost The cost.
 * @param infeasibility The infeasibility.
 * @param complementarity The complementarity.
 * @param δ The Hessian regularization factor.
 * @param primal_α The primal step size.
 * @param dual_α The dual step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void PrintIterationDiagnostics(int iterations, IterationType type,
                               const std::chrono::duration<Rep, Period>& time,
                               double error, double cost, double infeasibility,
                               double complementarity, double δ,
                               double primal_α, double dual_α) {
  if (iterations % 20 == 0) {
    if (iterations == 0) {
      sleipnir::print("┏");
    } else {
      sleipnir::print("┢");
    }
    sleipnir::print(
        "{:━^4}┯{:━^4}┯{:━^9}┯{:━^12}┯{:━^13}┯{:━^12}┯{:━^12}┯{:━^5}┯{:━^8}┯"
        "{:━^8}┯{:━^2}",
        "", "", "", "", "", "", "", "", "", "", "");
    if (iterations == 0) {
      sleipnir::println("┓");
    } else {
      sleipnir::println("┪");
    }
    sleipnir::println(
        "┃{:^4}│{:^4}│{:^9}│{:^12}│{:^13}│{:^12}│{:^12}│{:^5}│{:^8}│{:^8}│{:^2}"
        "┃",
        "iter", "type", "time (ms)", "error", "cost", "infeas.", "complement.",
        "reg", "primal α", "dual α", "↩");
    sleipnir::println(
        "┡{:━^4}┷{:━^4}┷{:━^9}┷{:━^12}┷{:━^13}┷{:━^12}┷{:━^12}┷{:━^5}┷{:━^8}┷"
        "{:━^8}┷{:━^2}┩",
        "", "", "", "", "", "", "", "", "", "", "");
  }

  constexpr const char* kIterationTypes[] = {"norm", "✓SOC", "XSOC"};
  sleipnir::print("│{:4} {:4} {:9.3f} {:12e} {:13e} {:12e} {:12e} ", iterations,
                  kIterationTypes[to_underlying(type)], ToMs(time), error, cost,
                  infeasibility, complementarity);

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
      wpi::SmallVector<int> digits;
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
  sleipnir::println(" {:.2e} {:.2e} {:2d}│", primal_α, dual_α,
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
 * Prints final diagnostics.
 *
 * @param iterations Number of iterations.
 * @param setupProfilers Setup profilers.
 * @param solveProfilers Solve profilers.
 */
inline void PrintFinalDiagnostics(
    int iterations, const wpi::SmallVector<SetupProfiler>& setupProfilers,
    const wpi::SmallVector<SolveProfiler>& solveProfilers) {
  // Print bottom of iteration diagnostics table
  sleipnir::println("└{:─^99}┘", "");

  // Print total time
  auto setupDuration = ToMs(setupProfilers[0].Duration());
  auto solveDuration = ToMs(solveProfilers[0].TotalDuration());
  sleipnir::println("\nTime: {:.3f} ms", setupDuration + solveDuration);
  sleipnir::println("  ↳ setup: {:.3f} ms", setupDuration);
  sleipnir::println("  ↳ solve: {:.3f} ms ({} iterations)", solveDuration,
                    iterations);

  // Print setup diagnostics
  sleipnir::println("\n┏{:━^21}┯{:━^18}┯{:━^10}┓", "", "", "");
  sleipnir::println("┃{:^21}│{:^18}│{:^10}┃", "trace", "percent", "total (ms)");
  sleipnir::println("┡{:━^21}┷{:━^18}┷{:━^10}┩", "", "", "");

  for (auto& profiler : setupProfilers) {
    double norm =
        ToMs(profiler.Duration()) / ToMs(setupProfilers[0].Duration());
    sleipnir::println("│{:<21} {:>6.2f}%▕{}▏ {:>10.3f}│", profiler.name,
                      norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.Duration()));
  }

  sleipnir::println("└{:─^51}┘", "");

  // Print solve diagnostics
  sleipnir::println("┏{:━^21}┯{:━^18}┯{:━^10}┯{:━^9}┯{:━^4}┓", "", "", "", "",
                    "");
  sleipnir::println("┃{:^21}│{:^18}│{:^10}│{:^9}│{:^4}┃", "trace", "percent",
                    "total (ms)", "each (ms)", "runs");
  sleipnir::println("┡{:━^21}┷{:━^18}┷{:━^10}┷{:━^9}┷{:━^4}┩", "", "", "", "",
                    "");

  for (auto& profiler : solveProfilers) {
    double norm = ToMs(profiler.TotalDuration()) /
                  ToMs(solveProfilers[0].TotalDuration());
    sleipnir::println("│{:<21} {:>6.2f}%▕{}▏ {:>10.3f} {:>9.3f} {:>4}│",
                      profiler.name, norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.TotalDuration()),
                      ToMs(profiler.AverageDuration()), profiler.NumSolves());
  }

  sleipnir::println("└{:─^66}┘\n", "");
}

}  // namespace sleipnir
