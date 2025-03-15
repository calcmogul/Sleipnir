// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/util/print.hpp"
#include "sleipnir/util/setup_profiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"

namespace slp {

/**
 * Iteration type.
 */
enum class IterationType : uint8_t {
  /// Normal iteration.
  NORMAL,
  /// Accepted second-order correction iteration.
  ACCEPTED_SOC,
  /// Rejected second-order correction iteration.
  REJECTED_SOC
};

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
constexpr double to_ms(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1e3;
}

/**
 * Renders value as power of 10.
 *
 * @param value Value.
 */
inline std::string power_of_10(double value) {
  if (value == 0.0) {
    return " 0";
  } else {
    int exponent = std::log10(value);

    if (exponent == 0) {
      return " 1";
    } else if (exponent == 1) {
      return "10";
    } else {
      // Gather exponent digits
      int n = std::abs(exponent);
      small_vector<int> digits;
      do {
        digits.emplace_back(n % 10);
        n /= 10;
      } while (n > 0);

      std::string output = "10";

      // Append exponent
      if (exponent < 0) {
        output += "⁻";
      }
      constexpr std::array strs = {"⁰", "¹", "²", "³", "⁴",
                                   "⁵", "⁶", "⁷", "⁸", "⁹"};
      for (const auto& digit : digits | std::views::reverse) {
        output += strs[digit];
      }

      return output;
    }
  }
}

/**
 * Prints error for too many degrees of freedom.
 *
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 */
inline void print_too_many_dofs_error(const Eigen::VectorXd& c_e) {
  slp::println("The problem has too few degrees of freedom.");
  slp::println("Violated constraints (cₑ(x) = 0) in order of declaration:");
  for (int row = 0; row < c_e.rows(); ++row) {
    if (c_e(row) < 0.0) {
      slp::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e(row));
    }
  }
}

/**
 * Prints equality constraint local infeasibility error.
 *
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 */
inline void print_c_e_local_infeasibility_error(const Eigen::VectorXd& c_e) {
  slp::println(
      "The problem is locally infeasible due to violated equality "
      "constraints.");
  slp::println("Violated constraints (cₑ(x) = 0) in order of declaration:");
  for (int row = 0; row < c_e.rows(); ++row) {
    if (c_e[row] < 0.0) {
      slp::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e[row]);
    }
  }
}

/**
 * Prints inequality constraint local infeasibility error.
 *
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 */
inline void print_c_i_local_infeasibility_error(const Eigen::VectorXd& c_i) {
  slp::println(
      "The problem is infeasible due to violated inequality "
      "constraints.");
  slp::println("Violated constraints (cᵢ(x) ≥ 0) in order of declaration:");
  for (int row = 0; row < c_i.rows(); ++row) {
    if (c_i[row] < 0.0) {
      slp::println("  {}/{}: {} ≥ 0", row + 1, c_i.rows(), c_i[row]);
    }
  }
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
 * @param μ The barrier parameter.
 * @param δ The Hessian regularization factor.
 * @param primal_α The primal step size.
 * @param primal_α_max The max primal step size.
 * @param dual_α The dual step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void print_iteration_diagnostics(int iterations, IterationType type,
                                 const std::chrono::duration<Rep, Period>& time,
                                 double error, double cost,
                                 double infeasibility, double complementarity,
                                 double μ, double δ, double primal_α,
                                 double primal_α_max, double dual_α) {
  if (iterations % 20 == 0) {
    if (iterations == 0) {
      slp::print("┏");
    } else {
      slp::print("┢");
    }
    slp::print(
        "{:━^4}┯{:━^4}┯{:━^9}┯{:━^12}┯{:━^13}┯{:━^12}┯{:━^12}┯{:━^8}┯{:━^5}┯"
        "{:━^8}┯{:━^8}┯{:━^2}",
        "", "", "", "", "", "", "", "", "", "", "", "");
    if (iterations == 0) {
      slp::println("┓");
    } else {
      slp::println("┪");
    }
    slp::println(
        "┃{:^4}│{:^4}│{:^9}│{:^12}│{:^13}│{:^12}│{:^12}│{:^8}│{:^5}│{:^8}│{:^8}"
        "│{:^2}┃",
        "iter", "type", "time (ms)", "error", "cost", "infeas.", "complement.",
        "μ", "reg", "primal α", "dual α", "↩");
    slp::println(
        "┡{:━^4}┷{:━^4}┷{:━^9}┷{:━^12}┷{:━^13}┷{:━^12}┷{:━^12}┷{:━^8}┷{:━^5}┷"
        "{:━^8}┷{:━^8}┷{:━^2}┩",
        "", "", "", "", "", "", "", "", "", "", "", "");
  }

  // For the number of backtracks, we want x such that:
  //
  //   α_max 2⁻ˣ = α
  //   2⁻ˣ = α/α_max
  //   −x = std::log2(α/α_max)
  //   x = −std::log2(α/α_max)
  int backtracks = static_cast<int>(-std::log2(primal_α / primal_α_max));

  constexpr std::array ITERATION_TYPES = {"norm", "✓SOC", "XSOC"};
  slp::println(
      "│{:4} {:4} {:9.3f} {:12e} {:13e} {:12e} {:12e} {:.2e} {:<5} {:.2e} "
      "{:.2e} {:2d}│",
      iterations, ITERATION_TYPES[std::to_underlying(type)], to_ms(time), error,
      cost, infeasibility, complementarity, μ, power_of_10(δ), primal_α, dual_α,
      backtracks);
}

/**
 * Renders histogram of the given normalized value.
 *
 * @tparam Width Width of the histogram in characters.
 * @param value Normalized value from 0 to 1.
 */
template <int Width>
  requires(Width > 0)
std::string histogram(double value) {
  value = std::clamp(value, 0.0, 1.0);

  double ipart;
  int fpart = static_cast<int>(std::modf(value * Width, &ipart) * 8);

  constexpr std::array strs = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
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
 * @param setup_profilers Setup profilers.
 * @param solve_profilers Solve profilers.
 */
inline void print_final_diagnostics(
    int iterations, const small_vector<SetupProfiler>& setup_profilers,
    const small_vector<SolveProfiler>& solve_profilers) {
  // Print bottom of iteration diagnostics table
  slp::println("└{:─^108}┘", "");

  // Print total time
  auto setup_duration = to_ms(setup_profilers[0].duration());
  auto solve_duration = to_ms(solve_profilers[0].total_duration());
  slp::println("\nTime: {:.3f} ms", setup_duration + solve_duration);
  slp::println("  ↳ setup: {:.3f} ms", setup_duration);
  slp::println("  ↳ solve: {:.3f} ms ({} iterations)", solve_duration,
               iterations);

  // Print setup diagnostics
  slp::println("\n┏{:━^23}┯{:━^18}┯{:━^10}┓", "", "", "");
  slp::println("┃{:^23}│{:^18}│{:^10}┃", "trace", "percent", "total (ms)");
  slp::println("┡{:━^23}┷{:━^18}┷{:━^10}┩", "", "", "");

  for (auto& profiler : setup_profilers) {
    double norm = setup_duration == 0.0
                      ? (&profiler == &setup_profilers[0] ? 1.0 : 0.0)
                      : to_ms(profiler.duration()) / setup_duration;
    slp::println("│{:<23} {:>6.2f}%▕{}▏ {:>10.3f}│", profiler.name,
                 norm * 100.0, histogram<9>(norm), to_ms(profiler.duration()));
  }

  slp::println("└{:─^53}┘", "");

  // Print solve diagnostics
  slp::println("┏{:━^23}┯{:━^18}┯{:━^10}┯{:━^9}┯{:━^4}┓", "", "", "", "", "");
  slp::println("┃{:^23}│{:^18}│{:^10}│{:^9}│{:^4}┃", "trace", "percent",
               "total (ms)", "each (ms)", "runs");
  slp::println("┡{:━^23}┷{:━^18}┷{:━^10}┷{:━^9}┷{:━^4}┩", "", "", "", "", "");

  for (auto& profiler : solve_profilers) {
    double norm = solve_duration == 0.0
                      ? (&profiler == &solve_profilers[0] ? 1.0 : 0.0)
                      : to_ms(profiler.total_duration()) / solve_duration;
    slp::println("│{:<23} {:>6.2f}%▕{}▏ {:>10.3f} {:>9.3f} {:>4}│",
                 profiler.name, norm * 100.0, histogram<9>(norm),
                 to_ms(profiler.total_duration()),
                 to_ms(profiler.average_duration()), profiler.num_solves());
  }

  slp::println("└{:─^68}┘", "");
}

}  // namespace slp
