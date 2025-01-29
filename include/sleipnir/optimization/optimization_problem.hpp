// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>

#include <Eigen/Core>
#include <wpi/SmallVector.h>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/optimization/solver/interior_point.hpp"
#include "sleipnir/optimization/solver/newton.hpp"
#include "sleipnir/optimization/solver/sqp.hpp"
#include "sleipnir/optimization/solver_config.hpp"
#include "sleipnir/optimization/solver_iteration_info.hpp"
#include "sleipnir/optimization/solver_status.hpp"
#include "sleipnir/util/symbol_exports.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/print.hpp"
#endif

namespace sleipnir {

/**
 * This class allows the user to pose a constrained nonlinear optimization
 * problem in natural mathematical notation and solve it.
 *
 * This class supports problems of the form:
@verbatim
      minₓ f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
@endverbatim
 *
 * where f(x) is the scalar cost function, x is the vector of decision variables
 * (variables the solver can tweak to minimize the cost function), cᵢ(x) are the
 * inequality constraints, and cₑ(x) are the equality constraints. Constraints
 * are equations or inequalities of the decision variables that constrain what
 * values the solver is allowed to use when searching for an optimal solution.
 *
 * The nice thing about this class is users don't have to put their system in
 * the form shown above manually; they can write it in natural mathematical form
 * and it'll be converted for them.
 */
class SLEIPNIR_DLLEXPORT OptimizationProblem {
 public:
  /**
   * Construct the optimization problem.
   */
  OptimizationProblem() noexcept = default;

  /**
   * Create a decision variable in the optimization problem.
   *
   * @return A decision variable in the optimization problem.
   */
  [[nodiscard]]
  Variable decision_variable() {
    m_decision_variables.emplace_back();
    return m_decision_variables.back();
  }

  /**
   * Create a matrix of decision variables in the optimization problem.
   *
   * @param rows Number of matrix rows.
   * @param cols Number of matrix columns.
   * @return A matrix of decision variables in the optimization problem.
   */
  [[nodiscard]]
  VariableMatrix decision_variable(int rows, int cols = 1) {
    m_decision_variables.reserve(m_decision_variables.size() + rows * cols);

    VariableMatrix vars{rows, cols};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        m_decision_variables.emplace_back();
        vars(row, col) = m_decision_variables.back();
      }
    }

    return vars;
  }

  /**
   * Create a symmetric matrix of decision variables in the optimization
   * problem.
   *
   * Variable instances are reused across the diagonal, which helps reduce
   * problem dimensionality.
   *
   * @param rows Number of matrix rows.
   * @return A symmetric matrix of decision varaibles in the optimization
   *   problem.
   */
  [[nodiscard]]
  VariableMatrix symmetric_decision_variable(int rows) {
    // We only need to store the lower triangle of an n x n symmetric matrix;
    // the other elements are duplicates. The lower triangle has (n² + n)/2
    // elements.
    //
    //   n
    //   Σ k = (n² + n)/2
    //  k=1
    m_decision_variables.reserve(m_decision_variables.size() +
                                 (rows * rows + rows) / 2);

    VariableMatrix vars{rows, rows};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col <= row; ++col) {
        m_decision_variables.emplace_back();
        vars(row, col) = m_decision_variables.back();
        vars(col, row) = m_decision_variables.back();
      }
    }

    return vars;
  }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void minimize(const Variable& cost) { m_f = cost; }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void minimize(Variable&& cost) { m_f = std::move(cost); }

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize.
   */
  void maximize(const Variable& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -objective;
  }

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize.
   */
  void maximize(Variable&& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -std::move(objective);
  }

  /**
   * Tells the solver to solve the problem while satisfying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void subject_to(const EqualityConstraints& constraint) {
    m_equality_constraints.reserve(m_equality_constraints.size() +
                                   constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_equality_constraints));
  }

  /**
   * Tells the solver to solve the problem while satisfying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void subject_to(EqualityConstraints&& constraint) {
    m_equality_constraints.reserve(m_equality_constraints.size() +
                                   constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_equality_constraints));
  }

  /**
   * Tells the solver to solve the problem while satisfying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void subject_to(const InequalityConstraints& constraint) {
    m_inequality_constraints.reserve(m_inequality_constraints.size() +
                                     constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_inequality_constraints));
  }

  /**
   * Tells the solver to solve the problem while satisfying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void subject_to(InequalityConstraints&& constraint) {
    m_inequality_constraints.reserve(m_inequality_constraints.size() +
                                     constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_inequality_constraints));
  }

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   *
   * @param config Configuration options for the solver.
   * @return The solver status.
   */
  SolverStatus solve(const SolverConfig& config = SolverConfig{}) {
    SolverStatus status;

    // Create the initial value column vector
    Eigen::VectorXd x{m_decision_variables.size()};
    for (size_t i = 0; i < m_decision_variables.size(); ++i) {
      x[i] = m_decision_variables[i].value();
    }

    if (m_f) {
      status.cost_function_type = m_f.value().type();
    } else {
      // If there's no cost function, make it zero and continue
      m_f = 0.0;
    }

    // Get the highest order constraint expression types
    if (!m_equality_constraints.empty()) {
      status.equality_constraint_type =
          std::ranges::max(m_equality_constraints, {}, &Variable::type).type();
    }
    if (!m_inequality_constraints.empty()) {
      status.inequality_constraint_type =
          std::ranges::max(m_inequality_constraints, {}, &Variable::type)
              .type();
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      // Print possible solver exit conditions
      sleipnir::println("User-configured solver exit conditions:");
      sleipnir::println("  ↳ error below {}", config.tolerance);
      sleipnir::println("  ↳ error below {} for {} iterations",
                        config.acceptable_tolerance,
                        config.max_acceptable_iterations);
      if (!m_callbacks.empty()) {
        sleipnir::println("  ↳ user callback requested stop");
      }
      if (std::isfinite(config.max_iterations)) {
        sleipnir::println("  ↳ executed {} iterations", config.max_iterations);
      }
      if (std::isfinite(config.timeout.count())) {
        sleipnir::println("  ↳ {} s elapsed", config.timeout.count());
      }

      sleipnir::println("\n{} decision variables", m_decision_variables.size());

      auto print_constraint_types =
          [](const wpi::SmallVector<Variable>& constraints) {
            std::array<size_t, 5> type_counts{};
            for (const auto& constraint : constraints) {
              ++type_counts[static_cast<uint8_t>(constraint.type())];
            }
            for (const auto& [count, name] : std::views::zip(
                     type_counts, std::array{"empty", "constant", "linear",
                                             "quadratic", "nonlinear"})) {
              if (count > 0) {
                sleipnir::println("  ↳ {} {}", count, name);
              }
            }
          };

      // Print constraint types
      sleipnir::println("{} equality constraints",
                        m_equality_constraints.size());
      print_constraint_types(m_equality_constraints);
      sleipnir::println("{} inequality constraints",
                        m_inequality_constraints.size());
      print_constraint_types(m_inequality_constraints);
    }

    auto print_chosen_solver = [](std::string_view solver_name,
                                  const SolverStatus& status) {
      sleipnir::println("\nUsing {} solver due to:", solver_name);
      sleipnir::println("  ↳ {} cost function",
                        ToMessage(status.cost_function_type));
      sleipnir::println("  ↳ {} equality constraints",
                        ToMessage(status.equality_constraint_type));
      sleipnir::println("  ↳ {} inequality constraints",
                        ToMessage(status.inequality_constraint_type));
      sleipnir::println("");
    };
#endif

    // If the problem is empty or constant, there's nothing to do
    if (status.cost_function_type <= ExpressionType::CONSTANT &&
        status.equality_constraint_type <= ExpressionType::CONSTANT &&
        status.inequality_constraint_type <= ExpressionType::CONSTANT) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        print_chosen_solver("no-op", status);
      }
#endif
      return status;
    }

    // Solve the optimization problem
    if (m_equality_constraints.empty() && m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        print_chosen_solver("Newton", status);
      }
#endif
      newton(m_decision_variables, m_f.value(), m_callbacks, config, x,
             &status);
    } else if (m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        print_chosen_solver("SQP", status);
      }
#endif
      sqp(m_decision_variables, m_equality_constraints, m_f.value(),
          m_callbacks, config, x, &status);
    } else {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        print_chosen_solver("IPM", status);
      }
#endif
      interior_point(m_decision_variables, m_equality_constraints,
                     m_inequality_constraints, m_f.value(), m_callbacks, config,
                     x, &status);
    }

    // Assign the solution to the original Variable instances
    VariableMatrix{m_decision_variables}.set_value(x);

    return status;
  }

  /**
   * Adds a callback to be called at each solver iteration.
   *
   * The callback for this overload should return void.
   *
   * @param callback The callback.
   */
  template <typename F>
    requires requires(F callback, const SolverIterationInfo& info) {
      { callback(info) } -> std::same_as<void>;
    }
  void add_callback(F&& callback) {
    m_callbacks.emplace_back([=, callback = std::forward<F>(callback)](
                                 const SolverIterationInfo& info) {
      callback(info);
      return false;
    });
  }

  /**
   * Adds a callback to be called at each solver iteration.
   *
   * The callback for this overload should return bool.
   *
   * @param callback The callback. Returning true from the callback causes the
   *   solver to exit early with the solution it has so far.
   */
  template <typename F>
    requires requires(F callback, const SolverIterationInfo& info) {
      { callback(info) } -> std::same_as<bool>;
    }
  void add_callback(F&& callback) {
    m_callbacks.emplace_back(std::forward<F>(callback));
  }

  /**
   * Clears the registered callbacks.
   */
  void clear_callbacks() { m_callbacks.clear(); }

 private:
  // The list of decision variables, which are the root of the problem's
  // expression tree
  wpi::SmallVector<Variable> m_decision_variables;

  // The cost function: f(x)
  std::optional<Variable> m_f;

  // The list of equality constraints: cₑ(x) = 0
  wpi::SmallVector<Variable> m_equality_constraints;

  // The list of inequality constraints: cᵢ(x) ≥ 0
  wpi::SmallVector<Variable> m_inequality_constraints;

  // The user callback
  wpi::SmallVector<std::function<bool(const SolverIterationInfo& info)>>
      m_callbacks;
};

}  // namespace sleipnir
