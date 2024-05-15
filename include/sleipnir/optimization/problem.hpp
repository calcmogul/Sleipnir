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

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/newton.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/symbol_exports.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/print.hpp"
#endif

namespace slp {

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
class SLEIPNIR_DLLEXPORT Problem {
 public:
  /**
   * Construct the optimization problem.
   */
  Problem() noexcept = default;

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
        vars[row, col] = m_decision_variables.back();
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
        vars[row, col] = m_decision_variables.back();
        vars[col, row] = m_decision_variables.back();
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
   * Returns the cost function's type.
   *
   * @return The cost function's type.
   */
  ExpressionType cost_function_type() const {
    if (m_f) {
      return m_f.value().type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /**
   * Returns the type of the highest order equality constraint.
   *
   * @return The type of the highest order equality constraint.
   */
  ExpressionType equality_constraint_type() const {
    if (!m_equality_constraints.empty()) {
      return std::ranges::max(m_equality_constraints, {}, &Variable::type)
          .type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /**
   * Returns the type of the highest order inequality constraint.
   *
   * @return The type of the highest order inequality constraint.
   */
  ExpressionType inequality_constraint_type() const {
    if (!m_inequality_constraints.empty()) {
      return std::ranges::max(m_inequality_constraints, {}, &Variable::type)
          .type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   *
   * @param options Solver options.
   * @return The solver status.
   */
  ExitStatus solve(const Options& options = Options{}) {
    // Create the initial value column vector
    Eigen::VectorXd x{m_decision_variables.size()};
    for (size_t i = 0; i < m_decision_variables.size(); ++i) {
      x[i] = m_decision_variables[i].value();
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      // Print possible exit conditions
      slp::println("User-configured exit conditions:");
      slp::println("  ↳ error below {}", options.tolerance);
      slp::println("  ↳ error below {} for {} iterations",
                   options.acceptable_tolerance,
                   options.max_acceptable_iterations);
      if (!m_callbacks.empty()) {
        slp::println("  ↳ user callback requested stop");
      }
      if (std::isfinite(options.max_iterations)) {
        slp::println("  ↳ executed {} iterations", options.max_iterations);
      }
      if (std::isfinite(options.timeout.count())) {
        slp::println("  ↳ {} elapsed", options.timeout);
      }

      if (m_decision_variables.size() == 1) {
        slp::println("\n1 decision variable");
      } else {
        slp::println("\n{} decision variables", m_decision_variables.size());
      }

      auto print_constraint_types =
          [](const small_vector<Variable>& constraints) {
            std::array<size_t, 5> type_counts{};
            for (const auto& constraint : constraints) {
              ++type_counts[std::to_underlying(constraint.type())];
            }
            for (const auto& [count, name] : std::views::zip(
                     type_counts, std::array{"empty", "constant", "linear",
                                             "quadratic", "nonlinear"})) {
              if (count > 0) {
                slp::println("  ↳ {} {}", count, name);
              }
            }
          };

      // Print constraint types
      if (m_equality_constraints.size() == 1) {
        slp::println("1 equality constraint");
      } else {
        slp::println("{} equality constraints", m_equality_constraints.size());
      }
      print_constraint_types(m_equality_constraints);
      if (m_inequality_constraints.size() == 1) {
        slp::println("1 inequality constraint");
      } else {
        slp::println("{} inequality constraints",
                     m_inequality_constraints.size());
      }
      print_constraint_types(m_inequality_constraints);
    }

    auto print_chosen_solver = [](std::string_view solver_name,
                                  const ExpressionType& f_type,
                                  const ExpressionType& c_e_type,
                                  const ExpressionType& c_i_type) {
      constexpr std::array types{"no", "constant", "linear", "quadratic",
                                 "nonlinear"};

      slp::println("\nUsing {} solver due to:", solver_name);
      slp::println("  ↳ {} cost function", types[std::to_underlying(f_type)]);
      slp::println("  ↳ {} equality constraints",
                   types[std::to_underlying(c_e_type)]);
      slp::println("  ↳ {} inequality constraints",
                   types[std::to_underlying(c_i_type)]);
      slp::println("");
    };
#endif

    // Get the highest order constraint expression types
    auto f_type = cost_function_type();
    auto c_e_type = equality_constraint_type();
    auto c_i_type = inequality_constraint_type();

    // If the problem is empty or constant, there's nothing to do
    if (f_type <= ExpressionType::CONSTANT &&
        c_e_type <= ExpressionType::CONSTANT &&
        c_i_type <= ExpressionType::CONSTANT) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_chosen_solver("no-op", f_type, c_e_type, c_i_type);
      }
#endif
      return ExitStatus::SUCCESS;
    }

    // Solve the optimization problem
    ExitStatus status;
    if (m_equality_constraints.empty() && m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_chosen_solver("Newton", f_type, c_e_type, c_i_type);
      }
#endif
      if (m_f) {
        status =
            newton(m_decision_variables, m_f.value(), m_callbacks, options, x);
      } else {
        Variable zero = 0.0;
        status = newton(m_decision_variables, zero, m_callbacks, options, x);
      }
    } else if (m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_chosen_solver("SQP", f_type, c_e_type, c_i_type);
      }
#endif
      if (m_f) {
        status = sqp(m_decision_variables, m_equality_constraints, m_f.value(),
                     m_callbacks, options, x);
      } else {
        Variable zero = 0.0;
        status = sqp(m_decision_variables, m_equality_constraints, zero,
                     m_callbacks, options, x);
      }
    } else {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_chosen_solver("IPM", f_type, c_e_type, c_i_type);
      }
#endif
      if (m_f) {
        status = interior_point(m_decision_variables, m_equality_constraints,
                                m_inequality_constraints, m_f.value(),
                                m_callbacks, options, x);
      } else {
        Variable zero = 0.0;
        status = interior_point(m_decision_variables, m_equality_constraints,
                                m_inequality_constraints, zero, m_callbacks,
                                options, x);
      }
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      slp::println("\nExit: {}", ToMessage(status));
    }
#endif

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
    requires requires(F callback, const IterationInfo& info) {
      { callback(info) } -> std::same_as<void>;
    }
  void add_callback(F&& callback) {
    m_callbacks.emplace_back(
        [=, callback = std::forward<F>(callback)](const IterationInfo& info) {
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
    requires requires(F callback, const IterationInfo& info) {
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
  small_vector<Variable> m_decision_variables;

  // The cost function: f(x)
  std::optional<Variable> m_f;

  // The list of equality constraints: cₑ(x) = 0
  small_vector<Variable> m_equality_constraints;

  // The list of inequality constraints: cᵢ(x) ≥ 0
  small_vector<Variable> m_inequality_constraints;

  // The user callback
  small_vector<std::function<bool(const IterationInfo& info)>> m_callbacks;
};

}  // namespace slp
