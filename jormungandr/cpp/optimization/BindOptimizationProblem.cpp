// Copyright (c) Sleipnir contributors

#include <format>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/optimization/SolverConfig.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindOptimizationProblem(nb::class_<OptimizationProblem>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(),
          DOC(sleipnir, OptimizationProblem, OptimizationProblem));
  cls.def("decision_variable",
          nb::overload_cast<>(&OptimizationProblem::DecisionVariable),
          DOC(sleipnir, OptimizationProblem, DecisionVariable));
  cls.def("decision_variable",
          nb::overload_cast<int, int>(&OptimizationProblem::DecisionVariable),
          "rows"_a, "cols"_a = 1,
          DOC(sleipnir, OptimizationProblem, DecisionVariable, 2));
  cls.def("symmetric_decision_variable",
          &OptimizationProblem::SymmetricDecisionVariable, "rows"_a,
          DOC(sleipnir, OptimizationProblem, SymmetricDecisionVariable));
  cls.def("minimize",
          nb::overload_cast<const Variable&>(&OptimizationProblem::Minimize),
          "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def(
      "minimize",
      [](OptimizationProblem& self, const VariableMatrix& cost) {
        self.Minimize(cost);
      },
      "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def(
      "minimize",
      [](OptimizationProblem& self, double cost) { self.Minimize(cost); },
      "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def("maximize",
          nb::overload_cast<const Variable&>(&OptimizationProblem::Maximize),
          "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def(
      "maximize",
      [](OptimizationProblem& self, const VariableMatrix& objective) {
        self.Maximize(objective);
      },
      "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def(
      "maximize",
      [](OptimizationProblem& self, double objective) {
        self.Maximize(objective);
      },
      "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def("subject_to",
          nb::overload_cast<const EqualityConstraints&>(
              &OptimizationProblem::SubjectTo),
          "constraint"_a, DOC(sleipnir, OptimizationProblem, SubjectTo));
  cls.def("subject_to",
          nb::overload_cast<const InequalityConstraints&>(
              &OptimizationProblem::SubjectTo),
          "constraint"_a, DOC(sleipnir, OptimizationProblem, SubjectTo, 3));
  cls.def(
      "solve",
      [](OptimizationProblem& self, const nb::kwargs& kwargs) {
        SolverConfig config;

        for (auto [key, value] : kwargs) {
          // XXX: The keyword arguments are manually copied from the struct
          // members in include/sleipnir/optimization/SolverConfig.hpp.
          //
          // C++'s OptimizationProblem::Solve() takes a SolverConfig object
          // instead of keyword arguments, so there's no compile-time checking
          // that the arguments match.
          auto key_str = nb::cast<std::string>(key);
          if (key_str == "tolerance") {
            config.tolerance = nb::cast<double>(value);
          } else if (key_str == "max_iterations") {
            config.maxIterations = nb::cast<int>(value);
          } else if (key_str == "acceptable_tolerance") {
            config.acceptableTolerance = nb::cast<double>(value);
          } else if (key_str == "max_acceptable_iterations") {
            config.maxAcceptableIterations = nb::cast<int>(value);
          } else if (key_str == "timeout") {
            config.timeout =
                std::chrono::duration<double>{nb::cast<double>(value)};
          } else if (key_str == "num_threads") {
            config.numThreads = nb::cast<int>(value);
          } else if (key_str == "feasible_ipm") {
            config.feasibleIPM = nb::cast<bool>(value);
          } else if (key_str == "diagnostics") {
            config.diagnostics = nb::cast<bool>(value);
          } else if (key_str == "spy") {
            config.spy = nb::cast<bool>(value);
          } else {
            throw nb::key_error(
                std::format("Invalid keyword argument: {}", key_str).c_str());
          }
        }

        return self.Solve(config);
      },
      // XXX: The keyword argument docs are manually copied from the struct
      // member docs in include/sleipnir/optimization/SolverConfig.hpp.
      //
      // C++'s OptimizationProblem::Solve() takes a SolverConfig object instead
      // of keyword arguments, so nanobind-mkdoc generates the wrong docs.
      R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``tolerance``:
    The solver will stop once the error is below this tolerance.
    (default: 1e-8)

Parameter ``max_iterations``:
    The maximum number of solver iterations before returning a solution.
    (default: 5000)

Parameter ``acceptable_tolerance``:
    The solver will stop once the error is below this tolerance for
    `acceptable_iterations` iterations. This is useful in cases where the
    solver might not be able to achieve the desired level of accuracy due to
    floating-point round-off.
    (default: 1e-6)

Parameter ``max_acceptable_iterations``:
    The solver will stop once the error is below `acceptable_tolerance` for
    this many iterations.
    (default: 15)

Parameter ``timeout``:
    The maximum elapsed wall clock time before returning a solution.
    (default: infinity)

Parameter ``num_threads``:
    The number of threads to use for parallel backtracking line searches
    (default: CPU core count).

Parameter ``feasible_ipm``:
    Enables the feasible interior-point method. When the inequality
    constraints are all feasible, step sizes are reduced when necessary to
    prevent them becoming infeasible again. This is useful when parts of the
    problem are ill-conditioned in infeasible regions (e.g., square root of a
    negative value). This can slow or prevent progress toward a solution
    though, so only enable it if necessary.
    (default: False)

Parameter ``diagnostics``:
    Enables diagnostic prints.
    (default: False)

Parameter ``spy``:
    Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named H.spy,
    A_e.spy, and A_i.spy respectively during solve.

    Use tools/spy.py to plot them.
    (default: False))doc");
  cls.def(
      "callback",
      [](OptimizationProblem& self,
         std::function<bool(const SolverIterationInfo& info)> callback) {
        self.Callback(std::move(callback));
      },
      "callback"_a, DOC(sleipnir, OptimizationProblem, Callback, 2));
}

}  // namespace sleipnir
