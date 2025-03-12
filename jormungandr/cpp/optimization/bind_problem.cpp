// Copyright (c) Sleipnir contributors

#include <format>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/options.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_problem(nb::class_<Problem>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(slp, Problem, Problem));
  cls.def("decision_variable", nb::overload_cast<>(&Problem::decision_variable),
          DOC(slp, Problem, decision_variable));
  cls.def("decision_variable",
          nb::overload_cast<int, int>(&Problem::decision_variable), "rows"_a,
          "cols"_a = 1, DOC(slp, Problem, decision_variable, 2));
  cls.def("symmetric_decision_variable", &Problem::symmetric_decision_variable,
          "rows"_a, DOC(slp, Problem, symmetric_decision_variable));
  cls.def("minimize", nb::overload_cast<const Variable&>(&Problem::minimize),
          "cost"_a, DOC(slp, Problem, minimize));
  cls.def(
      "minimize",
      [](Problem& self, const VariableMatrix& cost) { self.minimize(cost); },
      "cost"_a, DOC(slp, Problem, minimize));
  cls.def(
      "minimize", [](Problem& self, double cost) { self.minimize(cost); },
      "cost"_a, DOC(slp, Problem, minimize));
  cls.def("maximize", nb::overload_cast<const Variable&>(&Problem::maximize),
          "objective"_a, DOC(slp, Problem, maximize));
  cls.def(
      "maximize",
      [](Problem& self, const VariableMatrix& objective) {
        self.maximize(objective);
      },
      "objective"_a, DOC(slp, Problem, maximize));
  cls.def(
      "maximize",
      [](Problem& self, double objective) { self.maximize(objective); },
      "objective"_a, DOC(slp, Problem, maximize));
  cls.def("subject_to",
          nb::overload_cast<const EqualityConstraints&>(&Problem::subject_to),
          "constraint"_a, DOC(slp, Problem, subject_to));
  cls.def("subject_to",
          nb::overload_cast<const InequalityConstraints&>(&Problem::subject_to),
          "constraint"_a, DOC(slp, Problem, subject_to, 3));
  cls.def("cost_function_type", &Problem::cost_function_type,
          DOC(slp, Problem, cost_function_type));
  cls.def("equality_constraint_type", &Problem::equality_constraint_type,
          DOC(slp, Problem, equality_constraint_type));
  cls.def("inequality_constraint_type", &Problem::inequality_constraint_type,
          DOC(slp, Problem, inequality_constraint_type));
  cls.def(
      "solve",
      [](Problem& self, const nb::kwargs& kwargs) {
        // Make Python signals (e.g., SIGINT from Ctrl-C) abort the solve
        self.add_callback([](const IterationInfo&) -> bool {
          if (PyErr_CheckSignals() != 0) {
            throw nb::python_error();
          }
          return false;
        });

        Options options;
        bool spy = false;

        for (auto [key, value] : kwargs) {
          // XXX: The keyword arguments are manually copied from the struct
          // members in include/sleipnir/optimization/solver/options.hpp.
          //
          // C++'s Problem::Solve() takes an Options object instead of keyword
          // arguments, so there's no compile-time checking that the arguments
          // match.
          auto key_str = nb::cast<std::string>(key);
          if (key_str == "tolerance") {
            options.tolerance = nb::cast<double>(value);
          } else if (key_str == "max_iterations") {
            options.max_iterations = nb::cast<int>(value);
          } else if (key_str == "timeout") {
            options.timeout =
                std::chrono::duration<double>{nb::cast<double>(value)};
          } else if (key_str == "feasible_ipm") {
            options.feasible_ipm = nb::cast<bool>(value);
          } else if (key_str == "diagnostics") {
            options.diagnostics = nb::cast<bool>(value);
          } else if (key_str == "spy") {
            spy = nb::cast<bool>(value);
          } else {
            throw nb::key_error(
                std::format("Invalid keyword argument: {}", key_str).c_str());
          }
        }

        return self.solve(options, spy);
      },
      // XXX: The keyword argument docs are manually copied from the struct
      // member docs in include/sleipnir/optimization/solver/options.hpp.
      //
      // C++'s Problem::Solve() takes an Options object instead of keyword
      // arguments, so pybind11_mkdoc generates the wrong docs.
      R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``tolerance``:
    The solver will stop once the error is below this tolerance.
    (default: 1e-8)

Parameter ``max_iterations``:
    The maximum number of solver iterations before returning a solution.
    (default: 5000)

Parameter ``timeout``:
    The maximum elapsed wall clock time before returning a solution.
    (default: infinity)

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

    <table>
      <tr>
        <th>Heading</th>
        <th>Description</th>
      </tr>
      <tr>
        <td>iter</td>
        <td>Iteration number</td>
      </tr>
      <tr>
        <td>type</td>
        <td>Iteration type (normal, accepted second-order correction, rejected second-order correction)</td>
      </tr>
      <tr>
        <td>time (ms)</td>
        <td>Duration of iteration in milliseconds</td>
      </tr>
      <tr>
        <td>error</td>
        <td>Error estimate</td>
      </tr>
      <tr>
        <td>cost</td>
        <td>Cost function value at current iterate</td>
      </tr>
      <tr>
        <td>infeas.</td>
        <td>Constraint infeasibility at current iterate</td>
      </tr>
      <tr>
        <td>μ</td>
        <td>Barrier parameter</td>
      </tr>
      <tr>
        <td>reg</td>
        <td>Iteration matrix regularization</td>
      </tr>
      <tr>
        <td>primal α</td>
        <td>Primal step size</td>
      </tr>
      <tr>
        <td>dual α</td>
        <td>Dual step size</td>
      </tr>
      <tr>
        <td>↩</td>
        <td>Number of line search backtracks</td>
      </tr>
    </table>
    (default: False)

Parameter ``spy``:
    Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named H.spy,
    A_e.spy, and A_i.spy respectively during solve. Use tools/spy.py to plot them.
    (default: False))doc");
  cls.def(
      "add_callback",
      [](Problem& self,
         std::function<bool(const IterationInfo& info)> callback) {
        self.add_callback(std::move(callback));
      },
      "callback"_a, DOC(slp, Problem, add_callback, 2));
  cls.def("clear_callbacks", &Problem::clear_callbacks,
          DOC(slp, Problem, clear_callbacks));
}

}  // namespace slp
