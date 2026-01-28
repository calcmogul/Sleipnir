// Copyright (c) Sleipnir contributors

#include <format>
#include <string>
#include <utility>

#include <emscripten/bind.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/options.hpp>

#include "for_each_type.hpp"

namespace em = emscripten;

namespace slp {

void bind_problem(em::class_<Problem<double>>& cls) {
  using namespace em::literals;

  cls.function("__init__", [](Problem<double>* self) {
    new (self) Problem<double>();

    // Make Python signals (e.g., SIGINT from Ctrl-C) abort the solve
    self->add_persistent_callback([](const IterationInfo<double>&) -> bool {
      if (PyErr_CheckSignals() != 0) {
        throw em::python_error();
      }
      return false;
    });
  });
  cls.function("decision_variable",
               em::overload_cast<>(&Problem<double>::decision_variable));
  cls.function("decision_variable",
               em::overload_cast<int, int>(&Problem<double>::decision_variable),
               "rows"_a, "cols"_a = 1);
  cls.function("symmetric_decision_variable",
               &Problem<double>::symmetric_decision_variable, "rows"_a);
  for_each_type<double, const Variable<double>&, const VariableMatrix<double>&>(
      [&]<typename T> {
        cls.function(
            "minimize",
            [](Problem<double>& self, T cost) { self.minimize(cost); },
            "cost"_a);
        cls.function(
            "maximize",
            [](Problem<double>& self, T objective) {
              self.maximize(objective);
            },
            "objective"_a);
      });
  cls.function("subject_to",
               em::overload_cast<const EqualityConstraints<double>&>(
                   &Problem<double>::subject_to),
               "constraint"_a);
  cls.function("subject_to",
               em::overload_cast<const InequalityConstraints<double>&>(
                   &Problem<double>::subject_to),
               "constraint"_a);
  cls.function("cost_function_type", &Problem<double>::cost_function_type);
  cls.function("equality_constraint_type",
               &Problem<double>::equality_constraint_type);
  cls.function("inequality_constraint_type",
               &Problem<double>::inequality_constraint_type);
  cls.function(
      "solve",
      [](Problem<double>& self, const em::kwargs& kwargs) {
        Options options;
        bool spy = false;

        for (auto [key, value] : kwargs) {
          // XXX: The keyword arguments are manually copied from the struct
          // members in include/sleipnir/optimization/solver/options.hpp.
          //
          // C++'s Problem<double>::solve() takes an Options object instead of
          // keyword arguments, so there's no compile-time checking that the
          // arguments match.
          auto key_str = em::cast<std::string>(key);
          if (key_str == "tolerance") {
            options.tolerance = em::cast<double>(value);
          } else if (key_str == "max_iterations") {
            options.max_iterations = em::cast<int>(value);
          } else if (key_str == "timeout") {
            options.timeout =
                std::chrono::duration<double>{em::cast<double>(value)};
          } else if (key_str == "feasible_ipm") {
            options.feasible_ipm = em::cast<bool>(value);
          } else if (key_str == "diagnostics") {
            options.diagnostics = em::cast<bool>(value);
          } else if (key_str == "spy") {
            spy = em::cast<bool>(value);
          } else {
            throw em::key_error(
                std::format("Invalid keyword argument: {}", key_str).c_str());
          }
        }

        return self.solve(options, spy);
      },
      // XXX: The keyword argument docs are manually copied from the struct
      // member docs in include/sleipnir/optimization/solver/options.hpp.
      //
      // C++'s Problem<double>::solve() takes an Options object instead of
      // keyword arguments, so pybind11_mkdoc generates the wrong docs.
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
        <td>complement.</td>
        <td>Complementary slackness at current iterate (sᵀz)</td>
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
  cls.function(
      "add_callback",
      [](Problem<double>& self,
         std::function<bool(const IterationInfo<double>& info)> callback) {
        self.add_callback(std::move(callback));
      },
      "callback"_a);
  cls.function("clear_callbacks", &Problem<double>::clear_callbacks);
}

}  // namespace slp
