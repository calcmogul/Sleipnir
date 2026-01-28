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
  cls.constructor<>();
  cls.function("decision_variable",
               em::overload_cast<>(&Problem<double>::decision_variable));
  cls.function("decision_variable", em::overload_cast<int, int>(
                                        &Problem<double>::decision_variable));
  cls.function("symmetric_decision_variable",
               &Problem<double>::symmetric_decision_variable);
  for_each_type<double, const Variable<double>&, const VariableMatrix<double>&>(
      [&]<typename T> {
        cls.function("minimize", [](Problem<double>& self, T cost) {
          self.minimize(cost);
        });
        cls.function("maximize", [](Problem<double>& self, T objective) {
          self.maximize(objective);
        });
      });
  cls.function("subject_to",
               em::overload_cast<const EqualityConstraints<double>&>(
                   &Problem<double>::subject_to));
  cls.function("subject_to",
               em::overload_cast<const InequalityConstraints<double>&>(
                   &Problem<double>::subject_to));
  cls.function("cost_function_type", &Problem<double>::cost_function_type);
  cls.function("equality_constraint_type",
               &Problem<double>::equality_constraint_type);
  cls.function("inequality_constraint_type",
               &Problem<double>::inequality_constraint_type);
  cls.function("solve", [](Problem<double>& self, const em::kwargs& kwargs) {
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
  });
  cls.function(
      "add_callback",
      [](Problem<double>& self,
         std::function<bool(const IterationInfo<double>& info)> callback) {
        self.add_callback(std::move(callback));
      });
  cls.function("clear_callbacks", &Problem<double>::clear_callbacks);
}

}  // namespace slp
