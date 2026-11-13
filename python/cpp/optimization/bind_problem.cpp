// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>
#include <sleipnir/optimization/solver/options.hpp>

#include "arity.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "text_wrap.hpp"

namespace nb = nanobind;

namespace slp {

namespace {

/// Name and docs of an Options member.
struct OptionDoc {
  /// Member name, which is also the solve() binding's keyword argument name.
  const char* name;

  /// Member docs.
  const char* doc;
};

/// Name and docs of each Options member in declaration order.
///
/// See include/sleipnir/optimization/solver/options.hpp.
constexpr OptionDoc option_docs[] = {
    {"tolerance", DOC(slp, Options, tolerance)},
    {"max_iterations", DOC(slp, Options, max_iterations)},
    {"timeout", DOC(slp, Options, timeout)},
    {"feasible_ipm", DOC(slp, Options, feasible_ipm)},
    {"diagnostics", DOC(slp, Options, diagnostics)}};
static_assert(member_count<Options<double>>() == std::size(option_docs),
              "Every Options member must be documented here and be a "
              "keyword argument of the solve() binding");

/// Returns the Python docstring for Problem.solve().
///
/// C++'s Problem<double>::solve() takes an Options object instead of keyword
/// arguments, so its generated docstring documents a single `options` argument.
/// This replaces that entry with the docs for each Options member.
consteval const char* solve_doc() {
  static constexpr auto doc = [] {
    constexpr auto make_doc = [] {
      std::string args;
      for (const auto& [name, option_doc] : option_docs) {
        // The Options member docs were wrapped without the "    name: " prefix,
        // so they need to be rewrapped. pybind11_mkdoc's default docstring
        // width is 70.
        args += wrap(std::string{"    "} + std::string{name} + ": ", option_doc,
                     70);
        args += '\n';
      }

      // Find the start of the `options` entry
      std::string result = DOC(slp, Problem, solve);
      constexpr std::string_view options_entry = "    options: ";
      auto begin = result.find(options_entry);
      if (begin == std::string::npos) {
        throw std::logic_error{
            "Problem::solve() docs have no options argument"};
      }

      // Find the end of the `options` entry's last line. Each following line
      // indented past the argument names is a continuation of the entry, as are
      // any blank lines before it.
      std::string_view rest = std::string_view{result}.substr(begin);
      auto lines = rest | std::views::split('\n');
      auto entry_end = std::ranges::end(lines.front());
      for (auto line :
           lines | std::views::drop(1) | std::views::take_while([](auto line) {
             return line.empty() || std::string_view{line}.starts_with("     ");
           })) {
        if (!line.empty()) {
          entry_end = line.end();
        }
      }

      // Include the entry's trailing newline if it has one
      size_t end = entry_end - rest.begin();
      if (entry_end != rest.end()) {
        ++end;
      }

      // Replace the `options` entry with the Options member entries
      result.replace(begin, end, args);

      return result;
    };

    std::array<char, make_doc().size() + 1> doc;
    std::ranges::copy(make_doc(), doc.begin());
    doc.back() = '\0';
    return doc;
  }();

  return doc.data();
}

}  // namespace

void bind_problem(nb::class_<Problem<double>>& cls) {
  using namespace nb::literals;

  cls.def(
      "__init__",
      [](Problem<double>* self) {
        new (self) Problem<double>();

        // Make Python signals (e.g., SIGINT from Ctrl-C) abort the solve
        self->add_persistent_callback([](const IterationInfo<double>&) -> bool {
          if (PyErr_CheckSignals() != 0) {
            throw nb::python_error();
          }
          return false;
        });
      },
      DOC(slp, Problem, Problem));
  cls.def("decision_variable",
          nb::overload_cast<>(&Problem<double>::decision_variable),
          DOC(slp, Problem, decision_variable));
  cls.def("decision_variable",
          nb::overload_cast<int, int>(&Problem<double>::decision_variable),
          "rows"_a, "cols"_a = 1, DOC(slp, Problem, decision_variable, 2));
  cls.def(
      "decision_variable",
      [](Problem<double>& self, sphere_t tag, int rows) {
        return self.decision_variable(tag, rows);
      },
      "manifold"_a, "rows"_a, DOC(slp, Problem, decision_variable, 3));
  cls.def(
      "decision_variable",
      [](Problem<double>& self, so2_t tag) {
        return self.decision_variable(tag);
      },
      "manifold"_a, DOC(slp, Problem, decision_variable, 4));
  cls.def(
      "decision_variable",
      [](Problem<double>& self, so3_t tag) {
        return self.decision_variable(tag);
      },
      "manifold"_a, DOC(slp, Problem, decision_variable, 5));
  cls.def(
      "decision_variable",
      [](Problem<double>& self, se2_t tag) {
        return self.decision_variable(tag);
      },
      "manifold"_a, DOC(slp, Problem, decision_variable, 6));
  cls.def(
      "decision_variable",
      [](Problem<double>& self, se3_t tag) {
        return self.decision_variable(tag);
      },
      "manifold"_a, DOC(slp, Problem, decision_variable, 7));
  cls.def("symmetric_decision_variable",
          &Problem<double>::symmetric_decision_variable, "rows"_a,
          DOC(slp, Problem, symmetric_decision_variable));
  for_each_type<double, const Variable<double>&, const VariableMatrix<double>&>(
      [&]<typename T> {
        cls.def(
            "minimize",
            [](Problem<double>& self, T cost) { self.minimize(cost); },
            "cost"_a, DOC(slp, Problem, minimize));
        cls.def(
            "maximize",
            [](Problem<double>& self, T objective) {
              self.maximize(objective);
            },
            "objective"_a, DOC(slp, Problem, maximize));
      });
  cls.def("subject_to",
          nb::overload_cast<const EqualityConstraints<double>&>(
              &Problem<double>::subject_to),
          "constraint"_a, DOC(slp, Problem, subject_to));
  cls.def("subject_to",
          nb::overload_cast<const InequalityConstraints<double>&>(
              &Problem<double>::subject_to),
          "constraint"_a, DOC(slp, Problem, subject_to, 3));
  cls.def("cost_function_type", &Problem<double>::cost_function_type,
          DOC(slp, Problem, cost_function_type));
  cls.def("equality_constraint_type",
          &Problem<double>::equality_constraint_type,
          DOC(slp, Problem, equality_constraint_type));
  cls.def("inequality_constraint_type",
          &Problem<double>::inequality_constraint_type,
          DOC(slp, Problem, inequality_constraint_type));

  // Maps Python's Problem.solve(**kwargs) to
  // C++'s Problem::solve(Options, bool)
  constexpr auto solve = [](Problem<double>& self, double tolerance,
                            int max_iterations, double timeout,
                            bool feasible_ipm, bool diagnostics, bool spy) {
    return self.solve({.tolerance = tolerance,
                       .max_iterations = max_iterations,
                       .timeout = std::chrono::duration<double>{timeout},
                       .feasible_ipm = feasible_ipm,
                       .diagnostics = diagnostics},
                      spy);
  };
  static_assert(
      lambda_param_count<decltype(solve)>() ==
          1 + member_count<Options<double>>() + 1,
      "Every Options member must be a keyword argument of the solve() binding");

  cls.def("solve", solve, nb::kw_only(),
          nb::arg(option_docs[0].name) = Options<double>{}.tolerance,
          nb::arg(option_docs[1].name) = Options<double>{}.max_iterations,
          // nanobind renders defaults in signatures and stubs via repr(), and
          // repr(float("inf")) is "inf", which isn't valid Python. Override the
          // displayed default with sig().
          nb::arg(option_docs[2].name).sig("float('inf')") =
              Options<double>{}.timeout.count(),
          nb::arg(option_docs[3].name) = Options<double>{}.feasible_ipm,
          nb::arg(option_docs[4].name) = Options<double>{}.diagnostics,
          "spy"_a = false, solve_doc());

  cls.def(
      "add_callback",
      [](Problem<double>& self,
         std::function<bool(const IterationInfo<double>& info)> callback) {
        self.add_callback(std::move(callback));
      },
      "callback"_a, DOC(slp, Problem, add_callback, 2));
  cls.def("clear_callbacks", &Problem<double>::clear_callbacks,
          DOC(slp, Problem, clear_callbacks));
}

}  // namespace slp
