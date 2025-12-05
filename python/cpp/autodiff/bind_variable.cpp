// Copyright (c) Sleipnir contributors

#include <concepts>
#include <utility>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <sleipnir/autodiff/variable.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

#if defined(__APPLE__) && defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#elif defined(__clang__) && __clang_major__ >= 7
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

namespace slp {

/// Bind unary math function.
template <typename F, typename... Args>
void def_unary_math(nb::module_& autodiff, const char* name, F&& f,
                    Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename X> {
    if constexpr (std::same_as<X, const V&>) {
      autodiff.def(name, f, std::forward<Args>(args)...);
    } else {
      autodiff.def(
          name, [=](X&& x) { return f(x); }, std::forward<Args>(args)...);
    }
  });
}

/// Bind binary math function.
template <typename F, typename... Args>
void def_binary_math(nb::module_& autodiff, const char* name, F&& f,
                     Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename L> {
    for_each_type<double, const V&>([&]<typename R> {
      if constexpr (std::same_as<L, const V&> && std::same_as<R, const V&>) {
        autodiff.def(name, f, std::forward<Args>(args)...);
      } else {
        autodiff.def(
            name, [=](L&& l, R&& r) { return f(l, r); },
            std::forward<Args>(args)...);
      }
    });
  });
}

/// Bind ternary math function.
template <typename F, typename... Args>
void def_ternary_math(nb::module_& autodiff, const char* name, F&& f,
                      Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename L> {
    for_each_type<double, const V&>([&]<typename M> {
      for_each_type<double, const V&>([&]<typename R> {
        if constexpr (std::same_as<L, const V&> && std::same_as<M, const V&> &&
                      std::same_as<R, const V&>) {
          autodiff.def(name, f, std::forward<Args>(args)...);
        } else {
          autodiff.def(
              name, [=](L&& l, M&& m, R&& r) { return f(l, m, r); },
              std::forward<Args>(args)...);
        }
      });
    });
  });
}

void bind_variable(nb::module_& autodiff, nb::class_<Variable<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(slp, Variable, Variable));
  cls.def(nb::init<double>(), "value"_a, DOC(slp, Variable, Variable, 2));
  cls.def("set_value", nb::overload_cast<double>(&Variable<double>::set_value),
          "value"_a, DOC(slp, Variable, set_value));
  cls.def("value", &Variable<double>::value, DOC(slp, Variable, value));
  cls.def("type", &Variable<double>::type, DOC(slp, Variable, type));

  for_each_type<nb::detail::self_t, double, int>([&]<typename T> {
    cls.def(nb::self * T(), "rhs"_a);
    cls.def(nb::self *= T(), "rhs"_a, DOC(slp, Variable, operator, imul));
    cls.def(nb::self / T(), "rhs"_a);
    cls.def(nb::self /= T(), "rhs"_a, DOC(slp, Variable, operator, idiv));
    cls.def(nb::self + T(), "rhs"_a);
    cls.def(nb::self += T(), "rhs"_a, DOC(slp, Variable, operator, iadd));
    cls.def(nb::self - T(), "rhs"_a);
    cls.def(nb::self -= T(), "rhs"_a, DOC(slp, Variable, operator, isub));
    if constexpr (!std::same_as<T, nb::detail::self_t>) {
      cls.def(T() * nb::self, "lhs"_a);
      cls.def(T() / nb::self, "lhs"_a);
      cls.def(T() + nb::self, "lhs"_a);
      cls.def(T() - nb::self, "lhs"_a);
    }
  });

  cls.def(
      "__pow__",
      [](const Variable<double>& self, int power) { return pow(self, power); },
      nb::is_operator(), "power"_a);
  cls.def(-nb::self);
  cls.def(+nb::self);

  // Comparison operators
  for_each_type<nb::detail::self_t, double, int>([&]<typename T> {
    cls.def(nb::self == T(), "rhs"_a, DOC(slp, operator, eq));
    cls.def(nb::self < T(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(nb::self <= T(), "rhs"_a, DOC(slp, operator, le));
    cls.def(nb::self > T(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(nb::self >= T(), "rhs"_a, DOC(slp, operator, ge));
    if constexpr (!std::same_as<T, nb::detail::self_t>) {
      cls.def(T() == nb::self, "lhs"_a, DOC(slp, operator, eq));
      cls.def(T() < nb::self, "lhs"_a, DOC(slp, operator, lt));
      cls.def(T() <= nb::self, "lhs"_a, DOC(slp, operator, le));
      cls.def(T() > nb::self, "lhs"_a, DOC(slp, operator, gt));
      cls.def(T() >= nb::self, "lhs"_a, DOC(slp, operator, ge));
    }
  });

  // Math functions
  using V = Variable<double>;
  def_unary_math(autodiff, "abs", &abs<double>, "x"_a, DOC(slp, abs));
  def_unary_math(autodiff, "acos", &acos<double>, "x"_a, DOC(slp, acos));
  def_unary_math(autodiff, "asin", &asin<double>, "x"_a, DOC(slp, asin));
  def_unary_math(autodiff, "atan", &atan<double>, "x"_a, DOC(slp, atan));
  def_binary_math(autodiff, "atan2", &atan2<double>, "y"_a, "x"_a,
                  DOC(slp, atan2));
  def_unary_math(autodiff, "cbrt", &cbrt<double>, "x"_a, DOC(slp, cbrt));
  def_unary_math(autodiff, "cos", &cos<double>, "x"_a, DOC(slp, cos));
  def_unary_math(autodiff, "cosh", &cosh<double>, "x"_a, DOC(slp, cosh));
  def_unary_math(autodiff, "erf", &erf<double>, "x"_a, DOC(slp, erf));
  def_unary_math(autodiff, "exp", &exp<double>, "x"_a, DOC(slp, exp));
  def_binary_math(autodiff, "hypot",
                  static_cast<V (*)(const V&, const V&)>(&hypot<double>), "x"_a,
                  "y"_a, DOC(slp, hypot));
  def_ternary_math(
      autodiff, "hypot",
      static_cast<V (*)(const V&, const V&, const V&)>(&hypot<double>), "x"_a,
      "y"_a, "z"_a, DOC(slp, hypot, 2));
  def_unary_math(autodiff, "log", &log<double>, "x"_a, DOC(slp, log));
  def_unary_math(autodiff, "log10", &log10<double>, "x"_a, DOC(slp, log10));
  def_binary_math(autodiff, "pow", &pow<double>, "base"_a, "power"_a,
                  DOC(slp, pow));
  def_unary_math(autodiff, "sign", &sign<double>, "x"_a, DOC(slp, sign));
  def_unary_math(autodiff, "sin", &sin<double>, "x"_a, DOC(slp, sin));
  def_unary_math(autodiff, "sinh", &sinh<double>, "x"_a, DOC(slp, sinh));
  def_unary_math(autodiff, "sqrt", &sqrt<double>, "x"_a, DOC(slp, sqrt));
  def_unary_math(autodiff, "tan", &tan<double>, "x"_a, DOC(slp, tan));
  def_unary_math(autodiff, "tanh", &tanh<double>, "x"_a, DOC(slp, tanh));
}

}  // namespace slp
