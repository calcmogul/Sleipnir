// Copyright (c) Sleipnir contributors

#include <concepts>
#include <utility>

#include <emscripten/bind.h>
#include <sleipnir/autodiff/variable.hpp>

#include "for_each_type.hpp"

namespace em = emscripten;

#if defined(__APPLE__) && defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#elif defined(__clang__) && __clang_major__ >= 7
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

namespace slp {

/// Bind unary math function.
template <typename F, typename... Args>
void def_unary_math(em::module_& autodiff, const char* name, F&& f,
                    Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename X> {
    if constexpr (std::same_as<X, const V&>) {
      autodiff.function(name, f, std::forward<Args>(args)...);
    } else {
      autodiff.function(
          name, [=](X&& x) { return f(x); }, std::forward<Args>(args)...);
    }
  });
}

/// Bind binary math function.
template <typename F, typename... Args>
void def_binary_math(em::module_& autodiff, const char* name, F&& f,
                     Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename L> {
    for_each_type<double, const V&>([&]<typename R> {
      if constexpr (std::same_as<L, const V&> && std::same_as<R, const V&>) {
        autodiff.function(name, f, std::forward<Args>(args)...);
      } else {
        autodiff.function(
            name, [=](L&& l, R&& r) { return f(l, r); },
            std::forward<Args>(args)...);
      }
    });
  });
}

/// Bind ternary math function.
template <typename F, typename... Args>
void def_ternary_math(em::module_& autodiff, const char* name, F&& f,
                      Args&&... args) {
  using V = Variable<double>;
  for_each_type<double, const V&>([&]<typename L> {
    for_each_type<double, const V&>([&]<typename M> {
      for_each_type<double, const V&>([&]<typename R> {
        if constexpr (std::same_as<L, const V&> && std::same_as<M, const V&> &&
                      std::same_as<R, const V&>) {
          autodiff.function(name, f, std::forward<Args>(args)...);
        } else {
          autodiff.function(
              name, [=](L&& l, M&& m, R&& r) { return f(l, m, r); },
              std::forward<Args>(args)...);
        }
      });
    });
  });
}

void bind_variable(em::module_& autodiff, em::class_<Variable<double>>& cls) {
  using namespace em::literals;

  cls.function(em::init<>());
  cls.function(em::init<double>(), "value"_a);
  cls.function("set_value",
               em::overload_cast<double>(&Variable<double>::set_value),
               "value"_a);
  cls.function("value", &Variable<double>::value);
  cls.function("type", &Variable<double>::type);

  for_each_type<em::detail::self_t, double, int>([&]<typename T> {
    cls.function(em::self * T(), "rhs"_a);
    cls.function(em::self *= T(), "rhs"_a);
    cls.function(em::self / T(), "rhs"_a);
    cls.function(em::self /= T(), "rhs"_a);
    cls.function(em::self + T(), "rhs"_a);
    cls.function(em::self += T(), "rhs"_a);
    cls.function(em::self - T(), "rhs"_a);
    cls.function(em::self -= T(), "rhs"_a);
    if constexpr (!std::same_as<T, em::detail::self_t>) {
      cls.function(T() * em::self, "lhs"_a);
      cls.function(T() / em::self, "lhs"_a);
      cls.function(T() + em::self, "lhs"_a);
      cls.function(T() - em::self, "lhs"_a);
    }
  });

  cls.function(
      "__pow__",
      [](const Variable<double>& self, int power) { return pow(self, power); },
      em::is_operator(), "power"_a);
  cls.function(-em::self);
  cls.function(+em::self);

  // Comparison operators
  for_each_type<em::detail::self_t, double, int>([&]<typename T> {
    cls.function(em::self == T(), "rhs"_a);
    cls.function(em::self < T(), "rhs"_a);
    cls.function(em::self <= T(), "rhs"_a);
    cls.function(em::self > T(), "rhs"_a);
    cls.function(em::self >= T(), "rhs"_a);
    if constexpr (!std::same_as<T, em::detail::self_t>) {
      cls.function(T() == em::self, "lhs"_a);
      cls.function(T() < em::self, "lhs"_a);
      cls.function(T() <= em::self, "lhs"_a);
      cls.function(T() > em::self, "lhs"_a);
      cls.function(T() >= em::self, "lhs"_a);
    }
  });

  // Math functions
  using V = Variable<double>;
  def_unary_math(autodiff, "abs", &abs<double>, "x"_a);
  def_unary_math(autodiff, "acos", &acos<double>, "x"_a);
  def_unary_math(autodiff, "asin", &asin<double>, "x"_a);
  def_unary_math(autodiff, "atan", &atan<double>, "x"_a);
  def_binary_math(autodiff, "atan2", &atan2<double>, "y"_a, "x"_a);
  def_unary_math(autodiff, "cbrt", &cbrt<double>, "x"_a);
  def_unary_math(autodiff, "cos", &cos<double>, "x"_a);
  def_unary_math(autodiff, "cosh", &cosh<double>, "x"_a);
  def_unary_math(autodiff, "erf", &erf<double>, "x"_a);
  def_unary_math(autodiff, "exp", &exp<double>, "x"_a);
  def_binary_math(autodiff, "hypot",
                  em::overload_cast<const V&, const V&>(&hypot<double>), "x"_a,
                  "y"_a);
  def_ternary_math(
      autodiff, "hypot",
      em::overload_cast<const V&, const V&, const V&>(&hypot<double>), "x"_a,
      "y"_a, "z"_a);
  def_unary_math(autodiff, "log", &log<double>, "x"_a);
  def_unary_math(autodiff, "log10", &log10<double>, "x"_a);
  def_binary_math(autodiff, "pow", &pow<double>, "base"_a, "power"_a);
  def_unary_math(autodiff, "sign", &sign<double>, "x"_a);
  def_unary_math(autodiff, "sin", &sin<double>, "x"_a);
  def_unary_math(autodiff, "sinh", &sinh<double>, "x"_a);
  def_unary_math(autodiff, "sqrt", &sqrt<double>, "x"_a);
  def_unary_math(autodiff, "tan", &tan<double>, "x"_a);
  def_unary_math(autodiff, "tanh", &tanh<double>, "x"_a);
}

}  // namespace slp
