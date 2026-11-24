// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <string>
#include <utility>

#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>

#include "complex_scalar.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "try_cast.hpp"

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
  // double overload
  autodiff.def(
      name, [=](double x) { return f(x); }, std::forward<Args>(args)...);

  // Real overload
  autodiff.def(name, f, std::forward<Args>(args)...);

  // RealMatrix overload
  autodiff.def(
      name, [=](const RealMatrix<double>& x) { return x.cwise_transform(f); },
      std::forward<Args>(args)...);

  // RealSubmatrix overload
  autodiff.def(
      name,
      [=](const RealSubmatrix<RealMatrix<double>>& x) {
        return x.cwise_transform(f);
      },
      std::forward<Args>(args)...);
}

/// Bind binary math function.
template <typename F, typename... Args>
void def_binary_math(nb::module_& autodiff, const char* name, F&& f,
                     Args&&... args) {
  using V = Real<double>;
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
  using V = Real<double>;
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

void bind_real(nb::module_& autodiff, nb::class_<Real<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(slp, Real, Real));
  cls.def(nb::init_implicit<double>(), "value"_a, DOC(slp, Real, Real, 2));
  cls.def("set_value", nb::overload_cast<double>(&Real<double>::set_value),
          "value"_a, DOC(slp, Real, set_value));
  cls.def("value", &Real<double>::value, DOC(slp, Real, value));
  cls.def("type", &Real<double>::type, DOC(slp, Real, type));

  for_each_type<nb::detail::self_t, double, int>([&]<typename T> {
    cls.def(nb::self * T(), "rhs"_a);
    cls.def(nb::self *= T(), "rhs"_a, DOC(slp, Real, operator, imul));
    cls.def(nb::self / T(), "rhs"_a);
    cls.def(nb::self /= T(), "rhs"_a, DOC(slp, Real, operator, idiv));
    cls.def(nb::self + T(), "rhs"_a);
    cls.def(nb::self += T(), "rhs"_a, DOC(slp, Real, operator, iadd));
    cls.def(nb::self - T(), "rhs"_a);
    cls.def(nb::self -= T(), "rhs"_a, DOC(slp, Real, operator, isub));
    if constexpr (!std::same_as<T, nb::detail::self_t>) {
      cls.def(T() * nb::self, "lhs"_a);
      cls.def(T() / nb::self, "lhs"_a);
      cls.def(T() + nb::self, "lhs"_a);
      cls.def(T() - nb::self, "lhs"_a);
    }
  });

  // Scalar-matrix multiplication and division with NumPy arrays. These are also
  // handled by __array_ufunc__ below, but type checkers only see these
  // overloads.
  cls.def(
      "__mul__",
      [](const Real<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__imul__",
      [](const Real<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const Real<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__rtruediv__",
      [](const Real<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs) {
        return lhs / rhs;
      },
      nb::is_operator(), "lhs"_a);

  // Map NumPy ufunc reprs to the equivalent function names in Python's
  // operator module
  static constexpr std::array scalar_operators{
      std::pair{"<ufunc 'add'>", "add"},
      std::pair{"<ufunc 'subtract'>", "sub"},
      std::pair{"<ufunc 'multiply'>", "mul"},
      std::pair{"<ufunc 'divide'>", "truediv"},
      std::pair{"<ufunc 'equal'>", "eq"},
      std::pair{"<ufunc 'less'>", "lt"},
      std::pair{"<ufunc 'less_equal'>", "le"},
      std::pair{"<ufunc 'greater'>", "gt"},
      std::pair{"<ufunc 'greater_equal'>", "ge"}};

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](Real<double>& self, nb::object ufunc, nb::str method, nb::args inputs,
         const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

        // RealMatrix only represents 2D arrays
        auto is_2d_array = [](const nb::handle& obj) {
          return nb::isinstance<nb::ndarray<>>(obj) &&
                 nb::cast<nb::ndarray<>>(obj).ndim() == 2;
        };

        if (method_name == "__call__" && inputs.size() == 2) {
          if (auto op = std::ranges::find(
                  scalar_operators, ufunc_name,
                  [](const auto& pair) { return pair.first; });
              op != scalar_operators.end()) {
            auto np = nb::module_::import_("numpy");

            auto is_0d = [&](const nb::handle& obj) {
              // np.ndim() returns 0 for any non-array object
              return !nb::isinstance<Real<double>>(obj) &&
                     nb::cast<int>(np.attr("ndim")(obj)) == 0;
            };

            // Convert NumPy scalars and 0D arrays to the equivalent Python
            // scalar (e.g., np.float64 to float) and return other objects
            // unchanged
            auto numpy_to_python_scalar =
                [&](const nb::handle& obj) -> nb::object {
              nb::object numpy_types =
                  nb::make_tuple(np.attr("generic"), np.attr("ndarray"));
              if (PyObject_IsInstance(obj.ptr(), numpy_types.ptr()) == 1) {
                return obj.attr("item")();
              } else {
                return nb::borrow(obj);
              }
            };

            // If an operand is 0D, apply the Python operator to its Python
            // scalar and the other operand, which dispatches to Real's
            // scalar overloads
            if (is_0d(inputs[0])) {
              return nb::module_::import_("operator")
                  .attr(op->second)(numpy_to_python_scalar(inputs[0]),
                                    inputs[1]);
            } else if (is_0d(inputs[1])) {
              return nb::module_::import_("operator")
                  .attr(op->second)(inputs[0],
                                    numpy_to_python_scalar(inputs[1]));
            }
          }

          if (ufunc_name == "<ufunc 'multiply'>") {
            if (is_2d_array(inputs[0])) {
              if (auto result = apply_eigen_op(
                      [&](auto& lhs) {
                        return lhs.template cast<double>() * self;
                      },
                      inputs[0])) {
                return result.value();
              }
            } else if (is_2d_array(inputs[1])) {
              if (auto result = apply_eigen_op(
                      [&](auto& rhs) {
                        return self * rhs.template cast<double>();
                      },
                      inputs[1])) {
                return result.value();
              }
            }
          } else if (ufunc_name == "<ufunc 'divide'>") {
            if (is_2d_array(inputs[0])) {
              if (auto result = apply_eigen_op(
                      [&](auto& lhs) {
                        return lhs.template cast<double>() / self;
                      },
                      inputs[0])) {
                return result.value();
              }
            }
          }
        }

        return nb::not_implemented();
      },
      // nanobind can't annotate nb::args or nb::kwargs, so the stub needs an
      // explicit signature
      nb::sig("def __array_ufunc__(self, ufunc: object, method: str, "
              "*inputs: object, **kwargs: object) -> object"),
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(
      "__pow__",
      [](const Real<double>& self, int power) { return pow(self, power); },
      nb::is_operator(), "power"_a);
  cls.def(-nb::self);
  cls.def(+nb::self);

  // Comparison operators
  for_each_type<nb::detail::self_t, double, int>([&]<typename T> {
    cls.def(nb::self == T(), "rhs"_a, DOC(slp, operator, eq, 7));
    cls.def(nb::self < T(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(nb::self <= T(), "rhs"_a, DOC(slp, operator, le));
    cls.def(nb::self > T(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(nb::self >= T(), "rhs"_a, DOC(slp, operator, ge));
    if constexpr (!std::same_as<T, nb::detail::self_t>) {
      cls.def(T() == nb::self, "lhs"_a, DOC(slp, operator, eq, 7));
      cls.def(T() < nb::self, "lhs"_a, DOC(slp, operator, lt));
      cls.def(T() <= nb::self, "lhs"_a, DOC(slp, operator, le));
      cls.def(T() > nb::self, "lhs"_a, DOC(slp, operator, gt));
      cls.def(T() >= nb::self, "lhs"_a, DOC(slp, operator, ge));
    }
  });

  // Math functions
  using V = Real<double>;
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
                  nb::overload_cast<const V&, const V&>(&hypot<double>), "x"_a,
                  "y"_a, DOC(slp, hypot));
  def_ternary_math(
      autodiff, "hypot",
      nb::overload_cast<const V&, const V&, const V&>(&hypot<double>), "x"_a,
      "y"_a, "z"_a, DOC(slp, hypot));
  autodiff.def(
      "if_else",
      [](const EqualityConstraints<double>& cond, const V& t, const V& f) {
        return if_else(cond, t, f);
      },
      "cond"_a, "t"_a, "f"_a, DOC(slp, if_else));
  autodiff.def(
      "if_else",
      [](const InequalityConstraints<double>& cond, const V& t, const V& f) {
        return if_else(cond, t, f);
      },
      "cond"_a, "t"_a, "f"_a, DOC(slp, if_else, 2));
  def_unary_math(autodiff, "log", &log<double>, "x"_a, DOC(slp, log));
  def_unary_math(autodiff, "log10", &log10<double>, "x"_a, DOC(slp, log10));
  def_binary_math(autodiff, "max",
                  nb::overload_cast<const V&, const V&>(&max<double>), "a"_a,
                  "b"_a, DOC(slp, max));
  def_binary_math(autodiff, "min",
                  nb::overload_cast<const V&, const V&>(&min<double>), "a"_a,
                  "b"_a, DOC(slp, min));
  def_binary_math(autodiff, "pow", &pow<double>, "base"_a, "power"_a,
                  DOC(slp, pow));
  def_unary_math(autodiff, "sign", &sign<double>, "x"_a, DOC(slp, sign));
  def_unary_math(autodiff, "sin", &sin<double>, "x"_a, DOC(slp, sin));
  def_unary_math(autodiff, "sinh", &sinh<double>, "x"_a, DOC(slp, sinh));
  def_unary_math(autodiff, "sqrt", &sqrt<double>, "x"_a, DOC(slp, sqrt));
  def_unary_math(autodiff, "tan", &tan<double>, "x"_a, DOC(slp, tan));
  def_unary_math(autodiff, "tanh", &tanh<double>, "x"_a, DOC(slp, tanh));

  // Arithmetic with constant complex numbers
  using C = ComplexScalar;
  cls.def(
      "__add__", [](const Real<double>& lhs, C rhs) { return lhs + rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 3));
  cls.def(
      "__radd__",
      [](const Real<double>& rhs, C lhs) { return lhs.value + rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, add, 4));
  cls.def(
      "__sub__", [](const Real<double>& lhs, C rhs) { return lhs - rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 3));
  cls.def(
      "__rsub__",
      [](const Real<double>& rhs, C lhs) { return lhs.value - rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, sub, 4));
  cls.def(
      "__mul__", [](const Real<double>& lhs, C rhs) { return lhs * rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 3));
  cls.def(
      "__rmul__",
      [](const Real<double>& rhs, C lhs) { return lhs.value * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 4));
  cls.def(
      "__truediv__",
      [](const Real<double>& lhs, C rhs) { return lhs / rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 3));
  cls.def(
      "__rtruediv__",
      [](const Real<double>& rhs, C lhs) { return lhs.value / rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, div, 4));

  // In-place operators with constant complex numbers return a Complex. Type
  // checkers require each in-place operator to have the same overloads as its
  // binary operator.
  cls.def(
      "__iadd__",
      [](const Real<double>& lhs, C rhs) { return lhs + rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 3));
  cls.def(
      "__isub__",
      [](const Real<double>& lhs, C rhs) { return lhs - rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 3));
  cls.def(
      "__imul__",
      [](const Real<double>& lhs, C rhs) { return lhs * rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 3));
  cls.def(
      "__itruediv__",
      [](const Real<double>& lhs, C rhs) { return lhs / rhs.value; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 3));
}

}  // namespace slp
