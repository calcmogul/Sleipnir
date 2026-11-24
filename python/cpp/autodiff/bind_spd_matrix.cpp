// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <concepts>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/autodiff/spd_matrix.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

namespace slp {

namespace {

using SPD = SPDMatrix<double>;
using VM = RealMatrix<double>;
using VB = RealSubmatrix<VM>;

/// Converts an operand to the type C++'s RealMatrix operators take.
///
/// SPDMatrices become Matrices, and Python scalars become Reals.
template <typename T>
decltype(auto) operand(T&& value) {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, SPD>) {
    VM result{value};
    return result;
  } else if constexpr (std::same_as<U, double> || std::same_as<U, int>) {
    return Real<double>{static_cast<double>(value)};
  } else {
    return std::forward<T>(value);
  }
}

/// Wraps a negative index and checks it's in bounds.
int wrap_index(int index, int size) {
  if (index < 0) {
    index += size;
  }
  if (index < 0 || index >= size) {
    throw std::out_of_range("Index out of bounds");
  }
  return index;
}

}  // namespace

void bind_spd_matrix(nb::class_<SPD>& cls) {
  using namespace nb::literals;

  cls.def(
      "__init__",
      [](SPD* self, nb::DRef<Eigen::MatrixXd> values) {
        new (self) SPD{values};
      },
      "values"_a, DOC(slp, SPDMatrix, SPDMatrix));

  cls.def("rows", &SPD::rows, DOC(slp, SPDMatrix, rows));
  cls.def("cols", &SPD::cols, DOC(slp, SPDMatrix, cols));
  cls.def_prop_ro("shape", [](const SPD& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def(
      "__getitem__",
      [](const SPD& self, std::tuple<int, int> slices) -> Real<double> {
        auto [row, col] = slices;
        return self[wrap_index(row, self.rows()), wrap_index(col, self.cols())];
      },
      "slices"_a, DOC(slp, SPDMatrix, operator, array));

  cls.def("value", nb::overload_cast<int, int>(&SPD::value), "row"_a, "col"_a,
          DOC(slp, SPDMatrix, value));
  cls.def("value", nb::overload_cast<>(&SPD::value),
          DOC(slp, SPDMatrix, value, 2));
  cls.def(
      "set_value",
      [](SPD& self, nb::DRef<Eigen::MatrixXd> values) {
        self.set_value(values);
      },
      "values"_a, DOC(slp, SPDMatrix, set_value));

  cls.def_prop_ro("T", &SPD::T, DOC(slp, SPDMatrix, T));
  cls.def("trace", &SPD::trace, DOC(slp, SPDMatrix, trace));

  cls.def("__pos__", [](const SPD& self) { return +self; });
  cls.def("__neg__", [](const SPD& self) -> VM { return -self; });

  // The sum of two SPDMatrices is an SPDMatrix
  cls.def(
      "__add__", [](const SPD& lhs, const SPD& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a);

  // Operations that don't preserve positive definiteness return a
  // RealMatrix. They convert the SPDMatrix to a RealMatrix and use its
  // operators.
  for_each_type<double, int, const Real<double>&, const VM&, const VB&,
                nb::DRef<Eigen::MatrixXd>, const SPD&>([&]<typename T> {
    constexpr bool is_scalar = std::same_as<T, double> ||
                               std::same_as<T, int> ||
                               std::same_as<T, const Real<double>&>;

    if constexpr (!std::same_as<T, const SPD&>) {
      cls.def(
          "__add__",
          [](const SPD& lhs, T rhs) -> VM {
            return operand(lhs) + operand(rhs);
          },
          nb::is_operator(), "rhs"_a);
    }
    cls.def(
        "__sub__",
        [](const SPD& lhs, T rhs) -> VM { return operand(lhs) - operand(rhs); },
        nb::is_operator(), "rhs"_a);

    if constexpr (is_scalar) {
      cls.def(
          "__radd__",
          [](const SPD& rhs, T lhs) -> VM {
            return operand(lhs) + operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__rsub__",
          [](const SPD& rhs, T lhs) -> VM {
            return operand(lhs) - operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__mul__",
          [](const SPD& lhs, T rhs) -> VM {
            return operand(lhs) * operand(rhs);
          },
          nb::is_operator(), "rhs"_a);
      cls.def(
          "__rmul__",
          [](const SPD& rhs, T lhs) -> VM {
            return operand(lhs) * operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__truediv__",
          [](const SPD& lhs, T rhs) -> VM {
            return operand(lhs) / operand(rhs);
          },
          nb::is_operator(), "rhs"_a);
    } else {
      for (const char* name : {"__mul__", "__matmul__"}) {
        cls.def(
            name,
            [](const SPD& lhs, T rhs) -> VM {
              return operand(lhs) * operand(rhs);
            },
            nb::is_operator(), "rhs"_a);
      }

      if constexpr (std::same_as<T, nb::DRef<Eigen::MatrixXd>>) {
        cls.def(
            "__eq__",
            [](const SPD& lhs, const Eigen::MatrixXd& rhs) {
              return lhs == rhs;
            },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 20));
      } else {
        cls.def(
            "__eq__", [](const SPD& lhs, T rhs) { return lhs == rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 20));
      }
    }
  });
  cls.def(
      "__eq__",
      [](const SPD& lhs, const SymmetricView<double>& rhs) {
        return lhs == rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 20));

  // Without this, an operand no overload above takes would fall back to
  // identity comparison. nanobind tries every overload without conversions
  // before trying any with them, so operands that need a conversion to a
  // matrix, like integer arrays, reach this too.
  cls.def(
      "__eq__",
      [](const SPD& lhs, nb::handle rhs) -> EqualityConstraints<double> {
        if (Eigen::MatrixXd values; nb::try_cast(rhs, values)) {
          return lhs == values;
        }
        throw nb::type_error(
            "An SPDMatrix can only equal a symmetric matrix. Constrain "
            "individual elements instead, like P[i, j] == c.");
      },
      nb::is_operator(),
      nb::sig("def __eq__(self, rhs: object) -> typing.NoReturn"), "rhs"_a,
      "Raises TypeError, since an SPDMatrix can only equal a symmetric "
      "matrix. Constrain individual elements instead, like P[i, j] == c.");

  // Matrices on the left, whose operators don't take an SPDMatrix
  for_each_type<const VM&, const VB&, nb::DRef<Eigen::MatrixXd>>(
      [&]<typename T> {
        for (const char* name : {"__rmul__", "__rmatmul__"}) {
          cls.def(
              name,
              [](const SPD& rhs, T lhs) -> VM { return lhs * operand(rhs); },
              nb::is_operator(), "lhs"_a);
        }
        cls.def(
            "__radd__",
            [](const SPD& rhs, T lhs) -> VM { return lhs + operand(rhs); },
            nb::is_operator(), "lhs"_a);
        cls.def(
            "__rsub__",
            [](const SPD& rhs, T lhs) -> VM { return lhs - operand(rhs); },
            nb::is_operator(), "lhs"_a);
      });

  // Make NumPy return NotImplemented from its binary operators so Python calls
  // the reflected operators above instead of treating the SPDMatrix as a
  // sequence of Reals
  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.attr("__array_ufunc__") = nb::none();
}

}  // namespace slp
