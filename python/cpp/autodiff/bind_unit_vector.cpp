// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

namespace slp {

namespace {

using UV = UnitVector<double>;
using VM = RealMatrix<double>;
using VB = RealSubmatrix<VM>;

/// Converts an operand to the type C++'s RealMatrix operators take.
///
/// UnitVectors become Matrices, and Python scalars become Reals.
template <typename T>
decltype(auto) operand(T&& value) {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, UV>) {
    VM result = value;
    return result;
  } else if constexpr (std::same_as<U, double> || std::same_as<U, int>) {
    return Real<double>{static_cast<double>(value)};
  } else {
    return std::forward<T>(value);
  }
}

}  // namespace

void bind_unit_vector(nb::class_<UV>& cls) {
  using namespace nb::literals;

  cls.def(
      "__init__",
      [](UV* self, nb::DRef<Eigen::VectorXd> values) { new (self) UV{values}; },
      "values"_a, DOC(slp, UnitVector, UnitVector));

  cls.def("rows", &UV::rows, DOC(slp, UnitVector, rows));
  cls.def("cols", &UV::cols, DOC(slp, UnitVector, cols));
  cls.def_prop_ro(
      "shape", [](const UV& self) { return nb::make_tuple(self.rows(), 1); });
  cls.def("__len__", &UV::rows, DOC(slp, UnitVector, rows));
  cls.def(
      "__getitem__",
      [](const UV& self, int row) -> Real<double> {
        if (row < 0) {
          row += self.rows();
        }
        if (row < 0 || row >= self.rows()) {
          throw std::out_of_range("Index out of bounds");
        }
        return self[row];
      },
      "row"_a, DOC(slp, UnitVector, operator, array));
  cls.def(
      "__iter__",
      [](const UV& self) {
        return nb::make_iterator(nb::type<UV>(), "value_iterator", self.begin(),
                                 self.end());
      },
      nb::keep_alive<0, 1>());

  cls.def("value", nb::overload_cast<int>(&UV::value), "row"_a,
          DOC(slp, UnitVector, value));
  cls.def("value", nb::overload_cast<>(&UV::value),
          DOC(slp, UnitVector, value, 2));
  cls.def(
      "set_value",
      [](UV& self, nb::DRef<Eigen::VectorXd> values) {
        self.set_value(values);
      },
      "values"_a, DOC(slp, UnitVector, set_value));

  cls.def_prop_ro("T", &UV::T, DOC(slp, UnitVector, T));
  cls.def("dot", &UV::dot, "rhs"_a, DOC(slp, UnitVector, dot));
  cls.def(
      "dot",
      [](const UV& self, nb::DRef<Eigen::VectorXd> rhs) {
        return self.dot(VM(rhs));
      },
      "rhs"_a, DOC(slp, UnitVector, dot));
  cls.def("cross", &UV::cross, "rhs"_a, DOC(slp, UnitVector, cross));
  cls.def(
      "cross",
      [](const UV& self, nb::DRef<Eigen::VectorXd> rhs) {
        return self.cross(VM(rhs));
      },
      "rhs"_a, DOC(slp, UnitVector, cross));
  cls.def("rotate_by", &UV::rotate_by<Rotation3<double>>, "q"_a,
          DOC(slp, UnitVector, rotate_by));
  cls.def("rotate_by", &UV::rotate_by<Rotation2<double>>, "z"_a,
          DOC(slp, UnitVector, rotate_by, 2));

  cls.def("__neg__", [](const UV& self) { return -self; });
  cls.def("__pos__", [](const UV& self) { return self; });

  // Operations that don't preserve unit length return a RealMatrix. They
  // convert the UnitVector to a RealMatrix and use its operators.
  for_each_type<double, int, const Real<double>&, const VM&, const VB&,
                nb::DRef<Eigen::MatrixXd>, const UV&>([&]<typename T> {
    constexpr bool is_scalar = std::same_as<T, double> ||
                               std::same_as<T, int> ||
                               std::same_as<T, const Real<double>&>;

    cls.def(
        "__add__",
        [](const UV& lhs, T rhs) -> VM { return operand(lhs) + operand(rhs); },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__sub__",
        [](const UV& lhs, T rhs) -> VM { return operand(lhs) - operand(rhs); },
        nb::is_operator(), "rhs"_a);

    if constexpr (is_scalar) {
      cls.def(
          "__radd__",
          [](const UV& rhs, T lhs) -> VM {
            return operand(lhs) + operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__rsub__",
          [](const UV& rhs, T lhs) -> VM {
            return operand(lhs) - operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__mul__",
          [](const UV& lhs, T rhs) -> VM {
            return operand(lhs) * operand(rhs);
          },
          nb::is_operator(), "rhs"_a);
      cls.def(
          "__rmul__",
          [](const UV& rhs, T lhs) -> VM {
            return operand(lhs) * operand(rhs);
          },
          nb::is_operator(), "lhs"_a);
      cls.def(
          "__truediv__",
          [](const UV& lhs, T rhs) -> VM {
            return operand(lhs) / operand(rhs);
          },
          nb::is_operator(), "rhs"_a);
    } else {
      for (const char* name : {"__mul__", "__matmul__"}) {
        cls.def(
            name,
            [](const UV& lhs, T rhs) -> VM {
              return operand(lhs) * operand(rhs);
            },
            nb::is_operator(), "rhs"_a);
      }
    }

    // Comparison operators
    if constexpr (std::same_as<T, const UV&>) {
      cls.def(
          "__eq__", [](const UV& lhs, const UV& rhs) { return lhs == rhs; },
          nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 22));
    } else if constexpr (std::same_as<T, nb::DRef<Eigen::MatrixXd>>) {
      cls.def(
          "__eq__",
          [](const UV& lhs, const Eigen::VectorXd& rhs) { return lhs == rhs; },
          nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 23));
    }
    cls.def(
        "__lt__",
        [](const UV& lhs, T rhs) { return operand(lhs) < operand(rhs); },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(
        "__le__",
        [](const UV& lhs, T rhs) { return operand(lhs) <= operand(rhs); },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
    cls.def(
        "__gt__",
        [](const UV& lhs, T rhs) { return operand(lhs) > operand(rhs); },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(
        "__ge__",
        [](const UV& lhs, T rhs) { return operand(lhs) >= operand(rhs); },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));
  });

  // Without this, an operand no overload takes would fall back to identity
  // comparison. nanobind tries every overload without conversions before
  // trying any with them, so operands that need a conversion to a vector,
  // like integer arrays, reach this too.
  cls.def(
      "__eq__",
      [](const UV& lhs, nb::handle rhs) -> EqualityConstraints<double> {
        if (Eigen::VectorXd values; nb::try_cast(rhs, values)) {
          return lhs == values;
        }
        throw nb::type_error(
            "A UnitVector can only equal another UnitVector or a constant "
            "direction.");
      },
      nb::is_operator(),
      nb::sig("def __eq__(self, rhs: object) -> typing.NoReturn"), "rhs"_a,
      "Raises TypeError, since a UnitVector can only equal another "
      "UnitVector or a constant direction.");

  // NumPy array on the left
  for (const char* name : {"__rmul__", "__rmatmul__"}) {
    cls.def(
        name,
        [](const UV& rhs, nb::DRef<Eigen::MatrixXd> lhs) -> VM {
          return lhs * operand(rhs);
        },
        nb::is_operator(), "lhs"_a);
  }
  cls.def(
      "__radd__",
      [](const UV& rhs, nb::DRef<Eigen::MatrixXd> lhs) -> VM {
        return lhs + operand(rhs);
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__rsub__",
      [](const UV& rhs, nb::DRef<Eigen::MatrixXd> lhs) -> VM {
        return lhs - operand(rhs);
      },
      nb::is_operator(), "lhs"_a);

  // Make NumPy return NotImplemented from its binary operators so Python calls
  // the reflected operators above instead of treating the UnitVector as a
  // sequence of Reals
  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.attr("__array_ufunc__") = nb::none();
}

}  // namespace slp
