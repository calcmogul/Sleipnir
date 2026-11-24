// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <concepts>
#include <stdexcept>
#include <tuple>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

namespace slp {

void bind_symmetric_view(nb::class_<SymmetricView<double>>& cls) {
  using namespace nb::literals;

  using SV = SymmetricView<double>;
  using VM = RealMatrix<double>;
  using VB = RealSubmatrix<VM>;

  cls.def(nb::init<const VM&>(), "matrix"_a,
          DOC(slp, SymmetricView, SymmetricView));
  cls.def("rows", &SV::rows, DOC(slp, SymmetricView, rows));
  cls.def("cols", &SV::cols, DOC(slp, SymmetricView, cols));
  cls.def_prop_ro("shape", [](const SV& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def(
      "__getitem__",
      [](const SV& self, std::tuple<int, int> slices) -> Real<double> {
        auto [row, col] = slices;
        if (row < 0) {
          row += self.rows();
        }
        if (col < 0) {
          col += self.cols();
        }
        if (row < 0 || row >= self.rows() || col < 0 || col >= self.cols()) {
          throw std::out_of_range("Index out of bounds");
        }
        return self[row, col];
      },
      "slices"_a, DOC(slp, SymmetricView, operator, array));

  for_each_type<double, int, const Real<double>&, const VM&, const VB&,
                const SV&>([&]<typename T> {
    cls.def(
        "__eq__", [](const SV& lhs, T rhs) { return lhs == rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 12));
  });
  cls.def(
      "__eq__",
      [](const SV& lhs, const Eigen::MatrixXd& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 12));

  // Python calls the operators above reflected for other operands on the left,
  // which gives the same constraints since equality is symmetric

  // Make NumPy return NotImplemented from its binary operators so Python calls
  // the reflected operators above instead of treating the view as a sequence
  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.attr("__array_ufunc__") = nb::none();
}

}  // namespace slp
