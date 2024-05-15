// Copyright (c) Sleipnir contributors

#include <format>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"
#include "NumPy.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindVariableMatrix(py::module_& autodiff,
                        py::class_<VariableMatrix>& cls) {
  using namespace py::literals;

  cls.def(py::init<>(), DOC(sleipnir, VariableMatrix, VariableMatrix));
  cls.def(py::init<int>(), "rows"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 2));
  cls.def(py::init<int, int>(), "rows"_a, "cols"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 3));
  cls.def(py::init<const std::vector<std::vector<double>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 5));
  cls.def(py::init<const std::vector<std::vector<Variable>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 6));
  cls.def(py::init<const Variable&>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 9));
  cls.def(py::init<const VariableBlock<VariableMatrix>&>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 11));
  cls.def(
      "set_value",
      [](VariableMatrix& self, const Eigen::MatrixXd& values) {
        self.SetValue(values);
      },
      "values"_a, DOC(sleipnir, VariableMatrix, SetValue));
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, int row, const Variable& value) {
        return self[row] = value;
      },
      "row"_a, "value"_a);
  // TODO: Support slice stride other than 1
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, py::tuple slices, py::object value) {
        if (slices.size() != 2) {
          throw py::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()));
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        size_t start;
        size_t stop;
        size_t step;
        size_t sliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (py::isinstance<py::slice>(rowElem)) {
          const auto& rowSlice = rowElem.cast<py::slice>();
          if (!rowSlice.compute(self.Rows(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = rowElem.cast<int>();
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (py::isinstance<py::slice>(colElem)) {
          const auto& colSlice = colElem.cast<py::slice>();
          if (!colSlice.compute(self.Cols(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = colElem.cast<int>();
          blockCols = 1;
        }

        if (py::isinstance<VariableMatrix>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<VariableMatrix>();
        } else if (py::isinstance<VariableBlock<VariableMatrix>>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<VariableBlock<VariableMatrix>>();
        } else if (IsNumPyArithmeticArray(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<Eigen::MatrixXd>();
        } else if (py::isinstance<Variable>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<Variable>();
        } else if (py::isinstance<py::float_>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<double>();
        } else if (py::isinstance<py::int_>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              value.cast<int>();
        } else {
          throw py::value_error(
              "VariableMatrix.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](VariableMatrix& self, int row) -> Variable& {
        if (row < 0) {
          row = self.size() + row;
        }
        return self[row];
      },
      py::keep_alive<0, 1>(), "row"_a,
      DOC(sleipnir, VariableMatrix, operator, array, 3));
  // TODO: Support slice stride other than 1
  cls.def(
      "__getitem__",
      [](VariableMatrix& self, py::tuple slices) -> py::object {
        if (slices.size() != 2) {
          throw py::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()));
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (py::isinstance<py::int_>(slices[0]) &&
            py::isinstance<py::int_>(slices[1])) {
          int row = slices[0].cast<int>();
          int col = slices[1].cast<int>();

          if (row >= self.Rows() || col >= self.Cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row = self.Rows() + row;
          }
          if (col < 0) {
            col = self.Cols() + col;
          }
          return py::cast(self[row, col]);
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        size_t start;
        size_t stop;
        size_t step;
        size_t sliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (py::isinstance<py::slice>(rowElem)) {
          const auto& rowSlice = rowElem.cast<py::slice>();
          if (!rowSlice.compute(self.Rows(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = rowElem.cast<int>();
          if (rowOffset < 0) {
            rowOffset = self.Rows() + rowOffset;
          }
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (py::isinstance<py::slice>(colElem)) {
          const auto& colSlice = colElem.cast<py::slice>();
          if (!colSlice.compute(self.Cols(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = colElem.cast<int>();
          if (colOffset < 0) {
            colOffset = self.Cols() + colOffset;
          }
          blockCols = 1;
        }

        return py::cast(self.Block(rowOffset, colOffset, blockRows, blockCols));
      },
      py::keep_alive<0, 1>(), "slices"_a,
      DOC(sleipnir, VariableMatrix, operator, array));
  cls.def("row", py::overload_cast<int>(&VariableMatrix::Row), "row"_a,
          DOC(sleipnir, VariableMatrix, Row));
  cls.def("col", py::overload_cast<int>(&VariableMatrix::Col), "col"_a,
          DOC(sleipnir, VariableMatrix, Col));
  cls.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableMatrix& lhs) {
        return lhs * rhs;
      },
      py::is_operator(), "lhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      py::is_operator(), "rhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](VariableMatrix& self, py::object ufunc, py::str method,
         py::args inputs, const py::kwargs&) -> py::object {
        std::string method_name = method;
        std::string ufunc_name = ufunc.attr("__repr__")().cast<py::str>();

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() * self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self * inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() + self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self + inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() - self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self - inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() == self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self == inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() < self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self < inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() <= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self <= inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() > self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self > inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() >= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self >= inputs[1].cast<Eigen::MatrixXd>());
            }
          }
        }

        std::string input1_name = inputs[0].attr("__repr__")().cast<py::str>();
        std::string input2_name = inputs[1].attr("__repr__")().cast<py::str>();
        throw py::value_error(
            std::format("VariableMatrix: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name));
        return py::cast(VariableMatrix{self});
      },
      "ufunc"_a, "method"_a);

  cls.def(py::self * Variable(), "rhs"_a);
  cls.def(py::self * double(), "rhs"_a);
  cls.def(Variable() * py::self, "lhs"_a);
  cls.def(double() * py::self, "lhs"_a);

  cls.def(py::self / Variable(), "rhs"_a);
  cls.def(py::self / double(), "rhs"_a);
  cls.def(py::self /= Variable(), "rhs"_a,
          DOC(sleipnir, VariableMatrix, operator, idiv));
  cls.def(py::self /= double(), "rhs"_a,
          DOC(sleipnir, VariableMatrix, operator, idiv));

  cls.def(py::self + py::self, "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs + VariableMatrix{rhs};
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} + rhs;
      },
      py::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](double lhs, const VariableMatrix& rhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs, double lhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      py::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs,
         const Eigen::Ref<const Eigen::MatrixXd>& lhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator(), "lhs"_a);

  cls.def(py::self - py::self, "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs - VariableMatrix{rhs};
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} - rhs;
      },
      py::is_operator(), "lhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix& rhs,
         const Eigen::Ref<const Eigen::MatrixXd>& lhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator(), "lhs"_a);

  cls.def(-py::self);
  cls.def(
      "__pow__",
      [](const VariableMatrix& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator(), "power"_a);
  cls.def_property_readonly("T", &VariableMatrix::T,
                            DOC(sleipnir, VariableMatrix, T));
  cls.def("rows", &VariableMatrix::Rows, DOC(sleipnir, VariableMatrix, Rows));
  cls.def("cols", &VariableMatrix::Cols, DOC(sleipnir, VariableMatrix, Cols));
  cls.def_property_readonly("shape", [](const VariableMatrix& self) {
    return py::make_tuple(self.Rows(), self.Cols());
  });
  cls.def(
      "value",
      static_cast<double (VariableMatrix::*)(int, int)>(&VariableMatrix::Value),
      "row"_a, "col"_a, DOC(sleipnir, VariableMatrix, Value));
  cls.def("value",
          static_cast<double (VariableMatrix::*)(int)>(&VariableMatrix::Value),
          "index"_a, DOC(sleipnir, VariableMatrix, Value, 2));
  cls.def("value",
          static_cast<Eigen::MatrixXd (VariableMatrix::*)()>(
              &VariableMatrix::Value),
          DOC(sleipnir, VariableMatrix, Value, 3));
  cls.def(
      "cwise_transform",
      [](const VariableMatrix& self,
         const std::function<Variable(const Variable&)>& func) {
        return self.CwiseTransform(func);
      },
      "func"_a, DOC(sleipnir, VariableMatrix, CwiseTransform));
  cls.def_static("zero", &VariableMatrix::Zero, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, Zero));
  cls.def_static("ones", &VariableMatrix::Ones, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, Ones));
  cls.def(py::self == py::self, "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self == Variable(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self == double(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self == int(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(Variable() == py::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(double() == py::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(int() == py::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self < py::self, "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self < Variable(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self < double(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self < int(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(Variable() < py::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(double() < py::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(int() < py::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self <= py::self, "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self <= Variable(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self <= double(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self <= int(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(Variable() <= py::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(double() <= py::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(int() <= py::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self > py::self, "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self > Variable(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self > double(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self > int(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(Variable() > py::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(double() > py::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(int() > py::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self >= py::self, "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(py::self >= Variable(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(py::self >= double(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(py::self >= int(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(Variable() >= py::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(double() >= py::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(int() >= py::self, "lhs"_a, DOC(sleipnir, operator, ge));
  py::implicitly_convertible<VariableMatrix, Variable>();

  // VariableMatrix-VariableBlock overloads
  cls.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableBlock<VariableMatrix>& lhs) {
        return lhs * rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs + rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs - rhs;
      },
      py::is_operator(), "rhs"_a);
  cls.def(
      "__eq__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs == rhs;
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs < rhs;
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs <= rhs;
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs > rhs;
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs >= rhs;
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(
      "__eq__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs == rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs < rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs <= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs > rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs >= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), "rhs"_a, DOC(sleipnir, operator, ge));

  cls.def("__len__", &VariableMatrix::Rows,
          DOC(sleipnir, VariableMatrix, Rows));

  cls.def(
      "__iter__",
      [](const VariableMatrix& self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>());

  autodiff.def(
      "cwise_reduce",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs,
         const std::function<Variable(const Variable&, const Variable&)> func) {
        return CwiseReduce(lhs, rhs, func);
      },
      "lhs"_a, "rhs"_a, "func"_a, DOC(sleipnir, CwiseReduce));

  autodiff.def("block",
               static_cast<VariableMatrix (*)(
                   const std::vector<std::vector<VariableMatrix>>&)>(&Block),
               "list"_a, DOC(sleipnir, Block));

  autodiff.def("solve",
               static_cast<VariableMatrix (*)(const VariableMatrix&,
                                              const VariableMatrix&)>(&Solve),
               "A"_a, "B"_a, DOC(sleipnir, Solve));
}  // NOLINT(readability/fn_size)

}  // namespace sleipnir
