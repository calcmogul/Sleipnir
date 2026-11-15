// Copyright (c) Sleipnir contributors

#include "sleipnir/cpp/bridge.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "sleipnir/src/ffi.rs.h"

namespace sleipnir_rs {

namespace {

template <typename Scalar>
using Condition = bool (*)(Scalar a, Scalar b);

template <typename Scalar>
Condition<Scalar> condition_fn(ConditionKind kind) {
  switch (kind) {
    case ConditionKind::Less:
      return [](Scalar a, Scalar b) { return a < b; };
    case ConditionKind::LessEqual:
      return [](Scalar a, Scalar b) { return a <= b; };
    case ConditionKind::Greater:
      return [](Scalar a, Scalar b) { return a > b; };
    case ConditionKind::GreaterEqual:
      return [](Scalar a, Scalar b) { return a >= b; };
    case ConditionKind::Equal:
      return [](Scalar a, Scalar b) { return a == b; };
    case ConditionKind::NotEqual:
      return [](Scalar a, Scalar b) { return a != b; };
  }

  throw std::invalid_argument{"unknown condition kind"};
}

template <typename Scalar, typename Node>
slp::Variable<Scalar> build_node(ProblemHandle<Scalar>& handle,
                                 const std::vector<slp::Variable<Scalar>>& v,
                                 const Node& node) {
  if (node.kind != NodeKind::Decision && node.kind != NodeKind::Parameter &&
      node.kind != NodeKind::Constant) {
    for (auto operand : node.operands) {
      if (operand >= v.size()) {
        throw std::out_of_range{"expression node operand out of range"};
      }
    }
  }

  const auto& a = v[node.operands[0]];
  const auto& b = v[node.operands[1]];

  switch (node.kind) {
    case NodeKind::Decision: {
      auto x = handle.problem.decision_variable();
      x.set_value(node.value);
      handle.decision_variables.push_back(x);
      return x;
    }
    case NodeKind::Parameter: {
      slp::Variable<Scalar> p;
      p.set_value(node.value);
      handle.parameters.push_back(p);
      return p;
    }
    case NodeKind::Constant:
      return slp::Variable<Scalar>{node.value};
    case NodeKind::Neg:
      return -a;
    case NodeKind::Abs:
      return slp::abs(a);
    case NodeKind::Acos:
      return slp::acos(a);
    case NodeKind::Asin:
      return slp::asin(a);
    case NodeKind::Atan:
      return slp::atan(a);
    case NodeKind::Cbrt:
      return slp::cbrt(a);
    case NodeKind::Cos:
      return slp::cos(a);
    case NodeKind::Cosh:
      return slp::cosh(a);
    case NodeKind::Erf:
      return slp::erf(a);
    case NodeKind::Exp:
      return slp::exp(a);
    case NodeKind::Log:
      return slp::log(a);
    case NodeKind::Log10:
      return slp::log10(a);
    case NodeKind::Sign:
      return slp::sign(a);
    case NodeKind::Sin:
      return slp::sin(a);
    case NodeKind::Sinh:
      return slp::sinh(a);
    case NodeKind::Sqrt:
      return slp::sqrt(a);
    case NodeKind::Tan:
      return slp::tan(a);
    case NodeKind::Tanh:
      return slp::tanh(a);
    case NodeKind::Add:
      return a + b;
    case NodeKind::Sub:
      return a - b;
    case NodeKind::Mul:
      return a * b;
    case NodeKind::Div:
      return a / b;
    case NodeKind::Pow:
      return slp::pow(a, b);
    case NodeKind::Atan2:
      return slp::atan2(a, b);
    case NodeKind::Hypot:
      return slp::hypot(a, b);
    case NodeKind::Max:
      return slp::max(a, b);
    case NodeKind::Min:
      return slp::min(a, b);
    case NodeKind::IfElse:
      return slp::if_else(condition_fn<Scalar>(node.condition), a, b,
                          v[node.operands[2]], v[node.operands[3]]);
  }

  throw std::invalid_argument{"unknown expression node kind"};
}

template <typename Scalar, typename Handle, typename Node>
std::unique_ptr<Handle> build_problem(
    rust::Slice<const Node> nodes, rust::isize cost,
    rust::Slice<const size_t> equality_constraints,
    rust::Slice<const size_t> inequality_constraints) {
  auto handle = std::make_unique<Handle>();

  std::vector<slp::Variable<Scalar>> v;
  v.reserve(nodes.size());
  for (const auto& node : nodes) {
    v.push_back(build_node<Scalar>(*handle, v, node));
  }

  auto at = [&](size_t id) -> const slp::Variable<Scalar>& {
    if (id >= v.size()) {
      throw std::out_of_range{"constraint node out of range"};
    }
    return v[id];
  };

  if (cost >= 0) {
    handle->problem.minimize(at(static_cast<size_t>(cost)));
  }
  for (auto id : equality_constraints) {
    handle->problem.subject_to(at(id) == Scalar(0));
  }
  for (auto id : inequality_constraints) {
    handle->problem.subject_to(at(id) >= Scalar(0));
  }

  return handle;
}

/// Appends a column's entries sorted by row.
template <typename Scalar, typename Sparse>
void append_column(Sparse& out,
                   std::vector<std::pair<size_t, Scalar>>& column) {
  std::ranges::sort(column, {}, &std::pair<size_t, Scalar>::first);
  for (const auto& [row, value] : column) {
    out.row_indices.push_back(row);
    out.values.push_back(value);
  }
  out.col_offsets.push_back(out.row_indices.size());
  column.clear();
}

template <typename Sparse, typename Scalar>
Sparse to_rust(const Eigen::SparseMatrix<Scalar>& m) {
  Sparse out;
  out.rows = static_cast<size_t>(m.rows());
  out.cols = static_cast<size_t>(m.cols());
  out.col_offsets.push_back(0);

  std::vector<std::pair<size_t, Scalar>> column;
  for (int col = 0; col < m.outerSize(); ++col) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{m, col}; it;
         ++it) {
      column.emplace_back(static_cast<size_t>(it.row()), it.value());
    }
    append_column(out, column);
  }

  return out;
}

template <typename Sparse, typename Scalar>
Sparse to_rust(const Eigen::SparseVector<Scalar>& g) {
  Sparse out;
  out.rows = static_cast<size_t>(g.size());
  out.cols = 1;
  out.col_offsets.push_back(0);

  std::vector<std::pair<size_t, Scalar>> column;
  for (typename Eigen::SparseVector<Scalar>::InnerIterator it{g}; it; ++it) {
    column.emplace_back(static_cast<size_t>(it.index()), it.value());
  }
  append_column(out, column);

  return out;
}

template <typename Scalar, typename Sparse, typename Handle, typename Options,
          typename Callback>
int32_t solve_problem(Handle& handle, rust::Slice<const Scalar> decision_values,
                      rust::Slice<const Scalar> parameter_values,
                      const Options& options, bool has_callback,
                      Callback& callback, rust::Slice<Scalar> values) {
  if (decision_values.size() != handle.decision_variables.size() ||
      values.size() != handle.decision_variables.size() ||
      parameter_values.size() != handle.parameters.size()) {
    throw std::invalid_argument{"leaf count mismatch"};
  }

  for (size_t i = 0; i < decision_values.size(); ++i) {
    handle.decision_variables[i].set_value(decision_values[i]);
  }
  for (size_t i = 0; i < parameter_values.size(); ++i) {
    handle.parameters[i].set_value(parameter_values[i]);
  }

  handle.problem.clear_callbacks();
  if (has_callback) {
    handle.problem.add_callback([&](const slp::IterationInfo<Scalar>& info) {
      auto slice = [](const Eigen::VectorX<Scalar>& v) {
        return rust::Slice<const Scalar>{v.data(),
                                         static_cast<size_t>(v.size())};
      };
      return callback.call(info.iteration, slice(info.x), slice(info.s),
                           slice(info.y), slice(info.z),
                           to_rust<Sparse>(info.g), to_rust<Sparse>(info.H),
                           to_rust<Sparse>(info.A_e), to_rust<Sparse>(info.A_i))
                 ? slp::ControlFlow::BREAK
                 : slp::ControlFlow::CONTINUE;
    });
  }

  auto status = handle.problem.solve(
      slp::Options<Scalar>{
          .tolerance = options.tolerance,
          .max_iterations = options.max_iterations,
          .timeout = std::chrono::duration<double>{options.timeout},
          .feasible_ipm = options.feasible_ipm,
          .diagnostics = options.diagnostics},
      options.spy);
  handle.problem.clear_callbacks();

  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = handle.decision_variables[i].value();
  }

  return static_cast<int32_t>(status);
}

}  // namespace

std::unique_ptr<ProblemF64> build_problem_f64(
    rust::Slice<const NodeF64> nodes, rust::isize cost,
    rust::Slice<const size_t> equality_constraints,
    rust::Slice<const size_t> inequality_constraints) {
  return build_problem<double, ProblemF64>(nodes, cost, equality_constraints,
                                           inequality_constraints);
}

std::unique_ptr<ProblemF32> build_problem_f32(
    rust::Slice<const NodeF32> nodes, rust::isize cost,
    rust::Slice<const size_t> equality_constraints,
    rust::Slice<const size_t> inequality_constraints) {
  return build_problem<float, ProblemF32>(nodes, cost, equality_constraints,
                                          inequality_constraints);
}

int32_t solve_problem_f64(ProblemF64& problem,
                          rust::Slice<const double> decision_values,
                          rust::Slice<const double> parameter_values,
                          const OptionsF64& options, bool has_callback,
                          CallbackF64& callback, rust::Slice<double> values) {
  return solve_problem<double, SparseF64>(problem, decision_values,
                                          parameter_values, options,
                                          has_callback, callback, values);
}

int32_t solve_problem_f32(ProblemF32& problem,
                          rust::Slice<const float> decision_values,
                          rust::Slice<const float> parameter_values,
                          const OptionsF32& options, bool has_callback,
                          CallbackF32& callback, rust::Slice<float> values) {
  return solve_problem<float, SparseF32>(problem, decision_values,
                                         parameter_values, options,
                                         has_callback, callback, values);
}

}  // namespace sleipnir_rs
