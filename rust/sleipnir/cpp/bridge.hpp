// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "rust/cxx.h"

namespace sleipnir_rs {

struct NodeF64;
struct NodeF32;
struct OptionsF64;
struct OptionsF32;
struct CallbackF64;
struct CallbackF32;

/// Problem built from a Rust expression graph, with the leaves whose values
/// Rust sets before each solve.
template <typename Scalar>
struct ProblemHandle {
  slp::Problem<Scalar> problem;

  /// Decision variables in creation order.
  std::vector<slp::Variable<Scalar>> decision_variables;

  /// Parameters (unregistered leaves) in creation order.
  std::vector<slp::Variable<Scalar>> parameters;
};

class ProblemF64 : public ProblemHandle<double> {};
class ProblemF32 : public ProblemHandle<float> {};

std::unique_ptr<ProblemF64> build_problem_f64(
    rust::Slice<const NodeF64> nodes, rust::isize cost,
    rust::Slice<const size_t> equality_constraints,
    rust::Slice<const size_t> inequality_constraints);

std::unique_ptr<ProblemF32> build_problem_f32(
    rust::Slice<const NodeF32> nodes, rust::isize cost,
    rust::Slice<const size_t> equality_constraints,
    rust::Slice<const size_t> inequality_constraints);

int32_t solve_problem_f64(ProblemF64& problem,
                          rust::Slice<const double> decision_values,
                          rust::Slice<const double> parameter_values,
                          const OptionsF64& options, bool has_callback,
                          CallbackF64& callback, rust::Slice<double> values);

int32_t solve_problem_f32(ProblemF32& problem,
                          rust::Slice<const float> decision_values,
                          rust::Slice<const float> parameter_values,
                          const OptionsF32& options, bool has_callback,
                          CallbackF32& callback, rust::Slice<float> values);

}  // namespace sleipnir_rs
