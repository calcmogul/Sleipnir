// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/complex.hpp"
#include "sleipnir/autodiff/complex_matrix.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/pose2.hpp"
#include "sleipnir/autodiff/pose3.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/rotation2.hpp"
#include "sleipnir/autodiff/rotation3.hpp"
#include "sleipnir/autodiff/similarity2.hpp"
#include "sleipnir/autodiff/similarity3.hpp"
#include "sleipnir/autodiff/spd_matrix.hpp"
#include "sleipnir/autodiff/translation2.hpp"
#include "sleipnir/autodiff/translation3.hpp"
#include "sleipnir/autodiff/unit_vector.hpp"
#include "sleipnir/optimization/lint.hpp"
#include "sleipnir/optimization/solver/control_flow.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/ipm.hpp"
#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/newton.hpp"
#include "sleipnir/optimization/solver/newton_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/bounds.hpp"
#include "sleipnir/optimization/solver/util/manifold.hpp"
#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/optimization/solver/util/problem_scaling.hpp"
#include "sleipnir/optimization/solver/util/so2_manifold.hpp"
#include "sleipnir/optimization/solver/util/so3_manifold.hpp"
#include "sleipnir/optimization/solver/util/spd_manifold.hpp"
#include "sleipnir/optimization/solver/util/sphere_manifold.hpp"
#include "sleipnir/optimization/solver/util/time_trace.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/print.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/spy.hpp"
#include "sleipnir/util/symbol_exports.hpp"
#include "sleipnir/util/time_tracer.hpp"

namespace slp {

/// This class allows the user to pose a constrained nonlinear optimization
/// problem in natural mathematical notation and solve it.
///
/// This class supports problems of the form:
///
/// ```
///       minₓ f(x)
/// subject to cₑ(x) = 0
///            cᵢ(x) ≥ 0
/// ```
///
/// where f(x) is the scalar cost function, x is the vector of decision
/// variables (variables the solver can tweak to minimize the cost function),
/// cᵢ(x) are the inequality constraints, and cₑ(x) are the equality
/// constraints. Constraints are equations or inequalities of the decision
/// variables that constrain what values the solver is allowed to use when
/// searching for an optimal solution.
///
/// The nice thing about this class is users don't have to put their system in
/// the form shown above manually; they can write it in natural mathematical
/// form and it'll be converted for them.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Problem {
 public:
  /// Constructs the optimization problem.
  Problem() noexcept = default;

  /// Creates a decision variable in the optimization problem.
  ///
  /// Decision variables have an initial value of zero.
  ///
  /// @return A decision variable in the optimization problem.
  [[nodiscard]]
  Real<Scalar> real_decision_variable() {
    m_decision_variables.emplace_back();
    return m_decision_variables.back();
  }

  /// Creates a matrix of decision variables in the optimization problem.
  ///
  /// Decision variables have an initial value of zero.
  ///
  /// @param rows Number of matrix rows.
  /// @param cols Number of matrix columns.
  /// @return A matrix of decision variables in the optimization problem.
  [[nodiscard]]
  RealMatrix<Scalar> real_decision_variable(int rows, int cols = 1) {
    m_decision_variables.reserve(m_decision_variables.size() + rows * cols);

    RealMatrix<Scalar> vars{detail::empty, rows, cols};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        m_decision_variables.emplace_back();
        vars[row, col] = m_decision_variables.back();
      }
    }

    return vars;
  }

  /// Creates a parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// Parameters have an initial value of zero.
  ///
  /// @return A parameter in the optimization problem.
  [[nodiscard]]
  Real<Scalar> parameter() {
    return Real<Scalar>{
        detail::make_expression_ptr<detail::ParameterExpression<Scalar>>()};
  }

  /// Creates a matrix of parameters in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// Parameters have an initial value of zero.
  ///
  /// @param rows Number of matrix rows.
  /// @param cols Number of matrix columns.
  /// @return A matrix of parameters in the optimization problem.
  [[nodiscard]]
  RealMatrix<Scalar> parameter(int rows, int cols = 1) {
    RealMatrix<Scalar> params{detail::empty, rows, cols};

    for (auto& param : params) {
      param = parameter();
    }

    return params;
  }

  /// Creates a complex decision variable in the optimization problem.
  ///
  /// The real and imaginary parts are two real decision variables. Complex
  /// decision variables have an initial value of zero.
  ///
  /// @return A complex decision variable in the optimization problem.
  [[nodiscard]]
  auto complex_decision_variable()
    requires std::floating_point<Scalar>
  {
    auto real = real_decision_variable();
    auto imag = real_decision_variable();
    return Complex<Scalar>{std::move(real), std::move(imag)};
  }

  /// Creates a matrix of complex decision variables in the optimization
  /// problem.
  ///
  /// The real and imaginary parts of each entry are two real decision
  /// variables. Complex decision variables have an initial value of zero.
  ///
  /// @param rows Number of matrix rows.
  /// @param cols Number of matrix columns.
  /// @return A matrix of complex decision variables in the optimization
  ///     problem.
  [[nodiscard]]
  auto complex_decision_variable(int rows, int cols = 1)
    requires std::floating_point<Scalar>
  {
    ComplexMatrix<Scalar> vars{detail::empty, rows, cols};

    for (auto& var : vars) {
      var = complex_decision_variable();
    }

    return vars;
  }

  /// Creates a complex parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// Complex parameters have an initial value of zero.
  ///
  /// @return A complex parameter in the optimization problem.
  [[nodiscard]]
  auto complex_parameter()
    requires std::floating_point<Scalar>
  {
    return Complex<Scalar>{parameter(), parameter()};
  }

  /// Creates a matrix of complex parameters in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// Complex parameters have an initial value of zero.
  ///
  /// @param rows Number of matrix rows.
  /// @param cols Number of matrix columns.
  /// @return A matrix of complex parameters in the optimization problem.
  [[nodiscard]]
  auto complex_parameter(int rows, int cols = 1)
    requires std::floating_point<Scalar>
  {
    ComplexMatrix<Scalar> params{detail::empty, rows, cols};

    for (auto& param : params) {
      param = complex_parameter();
    }

    return params;
  }

  /// Creates a unit vector of decision variables in the optimization problem.
  ///
  /// The solver keeps the vector on the unit sphere by stepping along the
  /// sphere instead of through ℝⁿ, so no equality constraint is needed.
  ///
  /// The vector has an initial value of e₁ = [1, 0, …, 0]ᵀ. Values set
  /// elsewhere are normalized when the solve starts.
  ///
  /// @param rows The vector's dimension. Must be at least 2.
  /// @return A unit vector of decision variables in the optimization problem.
  [[nodiscard]]
  UnitVector<Scalar> unit_vector_decision_variable(int rows) {
    slp_assert(rows >= 2);

    m_manifold_blocks.emplace_back(
        sphere_manifold<Scalar>, static_cast<int>(m_decision_variables.size()),
        rows);

    RealMatrix<Scalar> vars = real_decision_variable(rows);
    vars[0].set_value(Scalar(1));
    return UnitVector<Scalar>{std::move(vars)};
  }

  /// Creates a unit vector of parameters in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The vector has an initial value of e₁ = [1, 0, …, 0]ᵀ, and set_value()
  /// normalizes its argument.
  ///
  /// @param rows The vector's dimension. Must be at least 2.
  /// @return A unit vector of parameters in the optimization problem.
  [[nodiscard]]
  UnitVector<Scalar> unit_vector_parameter(int rows) {
    slp_assert(rows >= 2);

    RealMatrix<Scalar> params = parameter(rows);
    params[0].set_value(Scalar(1));
    return UnitVector<Scalar>{std::move(params)};
  }

  /// Creates a 2D translation decision variable in the optimization problem.
  ///
  /// The translation is two ordinary decision variables.
  ///
  /// The translation has an initial value of zero.
  ///
  /// @return A 2D translation of decision variables in the optimization
  ///     problem.
  [[nodiscard]]
  Translation2<Scalar> translation2_decision_variable() {
    return Translation2<Scalar>{real_decision_variable(2)};
  }

  /// Creates a 2D translation parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The translation has an initial value of zero.
  ///
  /// @return A 2D translation of parameters in the optimization problem.
  [[nodiscard]]
  Translation2<Scalar> translation2_parameter() {
    return Translation2<Scalar>{parameter(2)};
  }

  /// Creates a 3D translation decision variable in the optimization problem.
  ///
  /// The translation is three ordinary decision variables.
  ///
  /// The translation has an initial value of zero.
  ///
  /// @return A 3D translation of decision variables in the optimization
  ///     problem.
  [[nodiscard]]
  Translation3<Scalar> translation3_decision_variable() {
    return Translation3<Scalar>{real_decision_variable(3)};
  }

  /// Creates a 3D translation parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The translation has an initial value of zero.
  ///
  /// @return A 3D translation of parameters in the optimization problem.
  [[nodiscard]]
  Translation3<Scalar> translation3_parameter() {
    return Translation3<Scalar>{parameter(3)};
  }

  /// Creates a 2D rotation decision variable in the optimization problem.
  ///
  /// The solver keeps the rotation's complex number unit-length by stepping
  /// along SO(2) instead of through ℝ², so no equality constraint is needed.
  ///
  /// The rotation has an initial value of zero radians. Values set elsewhere
  /// are normalized when the solve starts.
  ///
  /// @return A 2D rotation of decision variables in the optimization problem.
  [[nodiscard]]
  Rotation2<Scalar> rotation2_decision_variable() {
    m_manifold_blocks.emplace_back(
        so2_manifold<Scalar>, static_cast<int>(m_decision_variables.size()), 2);

    RealMatrix<Scalar> z = real_decision_variable(2);
    z[0].set_value(Scalar(1));
    return Rotation2<Scalar>{z[0], z[1]};
  }

  /// Creates a 2D rotation parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The rotation's complex number is two parameters. The rotation has an
  /// initial value of zero radians.
  ///
  /// @return A 2D rotation of parameters in the optimization problem.
  [[nodiscard]]
  Rotation2<Scalar> rotation2_parameter() {
    RealMatrix<Scalar> z = parameter(2);
    z[0].set_value(Scalar(1));
    return Rotation2<Scalar>{z[0], z[1]};
  }

  /// Creates a 3D rotation decision variable in the optimization problem.
  ///
  /// The solver keeps the quaternion unit-length by stepping along SO(3)
  /// instead of through ℝ⁴, so no equality constraint is needed.
  ///
  /// The rotation has an initial value of the identity. Values set elsewhere
  /// are normalized when the solve starts.
  ///
  /// @return A 3D rotation of decision variables in the optimization problem.
  [[nodiscard]]
  Rotation3<Scalar> rotation3_decision_variable() {
    m_manifold_blocks.emplace_back(
        so3_manifold<Scalar>, static_cast<int>(m_decision_variables.size()), 4);

    RealMatrix<Scalar> q = real_decision_variable(4);
    q[0].set_value(Scalar(1));
    return Rotation3<Scalar>{q[0], q[1], q[2], q[3]};
  }

  /// Creates a 3D rotation parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The rotation's quaternion is four parameters. The rotation has an initial
  /// value of the identity, and set_value() normalizes its argument.
  ///
  /// @return A 3D rotation of parameters in the optimization problem.
  [[nodiscard]]
  Rotation3<Scalar> rotation3_parameter() {
    RealMatrix<Scalar> q = parameter(4);
    q[0].set_value(Scalar(1));
    return Rotation3<Scalar>{q[0], q[1], q[2], q[3]};
  }

  /// Creates a 2D pose decision variable in the optimization problem.
  ///
  /// The translation is two ordinary decision variables, and the rotation is
  /// a 2D rotation decision variable (see rotation2_decision_variable()).
  ///
  /// The pose has an initial value of the identity.
  ///
  /// @return A 2D pose of decision variables in the optimization problem.
  [[nodiscard]]
  Pose2<Scalar> pose2_decision_variable() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation2_decision_variable();
    auto rotation = rotation2_decision_variable();
    return Pose2<Scalar>{std::move(translation), std::move(rotation)};
  }

  /// Creates a 2D pose parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The pose is a 2D translation parameter (see translation2_parameter()) and
  /// a 2D rotation parameter (see rotation2_parameter()). Set their values
  /// through translation() and rotation().
  ///
  /// The pose has an initial value of the identity.
  ///
  /// @return A 2D pose of parameters in the optimization problem.
  [[nodiscard]]
  Pose2<Scalar> pose2_parameter() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation2_parameter();
    auto rotation = rotation2_parameter();
    return Pose2<Scalar>{std::move(translation), std::move(rotation)};
  }

  /// Creates a 3D pose decision variable in the optimization problem.
  ///
  /// The translation is three ordinary decision variables, and the rotation
  /// is a 3D rotation decision variable (see rotation3_decision_variable()).
  ///
  /// The pose has an initial value of the identity.
  ///
  /// @return A 3D pose of decision variables in the optimization problem.
  [[nodiscard]]
  Pose3<Scalar> pose3_decision_variable() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation3_decision_variable();
    auto rotation = rotation3_decision_variable();
    return Pose3<Scalar>{std::move(translation), std::move(rotation)};
  }

  /// Creates a 3D pose parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The pose is a 3D translation parameter (see translation3_parameter()) and
  /// a 3D rotation parameter (see rotation3_parameter()). Set their values
  /// through translation() and rotation().
  ///
  /// The pose has an initial value of the identity.
  ///
  /// @return A 3D pose of parameters in the optimization problem.
  [[nodiscard]]
  Pose3<Scalar> pose3_parameter() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation3_parameter();
    auto rotation = rotation3_parameter();
    return Pose3<Scalar>{std::move(translation), std::move(rotation)};
  }

  /// Creates a 2D similarity transform decision variable in the optimization
  /// problem.
  ///
  /// The translation is two ordinary decision variables, the rotation is a
  /// 2D rotation decision variable (see rotation2_decision_variable()), and the
  /// log scale is an ordinary decision variable.
  ///
  /// The similarity transform has an initial value of the identity.
  ///
  /// @return A 2D similarity transform of decision variables in the
  ///     optimization problem.
  [[nodiscard]]
  Similarity2<Scalar> similarity2_decision_variable() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation2_decision_variable();
    auto rotation = rotation2_decision_variable();
    auto log_scale = real_decision_variable();
    return Similarity2<Scalar>{std::move(translation), std::move(rotation),
                               std::move(log_scale)};
  }

  /// Creates a 2D similarity transform parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The similarity transform is a 2D translation parameter (see
  /// translation2_parameter()), a 2D rotation parameter (see
  /// rotation2_parameter()), and a log scale parameter. Set their values
  /// through translation(), rotation(), and log_scale().
  ///
  /// The similarity transform has an initial value of the identity.
  ///
  /// @return A 2D similarity transform of parameters in the optimization
  ///     problem.
  [[nodiscard]]
  Similarity2<Scalar> similarity2_parameter() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation2_parameter();
    auto rotation = rotation2_parameter();
    auto log_scale = parameter();
    return Similarity2<Scalar>{std::move(translation), std::move(rotation),
                               std::move(log_scale)};
  }

  /// Creates a 3D similarity transform decision variable in the optimization
  /// problem.
  ///
  /// The translation is three ordinary decision variables, the rotation is a
  /// 3D rotation decision variable (see rotation3_decision_variable()), and the
  /// log scale is an ordinary decision variable.
  ///
  /// The similarity transform has an initial value of the identity.
  ///
  /// @return A 3D similarity transform of decision variables in the
  ///     optimization problem.
  [[nodiscard]]
  Similarity3<Scalar> similarity3_decision_variable() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation3_decision_variable();
    auto rotation = rotation3_decision_variable();
    auto log_scale = real_decision_variable();
    return Similarity3<Scalar>{std::move(translation), std::move(rotation),
                               std::move(log_scale)};
  }

  /// Creates a 3D similarity transform parameter in the optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The similarity transform is a 3D translation parameter (see
  /// translation3_parameter()), a 3D rotation parameter (see
  /// rotation3_parameter()), and a log scale parameter. Set their values
  /// through translation(), rotation(), and log_scale().
  ///
  /// The similarity transform has an initial value of the identity.
  ///
  /// @return A 3D similarity transform of parameters in the optimization
  ///     problem.
  [[nodiscard]]
  Similarity3<Scalar> similarity3_parameter() {
    // Create components separately because MSVC has incorrect evaluation order:
    // https://developercommunity.visualstudio.com/t/Incorrect-evaluation-order-of-initialize/10993665
    auto translation = translation3_parameter();
    auto rotation = rotation3_parameter();
    auto log_scale = parameter();
    return Similarity3<Scalar>{std::move(translation), std::move(rotation),
                               std::move(log_scale)};
  }

  /// Creates a symmetric positive definite matrix of decision variables in the
  /// optimization problem.
  ///
  /// The solver keeps the matrix positive definite by retracting each step
  /// onto the positive definite matrices instead of stepping through the
  /// symmetric matrices, so no semidefinite constraint is needed. Problems
  /// with these matrices are solved with the interior-point method, which adds
  /// a log-det barrier term to keep them away from singular matrices. The
  /// matrix can approach a singular matrix but never reach one, so problems
  /// whose solution is singular converge slowly or not at all.
  ///
  /// The lower triangle's decision variables are reused across the diagonal,
  /// so the matrix is symmetric by construction.
  ///
  /// The matrix has an initial value of the identity. Values set elsewhere are
  /// kept if they're numerically positive definite, with each eigenvalue at
  /// least nε times the largest. Otherwise, they're projected onto the positive
  /// definite matrices when the solve starts by raising the smaller eigenvalues
  /// to √ε times the largest one.
  ///
  /// When equality constraints determine the matrix once the other decision
  /// variables are fixed, like a Lyapunov equation in a controller's gains,
  /// set its initial value by solving those constraints at the other decision
  /// variables' initial values. Warm starting only the other decision
  /// variables leaves the initial iterate far from the constraints, and the
  /// solve can fail.
  ///
  /// @param rows The matrix's dimension. Must be at least 1.
  /// @return A symmetric positive definite matrix of decision variables in the
  ///     optimization problem.
  [[nodiscard]]
  SPDMatrix<Scalar> spd_decision_variable(int rows) {
    slp_assert(rows >= 1);

    m_manifold_blocks.emplace_back(
        spd_manifold<Scalar>, static_cast<int>(m_decision_variables.size()),
        rows, rows);

    // Only the lower triangle's (n² + n)/2 elements are stored
    m_decision_variables.reserve(m_decision_variables.size() +
                                 (rows * rows + rows) / 2);

    RealMatrix<Scalar> P{detail::empty, rows, rows};
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col <= row; ++col) {
        m_decision_variables.emplace_back();
        P[row, col] = m_decision_variables.back();
        P[col, row] = m_decision_variables.back();
      }
      P[row, row].set_value(Scalar(1));
    }
    return SPDMatrix<Scalar>{std::move(P)};
  }

  /// Creates a symmetric positive definite matrix of parameters in the
  /// optimization problem.
  ///
  /// A parameter is a constant whose value can be changed between solves with
  /// set_value(). Unlike constants, parameters aren't folded into expressions.
  ///
  /// The lower triangle's parameters are reused across the diagonal, so the
  /// matrix is symmetric by construction.
  ///
  /// The matrix has an initial value of the identity. Unlike for decision
  /// variables, values set elsewhere are used as is, so they must be positive
  /// definite.
  ///
  /// @param rows The matrix's dimension. Must be at least 1.
  /// @return A symmetric positive definite matrix of parameters in the
  ///     optimization problem.
  [[nodiscard]]
  SPDMatrix<Scalar> spd_parameter(int rows) {
    slp_assert(rows >= 1);

    RealMatrix<Scalar> P{detail::empty, rows, rows};
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col <= row; ++col) {
        P[row, col] = parameter();
        P[col, row] = P[row, col];
      }
      P[row, row].set_value(Scalar(1));
    }
    return SPDMatrix<Scalar>{std::move(P)};
  }

  /// Tells the solver to minimize the output of the given cost function.
  ///
  /// Note that this is optional. If only constraints are specified, the solver
  /// will find the closest solution to the initial conditions that's in the
  /// feasible set.
  ///
  /// @param cost The cost function to minimize. A 1x1 RealMatrix will
  ///     implicitly convert to a Real, and a non-1x1 RealMatrix will
  ///     raise an assertion.
  void minimize(const Real<Scalar>& cost) { m_f = cost; }

  /// Tells the solver to minimize the output of the given cost function.
  ///
  /// Note that this is optional. If only constraints are specified, the solver
  /// will find the closest solution to the initial conditions that's in the
  /// feasible set.
  ///
  /// @param cost The cost function to minimize. A 1x1 RealMatrix will
  ///     implicitly convert to a Real, and a non-1x1 RealMatrix will
  ///     raise an assertion.
  void minimize(Real<Scalar>&& cost) { m_f = std::move(cost); }

  /// Tells the solver to maximize the output of the given objective function.
  ///
  /// Note that this is optional. If only constraints are specified, the solver
  /// will find the closest solution to the initial conditions that's in the
  /// feasible set.
  ///
  /// @param objective The objective function to maximize. A 1x1 RealMatrix
  ///     will implicitly convert to a Real, and a non-1x1 RealMatrix
  ///     will raise an assertion.
  void maximize(const Real<Scalar>& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -objective;
  }

  /// Tells the solver to maximize the output of the given objective function.
  ///
  /// Note that this is optional. If only constraints are specified, the solver
  /// will find the closest solution to the initial conditions that's in the
  /// feasible set.
  ///
  /// @param objective The objective function to maximize. A 1x1 RealMatrix
  ///     will implicitly convert to a Real, and a non-1x1 RealMatrix
  ///     will raise an assertion.
  void maximize(Real<Scalar>&& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -std::move(objective);
  }

  /// Tells the solver to solve the problem while satisfying the given equality
  /// constraint.
  ///
  /// @param constraint The constraint to satisfy.
  void subject_to(const EqualityConstraints<Scalar>& constraint) {
    m_equality_constraints.reserve(m_equality_constraints.size() +
                                   constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_equality_constraints));
  }

  /// Tells the solver to solve the problem while satisfying the given equality
  /// constraint.
  ///
  /// @param constraint The constraint to satisfy.
  void subject_to(EqualityConstraints<Scalar>&& constraint) {
    m_equality_constraints.reserve(m_equality_constraints.size() +
                                   constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_equality_constraints));
  }

  /// Tells the solver to solve the problem while satisfying the given
  /// inequality constraint.
  ///
  /// @param constraint The constraint to satisfy.
  void subject_to(const InequalityConstraints<Scalar>& constraint) {
    m_inequality_constraints.reserve(m_inequality_constraints.size() +
                                     constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_inequality_constraints));
  }

  /// Tells the solver to solve the problem while satisfying the given
  /// inequality constraint.
  ///
  /// @param constraint The constraint to satisfy.
  void subject_to(InequalityConstraints<Scalar>&& constraint) {
    m_inequality_constraints.reserve(m_inequality_constraints.size() +
                                     constraint.constraints.size());
    std::ranges::copy(constraint.constraints,
                      std::back_inserter(m_inequality_constraints));
  }

  /// Returns the cost function's type.
  ///
  /// @return The cost function's type.
  ExpressionType cost_function_type() const {
    if (m_f) {
      return m_f.value().type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /// Returns the type of the highest order equality constraint.
  ///
  /// @return The type of the highest order equality constraint.
  ExpressionType equality_constraint_type() const {
    if (!m_equality_constraints.empty()) {
      return std::ranges::max(m_equality_constraints, {}, &Real<Scalar>::type)
          .type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /// Returns the type of the highest order inequality constraint.
  ///
  /// @return The type of the highest order inequality constraint.
  ExpressionType inequality_constraint_type() const {
    if (!m_inequality_constraints.empty()) {
      return std::ranges::max(m_inequality_constraints, {}, &Real<Scalar>::type)
          .type();
    } else {
      return ExpressionType::NONE;
    }
  }

  /// Checks the problem's formulation for issues and suggests improvements.
  ///
  /// Linting analyzes the cost function and constraints without changing or
  /// solving the problem. It reports:
  ///
  /// <ul>
  ///   <li>Decision variables that don't appear in the cost function or any
  ///     constraint.</li>
  ///   <li>Constraints that don't depend on any decision variables.</li>
  ///   <li>Equality constraints that keep decision variables on a sphere,
  ///     which a unit vector or rotation decision variable does without a
  ///     constraint.</li>
  ///   <li>Linear constraints on the same linear combination of decision
  ///     variables that are redundant, conflict, or form an equality
  ///     constraint.</li>
  ///   <li>Functions of decision variables that aren't differentiable
  ///     everywhere, in forms with a differentiable replacement: abs(a) ≤ b,
  ///     max(a, b) ≤ c, min(a, b) ≥ c, a norm compared to a constant, and a
  ///     cost function that's a norm.</li>
  ///   <li>Coefficients of the cost function or linear constraints that span
  ///     so many orders of magnitude that the solver loses precision, when
  ///     rescaling decision variables or constraints fixes it.</li>
  /// </ul>
  ///
  /// Values in the findings, such as constraint values and bounds, use the
  /// decision variables' and parameters' current values.
  ///
  /// @return The findings, which are empty if there's nothing to improve.
  [[nodiscard]]
  std::vector<LintFinding> lint() const {
    return detail::lint<Scalar>(m_decision_variables, m_manifold_blocks, m_f,
                                m_equality_constraints,
                                m_inequality_constraints);
  }

  /// Solves the optimization problem. The solution will be stored in the
  /// original variables used to construct the problem.
  ///
  /// @param options Solver options.
  /// @param spy Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files
  ///     named H.spy, A_e.spy, and A_i.spy respectively during solve. Use
  ///     tools/spy.py to plot them.
  /// @return The solver status.
  ExitStatus solve(const Options<Scalar>& options = Options<Scalar>{},
                   bool spy = false) {
    if (options.diagnostics) {
      print_exit_conditions(options);
      print_problem_analysis();
    }

    // Create the initial value column vector
    DenseVector x{m_decision_variables.size()};
    for (size_t i = 0; i < m_decision_variables.size(); ++i) {
      x[i] = m_decision_variables[i].value();
    }

    // Move the initial guess onto the manifold
    const Manifold<Scalar> manifold{m_manifold_blocks};
    manifold.project(x);

    // If the problem is empty or constant, there's nothing to do
    if (cost_function_type() <= ExpressionType::PARAMETER &&
        equality_constraint_type() <= ExpressionType::PARAMETER &&
        inequality_constraint_type() <= ExpressionType::PARAMETER) {
      if (options.diagnostics) {
        slp::println("\nInvoking no-op solver\n");
      }

      RealMatrix<Scalar>{m_decision_variables}.set_value(x);
      return ExitStatus::SUCCESS;
    }

    // Solve the optimization problem. Manifolds with a boundary need the
    // interior-point method's barrier function.
    ExitStatus status;
    if (manifold.has_boundary() || !m_inequality_constraints.empty()) {
      if (options.diagnostics) {
        slp::println("\nInvoking IPM solver\n");
      }

      status = solve_ipm(options, spy, manifold, x);
    } else if (m_equality_constraints.empty()) {
      if (options.diagnostics) {
        slp::println("\nInvoking Newton solver\n");
      }

      status = solve_newton(options, spy, manifold, x);
    } else {
      if (options.diagnostics) {
        slp::println("\nInvoking SQP solver\n");
      }

      status = solve_sqp(options, spy, manifold, x);
    }

    if (options.diagnostics) {
      slp::println("\nExit: {}", status);
    }

    // Assign the solution to the original Real instances
    RealMatrix<Scalar>{m_decision_variables}.set_value(x);

    return status;
  }

  /// Adds a callback to be called at the beginning of each solver iteration.
  ///
  /// The callback for this overload should return void.
  ///
  /// @param callback The callback.
  template <typename F>
    requires requires(F callback, const IterationInfo<Scalar>& info) {
      { callback(info) } -> std::same_as<void>;
    }
  void add_callback(F&& callback) {
    m_iteration_callbacks.emplace_back(
        [=, callback =
                std::forward<F>(callback)](const IterationInfo<Scalar>& info) {
          callback(info);
          return ControlFlow::CONTINUE;
        });
  }

  /// Adds a callback to be called at the beginning of each solver iteration.
  ///
  /// The callback for this overload should return ControlFlow.
  ///
  /// @param callback The callback. Returning ControlFlow::BREAK from the
  ///     callback stops solving and returns the solution so far.
  template <typename F>
    requires requires(F callback, const IterationInfo<Scalar>& info) {
      { callback(info) } -> std::same_as<ControlFlow>;
    }
  void add_callback(F&& callback) {
    m_iteration_callbacks.emplace_back(std::forward<F>(callback));
  }

  /// Clears the registered callbacks.
  void clear_callbacks() { m_iteration_callbacks.clear(); }

  /// Adds a callback to be called at the beginning of each solver iteration.
  ///
  /// Language bindings should call this in the Problem constructor to register
  /// callbacks that shouldn't be removed by clear_callbacks(). Persistent
  /// callbacks run after non-persistent callbacks.
  ///
  /// @param callback The callback. Returning ControlFlow::BREAK from the
  ///     callback stops solving and returns the solution so far.
  template <typename F>
    requires requires(F callback, const IterationInfo<Scalar>& info) {
      { callback(info) } -> std::same_as<ControlFlow>;
    }
  void add_persistent_callback(F&& callback) {
    m_persistent_iteration_callbacks.emplace_back(std::forward<F>(callback));
  }

 private:
  using DenseVector = Eigen::VectorX<Scalar>;

  // The list of decision variables, which are the root of the problem's
  // expression tree
  gch::small_vector<Real<Scalar>> m_decision_variables;

  // The decision variable blocks constrained to manifolds
  gch::small_vector<ManifoldBlock<Scalar>> m_manifold_blocks;

  // The cost function: f(x)
  std::optional<Real<Scalar>> m_f;

  // The list of equality constraints: cₑ(x) = 0
  gch::small_vector<Real<Scalar>> m_equality_constraints;

  // The list of inequality constraints: cᵢ(x) ≥ 0
  gch::small_vector<Real<Scalar>> m_inequality_constraints;

  // The iteration callbacks
  gch::small_vector<
      std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
      m_iteration_callbacks;
  gch::small_vector<
      std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
      m_persistent_iteration_callbacks;

  ExitStatus solve_newton(const Options<Scalar>& options,
                          [[maybe_unused]] bool spy,
                          const Manifold<Scalar>& manifold, DenseVector& x) {
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using SparseVector = Eigen::SparseVector<Scalar>;
    using enum SetupTimeTracerID;

    RealMatrix<Scalar> x_ad{m_decision_variables};

    // Set up cost function
    Real f = m_f.value_or(Scalar(0));

    int num_decision_variables = m_decision_variables.size();

    gch::small_vector<
        std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks;
    for (const auto& callback : m_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }
    for (const auto& callback : m_persistent_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }

    TimeTracerTree<SetupTimeTracer, newton_setup_time_trace> ad_setup_tracers{
        options.diagnostics};
    ScopedTimeTracer setup_tracer{ad_setup_tracers[SETUP]};

    // Set up gradient autodiff
    PhaseTimeTracer phase_tracer{ad_setup_tracers[GRADIENT]};
    Gradient g{f, x_ad};

    // Set up Lagrangian Hessian autodiff
    phase_tracer.next(ad_setup_tracers[HESSIAN]);
    Hessian<Scalar, Eigen::Lower> H{f, x_ad};

    phase_tracer.stop();
    setup_tracer.stop();

    if (options.diagnostics) {
      print_setup_diagnostics(ad_setup_tracers);
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Sparsity pattern files written when spy flag is set
    std::unique_ptr<Spy<Scalar>> H_spy;

    if (spy) {
      H_spy = std::make_unique<Spy<Scalar>>(
          "H.spy", "Hessian", "Decision variables", "Decision variables",
          manifold.tangent_dimension(num_decision_variables),
          manifold.tangent_dimension(num_decision_variables));
      iteration_callbacks.push_back([&](const IterationInfo<Scalar>& info) {
        H_spy->add(info.H);
        return ControlFlow::CONTINUE;
      });
    }
#endif

    // Automatically scale the cost. The problem scaling procedure is
    // described in more detail in docs/algorithms.md#problem-scaling.
    x_ad.set_value(x);
    const ProblemScaling<Scalar> scaling{g.value()};

    NewtonMatrixCallbacks<Scalar> matrix_callbacks{
        num_decision_variables,
        [&](const DenseVector& x) -> Scalar {
          x_ad.set_value(x);
          return scaling.f * f.value();
        },
        [&](const DenseVector& x) -> SparseVector {
          x_ad.set_value(x);
          return scaling.f * g.value();
        },
        [&](const DenseVector& x) -> SparseMatrix {
          x_ad.set_value(x);
          return scaling.f * H.value();
        },
        scaling,
        manifold};

    // Invoke Newton solver
    return newton<Scalar>(matrix_callbacks, iteration_callbacks, options, x);
  }

  ExitStatus solve_sqp(const Options<Scalar>& options,
                       [[maybe_unused]] bool spy,
                       const Manifold<Scalar>& manifold, DenseVector& x) {
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using SparseVector = Eigen::SparseVector<Scalar>;
    using enum SetupTimeTracerID;

    RealMatrix<Scalar> x_ad{m_decision_variables};

    // Set up cost function
    Real f = m_f.value_or(Scalar(0));

    int num_decision_variables = m_decision_variables.size();
    int num_equality_constraints = m_equality_constraints.size();

    gch::small_vector<
        std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks;
    for (const auto& callback : m_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }
    for (const auto& callback : m_persistent_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }

    RealMatrix<Scalar> c_e_ad{m_equality_constraints};
    RealMatrix<Scalar> y_ad(num_equality_constraints);

    TimeTracerTree<SetupTimeTracer, sqp_setup_time_trace> ad_setup_tracers{
        options.diagnostics};
    ScopedTimeTracer setup_tracer{ad_setup_tracers[SETUP]};

    // Set up gradient autodiff
    PhaseTimeTracer phase_tracer{ad_setup_tracers[GRADIENT]};
    Gradient g{f, x_ad};

    phase_tracer.next(ad_setup_tracers[HESSIAN]);

    // Set up cost part of Lagrangian Hessian autodiff
    PhaseTimeTracer hessian_phase_tracer{ad_setup_tracers[HESSIAN_COST]};
    Hessian<Scalar, Eigen::Lower> H_f{f, x_ad};

    // Set up constraint part of Lagrangian Hessian autodiff
    hessian_phase_tracer.next(ad_setup_tracers[HESSIAN_CONSTRAINTS]);
    Hessian<Scalar, Eigen::Lower> H_c{-y_ad.T() * c_e_ad, x_ad};

    hessian_phase_tracer.stop();

    // Set up equality constraint Jacobian autodiff
    phase_tracer.next(ad_setup_tracers[EQUALITY_JACOBIAN]);
    Jacobian A_e{c_e_ad, x_ad};

    phase_tracer.stop();
    setup_tracer.stop();

    if (options.diagnostics) {
      print_setup_diagnostics(ad_setup_tracers);
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Sparsity pattern files written when spy flag is set
    std::unique_ptr<Spy<Scalar>> H_spy;
    std::unique_ptr<Spy<Scalar>> A_e_spy;

    if (spy) {
      H_spy = std::make_unique<Spy<Scalar>>(
          "H.spy", "Hessian", "Decision variables", "Decision variables",
          manifold.tangent_dimension(num_decision_variables),
          manifold.tangent_dimension(num_decision_variables));
      A_e_spy = std::make_unique<Spy<Scalar>>(
          "A_e.spy", "Equality constraint Jacobian", "Constraints",
          "Decision variables", num_equality_constraints,
          manifold.tangent_dimension(num_decision_variables));
      iteration_callbacks.push_back([&](const IterationInfo<Scalar>& info) {
        H_spy->add(info.H);
        A_e_spy->add(info.A_e);
        return ControlFlow::CONTINUE;
      });
    }
#endif

    // Automatically scale the cost and constraints. The problem scaling
    // procedure is described in more detail in
    // docs/algorithms.md#problem-scaling.
    x_ad.set_value(x);
    const ProblemScaling<Scalar> scaling{g.value(), A_e.value()};

    SQPMatrixCallbacks<Scalar> matrix_callbacks{
        num_decision_variables,
        num_equality_constraints,
        [&](const DenseVector& x) -> Scalar {
          x_ad.set_value(x);
          return scaling.f * f.value();
        },
        [&](const DenseVector& x) -> SparseVector {
          x_ad.set_value(x);
          return scaling.f * g.value();
        },
        [&](const DenseVector& x, const DenseVector& y) -> SparseMatrix {
          x_ad.set_value(x);
          y_ad.set_value(scaling.c_e.cwiseProduct(y));
          return scaling.f * H_f.value() + H_c.value();
        },
        [&](const DenseVector& x, const DenseVector& y) -> SparseMatrix {
          x_ad.set_value(x);
          y_ad.set_value(scaling.c_e.cwiseProduct(y));
          return H_c.value();
        },
        [&](const DenseVector& x) -> DenseVector {
          x_ad.set_value(x);
          return scaling.c_e.cwiseProduct(c_e_ad.value());
        },
        [&](const DenseVector& x) -> SparseMatrix {
          x_ad.set_value(x);
          return scaling.c_e.asDiagonal() * A_e.value();
        },
        scaling,
        manifold};

    // Invoke SQP solver
    return sqp<Scalar>(matrix_callbacks, iteration_callbacks, options, x);
  }

  ExitStatus solve_ipm(const Options<Scalar>& options,
                       [[maybe_unused]] bool spy,
                       const Manifold<Scalar>& manifold, DenseVector& x) {
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using SparseVector = Eigen::SparseVector<Scalar>;
    using enum SetupTimeTracerID;

    RealMatrix<Scalar> x_ad{m_decision_variables};

    // Set up cost function
    Real f = m_f.value_or(Scalar(0));

    int num_decision_variables = m_decision_variables.size();
    int num_equality_constraints = m_equality_constraints.size();
    int num_inequality_constraints = m_inequality_constraints.size();

    gch::small_vector<
        std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks;
    for (const auto& callback : m_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }
    for (const auto& callback : m_persistent_iteration_callbacks) {
      iteration_callbacks.emplace_back(callback);
    }

    RealMatrix<Scalar> c_e_ad{m_equality_constraints};
    RealMatrix<Scalar> c_i_ad{m_inequality_constraints};
    RealMatrix<Scalar> y_ad(num_equality_constraints);
    RealMatrix<Scalar> z_ad(num_inequality_constraints);

    TimeTracerTree<SetupTimeTracer, ipm_setup_time_trace> ad_setup_tracers{
        options.diagnostics};
    ScopedTimeTracer setup_tracer{ad_setup_tracers[SETUP]};

    // Set up gradient autodiff
    PhaseTimeTracer phase_tracer{ad_setup_tracers[GRADIENT]};
    Gradient g{f, x_ad};

    phase_tracer.next(ad_setup_tracers[HESSIAN]);

    // Set up cost part of Lagrangian Hessian autodiff
    PhaseTimeTracer hessian_phase_tracer{ad_setup_tracers[HESSIAN_COST]};
    Hessian<Scalar, Eigen::Lower> H_f{f, x_ad};

    // Set up constraint part of Lagrangian Hessian autodiff
    hessian_phase_tracer.next(ad_setup_tracers[HESSIAN_CONSTRAINTS]);
    Hessian<Scalar, Eigen::Lower> H_c{-y_ad.T() * c_e_ad - z_ad.T() * c_i_ad,
                                      x_ad};

    hessian_phase_tracer.stop();

    // Set up equality constraint Jacobian autodiff
    phase_tracer.next(ad_setup_tracers[EQUALITY_JACOBIAN]);
    Jacobian A_e{c_e_ad, x_ad};

    // Set up inequality constraint Jacobian autodiff
    phase_tracer.next(ad_setup_tracers[INEQUALITY_JACOBIAN]);
    Jacobian A_i{c_i_ad, x_ad};

    phase_tracer.stop();
    setup_tracer.stop();

    if (options.diagnostics) {
      print_setup_diagnostics(ad_setup_tracers);
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Sparsity pattern files written when spy flag is set
    std::unique_ptr<Spy<Scalar>> H_spy;
    std::unique_ptr<Spy<Scalar>> A_e_spy;
    std::unique_ptr<Spy<Scalar>> A_i_spy;

    if (spy) {
      H_spy = std::make_unique<Spy<Scalar>>(
          "H.spy", "Hessian", "Decision variables", "Decision variables",
          manifold.tangent_dimension(num_decision_variables),
          manifold.tangent_dimension(num_decision_variables));
      A_e_spy = std::make_unique<Spy<Scalar>>(
          "A_e.spy", "Equality constraint Jacobian", "Constraints",
          "Decision variables", num_equality_constraints,
          manifold.tangent_dimension(num_decision_variables));
      A_i_spy = std::make_unique<Spy<Scalar>>(
          "A_i.spy", "Inequality constraint Jacobian", "Constraints",
          "Decision variables", num_inequality_constraints,
          manifold.tangent_dimension(num_decision_variables));
      iteration_callbacks.push_back([&](const IterationInfo<Scalar>& info) {
        H_spy->add(info.H);
        A_e_spy->add(info.A_e);
        A_i_spy->add(info.A_i);
        return ControlFlow::CONTINUE;
      });
    }
#endif

    const auto [bound_constraint_mask, bounds, conflicting_bound_indices] =
        get_bounds<Scalar>(m_decision_variables, m_inequality_constraints,
                           A_i.value(), manifold);
    if (!conflicting_bound_indices.empty()) {
      if (options.diagnostics) {
        print_bound_constraint_global_infeasibility_error(
            conflicting_bound_indices);
      }
      return ExitStatus::GLOBALLY_INFEASIBLE;
    }

#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    project_onto_bounds(x, bounds);
#endif

    // Automatically scale the cost and constraints. The problem scaling
    // procedure is described in more detail in
    // docs/algorithms.md#problem-scaling.
    x_ad.set_value(x);
    const ProblemScaling<Scalar> scaling{g.value(), A_e.value(), A_i.value()};

    IPMMatrixCallbacks<Scalar> matrix_callbacks{
        num_decision_variables,
        num_equality_constraints,
        num_inequality_constraints,
        [&](const DenseVector& x) -> Scalar {
          x_ad.set_value(x);
          return scaling.f * f.value();
        },
        [&](const DenseVector& x) -> SparseVector {
          x_ad.set_value(x);
          return scaling.f * g.value();
        },
        [&](const DenseVector& x, const DenseVector& y,
            const DenseVector& z) -> SparseMatrix {
          x_ad.set_value(x);
          y_ad.set_value(scaling.c_e.cwiseProduct(y));
          z_ad.set_value(scaling.c_i.cwiseProduct(z));
          return scaling.f * H_f.value() + H_c.value();
        },
        [&](const DenseVector& x, const DenseVector& y,
            const DenseVector& z) -> SparseMatrix {
          x_ad.set_value(x);
          y_ad.set_value(scaling.c_e.cwiseProduct(y));
          z_ad.set_value(scaling.c_i.cwiseProduct(z));
          return H_c.value();
        },
        [&](const DenseVector& x) -> DenseVector {
          x_ad.set_value(x);
          return scaling.c_e.cwiseProduct(c_e_ad.value());
        },
        [&](const DenseVector& x) -> SparseMatrix {
          x_ad.set_value(x);
          return scaling.c_e.asDiagonal() * A_e.value();
        },
        [&](const DenseVector& x) -> DenseVector {
          x_ad.set_value(x);
          return scaling.c_i.cwiseProduct(c_i_ad.value());
        },
        [&](const DenseVector& x) -> SparseMatrix {
          x_ad.set_value(x);
          return scaling.c_i.asDiagonal() * A_i.value();
        },
        scaling,
        manifold};

    // Invoke interior-point method solver
    return ipm<Scalar>(matrix_callbacks, iteration_callbacks, options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                       bound_constraint_mask,
#endif
                       x);
  }

  void print_exit_conditions(const Options<Scalar>& options) {
    // Print possible exit conditions
    slp::println("User-configured exit conditions:");
    slp::println("  ↳ error below {}", options.tolerance);
    if (!m_iteration_callbacks.empty() ||
        !m_persistent_iteration_callbacks.empty()) {
      slp::println("  ↳ iteration callback requested stop");
    }
    if (std::isfinite(options.max_iterations)) {
      slp::println("  ↳ executed {} iterations", options.max_iterations);
    }
    if (std::isfinite(options.timeout.count())) {
      slp::println("  ↳ {} elapsed", options.timeout);
    }
  }

  void print_problem_analysis() {
    constexpr std::array types{"no",     "constant",  "parameter",
                               "linear", "quadratic", "nonlinear"};

    // Print problem structure
    slp::println("\nProblem structure:");
    slp::println("  ↳ {} cost function",
                 types[std::to_underlying(cost_function_type())]);
    slp::println("  ↳ {} equality constraints",
                 types[std::to_underlying(equality_constraint_type())]);
    slp::println("  ↳ {} inequality constraints",
                 types[std::to_underlying(inequality_constraint_type())]);

    if (m_decision_variables.size() == 1) {
      slp::print("\n1 decision variable\n");
    } else {
      slp::print("\n{} decision variables\n", m_decision_variables.size());
    }

    auto print_constraint_types =
        [](const gch::small_vector<Real<Scalar>>& constraints) {
          std::array<size_t, 6> counts{};
          for (const auto& constraint : constraints) {
            ++counts[std::to_underlying(constraint.type())];
          }
          for (const auto& [count, name] : std::views::zip(
                   counts, std::array{"empty", "constant", "parameter",
                                      "linear", "quadratic", "nonlinear"})) {
            if (count > 0) {
              slp::println("  ↳ {} {}", count, name);
            }
          }
        };

    // Print constraint types
    if (m_equality_constraints.size() == 1) {
      slp::println("1 equality constraint");
    } else {
      slp::println("{} equality constraints", m_equality_constraints.size());
    }
    print_constraint_types(m_equality_constraints);
    if (m_inequality_constraints.size() == 1) {
      slp::println("1 inequality constraint");
    } else {
      slp::println("{} inequality constraints",
                   m_inequality_constraints.size());
    }
    print_constraint_types(m_inequality_constraints);
  }
};

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Problem<double>;

}  // namespace slp
