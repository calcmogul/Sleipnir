// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/empty.hpp"

namespace slp {

/// The kind of formulation issue a lint finding reports.
enum class LintKind : uint8_t {
  /// Decision variables don't appear in the cost function or any constraint.
  UNUSED_DECISION_VARIABLE,
  /// A constraint doesn't depend on any decision variables, so it's either
  /// always satisfied or never satisfied.
  CONSTANT_CONSTRAINT,
  /// An equality constraint keeps decision variables on a sphere, which a
  /// manifold decision variable does without a constraint.
  SPHERE_CONSTRAINT,
  /// Linear constraints are implied by other linear constraints on the same
  /// linear combination of decision variables.
  REDUNDANT_CONSTRAINTS,
  /// Linear constraints on the same linear combination of decision variables
  /// contradict each other, so the problem is infeasible.
  CONFLICTING_CONSTRAINTS,
  /// A lower bound and an upper bound on the same linear combination of
  /// decision variables are equal, so together they're an equality
  /// constraint.
  EQUALITY_AS_INEQUALITIES,
  /// The cost function or a constraint uses a function of decision variables
  /// that isn't differentiable everywhere, like abs() or a norm, in a form
  /// with a differentiable replacement.
  NONSMOOTH_FUNCTION,
  /// Coefficients of the cost function or linear constraints span so many
  /// orders of magnitude that the solver loses precision, and rescaling
  /// decision variables or constraints fixes it.
  BADLY_SCALED,
};

/// A suggested improvement to a problem's formulation from Problem::lint().
///
/// Decision variable indices count decision variables in the order they were
/// created. Constraint indices count scalar constraints in the order they were
/// added, with matrix constraints adding one per element. Messages refer to
/// decision variable i as x[i].
struct LintFinding {
  /// The kind of formulation issue.
  LintKind kind;

  /// The indices of the decision variables involved.
  std::vector<int> decision_variables;

  /// Whether the cost function is involved.
  bool cost = false;

  /// The indices of the equality constraints involved.
  std::vector<int> equality_constraints;

  /// The indices of the inequality constraints involved.
  std::vector<int> inequality_constraints;

  /// A description of the issue and how to fix it.
  std::string message;
};

namespace detail {

/// Joins items into an English list ("a", "a and b", or "a, b, and c").
///
/// @param items The items.
/// @return The list.
inline std::string join_english(std::span<const std::string> items) {
  std::string result;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      result += items.size() == 2 ? " and " : ", ";
      if (items.size() > 2 && i + 1 == items.size()) {
        result += "and ";
      }
    }
    result += items[i];
  }
  return result;
}

/// Describes decision variables as "x[i]", "x[i] and x[j]", and so on.
///
/// @param indices The decision variable indices.
/// @return The description.
inline std::string describe_decision_variables(std::span<const int> indices) {
  std::vector<std::string> items;
  for (int index : indices) {
    items.emplace_back(std::format("x[{}]", index));
  }
  return join_english(items);
}

/// Describes constraints as "equality constraint i", "inequality constraints i
/// and j", and so on.
///
/// @param kind "equality" or "inequality".
/// @param indices The constraint indices.
/// @return The description.
inline std::string describe_constraints(std::string_view kind,
                                        std::span<const int> indices) {
  std::vector<std::string> items;
  for (int index : indices) {
    items.emplace_back(std::format("{}", index));
  }
  return std::format("{} constraint{} {}", kind, indices.size() == 1 ? "" : "s",
                     join_english(items));
}

/// Returns the string with its first letter capitalized.
///
/// @param str The string.
/// @return The capitalized string.
inline std::string capitalize(std::string str) {
  if (!str.empty()) {
    str[0] =
        static_cast<char>(std::toupper(static_cast<unsigned char>(str[0])));
  }
  return str;
}

/// Returns true if two values are equal to within a relative tolerance.
///
/// @tparam Scalar Scalar type.
/// @param a The first value.
/// @param b The second value.
/// @param tolerance The relative tolerance.
/// @return True if the values are equal to within the tolerance.
template <typename Scalar>
bool near(Scalar a, Scalar b, Scalar tolerance) {
  using std::abs;
  return abs(a - b) <= tolerance * std::max({Scalar(1), abs(a), abs(b)});
}

/// Returns true if an expression is a sum of squares of expressions of decision
/// variables, like a vector's squared norm.
///
/// @tparam Scalar Scalar type.
/// @param node The expression.
/// @return True if the expression is a sum of squares.
template <typename Scalar>
bool is_sum_of_squares(const Expression<Scalar>* node) {
  using enum ExpressionType;

  std::array<const Expression<Scalar>*, 2> args{};
  size_t num_args = 0;
  node->visit_args([&](Expression<Scalar>* arg) {
    if (num_args < args.size()) {
      args[num_args] = arg;
    }
    ++num_args;
  });

  const auto name = node->name();
  if (name == "binary plus") {
    return is_sum_of_squares(args[0]) && is_sum_of_squares(args[1]);
  } else if (name == "multiplication") {
    return args[0] == args[1] && args[0]->type() > PARAMETER;
  } else if (name == "pow") {
    return args[0]->type() > PARAMETER && args[1]->is_constant(Scalar(2));
  } else {
    return false;
  }
}

/// Where an expression appears in a problem.
enum class ExpressionRole : uint8_t {
  /// The cost function.
  COST,
  /// An equality constraint.
  EQUALITY_CONSTRAINT,
  /// An inequality constraint.
  INEQUALITY_CONSTRAINT,
};

/// Returns the functions of decision variables in an expression that aren't
/// differentiable everywhere and appear in a form with a differentiable
/// replacement.
///
/// The forms and their replacements are:
///
/// <ul>
///   <li>"abs": abs(a) ≤ b → −b ≤ a ≤ b</li>
///   <li>"max": max(a, b) ≤ c → a ≤ c and b ≤ c</li>
///   <li>"min": min(a, b) ≥ c → a ≥ c and b ≥ c</li>
///   <li>"norm": a norm (sqrt() of a sum of squares or hypot()) compared to a
///     value that doesn't depend on decision variables → the squared norm
///     compared to the value's square</li>
///   <li>"cost norm": a cost function that's a norm → the squared norm, which
///     has the same minimizer</li>
/// </ul>
///
/// @tparam Scalar Scalar type.
/// @param x The expression.
/// @param role Where the expression appears in the problem.
/// @return The forms found, without duplicates.
template <typename Scalar>
std::vector<std::string_view> replaceable_nonsmooth_functions(
    const Real<Scalar>& x, ExpressionRole role) {
  using enum ExpressionType;

  std::vector<std::string_view> forms;
  if (x.expr == nullptr || x.expr->is_leaf) {
    return forms;
  }

  auto args_of = [](const Expression<Scalar>* node) {
    std::vector<const Expression<Scalar>*> args;
    node->visit_args([&](Expression<Scalar>* arg) { args.emplace_back(arg); });
    return args;
  };

  auto depends_on_x = [](const Expression<Scalar>* node) {
    return node->type() > PARAMETER;
  };

  // Returns true if the node is a norm of decision variables, which isn't
  // differentiable where it's 0
  auto is_norm = [&](const Expression<Scalar>* node) {
    const auto name = node->name();
    const auto args = args_of(node);
    if (name == "sqrt") {
      return is_sum_of_squares(args[0]);
    } else if (name == "hypot") {
      return std::ranges::all_of(args, depends_on_x);
    } else {
      return false;
    }
  };

  auto add = [&](std::string_view form) {
    if (std::ranges::find(forms, form) == forms.end()) {
      forms.emplace_back(form);
    }
  };

  const auto root = x.expr.get();

  if (role == ExpressionRole::COST) {
    if (is_norm(root)) {
      add("cost norm");
    }
    return forms;
  }

  // Constraints are lhs − rhs = 0 or lhs − rhs ≥ 0, and pruning turns 0 − rhs
  // into −rhs and lhs − 0 into lhs. Each side is bounded by the other.
  enum class Bound : uint8_t { EQUAL, LOWER, UPPER };
  struct Side {
    const Expression<Scalar>* expression;
    const Expression<Scalar>* other;
    Bound bound;
  };

  const bool is_equality = role == ExpressionRole::EQUALITY_CONSTRAINT;
  std::vector<Side> sides;
  const auto root_args = args_of(root);
  if (root->name() == "binary minus") {
    sides.push_back({root_args[0], root_args[1],
                     is_equality ? Bound::EQUAL : Bound::LOWER});
    sides.push_back({root_args[1], root_args[0],
                     is_equality ? Bound::EQUAL : Bound::UPPER});
  } else if (root->name() == "unary minus") {
    sides.push_back(
        {root_args[0], nullptr, is_equality ? Bound::EQUAL : Bound::UPPER});
  } else {
    sides.push_back({root, nullptr, is_equality ? Bound::EQUAL : Bound::LOWER});
  }

  for (const auto& [expression, other, bound] : sides) {
    if (expression->is_leaf || !depends_on_x(expression)) {
      continue;
    }

    const auto name = expression->name();
    if (name == "abs" && bound == Bound::UPPER) {
      add("abs");
    } else if (name == "max" && bound == Bound::UPPER) {
      add("max");
    } else if (name == "min" && bound == Bound::LOWER) {
      add("min");
    } else if (is_norm(expression) &&
               (other == nullptr || !depends_on_x(other))) {
      add("norm");
    }
  }

  return forms;
}

/// Checks a problem's formulation for issues and suggests improvements.
///
/// See Problem::lint().
///
/// @tparam Scalar Scalar type.
/// @param decision_variables The decision variables.
/// @param manifold_blocks The decision variable blocks constrained to
///     manifolds.
/// @param cost The cost function, if any.
/// @param equality_constraints The equality constraints cₑ(x) = 0.
/// @param inequality_constraints The inequality constraints cᵢ(x) ≥ 0.
/// @return The findings.
template <typename Scalar>
std::vector<LintFinding> lint(
    std::span<const Real<Scalar>> decision_variables,
    std::span<const ManifoldBlock<Scalar>> manifold_blocks,
    const std::optional<Real<Scalar>>& cost,
    std::span<const Real<Scalar>> equality_constraints,
    std::span<const Real<Scalar>> inequality_constraints) {
  using enum ExpressionType;
  using std::abs;
  using std::sqrt;

  const Scalar tolerance = Scalar(1e3) * std::numeric_limits<Scalar>::epsilon();

  std::vector<LintFinding> findings;

  const int num_decision_variables = decision_variables.size();
  const int num_equality_constraints = equality_constraints.size();
  const int num_constraints = num_equality_constraints +
                              static_cast<int>(inequality_constraints.size());

  // Returns constraint i, counting equality constraints first
  auto constraint = [&](int i) -> Real<Scalar> {
    if (i < num_equality_constraints) {
      return equality_constraints[i];
    } else {
      return inequality_constraints[i - num_equality_constraints];
    }
  };

  // Splits constraint indices into equality and inequality constraint indices
  auto split_constraints = [&](std::span<const int> indices) {
    std::pair<std::vector<int>, std::vector<int>> result;
    for (int i : indices) {
      if (i < num_equality_constraints) {
        result.first.emplace_back(i);
      } else {
        result.second.emplace_back(i - num_equality_constraints);
      }
    }
    return result;
  };

  // Describes constraints (counting equality constraints first)
  auto describe = [&](std::span<const int> indices) {
    auto [equality, inequality] = split_constraints(indices);
    std::vector<std::string> items;
    if (!equality.empty()) {
      items.emplace_back(describe_constraints("equality", equality));
    }
    if (!inequality.empty()) {
      items.emplace_back(describe_constraints("inequality", inequality));
    }
    return join_english(items);
  };

  // Describes the cost function (if involved) and constraints
  auto describe_with_cost = [&](bool involves_cost,
                                std::span<const int> indices) {
    std::vector<std::string> items;
    if (involves_cost) {
      items.emplace_back("the cost function");
    }
    if (!indices.empty()) {
      items.emplace_back(describe(indices));
    }
    return join_english(items);
  };

  auto add_finding = [&](LintKind kind, std::vector<int> variables,
                         std::span<const int> constraints, std::string message,
                         bool involves_cost = false) {
    auto [equality, inequality] = split_constraints(constraints);
    std::ranges::sort(equality);
    std::ranges::sort(inequality);
    findings.push_back({.kind = kind,
                        .decision_variables = std::move(variables),
                        .cost = involves_cost,
                        .equality_constraints = std::move(equality),
                        .inequality_constraints = std::move(inequality),
                        .message = capitalize(std::move(message))});
  };

  // Map each decision variable to its manifold block, or -1 if it has none
  std::vector<int> block_of(num_decision_variables, -1);
  for (size_t block = 0; block < manifold_blocks.size(); ++block) {
    const auto& manifold_block = manifold_blocks[block];
    for (int i = 0; i < manifold_block.size(); ++i) {
      block_of[manifold_block.offset() + i] = block;
    }
  }

  // Decision variable values
  Eigen::VectorX<Scalar> x_value{num_decision_variables};
  for (int i = 0; i < num_decision_variables; ++i) {
    x_value[i] = Real<Scalar>{decision_variables[i]}.value();
  }

  // Constraint values
  Eigen::VectorX<Scalar> c_value{num_constraints};
  for (int i = 0; i < num_constraints; ++i) {
    c_value[i] = constraint(i).value();
  }

  // The constraint Jacobian's rows hold each constraint's decision variables
  // and gradient at the current values. Its sparsity pattern is structural, so
  // it keeps decision variables whose partial derivatives are currently zero.
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A;
  if (num_decision_variables > 0 && num_constraints > 0) {
    RealMatrix<Scalar> x{decision_variables};
    RealMatrix<Scalar> c{detail::empty, num_constraints, 1};
    for (int i = 0; i < num_constraints; ++i) {
      c[i] = constraint(i);
    }
    A = Jacobian<Scalar>{c, x}.value();
  } else {
    A.resize(num_constraints, num_decision_variables);
  }

  // Returns constraint i's decision variables
  auto constraint_variables = [&](int i) {
    std::vector<int> variables;
    for (typename decltype(A)::InnerIterator it{A, i}; it; ++it) {
      variables.emplace_back(it.col());
    }
    return variables;
  };

  // The cost gradient at the current values, with a structural sparsity
  // pattern like the constraint Jacobian's
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> g;
  if (cost && num_decision_variables > 0) {
    RealMatrix<Scalar> x{decision_variables};
    g = Jacobian<Scalar>{cost.value(), x}.value();
  } else {
    g.resize(1, num_decision_variables);
  }

  // Unused decision variables
  {
    std::vector<bool> used(num_decision_variables, false);
    for (int k = 0; k < A.outerSize(); ++k) {
      for (typename decltype(A)::InnerIterator it{A, k}; it; ++it) {
        used[it.col()] = true;
      }
    }
    for (typename decltype(g)::InnerIterator it{g, 0}; it; ++it) {
      used[it.col()] = true;
    }

    // The manifold keeps a block's decision variables related, so a block is
    // used if any of its decision variables are
    std::vector<bool> block_used(manifold_blocks.size(), false);
    for (int i = 0; i < num_decision_variables; ++i) {
      if (used[i] && block_of[i] != -1) {
        block_used[block_of[i]] = true;
      }
    }

    std::vector<int> unused;
    for (int i = 0; i < num_decision_variables; ++i) {
      if (!used[i] && (block_of[i] == -1 || !block_used[block_of[i]])) {
        unused.emplace_back(i);
      }
    }

    if (!unused.empty()) {
      auto message = std::format(
          "decision variable{} {} {} appear in the cost function or any "
          "constraint. Remove {}, or use {} in the problem.",
          unused.size() == 1 ? "" : "s", describe_decision_variables(unused),
          unused.size() == 1 ? "doesn't" : "don't",
          unused.size() == 1 ? "it" : "them",
          unused.size() == 1 ? "it" : "them");
      add_finding(LintKind::UNUSED_DECISION_VARIABLE, std::move(unused), {},
                  std::move(message));
    }
  }

  // Constraints without decision variables
  for (int i = 0; i < num_constraints; ++i) {
    const auto type = constraint(i).type();
    if (type > PARAMETER) {
      continue;
    }

    const bool is_equality = i < num_equality_constraints;
    const Scalar value = c_value[i];
    const bool satisfied =
        is_equality ? value == Scalar(0) : value >= Scalar(0);

    std::string verdict;
    if (satisfied) {
      verdict = "it's always satisfied. Remove it.";
    } else {
      verdict = "it can't be satisfied, so the problem is infeasible.";
    }

    const int indices[] = {i};
    add_finding(
        LintKind::CONSTANT_CONSTRAINT, {}, indices,
        std::format("{} doesn't depend on any decision variables. {} is {}, "
                    "so {}",
                    describe(indices),
                    type == PARAMETER
                        ? "With the parameters' current values, its value"
                        : "Its value",
                    value, verdict));
  }

  // Sphere constraints: ±(x₁² + … + xₙ²) + k = 0 with the same sign on every
  // square and no other terms
  {
    struct Sphere {
      int constraint;
      std::vector<int> variables;
      Scalar radius;
    };
    std::vector<Sphere> spheres;

    for (int i = 0; i < num_equality_constraints; ++i) {
      if (equality_constraints[i].type() != QUADRATIC) {
        continue;
      }

      auto variables = constraint_variables(i);
      const int n = variables.size();
      if (n < 2 ||
          std::ranges::any_of(
              variables,
              [&](int variable) { return block_of[variable] != -1; })) {
        continue;
      }

      RealMatrix<Scalar> wrt{detail::empty, n, 1};
      Eigen::VectorX<Scalar> x{n};
      Eigen::VectorX<Scalar> g{n};
      for (int k = 0; k < n; ++k) {
        wrt[k] = decision_variables[variables[k]];
        x[k] = x_value[variables[k]];
        g[k] = A.coeff(i, variables[k]);
      }

      // A quadratic constraint's Hessian is constant. A sphere's is a·I.
      const Eigen::MatrixX<Scalar> H =
          Hessian<Scalar>{equality_constraints[i], wrt}.value().toDense();
      const Scalar a = H(0, 0);
      if (a == Scalar(0) ||
          !H.isApprox(a * Eigen::MatrixX<Scalar>::Identity(n, n), tolerance)) {
        continue;
      }

      // The gradient g = Hx + b has no linear term b
      const Eigen::VectorX<Scalar> Hx = H * x;
      const Eigen::VectorX<Scalar> b = g - Hx;
      if (b.template lpNorm<Eigen::Infinity>() >
          tolerance * std::max({abs(a), g.template lpNorm<Eigen::Infinity>(),
                                Hx.template lpNorm<Eigen::Infinity>()})) {
        continue;
      }

      // c(x) = a/2 ‖x‖² + k, so the radius is √(−2k/a)
      const Scalar k = c_value[i] - b.dot(x) - x.dot(Hx) / Scalar(2);
      const Scalar r_squared = Scalar(-2) * k / a;
      if (r_squared <= tolerance) {
        continue;
      }

      spheres.emplace_back(i, std::move(variables), sqrt(r_squared));
    }

    // Spheres that share decision variables aren't a product of spheres
    std::vector<int> sphere_count(num_decision_variables, 0);
    for (const auto& sphere : spheres) {
      for (int variable : sphere.variables) {
        ++sphere_count[variable];
      }
    }

    for (auto& sphere : spheres) {
      if (std::ranges::any_of(sphere.variables, [&](int variable) {
            return sphere_count[variable] > 1;
          })) {
        continue;
      }

      const int n = sphere.variables.size();
      const int indices[] = {sphere.constraint};

      std::string message;
      if (near(sphere.radius, Scalar(1), tolerance)) {
        message = std::format(
            "{} keeps {} on the unit sphere. Creating them with "
            "unit_vector_decision_variable({}) instead removes the constraint "
            "and keeps every iterate on the sphere.",
            describe(indices), describe_decision_variables(sphere.variables),
            n);
      } else {
        message = std::format(
            "{} keeps {} on a sphere of radius {}. Writing them as {} times "
            "unit_vector_decision_variable({}) instead removes the constraint "
            "and keeps every iterate on the sphere.",
            describe(indices), describe_decision_variables(sphere.variables),
            sphere.radius, sphere.radius, n);
      }
      if (n == 2) {
        message +=
            " If they're a rotation's cosine and sine, use "
            "rotation2_decision_variable() instead.";
      } else if (n == 4) {
        message +=
            " If they're a quaternion, use rotation3_decision_variable() "
            "instead.";
      }

      add_finding(LintKind::SPHERE_CONSTRAINT, std::move(sphere.variables),
                  indices, std::move(message));
    }
  }

  // Linear constraints on the same linear combination of decision variables.
  // Each is normalized to aᵀx = v, aᵀx ≥ v, or aᵀx ≤ v with a's first
  // coefficient equal to 1.
  {
    enum class Relation : uint8_t { EQUAL, GREATER, LESS };

    struct Member {
      int constraint;
      Relation relation;
      Scalar value;
    };

    struct Group {
      std::vector<int> variables;
      std::vector<Scalar> coefficients;
      std::vector<Member> members;
    };

    std::vector<Group> groups;
    std::map<std::vector<int>, std::vector<size_t>> groups_by_variables;

    for (int i = 0; i < num_constraints; ++i) {
      if (constraint(i).type() != LINEAR || A.row(i).nonZeros() == 0) {
        continue;
      }

      std::vector<int> variables;
      std::vector<Scalar> coefficients;
      Scalar offset = c_value[i];
      for (typename decltype(A)::InnerIterator it{A, i}; it; ++it) {
        variables.emplace_back(it.col());
        coefficients.emplace_back(it.value());
        offset -= it.value() * x_value[it.col()];
      }

      // Structurally present partial derivatives can be zero, so normalize by
      // the first nonzero one
      auto leading = std::ranges::find_if(coefficients, [](Scalar coefficient) {
        return coefficient != Scalar(0);
      });
      if (leading == coefficients.end()) {
        continue;
      }
      const Scalar scale = *leading;
      for (auto& coefficient : coefficients) {
        coefficient /= scale;
      }

      Relation relation;
      if (i < num_equality_constraints) {
        relation = Relation::EQUAL;
      } else if (scale > Scalar(0)) {
        relation = Relation::GREATER;
      } else {
        relation = Relation::LESS;
      }
      // Adding zero turns −0 into 0 for messages
      Member member{i, relation, -offset / scale + Scalar(0)};

      auto& candidates = groups_by_variables[variables];
      auto group = std::ranges::find_if(candidates, [&](size_t candidate) {
        return std::ranges::equal(
            groups[candidate].coefficients, coefficients,
            [&](Scalar lhs, Scalar rhs) { return near(lhs, rhs, tolerance); });
      });
      if (group != candidates.end()) {
        groups[*group].members.emplace_back(member);
      } else {
        candidates.emplace_back(groups.size());
        groups.emplace_back(std::move(variables), std::move(coefficients),
                            std::vector{member});
      }
    }

    for (const auto& group : groups) {
      if (group.members.size() < 2) {
        continue;
      }

      // Describe the linear combination as x[i] + 2·x[j] − x[k]
      std::string combination;
      for (size_t k = 0; k < group.variables.size(); ++k) {
        const Scalar coefficient = group.coefficients[k];
        if (coefficient == Scalar(0)) {
          continue;
        }
        if (!combination.empty()) {
          combination += coefficient < Scalar(0) ? " − " : " + ";
        } else if (coefficient < Scalar(0)) {
          combination += "−";
        }
        if (abs(coefficient) != Scalar(1)) {
          combination += std::format("{}·", abs(coefficient));
        }
        combination += std::format("x[{}]", group.variables[k]);
      }

      auto add_group_finding = [&](LintKind kind,
                                   std::span<const int> constraints,
                                   std::string message) {
        add_finding(kind, group.variables, constraints, std::move(message));
      };

      std::vector<const Member*> equal;
      std::vector<const Member*> greater;
      std::vector<const Member*> less;
      for (const auto& member : group.members) {
        switch (member.relation) {
          case Relation::EQUAL:
            equal.emplace_back(&member);
            break;
          case Relation::GREATER:
            greater.emplace_back(&member);
            break;
          case Relation::LESS:
            less.emplace_back(&member);
            break;
        }
      }

      auto indices_of = [](std::span<const Member* const> members) {
        std::vector<int> indices;
        for (const auto& member : members) {
          indices.emplace_back(member->constraint);
        }
        return indices;
      };

      if (!equal.empty()) {
        const Scalar value = equal[0]->value;

        if (std::ranges::any_of(equal, [&](const Member* member) {
              return !near(member->value, value, tolerance);
            })) {
          std::vector<std::string> values;
          for (const auto& member : equal) {
            values.emplace_back(std::format("{}", member->value));
          }
          const auto indices = indices_of(equal);
          add_group_finding(
              LintKind::CONFLICTING_CONSTRAINTS, indices,
              std::format("{} set {} to different values ({}), so the problem "
                          "is infeasible.",
                          describe(indices), combination,
                          join_english(values)));
          continue;
        }

        if (equal.size() > 1) {
          const auto indices = indices_of(equal);
          add_group_finding(
              LintKind::REDUNDANT_CONSTRAINTS, indices,
              std::format("{} set {} = {}. Keep one of them; duplicate "
                          "equality constraints make the constraint Jacobian "
                          "rank-deficient.",
                          describe(indices), combination, value));
        }

        // Inequality constraints that the equality constraint satisfies or
        // violates
        std::vector<int> implied;
        std::vector<int> violated;
        for (const auto& member : greater) {
          (member->value <= value || near(member->value, value, tolerance)
               ? implied
               : violated)
              .emplace_back(member->constraint);
        }
        for (const auto& member : less) {
          (member->value >= value || near(member->value, value, tolerance)
               ? implied
               : violated)
              .emplace_back(member->constraint);
        }
        std::ranges::sort(implied);
        std::ranges::sort(violated);

        const int equality_index[] = {equal[0]->constraint};
        if (!implied.empty()) {
          std::vector<int> indices{equal[0]->constraint};
          indices.insert(indices.end(), implied.begin(), implied.end());
          add_group_finding(
              LintKind::REDUNDANT_CONSTRAINTS, indices,
              std::format("{} sets {} = {}, which satisfies {}. Remove {}.",
                          describe(equality_index), combination, value,
                          describe(implied),
                          implied.size() == 1 ? "it" : "them"));
        }
        if (!violated.empty()) {
          std::vector<int> indices{equal[0]->constraint};
          indices.insert(indices.end(), violated.begin(), violated.end());
          add_group_finding(
              LintKind::CONFLICTING_CONSTRAINTS, indices,
              std::format("{} sets {} = {}, which violates {}, so the problem "
                          "is infeasible.",
                          describe(equality_index), combination, value,
                          describe(violated)));
        }

        continue;
      }

      // The tightest lower and upper bounds
      const Member* lower = greater.empty() ? nullptr
                                            : *std::ranges::max_element(
                                                  greater, {}, &Member::value);
      const Member* upper =
          less.empty() ? nullptr
                       : *std::ranges::min_element(less, {}, &Member::value);

      if (lower != nullptr && upper != nullptr) {
        std::vector<int> indices{lower->constraint, upper->constraint};
        std::ranges::sort(indices);
        if (near(lower->value, upper->value, tolerance)) {
          add_group_finding(
              LintKind::EQUALITY_AS_INEQUALITIES, indices,
              std::format("{} require {} ≥ {} and {} ≤ {}. Replace them with "
                          "the equality constraint {} = {}, which the solver "
                          "handles directly instead of through two inequality "
                          "constraints with no strict interior.",
                          describe(indices), combination, lower->value,
                          combination, upper->value, combination,
                          lower->value));
        } else if (lower->value > upper->value) {
          add_group_finding(
              LintKind::CONFLICTING_CONSTRAINTS, indices,
              std::format("{} require {} ≥ {} and {} ≤ {}, so the problem is "
                          "infeasible.",
                          describe(indices), combination, lower->value,
                          combination, upper->value));
        }
      }

      // Bounds implied by the tightest bound in the same direction
      auto add_implied_bounds = [&](const Member* tightest,
                                    std::span<const Member* const> bounds,
                                    std::string_view relation) {
        std::vector<int> implied;
        for (const auto& member : bounds) {
          if (member != tightest) {
            implied.emplace_back(member->constraint);
          }
        }
        if (implied.empty()) {
          return;
        }

        std::vector<int> indices{tightest->constraint};
        indices.insert(indices.end(), implied.begin(), implied.end());
        const int tightest_index[] = {tightest->constraint};
        add_group_finding(
            LintKind::REDUNDANT_CONSTRAINTS, indices,
            std::format("{} requires {} {} {}, which implies {}. Remove {}.",
                        describe(tightest_index), combination, relation,
                        tightest->value, describe(implied),
                        implied.size() == 1 ? "it" : "them"));
      };
      if (lower != nullptr) {
        add_implied_bounds(lower, greater, "≥");
      }
      if (upper != nullptr) {
        add_implied_bounds(upper, less, "≤");
      }
    }
  }

  // Functions that aren't differentiable everywhere, in forms with a
  // differentiable replacement
  {
    struct Uses {
      std::string_view form;
      bool in_cost = false;
      std::vector<int> constraints;
    };
    std::vector<Uses> uses;

    auto record = [&](const Real<Scalar>& expression, ExpressionRole role,
                      int constraint_index) {
      for (const auto& form :
           replaceable_nonsmooth_functions(expression, role)) {
        auto use = std::ranges::find(uses, form, &Uses::form);
        if (use == uses.end()) {
          uses.push_back({.form = form, .in_cost = false, .constraints = {}});
          use = uses.end() - 1;
        }
        if (role == ExpressionRole::COST) {
          use->in_cost = true;
        } else {
          use->constraints.emplace_back(constraint_index);
        }
      }
    };
    if (cost) {
      record(cost.value(), ExpressionRole::COST, -1);
    }
    for (int i = 0; i < num_constraints; ++i) {
      record(constraint(i),
             i < num_equality_constraints
                 ? ExpressionRole::EQUALITY_CONSTRAINT
                 : ExpressionRole::INEQUALITY_CONSTRAINT,
             i);
    }

    for (auto& use : uses) {
      const auto where = describe_with_cost(use.in_cost, use.constraints);
      const bool plural = use.constraints.size() > 1;
      const auto have = plural ? "have" : "has";
      const auto them = plural ? "them" : "it";

      std::string message;
      if (use.form == "abs") {
        message = std::format(
            "{} {} the form abs(a) ≤ b, where a and b are expressions. "
            "abs(a) isn't differentiable where a = 0. Write {} as −b ≤ a ≤ b "
            "instead.",
            where, have, them);
      } else if (use.form == "max") {
        message = std::format(
            "{} {} the form max(a, b) ≤ c, where a, b, and c are "
            "expressions. max(a, b) isn't differentiable where a = b. Write {} "
            "as a ≤ c and b ≤ c instead.",
            where, have, them);
      } else if (use.form == "min") {
        message = std::format(
            "{} {} the form min(a, b) ≥ c, where a, b, and c are "
            "expressions. min(a, b) isn't differentiable where a = b. Write {} "
            "as a ≥ c and b ≥ c instead.",
            where, have, them);
      } else if (use.form == "norm") {
        message = std::format(
            "{} {} a norm, like distance() or hypot(), to a value that "
            "doesn't depend on decision variables. The norm isn't "
            "differentiable where it's 0. Compare the squared norm, like "
            "squared_distance(), to the value's square instead.",
            where, plural ? "compare" : "compares");
      } else {
        message =
            "the cost function is a norm, like distance() or hypot(), which "
            "isn't differentiable where it's 0. Minimize the squared norm, "
            "like squared_distance(), instead, which has the same minimizer.";
      }

      add_finding(LintKind::NONSMOOTH_FUNCTION, {}, use.constraints,
                  std::move(message), use.in_cost);
    }
  }

  // Badly scaled coefficients. Beyond a ratio of 1/√ε, the solver's linear
  // algebra loses about half its precision. Only constant coefficients are
  // checked (linear constraints' gradients and a linear or quadratic cost's
  // gradient or Hessian), since other partial derivatives depend on where
  // they're evaluated.
  {
    using std::log10;
    using std::pow;
    using std::round;

    const Scalar max_ratio =
        Scalar(1) / sqrt(std::numeric_limits<Scalar>::epsilon());
    const Scalar max_log_ratio = log10(max_ratio);

    // A nonzero constant coefficient as log₁₀ of its magnitude. Group 0 is the
    // cost function, and group k + 1 is constraint k. A linear term's
    // coefficient is in variable i with j = −1, and a quadratic term's is in
    // variables i and j.
    struct Coefficient {
      int group;
      int i;
      int j;
      Scalar log_magnitude;
    };
    std::vector<Coefficient> coefficients;
    const int num_groups = num_constraints + 1;

    if (cost && num_decision_variables > 0) {
      if (cost->type() == LINEAR) {
        for (typename decltype(g)::InnerIterator it{g, 0}; it; ++it) {
          if (it.value() != Scalar(0)) {
            coefficients.push_back(
                {0, static_cast<int>(it.col()), -1, log10(abs(it.value()))});
          }
        }
      } else if (cost->type() == QUADRATIC) {
        RealMatrix<Scalar> x{decision_variables};
        Hessian<Scalar, Eigen::Lower> hessian{cost.value(), x};
        const Eigen::SparseMatrix<Scalar>& H = hessian.value();
        for (int k = 0; k < H.outerSize(); ++k) {
          for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{H, k}; it;
               ++it) {
            if (it.value() != Scalar(0)) {
              coefficients.push_back({0, static_cast<int>(it.row()),
                                      static_cast<int>(it.col()),
                                      log10(abs(it.value()))});
            }
          }
        }
      }
    }
    for (int i = 0; i < num_constraints; ++i) {
      if (constraint(i).type() != LINEAR) {
        continue;
      }
      for (typename decltype(A)::InnerIterator it{A, i}; it; ++it) {
        if (it.value() != Scalar(0)) {
          coefficients.push_back(
              {i + 1, static_cast<int>(it.col()), -1, log10(abs(it.value()))});
        }
      }
    }

    // Returns each group's coefficient spread as log₁₀ of the ratio between
    // its largest and smallest coefficients after substituting
    // x[i] = 10^(−s[i]) y[i]
    auto spreads = [&](std::span<const Scalar> s) {
      std::vector<Scalar> smallest(num_groups,
                                   std::numeric_limits<Scalar>::infinity());
      std::vector<Scalar> largest(num_groups,
                                  -std::numeric_limits<Scalar>::infinity());
      for (const auto& [group, i, j, log_magnitude] : coefficients) {
        const Scalar scaled =
            log_magnitude - s[i] - (j >= 0 ? s[j] : Scalar(0));
        smallest[group] = std::min(smallest[group], scaled);
        largest[group] = std::max(largest[group], scaled);
      }
      std::vector<Scalar> result(num_groups, Scalar(0));
      for (int group = 0; group < num_groups; ++group) {
        if (largest[group] >= smallest[group]) {
          result[group] = largest[group] - smallest[group];
        }
      }
      return result;
    };

    const std::vector<Scalar> unscaled_spreads =
        spreads(std::vector<Scalar>(num_decision_variables, Scalar(0)));
    std::vector<int> badly_scaled_groups;
    for (int group = 0; group < num_groups; ++group) {
      if (unscaled_spreads[group] > max_log_ratio) {
        badly_scaled_groups.emplace_back(group);
      }
    }

    if (!badly_scaled_groups.empty()) {
      // Fit log₁₀ magnitudes as a group scale plus variable scales by
      // alternating least squares
      std::vector<Scalar> c(num_decision_variables, Scalar(0));
      std::vector<Scalar> r(num_groups, Scalar(0));
      std::vector<bool> has_coefficients(num_decision_variables, false);
      for (const auto& coefficient : coefficients) {
        has_coefficients[coefficient.i] = true;
      }
      for (int iteration = 0; iteration < 50; ++iteration) {
        std::vector<Scalar> sum(num_groups, Scalar(0));
        std::vector<int> count(num_groups, 0);
        for (const auto& [group, i, j, log_magnitude] : coefficients) {
          sum[group] += log_magnitude - c[i] - (j >= 0 ? c[j] : Scalar(0));
          ++count[group];
        }
        for (int group = 0; group < num_groups; ++group) {
          if (count[group] > 0) {
            r[group] = sum[group] / Scalar(count[group]);
          }
        }

        std::vector<Scalar> variable_sum(num_decision_variables, Scalar(0));
        std::vector<int> variable_count(num_decision_variables, 0);
        for (const auto& [group, i, j, log_magnitude] : coefficients) {
          const Scalar residual = log_magnitude - r[group];
          if (j < 0) {
            variable_sum[i] += residual;
            ++variable_count[i];
          } else if (i == j) {
            variable_sum[i] += residual / Scalar(2);
            ++variable_count[i];
          } else {
            variable_sum[i] += residual - c[j];
            ++variable_count[i];
            variable_sum[j] += residual - c[i];
            ++variable_count[j];
          }
        }
        for (int i = 0; i < num_decision_variables; ++i) {
          if (variable_count[i] > 0) {
            c[i] = variable_sum[i] / Scalar(variable_count[i]);
          }
        }
      }

      // Variable scales are only determined up to a common offset. Try each
      // distinct variable scale as the offset, and keep the rescaling that
      // fixes every group with the fewest substitutions, then with
      // coefficients closest to 1.
      std::vector<Scalar> offsets;
      for (int i = 0; i < num_decision_variables; ++i) {
        if (has_coefficients[i] &&
            std::ranges::none_of(offsets, [&](Scalar offset) {
              return abs(offset - c[i]) < Scalar(0.5);
            })) {
          offsets.emplace_back(c[i]);
        }
      }

      std::optional<std::vector<Scalar>> best;
      int best_count = 0;
      Scalar best_deviation(0);
      for (const Scalar offset : offsets) {
        std::vector<Scalar> s(num_decision_variables, Scalar(0));
        int substitutions = 0;
        for (int i = 0; i < num_decision_variables; ++i) {
          // Only suggest substitutions of at least 100×
          if (has_coefficients[i] && abs(c[i] - offset) >= Scalar(2)) {
            s[i] = round(c[i] - offset);
            ++substitutions;
          }
        }

        const auto scaled_spreads = spreads(s);
        if (std::ranges::any_of(scaled_spreads, [&](Scalar spread) {
              return spread > max_log_ratio;
            })) {
          continue;
        }

        Scalar deviation(0);
        for (const auto& [group, i, j, log_magnitude] : coefficients) {
          deviation = std::max(deviation, abs(log_magnitude - s[i] -
                                              (j >= 0 ? s[j] : Scalar(0))));
        }

        if (!best || substitutions < best_count ||
            (substitutions == best_count && deviation < best_deviation)) {
          best = std::move(s);
          best_count = substitutions;
          best_deviation = deviation;
        }
      }

      if (best && best_count > 0) {
        const auto& s = best.value();
        const auto scaled_spreads = spreads(s);

        Scalar before(0);
        Scalar after(0);
        bool involves_cost = false;
        std::vector<int> badly_scaled_constraints;
        for (int group : badly_scaled_groups) {
          before = std::max(before, unscaled_spreads[group]);
          after = std::max(after, scaled_spreads[group]);
          if (group == 0) {
            involves_cost = true;
          } else {
            badly_scaled_constraints.emplace_back(group - 1);
          }
        }

        std::vector<int> variables;
        std::vector<std::string> substitutions;
        for (int i = 0; i < num_decision_variables; ++i) {
          if (s[i] != Scalar(0)) {
            variables.emplace_back(i);
            substitutions.emplace_back(std::format("x[{}] = {:.0e}·y[{}]", i,
                                                   pow(Scalar(10), -s[i]), i));
          }
        }

        add_finding(
            LintKind::BADLY_SCALED, variables, badly_scaled_constraints,
            std::format(
                "coefficients in {} span a ratio of up to {:.1e}, "
                "which costs the solver about half its precision. "
                "Substituting {} for {}, like by changing units, "
                "reduces it to {:.1e}.",
                describe_with_cost(involves_cost, badly_scaled_constraints),
                pow(Scalar(10), before), join_english(substitutions),
                variables.size() == 1
                    ? std::format("a new decision variable y[{}]", variables[0])
                    : std::string{"new decision variables y"},
                pow(Scalar(10), after)),
            involves_cost);
      }
    }

    // Linear constraint gradients much smaller than the largest one, after the
    // solver scales down gradients larger than 100
    constexpr Scalar max_scaled_norm(100);
    std::vector<std::pair<int, Scalar>> norms;
    Scalar largest_norm(0);
    int largest_constraint = -1;
    for (int i = 0; i < num_constraints; ++i) {
      if (constraint(i).type() != LINEAR) {
        continue;
      }
      Scalar norm(0);
      for (typename decltype(A)::InnerIterator it{A, i}; it; ++it) {
        norm = std::max(norm, abs(it.value()));
      }
      if (norm == Scalar(0)) {
        continue;
      }
      norm = std::min(norm, max_scaled_norm);
      norms.emplace_back(i, norm);
      if (norm > largest_norm) {
        largest_norm = norm;
        largest_constraint = i;
      }
    }

    std::vector<int> small_constraints;
    int smallest_constraint = -1;
    Scalar smallest_norm = std::numeric_limits<Scalar>::infinity();
    for (const auto& [i, norm] : norms) {
      if (largest_norm / norm > max_ratio) {
        small_constraints.emplace_back(i);
        if (norm < smallest_norm) {
          smallest_norm = norm;
          smallest_constraint = i;
        }
      }
    }

    if (!small_constraints.empty()) {
      const int largest[] = {largest_constraint};
      const int smallest[] = {smallest_constraint};
      const Scalar factor = pow(Scalar(10), round(-log10(smallest_norm)));

      std::string message;
      if (small_constraints.size() == 1) {
        message = std::format(
            "{} has a gradient {:.1e} times smaller than {}'s, even after the "
            "solver scales down gradients larger than {}. Multiply it by "
            "{:.0e} so its gradient is closer to 1.",
            describe(smallest), largest_norm / smallest_norm, describe(largest),
            max_scaled_norm, factor);
      } else {
        message = std::format(
            "{} have gradients up to {:.1e} times smaller than {}'s, even "
            "after the solver scales down gradients larger than {}. Multiply "
            "them by constants so their gradients are closer to 1, like {} by "
            "{:.0e}.",
            describe(small_constraints), largest_norm / smallest_norm,
            describe(largest), max_scaled_norm, describe(smallest), factor);
      }

      add_finding(LintKind::BADLY_SCALED, {}, small_constraints,
                  std::move(message));
    }
  }

  return findings;
}

}  // namespace detail

}  // namespace slp

/// Formatter for LintFinding.
template <>
struct std::formatter<slp::LintFinding> {
  /// Parses format string.
  ///
  /// @param ctx Format parse context.
  /// @return Format parse context iterator.
  constexpr auto parse(std::format_parse_context& ctx) {
    return m_underlying.parse(ctx);
  }

  /// Formats LintFinding as its message.
  ///
  /// @tparam FmtContext Format context type.
  /// @param finding Lint finding.
  /// @param ctx Format context.
  /// @return Format context iterator.
  template <typename FmtContext>
  auto format(const slp::LintFinding& finding, FmtContext& ctx) const {
    return m_underlying.format(finding.message, ctx);
  }

 private:
  std::formatter<std::string_view> m_underlying;
};
