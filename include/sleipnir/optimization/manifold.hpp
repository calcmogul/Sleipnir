// Copyright (c) Sleipnir contributors

#pragma once

namespace slp {

/// Tag type for decision variables on the unit sphere.
///
/// Problem::decision_variable(sphere, n) returns a UnitVector in ℝⁿ.
struct sphere_t {
  /// Constructs the tag.
  explicit sphere_t() = default;
};

/// Tag for decision variables on the unit sphere.
inline constexpr sphere_t sphere{};

/// Tag type for decision variables in the rotation group SO(2).
///
/// Problem::decision_variable(so2) returns a Rotation2.
struct so2_t {
  /// Constructs the tag.
  explicit so2_t() = default;
};

/// Tag for decision variables in the rotation group SO(2).
inline constexpr so2_t so2{};

/// Tag type for decision variables in the rotation group SO(3).
///
/// Problem::decision_variable(so3) returns a Rotation3.
struct so3_t {
  /// Constructs the tag.
  explicit so3_t() = default;
};

/// Tag for decision variables in the rotation group SO(3).
inline constexpr so3_t so3{};

/// Tag type for decision variables in the rigid transform group SE(2).
///
/// Problem::decision_variable(se2) returns a Pose2.
struct se2_t {
  /// Constructs the tag.
  explicit se2_t() = default;
};

/// Tag for decision variables in the rigid transform group SE(2).
inline constexpr se2_t se2{};

/// Tag type for decision variables in the rigid transform group SE(3).
///
/// Problem::decision_variable(se3) returns a Pose3.
struct se3_t {
  /// Constructs the tag.
  explicit se3_t() = default;
};

/// Tag for decision variables in the rigid transform group SE(3).
inline constexpr se3_t se3{};

}  // namespace slp
