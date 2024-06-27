// Copyright (c) Sleipnir contributors

#pragma once

#include <memory>

#include <rust/cxx.h>

#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnirrust {

struct HolonomicTrajectory;
struct Pose2d;
struct SwerveDrivetrain;

class Variable {
 public:
  Variable() = default;

 private:
  sleipnir::Variable variable;
};

std::unique_ptr<Variable> new_variable();

}  // namespace sleipnirrust
