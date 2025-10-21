// Copyright (c) Sleipnir contributors

#pragma once

#include <span>
#include <vector>

#include <sleipnir/optimization/problem.hpp>

/**
 * This class computes the optimal current allocation for a list of subsystems
 * given a list of their desired currents and current tolerances that determine
 * which subsystem gets less current if the current budget is exceeded.
 * Subsystems with a smaller tolerance are given higher priority.
 */
class CurrentManager {
 public:
  /**
   * Constructs a CurrentManager.
   *
   * @param current_tolerances The relative current tolerance of each subsystem.
   * @param max_current The current budget to allocate between subsystems.
   */
  CurrentManager(std::span<const double> current_tolerances,
                 double max_current);

  /**
   * Returns the optimal current allocation for a list of subsystems given a
   * list of their desired currents and current tolerances that determine which
   * subsystem gets less current if the current budget is exceeded. Subsystems
   * with a smaller tolerance are given higher priority.
   *
   * @param desired_currents The desired current for each subsystem.
   * @throws std::runtime_error if the number of desired currents doesn't equal
   *         the number of tolerances passed in the constructor.
   */
  std::vector<double> calculate(std::span<const double> desired_currents);

 private:
  slp::Problem<double> m_problem;
  slp::VariableMatrix<double> m_desired_currents;
  slp::VariableMatrix<double> m_allocated_currents;
};
