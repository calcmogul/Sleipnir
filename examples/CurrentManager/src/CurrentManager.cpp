// Copyright (c) Sleipnir contributors

#include "CurrentManager.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

CurrentManager::CurrentManager(std::span<const double> currentTolerances,
                               double maxCurrent)
    : m_desiredCurrents{static_cast<int>(currentTolerances.size()), 1},
      m_allocatedCurrents{
          m_problem.DecisionVariable(currentTolerances.size())} {
  // Ensure m_desiredCurrents contains initialized Variables
  for (int row = 0; row < m_desiredCurrents.Rows(); ++row) {
    // Don't initialize to 0 or 1, because those will get folded by Sleipnir
    m_desiredCurrents[row] = std::numeric_limits<double>::infinity();
  }

  sleipnir::Variable J = 0.0;
  sleipnir::Variable currentSum = 0.0;
  for (size_t i = 0; i < currentTolerances.size(); ++i) {
    // The weight is 1/tolᵢ² where tolᵢ is the tolerance between the desired
    // and allocated current for subsystem i
    auto error = m_desiredCurrents[i] - m_allocatedCurrents[i];
    J += error * error / (currentTolerances[i] * currentTolerances[i]);

    currentSum += m_allocatedCurrents[i];

    // Currents must be nonnegative
    m_problem.SubjectTo(m_allocatedCurrents[i] >= 0.0);
  }
  m_problem.Minimize(J);

  // Keep total current below maximum
  m_problem.SubjectTo(currentSum <= maxCurrent);
}

std::vector<double> CurrentManager::Calculate(
    std::span<const double> desiredCurrents) {
  if (m_desiredCurrents.Rows() != static_cast<int>(desiredCurrents.size())) {
    throw std::runtime_error(
        "Number of desired currents must equal the number of tolerances "
        "passed in the constructor.");
  }

  for (size_t i = 0; i < desiredCurrents.size(); ++i) {
    m_desiredCurrents[i].SetValue(desiredCurrents[i]);
  }

  m_problem.Solve();

  std::vector<double> result;
  for (size_t i = 0; i < desiredCurrents.size(); ++i) {
    result.emplace_back(std::max(m_allocatedCurrents.Value(i), 0.0));
  }

  return result;
}
