// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

#include <sleipnir/optimization/problem.hpp>

/**
 * Creates a flywheel quadratic optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
slp::Problem<double> flywheel_sleipnir(std::chrono::duration<double> dt, int N);

/**
 * Creates a cart-pole nonlinear optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
slp::Problem<double> cart_pole_sleipnir(std::chrono::duration<double> dt,
                                        int N);
