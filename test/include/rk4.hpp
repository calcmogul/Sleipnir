// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

/**
 * Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.
 *
 * @param f  The function to integrate. It must take two arguments x and u.
 * @param x  The initial value of x.
 * @param u  The value u held constant over the integration period.
 * @param dt The time over which to integrate.
 */
template <typename Scalar, typename F, typename T, typename U>
T rk4(F&& f, T x, U u, std::chrono::duration<Scalar> dt) {
  const auto h = dt.count();

  T k1 = f(x, u);
  T k2 = f(x + h * Scalar(0.5) * k1, u);
  T k3 = f(x + h * Scalar(0.5) * k2, u);
  T k4 = f(x + h * k3, u);

  return x + h / Scalar(6.0) * (k1 + Scalar(2.0) * k2 + Scalar(2.0) * k3 + k4);
}
