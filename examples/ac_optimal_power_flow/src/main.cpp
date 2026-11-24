// Copyright (c) Sleipnir contributors

// Finds the cheapest generator dispatch for the WSCC 9-bus system (MATPOWER's
// case9) subject to the AC power flow equations.
//
// Bus voltages V are complex decision variables in rectangular form. The power
// injected at each bus is S = V ∘ conj(Y V), where Y is the bus admittance
// matrix, and it must equal generation minus load. Voltage magnitudes,
// generator outputs, and line apparent power flows are limited. The cost is
// each generator's quadratic fuel cost.
//
// Quantities are per unit on a 100 MVA base except where noted.
//
// The data is from MATPOWER's case9:
// https://github.com/MATPOWER/matpower/blob/master/data/case9.m
//
// The problem is the standard AC OPF in section 6.1 of the MATPOWER User's
// Manual (version 8.1), with Cartesian voltages as in section 6.1.1. Branches
// use the π model in section 3.2 with no transformers, and the network
// equations are in section 3.6:
// https://matpower.org/docs/MATPOWER-manual.pdf
//
// MATPOWER's runopf('case9') solves the same problem.

#include <complex>
#include <numbers>
#include <print>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/optimization/problem.hpp>

struct Bus {
  // Load in MW and MVAr
  double P_load;
  double Q_load;
};

struct Generator {
  int bus;
  // Real power limits in MW
  double P_min;
  double P_max;
  // Reactive power limits in MVAr
  double Q_min;
  double Q_max;
  // Cost c₂P² + c₁P + c₀ in $/h with P in MW
  double c2;
  double c1;
  double c0;
};

struct Branch {
  int from;
  int to;
  // Series resistance and reactance
  double r;
  double x;
  // Total line charging susceptance
  double b;
  // Apparent power limit in MVA
  double rating;
};

int main() {
  constexpr double base_MVA = 100.0;

  std::vector<Bus> buses{{0, 0}, {0, 0},    {0, 0}, {0, 0},   {90, 30},
                         {0, 0}, {100, 35}, {0, 0}, {125, 50}};
  std::vector<Generator> generators{
      {0, 10, 250, -300, 300, 0.11, 5, 150},
      {1, 10, 300, -300, 300, 0.085, 1.2, 600},
      {2, 10, 270, -300, 300, 0.1225, 1, 335},
  };
  std::vector<Branch> branches{
      {0, 3, 0, 0.0576, 0, 250},          {3, 4, 0.017, 0.092, 0.158, 250},
      {4, 5, 0.039, 0.17, 0.358, 150},    {2, 5, 0, 0.0586, 0, 300},
      {5, 6, 0.0119, 0.1008, 0.209, 150}, {6, 7, 0.0085, 0.072, 0.149, 250},
      {7, 1, 0, 0.0625, 0, 250},          {7, 8, 0.032, 0.161, 0.306, 250},
      {8, 3, 0.01, 0.085, 0.176, 250},
  };

  const int n_bus = buses.size();
  const int n_gen = generators.size();
  const int n_branch = branches.size();

  // Each branch is a π model: a series admittance y between its buses and half
  // the line charging to ground at each end. The currents into branch k at its
  // ends are
  //
  //   [I_from,k]   [y + jb/2     −y   ] [V_f]
  //   [I_to,k  ] = [  −y      y + jb/2] [V_t]
  //
  // C_from and C_to select each branch's from and to bus voltages. The current
  // into the branch at each end is I_from = Y_from V and I_to = Y_to V, and the
  // bus admittance matrix Y sums the currents each branch draws from each bus,
  // so I = Y V.
  Eigen::SparseMatrix<double> C_from{n_branch, n_bus};
  Eigen::SparseMatrix<double> C_to{n_branch, n_bus};
  Eigen::VectorXcd y_series{n_branch};
  Eigen::VectorXcd y_shunt{n_branch};
  for (int k = 0; k < n_branch; ++k) {
    const auto& [f, t, r, x, b, rating] = branches[k];
    C_from.insert(k, f) = 1.0;
    C_to.insert(k, t) = 1.0;
    y_series[k] = 1.0 / std::complex<double>{r, x};
    y_shunt[k] = std::complex<double>{0.0, b / 2.0};
  }

  Eigen::SparseMatrix<std::complex<double>> Y_from =
      (y_series + y_shunt).asDiagonal() * C_from - y_series.asDiagonal() * C_to;
  Eigen::SparseMatrix<std::complex<double>> Y_to =
      (-y_series).asDiagonal() * C_from +
      (y_series + y_shunt).asDiagonal() * C_to;
  Eigen::SparseMatrix<std::complex<double>> Y =
      C_from.transpose() * Y_from + C_to.transpose() * Y_to;

  // Maps generator outputs to the buses they inject into
  Eigen::SparseMatrix<double> C_gen{n_bus, n_gen};
  for (int g = 0; g < n_gen; ++g) {
    C_gen.insert(generators[g].bus, g) = 1.0;
  }

  Eigen::VectorXcd S_load{n_bus};
  for (int i = 0; i < n_bus; ++i) {
    S_load[i] =
        std::complex<double>{buses[i].P_load, buses[i].Q_load} / base_MVA;
  }

  slp::Problem<double> problem;

  // Bus voltages start at 1∠0 p.u., and generator outputs start at zero
  auto V = problem.complex_decision_variable(n_bus);
  V.set_value(Eigen::VectorXd::Ones(n_bus));
  auto S_gen = problem.complex_decision_variable(n_gen);

  // Generator fuel cost
  slp::Real cost = 0.0;
  for (int g = 0; g < n_gen; ++g) {
    const auto& gen = generators[g];
    auto P = S_gen[g].real() * base_MVA;
    cost += gen.c2 * P * P + gen.c1 * P + gen.c0;
  }
  problem.minimize(cost);

  // Bus 1's voltage angle is the zero reference
  problem.subject_to(V[0].real() >= 0);
  problem.subject_to(V[0].imag() == 0);

  // Power balance at each bus
  problem.subject_to(V.cwise_product((Y * V).conj()) == C_gen * S_gen - S_load);

  // Voltage magnitudes are within 0.9 to 1.1 p.u.
  auto V_abs2 = V.abs2();
  problem.subject_to(V_abs2 >= 0.9 * 0.9);
  problem.subject_to(V_abs2 <= 1.1 * 1.1);

  // Generator output limits
  for (int g = 0; g < n_gen; ++g) {
    const auto& gen = generators[g];
    problem.subject_to(slp::bounds(gen.P_min / base_MVA, S_gen[g].real(),
                                   gen.P_max / base_MVA));
    problem.subject_to(slp::bounds(gen.Q_min / base_MVA, S_gen[g].imag(),
                                   gen.Q_max / base_MVA));
  }

  // Apparent power flowing into each end of each branch is within its rating
  auto S_from = (C_from * V).cwise_product((Y_from * V).conj());
  auto S_to = (C_to * V).cwise_product((Y_to * V).conj());
  for (int k = 0; k < n_branch; ++k) {
    double rating = branches[k].rating / base_MVA;
    problem.subject_to(S_from[k].abs2() <= rating * rating);
    problem.subject_to(S_to[k].abs2() <= rating * rating);
  }

  problem.solve({.diagnostics = true});

  std::println("cost = {:.2f} $/h", cost.value());
  for (int g = 0; g < n_gen; ++g) {
    auto s = S_gen[g].value() * base_MVA;
    std::println("generator {}: P = {:.2f} MW, Q = {:.2f} MVAr", g + 1,
                 s.real(), s.imag());
  }
  for (int i = 0; i < n_bus; ++i) {
    auto v = V[i];
    std::println("bus {}: |V| = {:.4f} p.u., ∠V = {:.3f}°", i + 1,
                 v.abs().value(), v.arg().value() * 180.0 / std::numbers::pi);
  }
}
