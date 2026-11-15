# Sleipnir for Rust

Rust bindings for [Sleipnir](https://github.com/SleipnirGroup/Sleipnir), a
reverse mode autodiff library, interior-point method, and NLP solver DSL. The
DSL automatically chooses the best solver based on the problem structure.

```rust
use sleipnir::{Options, Problem, SolveStatus, constrain};

let problem: Problem = Problem::new();

let x = problem.decision_variable();
let y = problem.decision_variable();

problem.minimize(x * x + (y - 3.0) * (y - 3.0));
problem.subject_to(constrain!(x + y == 1.0));
problem.subject_to(constrain!(-12.0 <= x <= 12.0));

assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
println!("x = {}, y = {}", x.value(), y.value());
```

## Features

* `Problem` with decision variables, symmetric matrices of them, and costs to
  minimize or maximize
* Parameters for re-solving the same problem with new data, without
  rebuilding it
* `constrain!` for equality and (chained) inequality constraints between
  scalars, `Variable`s, `VariableMatrix`es, views, nested arrays, and
  nalgebra matrices
* `VariableMatrix` with views (`VariableBlock`), strided slices, factories,
  `solve()`, and operators mixing in nested arrays and nalgebra matrices
* `Ocp` for optimal control problems with direct transcription, direct
  collocation, or single shooting
* `Gradient`, `Jacobian`, and `Hessian` returning nalgebra-sparse matrices
* Iteration callbacks with the solver's iterates, gradient, Hessian, and
  constraint Jacobians
* `multistart` for solving from several initial guesses in parallel
* `f64` and `f32` problems

## Example: optimal control

```rust
use std::time::Duration;

use sleipnir::{Ocp, OcpConfig, Options, Problem};

// Flywheel, dx/dt = −x + u, driven from rest toward 10 rad/s with |u| ≤ 12 V
let problem: Problem = Problem::new();
let ocp = Ocp::new(
    &problem,
    OcpConfig::new(1, 1, Duration::from_millis(5), 1000),
    |x, u| -x + u,
);
ocp.constrain_initial_state(0.0);
ocp.set_lower_input_bound(-12.0);
ocp.set_upper_input_bound(12.0);

let error = ocp.x().map(|x| x - 10.0);
problem.minimize(error.dot(&error));
problem.solve(&Options::default()).unwrap();
```

More examples are in [examples/](examples).

## Building

The crate builds the C++ library from source, which requires CMake 3.28 or
newer and one of the C++ library's supported toolchains: Visual Studio 2022
17.13 on Windows, GCC 14 on Linux, or Apple Clang 16 from Xcode 16.2 on
macOS.

CMake downloads Eigen and small_vector during the build. To build offline,
point it at existing checkouts:

```sh
FETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen \
FETCHCONTENT_SOURCE_DIR_SMALL_VECTOR=/path/to/small_vector \
cargo build
```

Inside the Sleipnir repository, the crate uses the repository's C++ sources.
Set `SLEIPNIR_SOURCE_DIR` to build against another checkout. Packages for
crates.io carry a copy of the sources made by `vendor_cpp.py`.
