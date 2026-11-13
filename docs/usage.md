# Usage

## Output

This section documents Sleipnir's diagnostic output when the `diagnostics` option is set to true. We'll show the diagnostics for the following problem:
```
       max xy
       x,y
subject to x + 3y = 36
```
```python
from sleipnir.optimization import Problem

problem = Problem()

x, y = problem.decision_variable(2)

problem.maximize(x * y)
problem.subject_to(x + 3 * y == 36)

problem.solve(diagnostics=True)
```

### Exit conditions

First, Sleipnir prints the user-configured exit conditions.
```
User-configured exit conditions:
  ↳ error below 1e-08
  ↳ iteration callback requested stop
  ↳ executed 5000 iterations
```

The user-configurable exit conditions include the error tolerance, maximum iterations, and timeout passed to the `solve()` call; and iteration callbacks added to the `Problem` returning `true`.

### Problem size and structure

Then, Sleipnir prints the problem's size and structure.
```
Problem structure:
  ↳ quadratic cost function
  ↳ linear equality constraints
  ↳ no inequality constraints

2 decision variables
1 equality constraint
  ↳ 1 linear
0 inequality constraints
```

Then, Sleipnir prints the solver selected based on that information:
```
Invoking SQP solver
```

Available solvers include:

* No-op for trivial problems
* Newton for unconstrained problems
* Sequential Quadratic Programming (SQP) for equality-constrained problems
* Interior-point method (IPM) for inequality-constrained problems

### Setup time trace

Then, Sleipnir prints a time trace of its autodiff setup.
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     time trace           percentage     duration ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│setup                 100.00%▕█████████▏     0.011│
│↳ ∇f(x)                 0.00%▕         ▏     0.000│
│↳ ∇²ₓₓL                81.82%▕███████▎ ▏     0.009│
│  ↳ ∇²ₓₓL_f            36.36%▕███▎     ▏     0.004│
│  ↳ ∇²ₓₓL_c            36.36%▕███▎     ▏     0.004│
│↳ ∂cₑ/∂x                9.09%▕▊        ▏     0.001│
└──────────────────────────────────────────────────┘
```

The headings are defined as follows:
<table>
  <tr>
    <th>Heading</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>time trace</td>
    <td>Tree of setup steps</td>
  </tr>
  <tr>
    <td>percentage</td>
    <td>Percentage of setup time</td>
  </tr>
  <tr>
    <td>duration</td>
    <td>Duration of setup step in milliseconds</td>
  </tr>
</table>

### Iterations

The solver prints a row of iterate diagnostics before taking any steps and after taking each step.
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃iter   duration    error       cost      infeas.   complem.    μ       δ     γ    |p_pr|   |p_du|     α       α_z    ↩ ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│init             3.6000e+01 -0.0000e+00 3.6000e+01 0.00e+00 0.00e+00                                                   │
│   0       0.016 1.7998e-03 -1.0800e+02 6.0167e-10 0.00e+00 0.00e+00 10⁻⁴  10⁻¹⁰ 1.80e+01 6.00e+00 1.00e+00 1.00e+00  0│
│   1       0.003 1.1997e-07 -1.0800e+02 9.9476e-14 0.00e+00 0.00e+00 10⁻⁴  10⁻¹⁰ 2.40e-03 1.00e-03 1.00e+00 1.00e+00  0│
│   2       0.002 4.9987e-12 -1.0800e+02 0.0000e+00 0.00e+00 0.00e+00 10⁻⁴  10⁻¹⁰ 2.00e-07 5.33e-08 1.00e+00 1.00e+00  0│
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The headings are defined as follows:
<table>
  <tr>
    <th>Heading</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>iter</td>
    <td>Iteration number with one of the suffixes below denoting the iteration type
      <ul>
        <li>` ` = normal</li>
        <li>`s` = second-order correction</li>
        <li>`r` = feasibility restoration</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>duration</td>
    <td>Duration of iteration in milliseconds</td>
  </tr>
  <tr>
    <td>error</td>
    <td>Infinity norm of scaled KKT condition errors</td>
  </tr>
  <tr>
    <td>cost</td>
    <td>Cost function value at current iterate</td>
  </tr>
  <tr>
    <td>infeas.</td>
    <td>Constraint infeasibility at current iterate</td>
  </tr>
  <tr>
    <td>complem.</td>
    <td>Complementary slackness at current iterate (sᵀz)</td>
  </tr>
  <tr>
    <td>μ</td>
    <td>Barrier parameter</td>
  </tr>
  <tr>
    <td>δ</td>
    <td>Iteration matrix Hessian regularization factor (δ in lhs + [δI, 0; 0, -γI])</td>
  </tr>
  <tr>
    <td>γ</td>
    <td>Iteration matrix constraint Jacobian regularization factor (γ in lhs + [δI, 0; 0, -γI])</td>
  </tr>
  <tr>
    <td>|p_pr|</td>
    <td>Infinity norm of full primal step</td>
  </tr>
  <tr>
    <td>|p_du|</td>
    <td>Infinity norm of full dual step</td>
  </tr>
  <tr>
    <td>α</td>
    <td>Step size α ∈ [0, 1] that scales down the full step for the decision variables, inequality constraint slack variables, and equality constraint dual variables</td>
  </tr>
  <tr>
    <td>α_z</td>
    <td>Step size α_z ∈ [0, 1] that scales down the full step for the inequality constraint dual variables</td>
  </tr>
  <tr>
    <td>↩</td>
    <td>Number of line search backtracks</td>
  </tr>
</table>

### Solver time trace

At the end of the solve, the solver prints a time trace of itself.
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     time trace           percentage       total      each    runs┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│solver                100.00%▕█████████▏      0.055     0.055    1│
│↳ setup                 9.09%▕▊        ▏      0.005     0.005    1│
│↳ iteration            38.18%▕███▍     ▏      0.021     0.007    3│
│  ↳ callbacks           0.00%▕         ▏      0.000     0.000    3│
│  ↳ KKT matrix build    3.64%▕▎        ▏      0.002     0.000    3│
│  ↳ KKT matrix decomp   7.27%▕▋        ▏      0.004     0.001    3│
│  ↳ KKT system solve    1.82%▕▏        ▏      0.001     0.000    3│
│  ↳ line search        14.55%▕█▎       ▏      0.008     0.002    3│
│    ↳ SOC               0.00%▕         ▏      0.000     0.000    0│
│  ↳ feas. restoration   0.00%▕         ▏      0.000     0.000    0│
│  ↳ f(x)                0.00%▕         ▏      0.000     0.000    4│
│  ↳ ∇f(x)               5.45%▕▍        ▏      0.003     0.000    4│
│  ↳ ∇²ₓₓL               1.82%▕▏        ▏      0.001     0.000    4│
│  ↳ ∇²ₓₓL_c             0.00%▕         ▏      0.000     0.000    0│
│  ↳ cₑ(x)               1.82%▕▏        ▏      0.001     0.000    4│
│  ↳ ∂cₑ/∂x              0.00%▕         ▏      0.000     0.000    4│
└──────────────────────────────────────────────────────────────────┘
```

The headings are defined as follows:
<table>
  <tr>
    <th>Heading</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>time trace</td>
    <td>Tree of solver steps</td>
  </tr>
  <tr>
    <td>percentage</td>
    <td>Percentage of solve time</td>
  </tr>
  <tr>
    <td>total</td>
    <td>Total time across all runs in milliseconds</td>
  </tr>
  <tr>
    <td>each</td>
    <td>Average time for each run in milliseconds</td>
  </tr>
  <tr>
    <td>runs</td>
    <td>Number of runs</td>
  </tr>
</table>

The function evaluations are defined as follows:
<table>
  <tr>
    <th>Function</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>f(x)</td>
    <td>Cost function value</td>
  </tr>
  <tr>
    <td>∇f(x)</td>
    <td>Cost function gradient</td>
  </tr>
  <tr>
    <td>∇²ₓₓL</td>
    <td>Lagrangian Hessian</td>
  </tr>
  <tr>
    <td>∇²ₓₓL_f</td>
    <td>Cost part of Lagrangian Hessian</td>
  </tr>
  <tr>
    <td>∇²ₓₓL_c</td>
    <td>Constraint part of Lagrangian Hessian</td>
  </tr>
  <tr>
    <td>cₑ(x)</td>
    <td>Equality constraint value</td>
  </tr>
  <tr>
    <td>∂cₑ/∂x</td>
    <td>Equality constraint Jacobian</td>
  </tr>
  <tr>
    <td>cᵢ(x)</td>
    <td>Inequality constraint value</td>
  </tr>
  <tr>
    <td>∂cᵢ/∂x</td>
    <td>Inequality constraint Jacobian</td>
  </tr>
</table>

### Exit status

Finally, the solver prints its exit status.
```
Exit: success
```

Possible exit statuses include:
<table>
  <tr>
    <th>Status</th>
    <th>Value</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>SUCCESS</td>
    <td>0</td>
    <td>Solved the problem to the desired tolerance.</td>
  </tr>
  <tr>
    <td>CALLBACK_REQUESTED_STOP</td>
    <td>1</td>
    <td>The solver returned its solution so far after the user requested a stop.</td>
  </tr>
  <tr>
    <td>TOO_FEW_DOFS</td>
    <td>-1</td>
    <td>The solver determined the problem to be overconstrained and gave up.</td>
  </tr>
  <tr>
    <td>LOCALLY_INFEASIBLE</td>
    <td>-2</td>
    <td>The solver determined the problem to be locally infeasible and gave up.</td>
  </tr>
  <tr>
    <td>GLOBALLY_INFEASIBLE</td>
    <td>-3</td>
    <td>The problem setup frontend determined the problem to have an empty feasible region.</td>
  </tr>
  <tr>
    <td>FACTORIZATION_FAILED</td>
    <td>-4</td>
    <td>The linear system factorization failed.</td>
  </tr>
  <tr>
    <td>FEASIBILITY_RESTORATION_FAILED</td>
    <td>-5</td>
    <td>The solver failed to reach the desired tolerance, and feasibility restoration failed to converge.</td>
  </tr>
  <tr>
    <td>NONFINITE_INITIAL_GUESS</td>
    <td>-6</td>
    <td>The solver encountered nonfinite initial cost, constraints, or derivatives and gave up.</td>
  </tr>
  <tr>
    <td>DIVERGING_ITERATES</td>
    <td>-7</td>
    <td>The solver encountered diverging primal iterates xₖ and/or sₖ and gave up.</td>
  </tr>
  <tr>
    <td>MAX_ITERATIONS_EXCEEDED</td>
    <td>-8</td>
    <td>The solver returned its solution so far after exceeding the maximum number of iterations.</td>
  </tr>
  <tr>
    <td>TIMEOUT</td>
    <td>-9</td>
    <td>The solver returned its solution so far after exceeding the maximum elapsed wall clock time.</td>
  </tr>
</table>

Negative values indicate errors.

## Problem formulation tips

### Optimizing the problem formulation

Cost functions and constraints can have the following orders:

* none (i.e., there is no cost function or are no constraints)
* constant
* linear
* quadratic
* nonlinear

For nonlinear problems, the solver calculates the Hessian of the cost function and the Jacobians of the constraints at each iteration. However, problems with lower order cost functions and constraints can be solved faster. For example, the following only need to be computed once because they're constant:

* the Hessian of a quadratic or lower cost function
* the Jacobian of linear or lower constraints

A problem is constant if:

* the cost function is constant or lower
* the equality constraints are constant or lower
* the inequality constraints are constant or lower

A problem is a linear program (LP) if:

* the cost function is linear
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

A problem is a quadratic program (QP) if:

* the cost function is quadratic
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

All other problems are nonlinear programs (NLPs).

### Suppressing constant pruning

Sleipnir aggressively simplifies constant expressions in the expression graph. This includes:

* multiplication by 0 or 1
* division by 1
* addition or subtraction by 0
* math functions evaluating constants

If the user intends to reuse the problem formulation with different constants, the expression pruning can be suppressed in two ways.

1. Make an empty (linear) `Variable`, then set its value. This can negatively impact autodiff caching.
```python
a = Variable()
a.set_value(0)
```
2. Create a constant `Variable` with a value that won't be pruned (expression-dependent).
```python
a = Variable(2)
```

### Avoiding numerical issues

Instead of using distance (2-norm) for the cost function, use sum-of-squares. The distance calculation's square root is nonlinear and has a limited domain, whereas sum-of-squares has the same minimum, is quadratic, and has no domain restriction. In other words, use `minimize(x ** 2 + y ** 2 + z ** 2)` instead of `minimize(hypot(x, y, z))`.

### Deduplicating autodiff work

Store common subexpressions in intermediate variables and reuse them instead of writing out the subexpressions each time. This ensures common subexpressions in the expression tree are only traversed and updated once.

### Minimum-time problems

The obvious problem formulation for minimum-time problems uses one dt shared across all timesteps.

```python
import sleipnir as slp

N = 100
T_max = 5.0

problem = slp.optimization.Problem()

x = problem.decision_variable(N + 1)
v = problem.decision_variable(N)

dt = problem.decision_variable()
dt.set_value(T_max / N)
problem.subject_to(dt > 0)
problem.subject_to(dt < T_max / N)

for k in range(N):
    problem.subject_to(x[k + 1] == x[k] + v[k] * dt)

problem.minimize(dt)

problem.solve()
```
The nonzero initial value for dt avoids a degenerate case, and the upper bound prevents the solver exploiting discretization artifacts.

This formulation can have feasibility issues though per section 15.3 "Elimination of variables" of "Numerical Optimization, 2nd Ed.". Instead, we recommend using a separate dt for each timestep, with them all equality-constrained.

```python
import sleipnir as slp

N = 100
T_max = 5.0

problem = slp.optimization.Problem()

x = problem.decision_variable(N + 1)
v = problem.decision_variable(N)

dt = problem.decision_variable(N)
problem.subject_to(dt > 0)
problem.subject_to(dt < T_max / N)
for k in range(N - 1):
    problem.subject_to(dt[k] == dt[k + 1])

for k in range(N):
    dt[k].set_value(T_max / N)
    problem.subject_to(x[k + 1] == x[k] + v[k] * dt[k])

problem.minimize(sum(dt))

problem.solve()
```

### Manifold optimization

Unit vectors, rotations, and poses can be decision variables. Instead of constraining their unit-length parts with `n.T @ n == 1`, the solver steps along the set of valid values, so every iterate is valid, and the problem has fewer constraints and multipliers. Use them instead of angle decision variables too: rotations have no discontinuity at ±π, and they don't need `atan2()`.

| Decision variable                         | Type         | Value                                      | Initial value      |
|-------------------------------------------|--------------|--------------------------------------------|--------------------|
| `problem.decision_variable(sphere, rows)` | `UnitVector` | unit vector                                | e₁ = [1, 0, …, 0]ᵀ |
| `problem.decision_variable(so2)`          | `Rotation2`  | angle in radians, in (−π, π]               | 0                  |
| `problem.decision_variable(so3)`          | `Rotation3`  | unit quaternion [w, x, y, z]               | identity           |
| `problem.decision_variable(se2)`          | `Pose2`      | `Translation2` and `Rotation2`             | identity           |
| `problem.decision_variable(se3)`          | `Pose3`      | `Translation3` and `Rotation3`             | identity           |

`set_value()` takes the same form as `value()` and normalizes unit vectors and quaternions. The tags are in `sleipnir.optimization` (`slp::sphere`, etc. in C++), and the types are in `sleipnir.autodiff`.

```python
import math

import numpy as np
from sleipnir.autodiff import Pose2
from sleipnir.optimization import Problem, se2

problem = Problem()

robot = problem.decision_variable(se2)
robot.translation().set_value(np.array([-0.1, 0.0]))
robot.rotation().set_value(0.2)

goal = Pose2(np.array([3.0, 1.0]), math.pi / 2)
problem.minimize(robot.translation().squared_distance(goal.translation()))
problem.subject_to(robot.rotation() == goal.rotation())

problem.solve()

x, y = robot.translation().value()
θ = robot.rotation().value()
```

The types support these operations:

* Rotations and poses compose with `*` (`a * b` applies `b`, then `a`), and have `inverse()` and `relative_to(other)`, which is `other.inverse() * self`.
* Vectors rotate with `v.rotate_by(rotation)`, and points transform with `p.transform_by(pose)`.
* `Rotation2` has `cos()`, `sin()`, and `angle()`. `Rotation3` has `w()`, `x()`, `y()`, `z()`, and `to_vector()`, which returns the rotation vector (axis times angle). Both have `to_matrix()`.
* `Rotation2(angle)` and `Rotation3(rvec)` construct rotations from an angle or rotation vector, which can be constants or autodiff variables. `Rotation3(w, x, y, z)` constructs one from a quaternion.
* `Translation2` and `Translation3` have `x()`, `y()`, `z()`, `distance()`, and `squared_distance()`.

#### Constraints

Use `==` between two values of the same type to constrain them equal. It constrains their relative angle or rotation vector, which is one equation per degree of freedom with no spurious solutions. Elementwise alternatives are worse:

* `n == d` with `n` and `d` as `VariableMatrix` gives n equations for a unit vector with n − 1 degrees of freedom, so the constraints are redundant.
* `rotation.cos() == math.cos(θ)` and `rotation.sin() == math.sin(θ)` gives two equations for one degree of freedom.
* `rotation.sin() == 0` is also satisfied by the opposite heading.

If both sides of a `UnitVector` equality are decision variables, set their initial values before creating the constraint, and don't start them exactly opposite each other.

For tolerances, use quantities that are smooth near the bound. Angles and distances themselves have a kink at zero, where their derivatives are undefined, so compare squares instead.

```python
from sleipnir.autodiff import VariableMatrix
from sleipnir.optimization import bounds

# Heading within ±tol of target. The relative angle only wraps when the
# rotations are opposite, so ranges that cross ±π work.
problem.subject_to(bounds(-tol, z.relative_to(target).angle(), tol))

# 3D rotation within tol of target
e = q.relative_to(target).to_vector()
problem.subject_to(e.T @ e <= tol**2)

# Position within tol of target
problem.subject_to(t.squared_distance(target) <= tol**2)

# Unit vector within angle tol of the constant direction d, an n×1 array
problem.subject_to(n.dot(VariableMatrix(d)) >= math.cos(tol))
```

A heading range [θ_min, θ_max] is a center (θ_min + θ_max)/2 and a tolerance (θ_max − θ_min)/2.

#### Twists

A twist is a change in pose along a constant-curvature arc (2D) or screw motion (3D), in the frame of the pose it starts from. `Twist2(linear, angular)` has a 2D linear part and a scalar angular part, and `Twist3(linear, angular)` has 3D linear and angular parts. `twist.exp()` returns the pose reached by moving along the twist from the identity, and `pose.log()` is its inverse.

Since a body-frame velocity times a timestep is a twist, `pose * (velocity * dt).exp()` integrates a constant velocity exactly. For example, a differential drive's dynamics are

```python
from sleipnir.autodiff import Twist2, VariableMatrix

for k in range(N):
    twist = Twist2(VariableMatrix([[v[k]], [0.0]]), ω[k])
    problem.subject_to(poses[k + 1] == poses[k] * (twist * dt).exp())
```

and angular velocity integrates with `q[k + 1] == q[k] * Rotation3(ω[k] * dt)`. `log()` of a relative pose, `pose.relative_to(reference).log()`, is a residual for pose errors.

#### Iteration callbacks

`IterationInfo.x` contains the decision variables' values, with each manifold variable's components in order: a `Rotation2`'s cos θ and sin θ, a `Rotation3`'s w, x, y, and z, and a pose's translation followed by its rotation. The gradient, Hessian, and Jacobians are with respect to the manifold's tangent space, which has one dimension per degree of freedom, so they have fewer columns than `x` has rows when the problem has manifold variables.

See [algorithms.md](algorithms.md#manifold-optimization) for how the solver handles manifold variables.
