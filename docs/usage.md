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

x, y = problem.real_decision_variable(2)

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

The user-configurable exit conditions include the error tolerance, maximum iterations, and timeout passed to the `solve()` call; and iteration callbacks added to the `Problem` returning `ControlFlow::BREAK` (C++) or raising `StopIteration` (Python).

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
│  ↳ KKT matrix factor   7.27%▕▋        ▏      0.004     0.001    3│
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

### Linting

`problem.lint()` checks the problem's formulation and returns a list of suggested improvements. It doesn't change or solve the problem, so call it while writing the problem instead of before every solve.

```python
for finding in problem.lint():
    print(finding)
```

Each finding has a `kind` (a `LintKind`), the indices of the decision variables and constraints involved, whether the cost function is involved (`cost`), and a `message` that explains the issue and how to fix it. Decision variables are numbered in the order they were created, and messages call decision variable i `x[i]`. Constraints are numbered in the order they were added, and matrix constraints add one per element.

| `LintKind`                 | Issue                                                                                                                                                                                                                                                                                                        |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `UNUSED_DECISION_VARIABLE` | Decision variables don't appear in the cost function or any constraint.                                                                                                                                                                                                                                      |
| `CONSTANT_CONSTRAINT`      | A constraint doesn't depend on any decision variables, so it's always satisfied or never satisfied.                                                                                                                                                                                                          |
| `REDUNDANT_CONSTRAINTS`    | Linear constraints on the same linear combination of decision variables imply each other.                                                                                                                                                                                                                    |
| `CONFLICTING_CONSTRAINTS`  | Linear constraints on the same linear combination of decision variables contradict each other.                                                                                                                                                                                                               |
| `EQUALITY_AS_INEQUALITIES` | A lower and upper bound on the same linear combination of decision variables are equal.                                                                                                                                                                                                                      |
| `NONSMOOTH_FUNCTION`       | A function that isn't differentiable everywhere appears in a form with a differentiable replacement: `abs(a) <= b`, `max(a, b) <= c`, `min(a, b) >= c`, a norm (like `hypot()`) compared to a constant, or a cost function that's a norm.                                                                    |
| `BADLY_SCALED`             | Coefficients of a linear constraint or a linear or quadratic cost function span so many orders of magnitude that the solver loses precision, and the message names the variable substitution that fixes it; or a linear constraint's gradient is tiny, and the message names the constant to multiply it by. |

Values in messages use the decision variables' and parameters' current values.

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

### Constant pruning and parameters

Sleipnir aggressively simplifies constant expressions in the expression graph. This includes:

* multiplication by 0 or 1
* division by 1
* addition or subtraction by 0
* math functions evaluating constants

Use a parameter for constants that need to change between solves.
```python
problem = Problem()
a = problem.parameter()
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

x = problem.real_decision_variable(N + 1)
v = problem.real_decision_variable(N)

dt = problem.real_decision_variable()
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

x = problem.real_decision_variable(N + 1)
v = problem.real_decision_variable(N)

dt = problem.real_decision_variable(N)
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

Unit vectors, rotations, poses, and similarity transforms can be decision variables. Instead of constraining their unit-length parts with `n.T @ n == 1`, the solver steps along the set of valid values, so every iterate is valid, and the problem has fewer constraints and multipliers. Use them instead of angle decision variables too: rotations have no discontinuity at ±π, and they don't need `atan2()`.

| Decision variable                             | Type           | Value                                      | Initial value      |
|-----------------------------------------------|----------------|--------------------------------------------|--------------------|
| `problem.unit_vector_decision_variable(rows)` | `UnitVector`   | unit vector                                | e₁ = [1, 0, …, 0]ᵀ |
| `problem.translation2_decision_variable()`    | `Translation2` | [x, y]                                     | 0                  |
| `problem.translation3_decision_variable()`    | `Translation3` | [x, y, z]                                  | 0                  |
| `problem.rotation2_decision_variable()`       | `Rotation2`    | angle in radians, in (−π, π]               | 0                  |
| `problem.rotation3_decision_variable()`       | `Rotation3`    | unit quaternion [w, x, y, z]               | identity           |
| `problem.pose2_decision_variable()`           | `Pose2`        | `Translation2` and `Rotation2`             | identity           |
| `problem.pose3_decision_variable()`           | `Pose3`        | `Translation3` and `Rotation3`             | identity           |
| `problem.similarity2_decision_variable()`     | `Similarity2`  | `Translation2`, `Rotation2`, and log scale | identity           |
| `problem.similarity3_decision_variable()`     | `Similarity3`  | `Translation3`, `Rotation3`, and log scale | identity           |

`set_value()` takes the same form as `value()` and normalizes unit vectors and quaternions. The types are in `sleipnir.autodiff`.

Each decision variable has a parameter counterpart with the same initial value, like `problem.pose2_parameter()`, for known values that change between solves. Use them instead of building a type from parameters, like `Rotation2(problem.parameter())`, whose `set_value()` can't change its value and whose expressions evaluate cos and sin.

```python
import math

import numpy as np
from sleipnir.autodiff import Pose2
from sleipnir.optimization import Problem

problem = Problem()

robot = problem.pose2_decision_variable()
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

* Rotations, poses, and similarity transforms compose with `*` (`a * b` applies `b`, then `a`), and have `inverse()` and `relative_to(other)`, which is `other.inverse() * self`.
* Unit vectors and translations rotate with `v.rotate_by(rotation)`, and points transform with `p.transform_by(pose)` or `p.transform_by(similarity)`.
* `Rotation2` has `cos()`, `sin()`, and `angle()`. `Rotation3` has `w()`, `x()`, `y()`, `z()`, and `to_vector()`, which returns the rotation vector (axis times angle). Both have `to_matrix()`.
* `Rotation2(angle)` and `Rotation3(rvec)` construct rotations from an angle or rotation vector, which can be constants or autodiff variables. `Rotation3(w, x, y, z)` constructs one from a quaternion.
* `Translation2` and `Translation3` have `x()`, `y()`, `z()`, `distance()`, and `squared_distance()`.
* A similarity transform maps a point p to s R p + t. `Similarity2` and `Similarity3` store the scale s as its natural logarithm, so it stays positive without a constraint. `log_scale()` returns the log scale, and `scale()` returns exp(`log_scale()`). Set the scale with `log_scale().set_value(math.log(s))`. `Similarity2(translation, angle, scale)` constructs a constant one, and `Similarity2(translation, rotation, log_scale)` and `Similarity3(translation, rotation, log_scale)` construct one from parts.

#### Constraints

Use `==` between two values of the same type to constrain them equal. It constrains their relative angle or rotation vector, which is one equation per degree of freedom with no spurious solutions. Similarity transforms also constrain their log scales equal. A `UnitVector` can also equal a constant direction, but not a `RealMatrix`. Elementwise alternatives are worse:

* `n == d` with `n` and `d` as `RealMatrix` gives n equations for a unit vector with n − 1 degrees of freedom, so the constraints are redundant.
* `rotation.cos() == math.cos(θ)` and `rotation.sin() == math.sin(θ)` gives two equations for one degree of freedom.
* `rotation.sin() == 0` is also satisfied by the opposite heading.

If both sides of a `UnitVector` equality are decision variables, set their initial values before creating the constraint, and don't start them exactly opposite each other.

For tolerances, use quantities that are smooth near the bound. Angles and distances themselves have a kink at zero, where their derivatives are undefined, so compare squares instead.

```python
from sleipnir.autodiff import RealMatrix
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
problem.subject_to(n.dot(RealMatrix(d)) >= math.cos(tol))
```

A heading range [θ_min, θ_max] is a center (θ_min + θ_max)/2 and a tolerance (θ_max − θ_min)/2.

#### Twists

A twist is a change in pose along a constant-curvature arc (2D) or screw motion (3D), in the frame of the pose it starts from. `Twist2(linear, angular)` has a 2D linear part and a scalar angular part, and `Twist3(linear, angular)` has 3D linear and angular parts. `twist.exp()` returns the pose reached by moving along the twist from the identity, and `pose.log()` is its inverse.

Since a body-frame velocity times a timestep is a twist, `pose * (velocity * dt).exp()` integrates a constant velocity exactly. For example, a differential drive's dynamics are

```python
from sleipnir.autodiff import RealMatrix, Twist2

for k in range(N):
    twist = Twist2(RealMatrix([[v[k]], [0.0]]), ω[k])
    problem.subject_to(poses[k + 1] == poses[k] * (twist * dt).exp())
```

and angular velocity integrates with `q[k + 1] == q[k] * Rotation3(ω[k] * dt)`. `log()` of a relative pose, `pose.relative_to(reference).log()`, is a residual for pose errors.

Similarity transforms have similarity twists, which add a log-scale part dσ: `SimilarityTwist2(linear, angular, log_scale)` and `SimilarityTwist3(linear, angular, log_scale)`. `twist.exp()` returns a `Similarity2` or `Similarity3`, `similarity.log()` is its inverse, and `similarity * (velocity * dt).exp()` integrates a constant body-frame velocity and log-scale rate exactly.

#### Iteration callbacks

`IterationInfo.x` contains the decision variables' values, with each manifold variable's components in order: a `Rotation2`'s cos θ and sin θ, a `Rotation3`'s w, x, y, and z, a pose's translation followed by its rotation, and a similarity transform's translation, rotation, and log scale. The gradient, Hessian, and Jacobians are with respect to the manifold's tangent space, which has one dimension per degree of freedom, so they have fewer columns than `x` has rows when the problem has manifold variables.

See [algorithms.md](algorithms.md#manifold-optimization) for how the solver handles manifold variables.
