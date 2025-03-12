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
┏━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━┯━━━━┓
┃     setup trace     │     percent      │total (ms)│each (ms)│runs┃
┡━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━┩
│setup                 100.00%▕█████████▏      0.011     0.011    1│
│↳ ∇f(x)                 9.09%▕▊        ▏      0.001     0.001    1│
│↳ ∇²ₓₓL                81.82%▕███████▎ ▏      0.009     0.009    1│
│  ↳ ∇²ₓₓL_f            36.36%▕███▎     ▏      0.004     0.004    1│
│  ↳ ∇²ₓₓL_c            36.36%▕███▎     ▏      0.004     0.004    1│
│↳ ∂cₑ/∂x                9.09%▕▊        ▏      0.001     0.001    1│
└──────────────────────────────────────────────────────────────────┘
```

### Iterations

After the solver takes each step, it prints a row of iteration diagnostics in a table format.
```
┏━━━━┯━━━━┯━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━┯━━━━━┯━━━━━━━━┯━━━━━━━━┯━━┓
┃iter│type│time (ms)│   error    │    cost     │  infeas.   │   μ    │ reg │primal α│ dual α │↩ ┃
┡━━━━┷━━━━┷━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━┷━━━━━┷━━━━━━━━┷━━━━━━━━┷━━┩
│   0 norm     0.014 1.799760e-03 -1.080000e+02 6.016734e-10 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
│   1 norm     0.006 1.199700e-07 -1.080000e+02 9.947598e-14 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
│   2 norm     0.003 4.998668e-12 -1.080000e+02 0.000000e+00 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

The headings are defined as follows:
<table>
  <tr>
    <th>Heading</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>iter</td>
    <td>Iteration number</td>
  </tr>
  <tr>
    <td>type</td>
    <td>Iteration type
      <ul>
        <li>`norm` = normal</li>
        <li>`✓SOC` = accepted second-order correction</li>
        <li>`XSOC` = rejected second-order correction</li>
        <li>`rest` = feasibility restoration</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>time (ms)</td>
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
    <td>μ</td>
    <td>Barrier parameter</td>
  </tr>
  <tr>
    <td>reg</td>
    <td>Iteration matrix regularization</td>
  </tr>
  <tr>
    <td>primal α</td>
    <td>Primal step size</td>
  </tr>
  <tr>
    <td>dual α</td>
    <td>Dual step size</td>
  </tr>
  <tr>
    <td>↩</td>
    <td>Number of line search backtracks</td>
  </tr>
</table>

### Solver time trace

At the end of the solve, the solver prints a time trace of itself.
```
┏━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━┯━━━━┓
┃    solver trace     │     percent      │total (ms)│each (ms)│runs┃
┡━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━┩
│solver                100.00%▕█████████▏      0.058     0.058    1│
│↳ setup                 5.17%▕▍        ▏      0.003     0.003    1│
│↳ iteration            39.66%▕███▌     ▏      0.023     0.007    3│
│  ↳ feasibility check   0.00%▕         ▏      0.000     0.000    3│
│  ↳ callbacks           0.00%▕         ▏      0.000     0.000    3│
│  ↳ KKT matrix build    3.45%▕▎        ▏      0.002     0.000    3│
│  ↳ KKT matrix decomp   6.90%▕▌        ▏      0.004     0.001    3│
│  ↳ KKT system solve    1.72%▕▏        ▏      0.001     0.000    3│
│  ↳ line search        13.79%▕█▏       ▏      0.008     0.002    3│
│    ↳ SOC               0.00%▕         ▏      0.000     0.000    0│
│  ↳ next iter prep      0.00%▕         ▏      0.000     0.000    3│
│  ↳ f(x)                0.00%▕         ▏      0.000     0.000    7│
│  ↳ ∇f(x)               3.45%▕▎        ▏      0.002     0.000    4│
│  ↳ ∇²ₓₓL               1.72%▕▏        ▏      0.001     0.000    4│
│  ↳ ∇²ₓₓL_c             0.00%▕         ▏      0.000     0.000    0│
│  ↳ cₑ(x)               1.72%▕▏        ▏      0.001     0.000    7│
│  ↳ ∂cₑ/∂x              0.00%▕         ▏      0.000     0.000    4│
└──────────────────────────────────────────────────────────────────┘
```

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

For nonlinear problems, the solver calculates the Hessian of the cost
function and the Jacobians of the constraints at each iteration. However,
problems with lower order cost functions and constraints can be solved
faster. For example, the following only need to be computed once because
they're constant:

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

### Avoiding numerical issues

Instead of using distance (2-norm) for the cost function, use sum-of-squares.
The distance calculation's square root is nonlinear and has a limited domain,
whereas sum-of-squares has the same minimum, is quadratic, and has no domain
restriction. In other words, use `minimize(x ** 2 + y ** 2 + z ** 2)` instead of
`minimize(hypot(x, y, z))`.

### Deduplicating autodiff work

Store common subexpressions in intermediate variables and reuse them instead of
writing out the subexpressions each time. This ensures common subexpressions in
the expression tree are only traversed and updated once.

### Minimum-time problems

The obvious problem formulation for minimum-time problems uses one dt shared
across all timesteps.

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
The nonzero initial value for dt avoids a degenerate case, and the upper bound
prevents the solver exploiting discretization artifacts.

This formulation can have feasibility issues though per section 15.3
"Elimination of variables" of "Numerical Optimization, 2nd Ed.". Instead, we
recommend using a separate dt for each timestep, with them all
equality-constrained.

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
