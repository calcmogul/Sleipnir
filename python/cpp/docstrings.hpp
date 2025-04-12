/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_formatter = R"doc(Formatter for ExpressionType.)doc";

static const char *__doc_formatter_2 = R"doc(Formatter for ExitStatus.)doc";

static const char *__doc_formatter_format =
R"doc(Format ExpressionType.

Template parameter ``FmtContext``:
    Format context type.

Parameter ``type``:
    Expression type.

Parameter ``ctx``:
    Format context.

Returns:
    Format context iterator.)doc";

static const char *__doc_formatter_format_2 =
R"doc(Format ExitStatus.

Template parameter ``FmtContext``:
    Format context type.

Parameter ``exit_status``:
    Exit status.

Parameter ``ctx``:
    Format context.

Returns:
    Format context iterator.)doc";

static const char *__doc_formatter_m_underlying = R"doc()doc";

static const char *__doc_formatter_m_underlying_2 = R"doc()doc";

static const char *__doc_formatter_parse =
R"doc(Parse format string.

Parameter ``ctx``:
    Format parse context.

Returns:
    Format parse context iterator.)doc";

static const char *__doc_formatter_parse_2 =
R"doc(Parse format string.

Parameter ``ctx``:
    Format parse context.

Returns:
    Format parse context iterator.)doc";

static const char *__doc_slp = R"doc()doc";

static const char *__doc_slp_2 = R"doc()doc";

static const char *__doc_slp_3 = R"doc()doc";

static const char *__doc_slp_4 = R"doc()doc";

static const char *__doc_slp_5 = R"doc()doc";

static const char *__doc_slp_DynamicsType = R"doc(Enum describing a type of system dynamics constraints.)doc";

static const char *__doc_slp_DynamicsType_DISCRETE = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

static const char *__doc_slp_DynamicsType_EXPLICIT_ODE = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

static const char *__doc_slp_EqualityConstraints =
R"doc(A vector of equality constraints of the form cₑ(x) = 0.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_EqualityConstraints_EqualityConstraints =
R"doc(Concatenates multiple equality constraints.

Parameter ``equality_constraints``:
    The list of EqualityConstraints to concatenate.)doc";

static const char *__doc_slp_EqualityConstraints_EqualityConstraints_2 =
R"doc(Concatenates multiple equality constraints.

This overload is for Python bindings only.

Parameter ``equality_constraints``:
    The list of EqualityConstraints to concatenate.)doc";

static const char *__doc_slp_EqualityConstraints_EqualityConstraints_3 =
R"doc(Constructs an equality constraint from a left and right side.

The standard form for equality constraints is c(x) = 0. This function
takes a constraint of the form lhs = rhs and converts it to lhs - rhs
= 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_slp_EqualityConstraints_constraints = R"doc(A vector of scalar equality constraints.)doc";

static const char *__doc_slp_EqualityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_slp_ExitStatus = R"doc(Solver exit status. Negative values indicate failure.)doc";

static const char *__doc_slp_ExitStatus_CALLBACK_REQUESTED_STOP =
R"doc(The solver returned its solution so far after the user requested a
stop.)doc";

static const char *__doc_slp_ExitStatus_DIVERGING_ITERATES =
R"doc(The solver encountered diverging primal iterates xₖ and/or sₖ and gave
up.)doc";

static const char *__doc_slp_ExitStatus_FACTORIZATION_FAILED = R"doc(The linear system factorization failed.)doc";

static const char *__doc_slp_ExitStatus_GLOBALLY_INFEASIBLE =
R"doc(The problem setup frontend determined the problem to have an empty
feasible region.)doc";

static const char *__doc_slp_ExitStatus_LINE_SEARCH_FAILED =
R"doc(The backtracking line search failed, and the problem isn't locally
infeasible.)doc";

static const char *__doc_slp_ExitStatus_LOCALLY_INFEASIBLE =
R"doc(The solver determined the problem to be locally infeasible and gave
up.)doc";

static const char *__doc_slp_ExitStatus_MAX_ITERATIONS_EXCEEDED =
R"doc(The solver returned its solution so far after exceeding the maximum
number of iterations.)doc";

static const char *__doc_slp_ExitStatus_NONFINITE_INITIAL_COST_OR_CONSTRAINTS =
R"doc(The solver encountered nonfinite initial cost or constraints and gave
up.)doc";

static const char *__doc_slp_ExitStatus_SUCCESS = R"doc(Solved the problem to the desired tolerance.)doc";

static const char *__doc_slp_ExitStatus_TIMEOUT =
R"doc(The solver returned its solution so far after exceeding the maximum
elapsed wall clock time.)doc";

static const char *__doc_slp_ExitStatus_TOO_FEW_DOFS = R"doc(The solver determined the problem to be overconstrained and gave up.)doc";

static const char *__doc_slp_ExpressionType =
R"doc(Expression type.

Used for autodiff caching.)doc";

static const char *__doc_slp_ExpressionType_CONSTANT = R"doc(The expression is a constant.)doc";

static const char *__doc_slp_ExpressionType_LINEAR = R"doc(The expression is composed of linear and lower-order operators.)doc";

static const char *__doc_slp_ExpressionType_NONE = R"doc(There is no expression.)doc";

static const char *__doc_slp_ExpressionType_NONLINEAR = R"doc(The expression is composed of nonlinear and lower-order operators.)doc";

static const char *__doc_slp_ExpressionType_QUADRATIC = R"doc(The expression is composed of quadratic and lower-order operators.)doc";

static const char *__doc_slp_Gradient =
R"doc(This class calculates the gradient of a variable with respect to a
vector of variables.

The gradient is only recomputed if the variable expression is
quadratic or higher order.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_Gradient_2 = R"doc()doc";

static const char *__doc_slp_Gradient_Gradient =
R"doc(Constructs a Gradient object.

Parameter ``variable``:
    Variable of which to compute the gradient.

Parameter ``wrt``:
    Variable with respect to which to compute the gradient.)doc";

static const char *__doc_slp_Gradient_Gradient_2 = R"doc()doc";

static const char *__doc_slp_Gradient_get =
R"doc(Returns the gradient as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The gradient as a VariableMatrix.)doc";

static const char *__doc_slp_Gradient_m_g = R"doc()doc";

static const char *__doc_slp_Gradient_m_jacobian = R"doc()doc";

static const char *__doc_slp_Gradient_value =
R"doc(Evaluates the gradient at wrt's value.

Returns:
    The gradient at wrt's value.)doc";

static const char *__doc_slp_Hessian =
R"doc(This class calculates the Hessian of a variable with respect to a
vector of variables.

The gradient tree is cached so subsequent Hessian calculations are
faster, and the Hessian is only recomputed if the variable expression
is nonlinear.

Template parameter ``Scalar``:
    Scalar type.

Template parameter ``UpLo``:
    Which part of the Hessian to compute (Lower or Lower | Upper).)doc";

static const char *__doc_slp_Hessian_2 = R"doc()doc";

static const char *__doc_slp_Hessian_3 = R"doc()doc";

static const char *__doc_slp_Hessian_Hessian =
R"doc(Constructs a Hessian object.

Parameter ``variable``:
    Variable of which to compute the Hessian.

Parameter ``wrt``:
    Variable with respect to which to compute the Hessian.)doc";

static const char *__doc_slp_Hessian_Hessian_2 = R"doc()doc";

static const char *__doc_slp_Hessian_get =
R"doc(Returns the Hessian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Hessian as a VariableMatrix.)doc";

static const char *__doc_slp_Hessian_m_H = R"doc()doc";

static const char *__doc_slp_Hessian_m_cached_triplets = R"doc()doc";

static const char *__doc_slp_Hessian_m_graphs = R"doc()doc";

static const char *__doc_slp_Hessian_m_nonlinear_rows = R"doc()doc";

static const char *__doc_slp_Hessian_m_variables = R"doc()doc";

static const char *__doc_slp_Hessian_m_wrt = R"doc()doc";

static const char *__doc_slp_Hessian_value =
R"doc(Evaluates the Hessian at wrt's value.

Returns:
    The Hessian at wrt's value.)doc";

static const char *__doc_slp_InequalityConstraints =
R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_InequalityConstraints_InequalityConstraints =
R"doc(Concatenates multiple inequality constraints.

Parameter ``inequality_constraints``:
    The list of InequalityConstraints to concatenate.)doc";

static const char *__doc_slp_InequalityConstraints_InequalityConstraints_2 =
R"doc(Concatenates multiple inequality constraints.

This overload is for Python bindings only.

Parameter ``inequality_constraints``:
    The list of InequalityConstraints to concatenate.)doc";

static const char *__doc_slp_InequalityConstraints_InequalityConstraints_3 =
R"doc(Constructs an inequality constraint from a left and right side.

The standard form for inequality constraints is c(x) ≥ 0. This
function takes a constraints of the form lhs ≥ rhs and converts it to
lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_slp_InequalityConstraints_constraints = R"doc(A vector of scalar inequality constraints.)doc";

static const char *__doc_slp_InequalityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_slp_IterationInfo =
R"doc(Solver iteration information exposed to an iteration callback.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_IterationInfo_A_e = R"doc(The equality constraint Jacobian.)doc";

static const char *__doc_slp_IterationInfo_A_i = R"doc(The inequality constraint Jacobian.)doc";

static const char *__doc_slp_IterationInfo_H = R"doc(The Hessian of the Lagrangian.)doc";

static const char *__doc_slp_IterationInfo_g = R"doc(The gradient of the cost function.)doc";

static const char *__doc_slp_IterationInfo_iteration = R"doc(The solver iteration.)doc";

static const char *__doc_slp_IterationInfo_x = R"doc(The decision variables.)doc";

static const char *__doc_slp_Jacobian =
R"doc(This class calculates the Jacobian of a vector of variables with
respect to a vector of variables.

The Jacobian is only recomputed if the variable expression is
quadratic or higher order.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_Jacobian_2 = R"doc()doc";

static const char *__doc_slp_Jacobian_3 = R"doc()doc";

static const char *__doc_slp_Jacobian_Jacobian =
R"doc(Constructs a Jacobian object.

Parameter ``variable``:
    Variable of which to compute the Jacobian.

Parameter ``wrt``:
    Variable with respect to which to compute the Jacobian.)doc";

static const char *__doc_slp_Jacobian_Jacobian_2 = R"doc()doc";

static const char *__doc_slp_Jacobian_Jacobian_3 = R"doc()doc";

static const char *__doc_slp_Jacobian_get =
R"doc(Returns the Jacobian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Jacobian as a VariableMatrix.)doc";

static const char *__doc_slp_Jacobian_m_J = R"doc()doc";

static const char *__doc_slp_Jacobian_m_cached_triplets = R"doc(Cached triplets for gradients of linear rows)doc";

static const char *__doc_slp_Jacobian_m_graphs = R"doc()doc";

static const char *__doc_slp_Jacobian_m_nonlinear_rows =
R"doc(List of row indices for nonlinear rows whose graients will be computed
in value())doc";

static const char *__doc_slp_Jacobian_m_variables = R"doc()doc";

static const char *__doc_slp_Jacobian_m_wrt = R"doc()doc";

static const char *__doc_slp_Jacobian_value =
R"doc(Evaluates the Jacobian at wrt's value.

Returns:
    The Jacobian at wrt's value.)doc";

static const char *__doc_slp_OCP =
R"doc(This class allows the user to pose and solve a constrained optimal
control problem (OCP) in a variety of ways.

The system is transcripted by one of three methods (direct
transcription, direct collocation, or single-shooting) and additional
constraints can be added.

In direct transcription, each state is a decision variable constrained
to the integrated dynamics of the previous state. In direct
collocation, the trajectory is modeled as a series of cubic
polynomials where the centerpoint slope is constrained. In single-
shooting, states depend explicitly as a function of all previous
states and all previous inputs.

Explicit ODEs are integrated using RK4.

For explicit ODEs, the function must be in the form dx/dt = f(t, x,
u). For discrete state transition functions, the function must be in
the form xₖ₊₁ = f(t, xₖ, uₖ).

Direct collocation requires an explicit ODE. Direct transcription and
single-shooting can use either an ODE or state transition function.

https://underactuated.mit.edu/trajopt.html goes into more detail on
each transcription method.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_OCP_2 = R"doc()doc";

static const char *__doc_slp_OCP_OCP =
R"doc(Build an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Parameter ``num_states``:
    The number of system states.

Parameter ``num_inputs``:
    The number of system inputs.

Parameter ``dt``:
    The timestep for fixed-step integration.

Parameter ``num_steps``:
    The number of control points.

Parameter ``dynamics``:
    Function representing an explicit or implicit ODE, or a discrete
    state transition function.

* Explicit: dx/dt = f(x, u, *)

* Implicit: f([x dx/dt]', u, *) = 0

* State transition: xₖ₊₁ = f(xₖ, uₖ)

Parameter ``dynamics_type``:
    The type of system evolution function.

Parameter ``timestep_method``:
    The timestep method.

Parameter ``transcription_method``:
    The transcription method.)doc";

static const char *__doc_slp_OCP_OCP_2 =
R"doc(Build an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Parameter ``num_states``:
    The number of system states.

Parameter ``num_inputs``:
    The number of system inputs.

Parameter ``dt``:
    The timestep for fixed-step integration.

Parameter ``num_steps``:
    The number of control points.

Parameter ``dynamics``:
    Function representing an explicit or implicit ODE, or a discrete
    state transition function.

* Explicit: dx/dt = f(t, x, u, *)

* Implicit: f(t, [x dx/dt]', u, *) = 0

* State transition: xₖ₊₁ = f(t, xₖ, uₖ, dt)

Parameter ``dynamics_type``:
    The type of system evolution function.

Parameter ``timestep_method``:
    The timestep method.

Parameter ``transcription_method``:
    The transcription method.)doc";

static const char *__doc_slp_OCP_U =
R"doc(Get the input variables. After the problem is solved, this will
contain the inputs corresponding to the optimized trajectory.

Shaped (num_inputs)x(num_steps+1), although the last input step is
unused in the trajectory.

Returns:
    The input variable matrix.)doc";

static const char *__doc_slp_OCP_X =
R"doc(Get the state variables. After the problem is solved, this will
contain the optimized trajectory.

Shaped (num_states)x(num_steps+1).

Returns:
    The state variable matrix.)doc";

static const char *__doc_slp_OCP_constrain_direct_collocation = R"doc(Apply direct collocation dynamics constraints.)doc";

static const char *__doc_slp_OCP_constrain_direct_transcription = R"doc(Apply direct transcription dynamics constraints.)doc";

static const char *__doc_slp_OCP_constrain_final_state =
R"doc(Utility function to constrain the final state.

Parameter ``final_state``:
    the final state to constrain to.)doc";

static const char *__doc_slp_OCP_constrain_initial_state =
R"doc(Utility function to constrain the initial state.

Parameter ``initial_state``:
    the initial state to constrain to.)doc";

static const char *__doc_slp_OCP_constrain_single_shooting = R"doc(Apply single shooting dynamics constraints.)doc";

static const char *__doc_slp_OCP_dt =
R"doc(Get the timestep variables. After the problem is solved, this will
contain the timesteps corresponding to the optimized trajectory.

Shaped 1x(num_steps+1), although the last timestep is unused in the
trajectory.

Returns:
    The timestep variable matrix.)doc";

static const char *__doc_slp_OCP_final_state =
R"doc(Convenience function to get the final state in the trajectory.

Returns:
    The final state of the trajectory.)doc";

static const char *__doc_slp_OCP_for_each_step =
R"doc(Set the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``callback``:
    The callback f(x, u) where x is the state and u is the input
    vector.)doc";

static const char *__doc_slp_OCP_for_each_step_2 =
R"doc(Set the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``callback``:
    The callback f(t, x, u, dt) where t is time, x is the state
    vector, u is the input vector, and dt is the timestep duration.)doc";

static const char *__doc_slp_OCP_initial_state =
R"doc(Convenience function to get the initial state in the trajectory.

Returns:
    The initial state of the trajectory.)doc";

static const char *__doc_slp_OCP_m_DT = R"doc()doc";

static const char *__doc_slp_OCP_m_U = R"doc()doc";

static const char *__doc_slp_OCP_m_X = R"doc()doc";

static const char *__doc_slp_OCP_m_dynamics = R"doc()doc";

static const char *__doc_slp_OCP_m_dynamics_type = R"doc()doc";

static const char *__doc_slp_OCP_m_num_steps = R"doc()doc";

static const char *__doc_slp_OCP_rk4 =
R"doc(Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for
dt.

Parameter ``f``:
    The function to integrate. It must take two arguments x and u.

Parameter ``x``:
    The initial value of x.

Parameter ``u``:
    The value u held constant over the integration period.

Parameter ``t0``:
    The initial time.

Parameter ``dt``:
    The time over which to integrate.)doc";

static const char *__doc_slp_OCP_set_lower_input_bound =
R"doc(Convenience function to set a lower bound on the input.

Parameter ``lower_bound``:
    The lower bound that inputs must always be above. Must be shaped
    (num_inputs)x1.)doc";

static const char *__doc_slp_OCP_set_max_timestep =
R"doc(Convenience function to set an upper bound on the timestep.

Parameter ``max_timestep``:
    The maximum timestep.)doc";

static const char *__doc_slp_OCP_set_min_timestep =
R"doc(Convenience function to set a lower bound on the timestep.

Parameter ``min_timestep``:
    The minimum timestep.)doc";

static const char *__doc_slp_OCP_set_upper_input_bound =
R"doc(Convenience function to set an upper bound on the input.

Parameter ``upper_bound``:
    The upper bound that inputs must always be below. Must be shaped
    (num_inputs)x1.)doc";

static const char *__doc_slp_Options = R"doc(Solver options.)doc";

static const char *__doc_slp_Options_diagnostics =
R"doc(Enables diagnostic prints.

<table> <tr> <th>Heading</th> <th>Description</th> </tr> <tr>
<td>iter</td> <td>Iteration number</td> </tr> <tr> <td>type</td>
<td>Iteration type (normal, accepted second-order correction, rejected
second-order correction)</td> </tr> <tr> <td>time (ms)</td>
<td>Duration of iteration in milliseconds</td> </tr> <tr>
<td>error</td> <td>Error estimate</td> </tr> <tr> <td>cost</td>
<td>Cost function value at current iterate</td> </tr> <tr>
<td>infeas.</td> <td>Constraint infeasibility at current iterate</td>
</tr> <tr> <td>complement.</td> <td>Complementary slackness at current
iterate (sᵀz)</td> </tr> <tr> <td>μ</td> <td>Barrier parameter</td>
</tr> <tr> <td>reg</td> <td>Iteration matrix regularization</td> </tr>
<tr> <td>primal α</td> <td>Primal step size</td> </tr> <tr> <td>dual
α</td> <td>Dual step size</td> </tr> <tr> <td>↩</td> <td>Number of
line search backtracks</td> </tr> </table>)doc";

static const char *__doc_slp_Options_feasible_ipm =
R"doc(Enables the feasible interior-point method. When the inequality
constraints are all feasible, step sizes are reduced when necessary to
prevent them becoming infeasible again. This is useful when parts of
the problem are ill-conditioned in infeasible regions (e.g., square
root of a negative value). This can slow or prevent progress toward a
solution though, so only enable it if necessary.)doc";

static const char *__doc_slp_Options_max_iterations = R"doc(The maximum number of solver iterations before returning a solution.)doc";

static const char *__doc_slp_Options_timeout = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

static const char *__doc_slp_Options_tolerance = R"doc(The solver will stop once the error is below this tolerance.)doc";

static const char *__doc_slp_Problem =
R"doc(This class allows the user to pose a constrained nonlinear
optimization problem in natural mathematical notation and solve it.

This class supports problems of the form:

```
minₓ f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
```

where f(x) is the scalar cost function, x is the vector of decision
variables (variables the solver can tweak to minimize the cost
function), cᵢ(x) are the inequality constraints, and cₑ(x) are the
equality constraints. Constraints are equations or inequalities of the
decision variables that constrain what values the solver is allowed to
use when searching for an optimal solution.

The nice thing about this class is users don't have to put their
system in the form shown above manually; they can write it in natural
mathematical form and it'll be converted for them.

Template parameter ``Scalar``:
    Scalar type.)doc";

static const char *__doc_slp_Problem_2 = R"doc()doc";

static const char *__doc_slp_Problem_Problem = R"doc(Construct the optimization problem.)doc";

static const char *__doc_slp_Problem_add_callback =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return void.

Parameter ``callback``:
    The callback.)doc";

static const char *__doc_slp_Problem_add_callback_2 =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return bool.

Parameter ``callback``:
    The callback. Returning true from the callback causes the solver
    to exit early with the solution it has so far.)doc";

static const char *__doc_slp_Problem_add_persistent_callback =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

Language bindings should call this in the Problem constructor to
register callbacks that shouldn't be removed by clear_callbacks().
Persistent callbacks run after non-persistent callbacks.

Parameter ``callback``:
    The callback. Returning true from the callback causes the solver
    to exit early with the solution it has so far.)doc";

static const char *__doc_slp_Problem_clear_callbacks = R"doc(Clears the registered callbacks.)doc";

static const char *__doc_slp_Problem_cost_function_type =
R"doc(Returns the cost function's type.

Returns:
    The cost function's type.)doc";

static const char *__doc_slp_Problem_decision_variable =
R"doc(Create a decision variable in the optimization problem.

Returns:
    A decision variable in the optimization problem.)doc";

static const char *__doc_slp_Problem_decision_variable_2 =
R"doc(Create a matrix of decision variables in the optimization problem.

Parameter ``rows``:
    Number of matrix rows.

Parameter ``cols``:
    Number of matrix columns.

Returns:
    A matrix of decision variables in the optimization problem.)doc";

static const char *__doc_slp_Problem_equality_constraint_type =
R"doc(Returns the type of the highest order equality constraint.

Returns:
    The type of the highest order equality constraint.)doc";

static const char *__doc_slp_Problem_inequality_constraint_type =
R"doc(Returns the type of the highest order inequality constraint.

Returns:
    The type of the highest order inequality constraint.)doc";

static const char *__doc_slp_Problem_m_decision_variables = R"doc()doc";

static const char *__doc_slp_Problem_m_equality_constraints = R"doc()doc";

static const char *__doc_slp_Problem_m_f = R"doc()doc";

static const char *__doc_slp_Problem_m_inequality_constraints = R"doc()doc";

static const char *__doc_slp_Problem_m_iteration_callbacks = R"doc()doc";

static const char *__doc_slp_Problem_m_persistent_iteration_callbacks = R"doc()doc";

static const char *__doc_slp_Problem_maximize =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize. A 1x1 VariableMatrix will
    implicitly convert to a Variable, and a non-1x1 VariableMatrix
    will raise an assertion.)doc";

static const char *__doc_slp_Problem_maximize_2 =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize. A 1x1 VariableMatrix will
    implicitly convert to a Variable, and a non-1x1 VariableMatrix
    will raise an assertion.)doc";

static const char *__doc_slp_Problem_minimize =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize. A 1x1 VariableMatrix will
    implicitly convert to a Variable, and a non-1x1 VariableMatrix
    will raise an assertion.)doc";

static const char *__doc_slp_Problem_minimize_2 =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize. A 1x1 VariableMatrix will
    implicitly convert to a Variable, and a non-1x1 VariableMatrix
    will raise an assertion.)doc";

static const char *__doc_slp_Problem_print_exit_conditions = R"doc()doc";

static const char *__doc_slp_Problem_print_problem_analysis = R"doc()doc";

static const char *__doc_slp_Problem_solve =
R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``options``:
    Solver options.

Parameter ``spy``:
    Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named
    H.spy, A_e.spy, and A_i.spy respectively during solve. Use
    tools/spy.py to plot them.

Returns:
    The solver status.)doc";

static const char *__doc_slp_Problem_subject_to =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_slp_Problem_subject_to_2 =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_slp_Problem_subject_to_3 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_slp_Problem_subject_to_4 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_slp_Problem_symmetric_decision_variable =
R"doc(Create a symmetric matrix of decision variables in the optimization
problem.

Variable instances are reused across the diagonal, which helps reduce
problem dimensionality.

Parameter ``rows``:
    Number of matrix rows.

Returns:
    A symmetric matrix of decision varaibles in the optimization
    problem.)doc";

static const char *__doc_slp_TimestepMethod = R"doc(Enum describing the type of system timestep.)doc";

static const char *__doc_slp_TimestepMethod_FIXED = R"doc(The timestep is a fixed constant.)doc";

static const char *__doc_slp_TimestepMethod_VARIABLE = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

static const char *__doc_slp_TimestepMethod_VARIABLE_SINGLE =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

static const char *__doc_slp_TranscriptionMethod = R"doc(Enum describing an OCP transcription method.)doc";

static const char *__doc_slp_TranscriptionMethod_DIRECT_COLLOCATION =
R"doc(The trajectory is modeled as a series of cubic polynomials where the
centerpoint slope is constrained.)doc";

static const char *__doc_slp_TranscriptionMethod_DIRECT_TRANSCRIPTION =
R"doc(Each state is a decision variable constrained to the integrated
dynamics of the previous state.)doc";

static const char *__doc_slp_TranscriptionMethod_SINGLE_SHOOTING =
R"doc(States depend explicitly as a function of all previous states and all
previous inputs.)doc";

static const char *__doc_slp_Variable =
R"doc(An autodiff variable pointing to an expression node.

Template parameter ``Scalar_``:
    Scalar type.)doc";

static const char *__doc_slp_VariableBlock =
R"doc(A submatrix of autodiff variables with reference semantics.

Template parameter ``Mat``:
    The type of the matrix whose storage this class points to.)doc";

static const char *__doc_slp_VariableBlock_T =
R"doc(Returns the transpose of the variable matrix.

Returns:
    The transpose of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_VariableBlock = R"doc(Copy constructor.)doc";

static const char *__doc_slp_VariableBlock_VariableBlock_2 = R"doc(Move constructor.)doc";

static const char *__doc_slp_VariableBlock_VariableBlock_3 =
R"doc(Constructs a Variable block pointing to all of the given matrix.

Parameter ``mat``:
    The matrix to which to point.)doc";

static const char *__doc_slp_VariableBlock_VariableBlock_4 =
R"doc(Constructs a Variable block pointing to a subset of the given matrix.

Parameter ``mat``:
    The matrix to which to point.

Parameter ``row_offset``:
    The block's row offset.

Parameter ``col_offset``:
    The block's column offset.

Parameter ``block_rows``:
    The number of rows in the block.

Parameter ``block_cols``:
    The number of columns in the block.)doc";

static const char *__doc_slp_VariableBlock_VariableBlock_5 =
R"doc(Constructs a Variable block pointing to a subset of the given matrix.

Note that the slices are taken as is rather than adjusted.

Parameter ``mat``:
    The matrix to which to point.

Parameter ``row_slice``:
    The block's row slice.

Parameter ``row_slice_length``:
    The block's row length.

Parameter ``col_slice``:
    The block's column slice.

Parameter ``col_slice_length``:
    The block's column length.)doc";

static const char *__doc_slp_VariableBlock_begin =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

static const char *__doc_slp_VariableBlock_begin_2 =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

static const char *__doc_slp_VariableBlock_block =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.

Returns:
    A block of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_block_2 =
R"doc(Returns a block slice of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.

Returns:
    A block slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_cbegin =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.)doc";

static const char *__doc_slp_VariableBlock_cend =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.)doc";

static const char *__doc_slp_VariableBlock_col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.

Returns:
    A column slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.

Returns:
    A column slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_cols =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.)doc";

static const char *__doc_slp_VariableBlock_const_iterator = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_const_iterator_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_m_index = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_m_mat = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_dec = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_dec_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableBlock_crbegin =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.)doc";

static const char *__doc_slp_VariableBlock_crend =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.)doc";

static const char *__doc_slp_VariableBlock_cwise_transform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unary_op``:
    The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.)doc";

static const char *__doc_slp_VariableBlock_end =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

static const char *__doc_slp_VariableBlock_end_2 =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

static const char *__doc_slp_VariableBlock_iterator = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_iterator = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_iterator_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_m_index = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_m_mat = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_dec = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_dec_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableBlock_m_col_slice = R"doc()doc";

static const char *__doc_slp_VariableBlock_m_col_slice_length = R"doc()doc";

static const char *__doc_slp_VariableBlock_m_mat = R"doc()doc";

static const char *__doc_slp_VariableBlock_m_row_slice = R"doc()doc";

static const char *__doc_slp_VariableBlock_m_row_slice_length = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_Variable = R"doc(Implicit conversion operator from 1x1 VariableBlock to Variable.)doc";

static const char *__doc_slp_VariableBlock_operator_array =
R"doc(Returns a scalar subblock at the given index.

Parameter ``index``:
    The scalar subblock's index.

Returns:
    A scalar subblock at the given index.)doc";

static const char *__doc_slp_VariableBlock_operator_array_2 =
R"doc(Returns a scalar subblock at the given index.

Parameter ``index``:
    The scalar subblock's index.

Returns:
    A scalar subblock at the given index.)doc";

static const char *__doc_slp_VariableBlock_operator_assign =
R"doc(Assigns a VariableBlock to the block.

Parameter ``values``:
    VariableBlock of values.

Returns:
    This VariableBlock.)doc";

static const char *__doc_slp_VariableBlock_operator_assign_2 =
R"doc(Assigns a VariableBlock to the block.

Parameter ``values``:
    VariableBlock of values.

Returns:
    This VariableBlock.)doc";

static const char *__doc_slp_VariableBlock_operator_assign_3 = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_assign_4 =
R"doc(Assigns an Eigen matrix to the block.

Parameter ``values``:
    Eigen matrix of values to assign.

Returns:
    This VariableBlock.)doc";

static const char *__doc_slp_VariableBlock_operator_assign_5 =
R"doc(Assigns a VariableMatrix to the block.

Parameter ``values``:
    VariableMatrix of values.

Returns:
    This VariableBlock.)doc";

static const char *__doc_slp_VariableBlock_operator_assign_6 =
R"doc(Assigns a VariableMatrix to the block.

Parameter ``values``:
    VariableMatrix of values.

Returns:
    This VariableBlock.)doc";

static const char *__doc_slp_VariableBlock_operator_call =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.

Returns:
    A scalar subblock at the given row and column.)doc";

static const char *__doc_slp_VariableBlock_operator_call_2 =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.

Returns:
    A scalar subblock at the given row and column.)doc";

static const char *__doc_slp_VariableBlock_operator_call_3 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_operator_call_4 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_operator_call_5 =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Parameter ``row_slice``:
    The row slice.

Parameter ``row_slice_length``:
    The row slice length.

Parameter ``col_slice``:
    The column slice.

Parameter ``col_slice_length``:
    The column slice length.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_operator_call_6 =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Parameter ``row_slice``:
    The row slice.

Parameter ``row_slice_length``:
    The row slice length.

Parameter ``col_slice``:
    The column slice.

Parameter ``col_slice_length``:
    The column slice length.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_operator_iadd = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_iadd_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_idiv = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_idiv_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_imul = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_imul_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_isub = R"doc()doc";

static const char *__doc_slp_VariableBlock_operator_isub_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_rbegin =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.)doc";

static const char *__doc_slp_VariableBlock_rbegin_2 =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.)doc";

static const char *__doc_slp_VariableBlock_rend =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.)doc";

static const char *__doc_slp_VariableBlock_rend_2 =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.)doc";

static const char *__doc_slp_VariableBlock_row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.

Returns:
    A row slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.

Returns:
    A row slice of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_rows =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.)doc";

static const char *__doc_slp_VariableBlock_segment =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.

Returns:
    A segment of the variable vector.)doc";

static const char *__doc_slp_VariableBlock_segment_2 =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.

Returns:
    A segment of the variable vector.)doc";

static const char *__doc_slp_VariableBlock_set_value =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Parameter ``value``:
    Value to assign.)doc";

static const char *__doc_slp_VariableBlock_set_value_2 =
R"doc(Sets block's internal values.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_slp_VariableBlock_size =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.)doc";

static const char *__doc_slp_VariableBlock_value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.

Returns:
    An element of the variable matrix.)doc";

static const char *__doc_slp_VariableBlock_value_2 =
R"doc(Returns an element of the variable block.

Parameter ``index``:
    The index of the element to return.

Returns:
    An element of the variable block.)doc";

static const char *__doc_slp_VariableBlock_value_3 =
R"doc(Returns the contents of the variable matrix.

Returns:
    The contents of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix =
R"doc(A matrix of autodiff variables.

Template parameter ``Scalar_``:
    Scalar type.)doc";

static const char *__doc_slp_VariableMatrix_T =
R"doc(Returns the transpose of the variable matrix.

Returns:
    The transpose of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix = R"doc(Constructs an empty VariableMatrix.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_2 =
R"doc(Constructs a zero-initialized VariableMatrix column vector with the
given rows.

Parameter ``rows``:
    The number of matrix rows.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_3 =
R"doc(Constructs a zero-initialized VariableMatrix with the given
dimensions.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_4 =
R"doc(Constructs an empty VariableMatrix with the given dimensions.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_5 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_6 =
R"doc(Constructs a scalar VariableMatrix from a nested list of scalars.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_7 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_8 =
R"doc(Constructs a VariableMatrix from an Eigen matrix.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_9 =
R"doc(Constructs a VariableMatrix from an Eigen diagonal matrix.

Parameter ``values``:
    Diagonal matrix of values.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_10 =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Parameter ``variable``:
    Variable.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_11 =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Parameter ``variable``:
    Variable.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_12 =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_13 =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_14 =
R"doc(Constructs a column vector wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_15 =
R"doc(Constructs a matrix wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_slp_VariableMatrix_begin =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_begin_2 =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_block =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.

Returns:
    A block of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_block_2 =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.

Returns:
    A block of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_cbegin =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_cend =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.)doc";

static const char *__doc_slp_VariableMatrix_col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.

Returns:
    A column slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.

Returns:
    A column slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_cols =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.)doc";

static const char *__doc_slp_VariableMatrix_const_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_const_iterator_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_m_it = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_dec = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_dec_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableMatrix_constant =
R"doc(Returns a variable matrix filled with a constant.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.

Parameter ``constant``:
    The constant.

Returns:
    A variable matrix filled with a constant.)doc";

static const char *__doc_slp_VariableMatrix_crbegin =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_crend =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.)doc";

static const char *__doc_slp_VariableMatrix_cwise_transform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unary_op``:
    The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.)doc";

static const char *__doc_slp_VariableMatrix_end =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

static const char *__doc_slp_VariableMatrix_end_2 =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.)doc";

static const char *__doc_slp_VariableMatrix_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_iterator_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_m_it = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_dec = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_dec_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_cols = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_rows = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_storage = R"doc()doc";

static const char *__doc_slp_VariableMatrix_one =
R"doc(Returns a variable matrix filled with ones.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.

Returns:
    A variable matrix filled with ones.)doc";

static const char *__doc_slp_VariableMatrix_operator_Variable = R"doc(Implicit conversion operator from 1x1 VariableMatrix to Variable.)doc";

static const char *__doc_slp_VariableMatrix_operator_array =
R"doc(Returns the element at the given index.

Parameter ``index``:
    The index.

Returns:
    The element at the given index.)doc";

static const char *__doc_slp_VariableMatrix_operator_array_2 =
R"doc(Returns the element at the given index.

Parameter ``index``:
    The index.

Returns:
    The element at the given index.)doc";

static const char *__doc_slp_VariableMatrix_operator_assign =
R"doc(Assigns an Eigen matrix to a VariableMatrix.

Parameter ``values``:
    Eigen matrix of values.

Returns:
    This VariableMatrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_assign_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_call =
R"doc(Returns the element at the given row and column.

Parameter ``row``:
    The row.

Parameter ``col``:
    The column.

Returns:
    The element at the given row and column.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_2 =
R"doc(Returns the element at the given row and column.

Parameter ``row``:
    The row.

Parameter ``col``:
    The column.

Returns:
    The element at the given row and column.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_3 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_4 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_5 =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Parameter ``row_slice``:
    The row slice.

Parameter ``row_slice_length``:
    The row slice length.

Parameter ``col_slice``:
    The column slice.

Parameter ``col_slice_length``:
    The column slice length.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_6 =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Parameter ``row_slice``:
    The row slice.

Parameter ``row_slice_length``:
    The row slice length.

Parameter ``col_slice``:
    The column slice.

Parameter ``col_slice_length``:
    The column slice length.

Returns:
    A slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_iadd = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_iadd_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_idiv = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_imul = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_imul_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_isub = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_isub_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_rbegin =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_rbegin_2 =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_rend =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.)doc";

static const char *__doc_slp_VariableMatrix_rend_2 =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.)doc";

static const char *__doc_slp_VariableMatrix_row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.

Returns:
    A row slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.

Returns:
    A row slice of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_rows =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.)doc";

static const char *__doc_slp_VariableMatrix_segment =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.

Returns:
    A segment of the variable vector.)doc";

static const char *__doc_slp_VariableMatrix_segment_2 =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.

Returns:
    A segment of the variable vector.)doc";

static const char *__doc_slp_VariableMatrix_set_value =
R"doc(Sets the VariableMatrix's internal values.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_slp_VariableMatrix_size =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.)doc";

static const char *__doc_slp_VariableMatrix_value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.

Returns:
    An element of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_value_2 =
R"doc(Returns an element of the variable matrix.

Parameter ``index``:
    The index of the element to return.

Returns:
    An element of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_value_3 =
R"doc(Returns the contents of the variable matrix.

Returns:
    The contents of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_zero =
R"doc(Returns a variable matrix filled with zeroes.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.

Returns:
    A variable matrix filled with zeroes.)doc";

static const char *__doc_slp_Variable_Variable = R"doc(Constructs a linear Variable with a value of zero.)doc";

static const char *__doc_slp_Variable_Variable_2 = R"doc(Constructs an empty Variable.)doc";

static const char *__doc_slp_Variable_Variable_3 =
R"doc(Constructs a Variable from a scalar type.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_slp_Variable_Variable_4 = R"doc()doc";

static const char *__doc_slp_Variable_Variable_5 = R"doc()doc";

static const char *__doc_slp_Variable_Variable_6 = R"doc()doc";

static const char *__doc_slp_Variable_Variable_7 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_slp_Variable_Variable_8 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_slp_Variable_expr = R"doc(The expression node)doc";

static const char *__doc_slp_Variable_m_graph =
R"doc(Used to update the value of this variable based on the values of its
dependent variables)doc";

static const char *__doc_slp_Variable_m_graph_initialized = R"doc(Used for lazy initialization of m_graph)doc";

static const char *__doc_slp_Variable_operator_assign = R"doc()doc";

static const char *__doc_slp_Variable_operator_iadd =
R"doc(Variable-Variable compound addition operator.

Parameter ``rhs``:
    Operator right-hand side.

Returns:
    Result of addition.)doc";

static const char *__doc_slp_Variable_operator_idiv =
R"doc(Variable-Variable compound division operator.

Parameter ``rhs``:
    Operator right-hand side.

Returns:
    Result of division.)doc";

static const char *__doc_slp_Variable_operator_imul =
R"doc(Variable-Variable compound multiplication operator.

Parameter ``rhs``:
    Operator right-hand side.

Returns:
    Result of multiplication.)doc";

static const char *__doc_slp_Variable_operator_isub =
R"doc(Variable-Variable compound subtraction operator.

Parameter ``rhs``:
    Operator right-hand side.

Returns:
    Result of subtraction.)doc";

static const char *__doc_slp_Variable_set_value =
R"doc(Sets Variable's internal value.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_slp_Variable_type =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).

Returns:
    The type of this expression.)doc";

static const char *__doc_slp_Variable_value =
R"doc(Returns the value of this variable.

Returns:
    The value of this variable.)doc";

static const char *__doc_slp_abs =
R"doc(abs() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_acos =
R"doc(acos() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_asin =
R"doc(asin() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_atan =
R"doc(atan() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_atan2 =
R"doc(atan2() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_slp_atan2_2 =
R"doc(atan2() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_slp_atan2_3 =
R"doc(atan2() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_slp_block =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_slp_block_2 =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_slp_bounds =
R"doc(Helper function for creating bound constraints.

Parameter ``l``:
    Lower bound.

Parameter ``x``:
    Variable to bound.

Parameter ``u``:
    Upper bound.)doc";

static const char *__doc_slp_cbrt =
R"doc(cbrt() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_cos =
R"doc(cos() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_cosh =
R"doc(cosh() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_cwise_reduce =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``lhs``:
    The left-hand side of the binary operator.

Parameter ``rhs``:
    The right-hand side of the binary operator.

Parameter ``binary_op``:
    The binary operator to use for the reduce operation.)doc";

static const char *__doc_slp_detail_GradientExpressionGraph = R"doc()doc";

static const char *__doc_slp_erf =
R"doc(erf() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_exp =
R"doc(exp() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_hypot =
R"doc(hypot() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_hypot_2 =
R"doc(hypot() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_hypot_3 =
R"doc(hypot() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_hypot_4 =
R"doc(hypot() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.

Parameter ``z``:
    The z argument.)doc";

static const char *__doc_slp_log =
R"doc(log() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_log10 =
R"doc(log10() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_make_constraints = R"doc()doc";

static const char *__doc_slp_make_constraints_2 = R"doc()doc";

static const char *__doc_slp_make_constraints_3 = R"doc()doc";

static const char *__doc_slp_make_constraints_4 = R"doc()doc";

static const char *__doc_slp_max =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_max_2 =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_max_3 =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_min =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_min_2 =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_min_3 =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``a``:
    The a argument.

Parameter ``b``:
    The b argument.)doc";

static const char *__doc_slp_operator_eq =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_eq_2 =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_eq_3 =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_ge =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_ge_2 =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_ge_3 =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_gt =
R"doc(Greater-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_le =
R"doc(Less-than-or-equal-to comparison operator that returns an inequality
constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_operator_lt =
R"doc(Less-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_slp_pow =
R"doc(pow() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_pow_2 =
R"doc(pow() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_pow_3 =
R"doc(pow() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_sign =
R"doc(sign() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_sin =
R"doc(sin() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_sinh =
R"doc(sinh() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_solve =
R"doc(Solves the VariableMatrix equation AX = B for X.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``A``:
    The left-hand side.

Parameter ``B``:
    The right-hand side.

Returns:
    The solution X.)doc";

static const char *__doc_slp_sqrt =
R"doc(sqrt() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_tan =
R"doc(tan() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_tanh =
R"doc(tanh() for Variables.

Template parameter ``Scalar``:
    Scalar type.

Parameter ``x``:
    The argument.)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

