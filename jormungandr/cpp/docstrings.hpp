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


static const char *__doc_Eigen_NumTraits =
R"doc(NumTraits specialization that allows instantiating Eigen types with
Variable.)doc";

static const char *__doc_slp_DynamicsType = R"doc(Enum describing a type of system dynamics constraints.)doc";

static const char *__doc_slp_DynamicsType_DISCRETE = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

static const char *__doc_slp_DynamicsType_EXPLICIT_ODE = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

static const char *__doc_slp_EqualityConstraints = R"doc(A vector of equality constraints of the form cₑ(x) = 0.)doc";

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

static const char *__doc_slp_ExitStatus_SOLVED_TO_ACCEPTABLE_TOLERANCE =
R"doc(Solved the problem to an acceptable tolerance, but not the desired
one.)doc";

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
R"doc(This class calculates the gradient of a a variable with respect to a
vector of variables.

The gradient is only recomputed if the variable expression is
quadratic or higher order.)doc";

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

static const char *__doc_slp_Gradient_get_profilers =
R"doc(Returns the profiler.

Returns:
    The profiler.)doc";

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

Template parameter ``UpLo``:
    Which part of the Hessian to compute (Lower or Lower | Upper).)doc";

static const char *__doc_slp_Hessian_2 = R"doc()doc";

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

static const char *__doc_slp_Hessian_get_profilers =
R"doc(Returns the profilers.

Returns:
    The profilers.)doc";

static const char *__doc_slp_Hessian_m_H = R"doc()doc";

static const char *__doc_slp_Hessian_m_cached_triplets = R"doc()doc";

static const char *__doc_slp_Hessian_m_graphs = R"doc()doc";

static const char *__doc_slp_Hessian_m_nonlinear_rows = R"doc()doc";

static const char *__doc_slp_Hessian_m_profilers = R"doc()doc";

static const char *__doc_slp_Hessian_m_variables = R"doc()doc";

static const char *__doc_slp_Hessian_m_wrt = R"doc()doc";

static const char *__doc_slp_Hessian_value =
R"doc(Evaluates the Hessian at wrt's value.

Returns:
    The Hessian at wrt's value.)doc";

static const char *__doc_slp_InequalityConstraints = R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.)doc";

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

static const char *__doc_slp_IterationInfo = R"doc(Solver iteration information exposed to a user callback.)doc";

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
quadratic or higher order.)doc";

static const char *__doc_slp_Jacobian_2 = R"doc()doc";

static const char *__doc_slp_Jacobian_Jacobian =
R"doc(Constructs a Jacobian object.

Parameter ``variable``:
    Variable of which to compute the Jacobian.

Parameter ``wrt``:
    Variable with respect to which to compute the Jacobian.)doc";

static const char *__doc_slp_Jacobian_Jacobian_2 = R"doc()doc";

static const char *__doc_slp_Jacobian_get =
R"doc(Returns the Jacobian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Jacobian as a VariableMatrix.)doc";

static const char *__doc_slp_Jacobian_get_profilers =
R"doc(Returns the profilers.

Returns:
    The profilers.)doc";

static const char *__doc_slp_Jacobian_m_J = R"doc()doc";

static const char *__doc_slp_Jacobian_m_cached_triplets = R"doc()doc";

static const char *__doc_slp_Jacobian_m_graphs = R"doc()doc";

static const char *__doc_slp_Jacobian_m_nonlinear_rows = R"doc()doc";

static const char *__doc_slp_Jacobian_m_profilers = R"doc()doc";

static const char *__doc_slp_Jacobian_m_variables = R"doc()doc";

static const char *__doc_slp_Jacobian_m_wrt = R"doc()doc";

static const char *__doc_slp_Jacobian_value =
R"doc(Evaluates the Jacobian at wrt's value.

Returns:
    The Jacobian at wrt's value.)doc";

static const char *__doc_slp_Multistart =
R"doc(Solves an optimization problem from different starting points in
parallel, then returns the solution with the lowest cost.

Each solve is performed on a separate thread. Solutions from
successful solves are always preferred over solutions from
unsuccessful solves, and cost (lower is better) is the tiebreaker
between successful solves.

Template parameter ``DecisionVariables``:
    The type containing the decision variable initial guess.

Parameter ``solve``:
    A user-provided function that takes a decision variable initial
    guess and returns a MultistartResult.

Parameter ``initial_guesses``:
    A list of decision variable initial guesses to try.)doc";

static const char *__doc_slp_MultistartResult =
R"doc(The result of a multistart solve.

Template parameter ``DecisionVariables``:
    The type containing the decision variable initial guess.)doc";

static const char *__doc_slp_MultistartResult_cost = R"doc(The solution's cost.)doc";

static const char *__doc_slp_MultistartResult_status = R"doc(The solver exit status.)doc";

static const char *__doc_slp_MultistartResult_variables = R"doc(The decision variables.)doc";

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
each transcription method.)doc";

static const char *__doc_slp_OCP_ForEachStep =
R"doc(Set the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``callback``:
    The callback f(t, x, u, dt) where t is time, x is the state
    vector, u is the input vector, and dt is the timestep duration.)doc";

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
    state transition function. - Explicit: dx/dt = f(x, u, *) -
    Implicit: f([x dx/dt]', u, *) = 0 - State transition: xₖ₊₁ = f(xₖ,
    uₖ)

Parameter ``dynamics_type``:
    The type of system evolution function.

Parameter ``timestep_method``:
    The timestep method.

Parameter ``method``:
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
    state transition function. - Explicit: dx/dt = f(t, x, u, *) -
    Implicit: f(t, [x dx/dt]', u, *) = 0 - State transition: xₖ₊₁ =
    f(t, xₖ, uₖ, dt)

Parameter ``dynamics_type``:
    The type of system evolution function.

Parameter ``timestep_method``:
    The timestep method.

Parameter ``method``:
    The transcription method.)doc";

static const char *__doc_slp_OCP_SetMaxTimestep =
R"doc(Convenience function to set an upper bound on the timestep.

Parameter ``max_timestep``:
    The maximum timestep.)doc";

static const char *__doc_slp_OCP_SetMinTimestep =
R"doc(Convenience function to set a lower bound on the timestep.

Parameter ``min_timestep``:
    The minimum timestep.)doc";

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

static const char *__doc_slp_OCP_constrain_direct_collocation = R"doc()doc";

static const char *__doc_slp_OCP_constrain_direct_transcription = R"doc()doc";

static const char *__doc_slp_OCP_constrain_final_state =
R"doc(Utility function to constrain the final state.

Parameter ``final_state``:
    the final state to constrain to.)doc";

static const char *__doc_slp_OCP_constrain_initial_state =
R"doc(Utility function to constrain the initial state.

Parameter ``initial_state``:
    the initial state to constrain to.)doc";

static const char *__doc_slp_OCP_constrain_single_shooting = R"doc()doc";

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

static const char *__doc_slp_OCP_initial_state =
R"doc(Convenience function to get the initial state in the trajectory.

Returns:
    The initial state of the trajectory.)doc";

static const char *__doc_slp_OCP_m_DT = R"doc()doc";

static const char *__doc_slp_OCP_m_U = R"doc()doc";

static const char *__doc_slp_OCP_m_X = R"doc()doc";

static const char *__doc_slp_OCP_m_dt = R"doc()doc";

static const char *__doc_slp_OCP_m_dynamics_function = R"doc()doc";

static const char *__doc_slp_OCP_m_dynamics_type = R"doc()doc";

static const char *__doc_slp_OCP_m_num_inputs = R"doc()doc";

static const char *__doc_slp_OCP_m_num_states = R"doc()doc";

static const char *__doc_slp_OCP_m_num_steps = R"doc()doc";

static const char *__doc_slp_OCP_m_timestep_method = R"doc()doc";

static const char *__doc_slp_OCP_m_transcription_method = R"doc()doc";

static const char *__doc_slp_OCP_set_lower_input_bound =
R"doc(Convenience function to set a lower bound on the input.

Parameter ``lower_bound``:
    The lower bound that inputs must always be above. Must be shaped
    (num_inputs)x1.)doc";

static const char *__doc_slp_OCP_set_upper_input_bound =
R"doc(Convenience function to set an upper bound on the input.

Parameter ``upper_bound``:
    The upper bound that inputs must always be below. Must be shaped
    (num_inputs)x1.)doc";

static const char *__doc_slp_Options = R"doc(Solver options.)doc";

static const char *__doc_slp_Options_acceptable_tolerance =
R"doc(The solver will stop once the error is below this tolerance for
`acceptable_iterations` iterations. This is useful in cases where the
solver might not be able to achieve the desired level of accuracy due
to floating-point round-off.)doc";

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

static const char *__doc_slp_Options_max_acceptable_iterations =
R"doc(The solver will stop once the error is below `acceptable_tolerance`
for this many iterations.)doc";

static const char *__doc_slp_Options_max_iterations = R"doc(The maximum number of solver iterations before returning a solution.)doc";

static const char *__doc_slp_Options_spy =
R"doc(Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named
H.spy, A_e.spy, and A_i.spy respectively during solve.

Use tools/spy.py to plot them.)doc";

static const char *__doc_slp_Options_timeout = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

static const char *__doc_slp_Options_tolerance = R"doc(The solver will stop once the error is below this tolerance.)doc";

static const char *__doc_slp_Problem =
R"doc(This class allows the user to pose a constrained nonlinear
optimization problem in natural mathematical notation and solve it.

This class supports problems of the form: @verbatim minₓ f(x) subject
to cₑ(x) = 0 cᵢ(x) ≥ 0 @endverbatim

where f(x) is the scalar cost function, x is the vector of decision
variables (variables the solver can tweak to minimize the cost
function), cᵢ(x) are the inequality constraints, and cₑ(x) are the
equality constraints. Constraints are equations or inequalities of the
decision variables that constrain what values the solver is allowed to
use when searching for an optimal solution.

The nice thing about this class is users don't have to put their
system in the form shown above manually; they can write it in natural
mathematical form and it'll be converted for them.)doc";

static const char *__doc_slp_Problem_Problem = R"doc(Construct the optimization problem.)doc";

static const char *__doc_slp_Problem_add_callback =
R"doc(Adds a callback to be called at each solver iteration.

The callback for this overload should return void.

Parameter ``callback``:
    The callback.)doc";

static const char *__doc_slp_Problem_add_callback_2 =
R"doc(Adds a callback to be called at each solver iteration.

The callback for this overload should return bool.

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

static const char *__doc_slp_Problem_m_callbacks = R"doc()doc";

static const char *__doc_slp_Problem_m_decision_variables = R"doc()doc";

static const char *__doc_slp_Problem_m_equality_constraints = R"doc()doc";

static const char *__doc_slp_Problem_m_f = R"doc()doc";

static const char *__doc_slp_Problem_m_inequality_constraints = R"doc()doc";

static const char *__doc_slp_Problem_maximize =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_slp_Problem_maximize_2 =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_slp_Problem_minimize =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_slp_Problem_minimize_2 =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_slp_Problem_solve =
R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``options``:
    Solver options.

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

static const char *__doc_slp_Slice = R"doc(Represents a sequence of elements in an iterable object.)doc";

static const char *__doc_slp_Slice_Slice = R"doc(Constructs a Slice.)doc";

static const char *__doc_slp_Slice_Slice_2 = R"doc(Constructs a slice.)doc";

static const char *__doc_slp_Slice_Slice_3 =
R"doc(Constructs a slice.

Parameter ``start``:
    Slice start index (inclusive).)doc";

static const char *__doc_slp_Slice_Slice_4 =
R"doc(Constructs a slice.

Parameter ``start``:
    Slice start index (inclusive).

Parameter ``stop``:
    Slice stop index (exclusive).)doc";

static const char *__doc_slp_Slice_Slice_5 =
R"doc(Constructs a slice.

Parameter ``start``:
    Slice start index (inclusive).

Parameter ``stop``:
    Slice stop index (exclusive).

Parameter ``step``:
    Slice step.)doc";

static const char *__doc_slp_Slice_adjust =
R"doc(Adjusts start and end slice indices assuming a sequence of the
specified length.

Parameter ``length``:
    The sequence length.

Returns:
    The slice length.)doc";

static const char *__doc_slp_Slice_start = R"doc(Start index (inclusive).)doc";

static const char *__doc_slp_Slice_step = R"doc(Step.)doc";

static const char *__doc_slp_Slice_stop = R"doc(Stop index (exclusive).)doc";

static const char *__doc_slp_TimestepMethod = R"doc(Enum describing the type of system timestep.)doc";

static const char *__doc_slp_TimestepMethod_FIXED = R"doc(The timestep is a fixed constant.)doc";

static const char *__doc_slp_TimestepMethod_VARIABLE = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

static const char *__doc_slp_TimestepMethod_VARIABLE_SINGLE =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

static const char *__doc_slp_ToMessage =
R"doc(Returns user-readable message corresponding to the expression type.

Parameter ``type``:
    Expression type.)doc";

static const char *__doc_slp_ToMessage_2 =
R"doc(Returns user-readable message corresponding to the solver exit status.

Parameter ``exit_status``:
    Solver exit status.)doc";

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

static const char *__doc_slp_Variable = R"doc(An autodiff variable pointing to an expression node.)doc";

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
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

static const char *__doc_slp_VariableBlock_cend =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

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

static const char *__doc_slp_VariableBlock_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableBlock_const_iterator_operator_mul = R"doc()doc";

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
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.

Returns:
    A scalar subblock at the given row.)doc";

static const char *__doc_slp_VariableBlock_operator_array_2 =
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.

Returns:
    A scalar subblock at the given row.)doc";

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
R"doc(Assigns a double to the block.

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
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.

Returns:
    A row of the variable column vector.)doc";

static const char *__doc_slp_VariableBlock_value_3 =
R"doc(Returns the contents of the variable matrix.

Returns:
    The contents of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix = R"doc(A matrix of autodiff variables.)doc";

static const char *__doc_slp_VariableMatrix_T =
R"doc(Returns the transpose of the variable matrix.

Returns:
    The transpose of the variable matrix.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix = R"doc(Constructs an empty VariableMatrix.)doc";

static const char *__doc_slp_VariableMatrix_VariableMatrix_2 =
R"doc(Constructs a VariableMatrix column vector with the given rows.

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
R"doc(Constructs a scalar VariableMatrix from a nested list of doubles.

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
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

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
R"doc(Returns begin iterator.

Returns:
    Begin iterator.)doc";

static const char *__doc_slp_VariableMatrix_cend =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

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

static const char *__doc_slp_VariableMatrix_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableMatrix_cwise_transform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unary_op``:
    The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.)doc";

static const char *__doc_slp_VariableMatrix_empty_t = R"doc(Type tag used to designate an uninitialized VariableMatrix.)doc";

static const char *__doc_slp_VariableMatrix_end =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

static const char *__doc_slp_VariableMatrix_end_2 =
R"doc(Returns end iterator.

Returns:
    End iterator.)doc";

static const char *__doc_slp_VariableMatrix_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_iterator = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_iterator_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_m_it = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_eq = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_inc = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_iterator_operator_mul = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_cols = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_rows = R"doc()doc";

static const char *__doc_slp_VariableMatrix_m_storage = R"doc()doc";

static const char *__doc_slp_VariableMatrix_ones =
R"doc(Returns a variable matrix filled with ones.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.

Returns:
    A variable matrix filled with ones.)doc";

static const char *__doc_slp_VariableMatrix_operator_Variable = R"doc(Implicit conversion operator from 1x1 VariableMatrix to Variable.)doc";

static const char *__doc_slp_VariableMatrix_operator_array =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.

Returns:
    A block pointing to the given row.)doc";

static const char *__doc_slp_VariableMatrix_operator_array_2 =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.

Returns:
    A block pointing to the given row.)doc";

static const char *__doc_slp_VariableMatrix_operator_assign =
R"doc(Assigns an Eigen matrix to a VariableMatrix.

Parameter ``values``:
    Eigen matrix of values.

Returns:
    This VariableMatrix.)doc";

static const char *__doc_slp_VariableMatrix_operator_assign_2 = R"doc()doc";

static const char *__doc_slp_VariableMatrix_operator_call =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.

Returns:
    A block pointing to the given row and column.)doc";

static const char *__doc_slp_VariableMatrix_operator_call_2 =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.

Returns:
    A block pointing to the given row and column.)doc";

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
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.

Returns:
    A row of the variable column vector.)doc";

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

static const char *__doc_slp_Variable_Variable_3 = R"doc()doc";

static const char *__doc_slp_Variable_Variable_4 = R"doc()doc";

static const char *__doc_slp_Variable_Variable_5 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_slp_Variable_Variable_6 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_slp_Variable_expr = R"doc(The expression node.)doc";

static const char *__doc_slp_Variable_m_graph =
R"doc(Updates the value of this variable based on the values of its
dependent variables)doc";

static const char *__doc_slp_Variable_operator_assign =
R"doc(Assignment operator for double.

Parameter ``value``:
    The value of the Variable.

Returns:
    This variable.)doc";

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
R"doc(std::abs() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_acos =
R"doc(std::acos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_asin =
R"doc(std::asin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_atan =
R"doc(std::atan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_atan2 =
R"doc(std::atan2() for Variables.

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

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_slp_block_2 =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_slp_cos =
R"doc(std::cos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_cosh =
R"doc(std::cosh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_cwise_reduce =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Parameter ``lhs``:
    The left-hand side of the binary operator.

Parameter ``rhs``:
    The right-hand side of the binary operator.

Parameter ``binary_op``:
    The binary operator to use for the reduce operation.)doc";

static const char *__doc_slp_detail_AbsExpression = R"doc(Derived expression type for std::abs().)doc";

static const char *__doc_slp_detail_AbsExpression_AbsExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_AbsExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_AbsExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_AbsExpression_type = R"doc()doc";

static const char *__doc_slp_detail_AbsExpression_value = R"doc()doc";

static const char *__doc_slp_detail_AcosExpression = R"doc(Derived expression type for std::acos().)doc";

static const char *__doc_slp_detail_AcosExpression_AcosExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_AcosExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_AcosExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_AcosExpression_type = R"doc()doc";

static const char *__doc_slp_detail_AcosExpression_value = R"doc()doc";

static const char *__doc_slp_detail_AdjointExpressionGraph =
R"doc(This class is an adaptor type that performs value updates of an
expression's adjoint graph.)doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_2 = R"doc()doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_AdjointExpressionGraph =
R"doc(Generates the adjoint graph for the given expression.

Parameter ``root``:
    The root node of the expression.)doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_append_adjoint_triplets =
R"doc(Updates the adjoints in the expression graph (computes the gradient)
then appends the adjoints of wrt to the sparse matrix triplets.

Parameter ``triplets``:
    The sparse matrix triplets.

Parameter ``row``:
    The row of wrt.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the Jacobian.)doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_generate_gradient_tree =
R"doc(Returns the variable's gradient tree.

This function lazily allocates variables, so elements of the returned
VariableMatrix will be empty if the corresponding element of wrt had
no adjoint. Ensure Variable::expr isn't nullptr before calling member
functions.

Parameter ``wrt``:
    Variables with respect to which to compute the gradient.

Returns:
    The variable's gradient tree.)doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_m_col_list = R"doc()doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_m_top_list = R"doc()doc";

static const char *__doc_slp_detail_AdjointExpressionGraph_update_values =
R"doc(Update the values of all nodes in this adjoint graph based on the
values of their dependent nodes.)doc";

static const char *__doc_slp_detail_AsinExpression = R"doc(Derived expression type for std::asin().)doc";

static const char *__doc_slp_detail_AsinExpression_AsinExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_AsinExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_AsinExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_AsinExpression_type = R"doc()doc";

static const char *__doc_slp_detail_AsinExpression_value = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression = R"doc(Derived expression type for std::atan2().)doc";

static const char *__doc_slp_detail_Atan2Expression_Atan2Expression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_Atan2Expression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression_type = R"doc()doc";

static const char *__doc_slp_detail_Atan2Expression_value = R"doc()doc";

static const char *__doc_slp_detail_AtanExpression = R"doc(Derived expression type for std::atan().)doc";

static const char *__doc_slp_detail_AtanExpression_AtanExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_AtanExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_AtanExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_AtanExpression_type = R"doc()doc";

static const char *__doc_slp_detail_AtanExpression_value = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression =
R"doc(Derived expression type for binary minus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_BinaryMinusExpression_2 =
R"doc(Derived expression type for binary minus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_BinaryMinusExpression_BinaryMinusExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_BinaryMinusExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression_type = R"doc()doc";

static const char *__doc_slp_detail_BinaryMinusExpression_value = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression =
R"doc(Derived expression type for binary plus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_BinaryPlusExpression_2 =
R"doc(Derived expression type for binary plus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_BinaryPlusExpression_BinaryPlusExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_BinaryPlusExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression_type = R"doc()doc";

static const char *__doc_slp_detail_BinaryPlusExpression_value = R"doc()doc";

static const char *__doc_slp_detail_ConstExpression = R"doc(Derived expression type for constant.)doc";

static const char *__doc_slp_detail_ConstExpression_2 = R"doc(Derived expression type for constant.)doc";

static const char *__doc_slp_detail_ConstExpression_ConstExpression = R"doc(Constructs a constant expression with a value of zero.)doc";

static const char *__doc_slp_detail_ConstExpression_ConstExpression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_slp_detail_ConstExpression_type = R"doc()doc";

static const char *__doc_slp_detail_ConstExpression_value = R"doc()doc";

static const char *__doc_slp_detail_CosExpression = R"doc(Derived expression type for std::cos().)doc";

static const char *__doc_slp_detail_CosExpression_CosExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_CosExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_CosExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_CosExpression_type = R"doc()doc";

static const char *__doc_slp_detail_CosExpression_value = R"doc()doc";

static const char *__doc_slp_detail_CoshExpression = R"doc(Derived expression type for std::cosh().)doc";

static const char *__doc_slp_detail_CoshExpression_CoshExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_CoshExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_CoshExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_CoshExpression_type = R"doc()doc";

static const char *__doc_slp_detail_CoshExpression_value = R"doc()doc";

static const char *__doc_slp_detail_DecisionVariableExpression = R"doc(Derived expression type for decision variable.)doc";

static const char *__doc_slp_detail_DecisionVariableExpression_DecisionVariableExpression = R"doc(Constructs a decision variable expression with a value of zero.)doc";

static const char *__doc_slp_detail_DecisionVariableExpression_DecisionVariableExpression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_slp_detail_DecisionVariableExpression_type = R"doc()doc";

static const char *__doc_slp_detail_DecisionVariableExpression_value = R"doc()doc";

static const char *__doc_slp_detail_DivExpression =
R"doc(Derived expression type for binary division operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_DivExpression_2 =
R"doc(Derived expression type for binary division operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_DivExpression_DivExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_DivExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_DivExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_DivExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_DivExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_DivExpression_type = R"doc()doc";

static const char *__doc_slp_detail_DivExpression_value = R"doc()doc";

static const char *__doc_slp_detail_ErfExpression = R"doc(Derived expression type for std::erf().)doc";

static const char *__doc_slp_detail_ErfExpression_ErfExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_ErfExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_ErfExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_ErfExpression_type = R"doc()doc";

static const char *__doc_slp_detail_ErfExpression_value = R"doc()doc";

static const char *__doc_slp_detail_ExpExpression = R"doc(Derived expression type for std::exp().)doc";

static const char *__doc_slp_detail_ExpExpression_ExpExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_ExpExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_ExpExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_ExpExpression_type = R"doc()doc";

static const char *__doc_slp_detail_ExpExpression_value = R"doc()doc";

static const char *__doc_slp_detail_Expression = R"doc(An autodiff expression node.)doc";

static const char *__doc_slp_detail_Expression_2 = R"doc(An autodiff expression node.)doc";

static const char *__doc_slp_detail_Expression_Expression = R"doc(Constructs a constant expression with a value of zero.)doc";

static const char *__doc_slp_detail_Expression_Expression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_slp_detail_Expression_Expression_3 =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_Expression_Expression_4 =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_Expression_adjoint = R"doc(The adjoint of the expression node used during autodiff.)doc";

static const char *__doc_slp_detail_Expression_adjoint_expr =
R"doc(The adjoint of the expression node used during gradient expression
tree generation.)doc";

static const char *__doc_slp_detail_Expression_args = R"doc(Expression arguments.)doc";

static const char *__doc_slp_detail_Expression_col = R"doc(This expression's column in a Jacobian, or -1 otherwise.)doc";

static const char *__doc_slp_detail_Expression_grad_expr_l =
R"doc(Returns Expression adjoint of the left child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.

Returns:
    The Expression adjoint of the left child expression.)doc";

static const char *__doc_slp_detail_Expression_grad_expr_r =
R"doc(Returns Expression adjoint of the right child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.

Returns:
    The Expression adjoint of the right child expression.)doc";

static const char *__doc_slp_detail_Expression_grad_l =
R"doc(Returns double adjoint of the left child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.

Returns:
    The double adjoint of the left child expression.)doc";

static const char *__doc_slp_detail_Expression_grad_r =
R"doc(Returns double adjoint of the right child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.

Returns:
    The double adjoint of the right child expression.)doc";

static const char *__doc_slp_detail_Expression_incoming_edges = R"doc(Counts incoming edges for this node.)doc";

static const char *__doc_slp_detail_Expression_is_constant =
R"doc(Returns true if the expression is the given constant.

Parameter ``constant``:
    The constant.

Returns:
    True if the expression is the given constant.)doc";

static const char *__doc_slp_detail_Expression_ref_count = R"doc(Reference count for intrusive shared pointer.)doc";

static const char *__doc_slp_detail_Expression_type =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).

Returns:
    The type of this expression.)doc";

static const char *__doc_slp_detail_Expression_val = R"doc(The value of the expression node.)doc";

static const char *__doc_slp_detail_Expression_value =
R"doc(Either nullary operator with no arguments, unary operator with one
argument, or binary operator with two arguments. This operator is used
to update the node's value.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Returns:
    The node's value.)doc";

static const char *__doc_slp_detail_HypotExpression = R"doc(Derived expression type for std::hypot().)doc";

static const char *__doc_slp_detail_HypotExpression_HypotExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_HypotExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_HypotExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_HypotExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_HypotExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_HypotExpression_type = R"doc()doc";

static const char *__doc_slp_detail_HypotExpression_value = R"doc()doc";

static const char *__doc_slp_detail_Log10Expression = R"doc(Derived expression type for std::log10().)doc";

static const char *__doc_slp_detail_Log10Expression_Log10Expression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_Log10Expression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_Log10Expression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_Log10Expression_type = R"doc()doc";

static const char *__doc_slp_detail_Log10Expression_value = R"doc()doc";

static const char *__doc_slp_detail_LogExpression = R"doc(Derived expression type for std::log().)doc";

static const char *__doc_slp_detail_LogExpression_LogExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_LogExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_LogExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_LogExpression_type = R"doc()doc";

static const char *__doc_slp_detail_LogExpression_value = R"doc()doc";

static const char *__doc_slp_detail_MultExpression =
R"doc(Derived expression type for binary multiplication operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_MultExpression_2 =
R"doc(Derived expression type for binary multiplication operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_MultExpression_MultExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_MultExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_MultExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_MultExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_MultExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_MultExpression_type = R"doc()doc";

static const char *__doc_slp_detail_MultExpression_value = R"doc()doc";

static const char *__doc_slp_detail_PowExpression =
R"doc(Derived expression type for std::pow().

Template parameter ``Expression``:
    type.)doc";

static const char *__doc_slp_detail_PowExpression_PowExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_slp_detail_PowExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_PowExpression_grad_expr_r = R"doc()doc";

static const char *__doc_slp_detail_PowExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_PowExpression_grad_r = R"doc()doc";

static const char *__doc_slp_detail_PowExpression_type = R"doc()doc";

static const char *__doc_slp_detail_PowExpression_value = R"doc()doc";

static const char *__doc_slp_detail_SignExpression = R"doc(Derived expression type for sign().)doc";

static const char *__doc_slp_detail_SignExpression_SignExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_SignExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_SignExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_SignExpression_type = R"doc()doc";

static const char *__doc_slp_detail_SignExpression_value = R"doc()doc";

static const char *__doc_slp_detail_SinExpression = R"doc(Derived expression type for std::sin().)doc";

static const char *__doc_slp_detail_SinExpression_SinExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_SinExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_SinExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_SinExpression_type = R"doc()doc";

static const char *__doc_slp_detail_SinExpression_value = R"doc()doc";

static const char *__doc_slp_detail_SinhExpression = R"doc(Derived expression type for std::sinh().)doc";

static const char *__doc_slp_detail_SinhExpression_SinhExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_SinhExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_SinhExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_SinhExpression_type = R"doc()doc";

static const char *__doc_slp_detail_SinhExpression_value = R"doc()doc";

static const char *__doc_slp_detail_SqrtExpression = R"doc(Derived expression type for std::sqrt().)doc";

static const char *__doc_slp_detail_SqrtExpression_SqrtExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_SqrtExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_SqrtExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_SqrtExpression_type = R"doc()doc";

static const char *__doc_slp_detail_SqrtExpression_value = R"doc()doc";

static const char *__doc_slp_detail_TanExpression = R"doc(Derived expression type for std::tan().)doc";

static const char *__doc_slp_detail_TanExpression_TanExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_TanExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_TanExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_TanExpression_type = R"doc()doc";

static const char *__doc_slp_detail_TanExpression_value = R"doc()doc";

static const char *__doc_slp_detail_TanhExpression = R"doc(Derived expression type for std::tanh().)doc";

static const char *__doc_slp_detail_TanhExpression_TanhExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_TanhExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_TanhExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_TanhExpression_type = R"doc()doc";

static const char *__doc_slp_detail_TanhExpression_value = R"doc()doc";

static const char *__doc_slp_detail_UnaryMinusExpression =
R"doc(Derived expression type for unary minus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_UnaryMinusExpression_2 =
R"doc(Derived expression type for unary minus operator.

Template parameter ``T``:
    Expression type.)doc";

static const char *__doc_slp_detail_UnaryMinusExpression_UnaryMinusExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_slp_detail_UnaryMinusExpression_grad_expr_l = R"doc()doc";

static const char *__doc_slp_detail_UnaryMinusExpression_grad_l = R"doc()doc";

static const char *__doc_slp_detail_UnaryMinusExpression_type = R"doc()doc";

static const char *__doc_slp_detail_UnaryMinusExpression_value = R"doc()doc";

static const char *__doc_slp_detail_abs =
R"doc(std::abs() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_acos =
R"doc(std::acos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_asin =
R"doc(std::asin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_atan =
R"doc(std::atan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_atan2 =
R"doc(std::atan2() for Expressions.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_slp_detail_cos =
R"doc(std::cos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_cosh =
R"doc(std::cosh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_dec_ref_count =
R"doc(Refcount decrement for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_slp_detail_dec_ref_count_2 =
R"doc(Refcount decrement for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_slp_detail_erf =
R"doc(std::erf() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_exp =
R"doc(std::exp() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_exp_2 =
R"doc(std::exp() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_hypot =
R"doc(std::hypot() for Expressions.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_detail_hypot_2 =
R"doc(std::hypot() for Expressions.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_detail_inc_ref_count =
R"doc(Refcount increment for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_slp_detail_inc_ref_count_2 =
R"doc(Refcount increment for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_slp_detail_log =
R"doc(std::log() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_log10 =
R"doc(std::log10() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_make_expression_ptr =
R"doc(Creates an intrusive shared pointer to an expression from the global
pool allocator.

Template parameter ``T``:
    The derived expression type.

Parameter ``args``:
    Constructor arguments for Expression.)doc";

static const char *__doc_slp_detail_pow =
R"doc(std::pow() for Expressions.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_detail_pow_2 =
R"doc(std::pow() for Expressions.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_detail_sign =
R"doc(sign() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sin =
R"doc(std::sin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sin_2 =
R"doc(std::sin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sinh =
R"doc(std::sinh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sinh_2 =
R"doc(std::sinh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sqrt =
R"doc(std::sqrt() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_sqrt_2 =
R"doc(std::sqrt() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_tan =
R"doc(std::tan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_tanh =
R"doc(std::tanh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_detail_topological_sort =
R"doc(Generate a topological sort of an expression graph from parent to
child.

https://en.wikipedia.org/wiki/Topological_sorting

Parameter ``root``:
    The root node of the expression.)doc";

static const char *__doc_slp_detail_update_values =
R"doc(Update the values of all nodes in this graph based on the values of
their dependent nodes.

Parameter ``list``:
    Topological sort of graph from parent to child.)doc";

static const char *__doc_slp_erf =
R"doc(std::erf() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_exp =
R"doc(std::exp() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_hypot =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_slp_hypot_2 =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.

Parameter ``z``:
    The z argument.)doc";

static const char *__doc_slp_interior_point =
R"doc(Finds the optimal solution to a nonlinear program using the interior-
point method.

A nonlinear program has the form:

@verbatim min_x f(x) subject to cₑ(x) = 0 cᵢ(x) ≥ 0 @endverbatim

where f(x) is the cost function, cₑ(x) are the equality constraints,
and cᵢ(x) are the inequality constraints.

Parameter ``decision_variables``:
    The list of decision variables.

Parameter ``equality_constraints``:
    The list of equality constraints.

Parameter ``inequality_constraints``:
    The list of inequality constraints.

Parameter ``f``:
    The cost function.

Parameter ``callbacks``:
    The list of user callbacks.

Parameter ``options``:
    Solver options.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Returns:
    The exit status.)doc";

static const char *__doc_slp_log =
R"doc(std::log() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_log10 =
R"doc(std::log10() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_make_constraints =
R"doc(Make a list of constraints.

The standard form for equality constraints is c(x) = 0, and the
standard form for inequality constraints is c(x) ≥ 0. This function
takes constraints of the form lhs = rhs or lhs ≥ rhs and converts them
to lhs - rhs = 0 or lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_slp_newton =
R"doc(Finds the optimal solution to a nonlinear program using Newton's
method.

A nonlinear program has the form:

@verbatim min_x f(x) @endverbatim

where f(x) is the cost function.

Parameter ``decision_variables``:
    The list of decision variables.

Parameter ``f``:
    The cost function.

Parameter ``callbacks``:
    The list of user callbacks.

Parameter ``options``:
    Solver options.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Returns:
    The exit status.)doc";

static const char *__doc_slp_operator_eq =
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
R"doc(std::pow() for Variables.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_slp_rk4 =
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

static const char *__doc_slp_sign =
R"doc(sign() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_sin =
R"doc(std::sin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_sinh =
R"doc(std::sinh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_slicing_none_t = R"doc(Type tag used to designate an omitted argument of the slice.)doc";

static const char *__doc_slp_solve =
R"doc(Solves the VariableMatrix equation AX = B for X.

Parameter ``A``:
    The left-hand side.

Parameter ``B``:
    The right-hand side.

Returns:
    The solution X.)doc";

static const char *__doc_slp_sqp =
R"doc(Finds the optimal solution to a nonlinear program using Sequential
Quadratic Programming (SQP).

A nonlinear program has the form:

@verbatim min_x f(x) subject to cₑ(x) = 0 @endverbatim

where f(x) is the cost function and cₑ(x) are the equality
constraints.

Parameter ``decision_variables``:
    The list of decision variables.

Parameter ``equality_constraints``:
    The list of equality constraints.

Parameter ``f``:
    The cost function.

Parameter ``callbacks``:
    The list of user callbacks.

Parameter ``options``:
    Solver options.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Returns:
    The exit status.)doc";

static const char *__doc_slp_sqrt =
R"doc(std::sqrt() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_tan =
R"doc(std::tan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_slp_tanh =
R"doc(std::tanh() for Variables.

Parameter ``x``:
    The argument.)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

