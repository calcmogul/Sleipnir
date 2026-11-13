/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a ## b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


constexpr char mkd_doc_formatter[] = R"doc(Formatter for ExpressionType.)doc";

constexpr char mkd_doc_formatter_2[] = R"doc(Formatter for ExitStatus.)doc";

constexpr char mkd_doc_formatter_format[] =
R"doc(Formats ExpressionType.

Args:
    type: Expression type.
    ctx: Format context.

Template Args:
    FmtContext: Format context type.

Returns:
    Format context iterator.

)doc";

constexpr char mkd_doc_formatter_format_2[] =
R"doc(Formats ExitStatus.

Args:
    exit_status: Exit status.
    ctx: Format context.

Template Args:
    FmtContext: Format context type.

Returns:
    Format context iterator.

)doc";

constexpr char mkd_doc_formatter_m_underlying[] = R"doc()doc";

constexpr char mkd_doc_formatter_m_underlying_2[] = R"doc()doc";

constexpr char mkd_doc_formatter_parse[] =
R"doc(Parse format string.

Args:
    ctx: Format parse context.

Returns:
    Format parse context iterator.

)doc";

constexpr char mkd_doc_formatter_parse_2[] =
R"doc(Parses format string.

Args:
    ctx: Format parse context.

Returns:
    Format parse context iterator.

)doc";

constexpr char mkd_doc_slp[] = R"doc()doc";

constexpr char mkd_doc_slp_2[] = R"doc()doc";

constexpr char mkd_doc_slp_3[] = R"doc()doc";

constexpr char mkd_doc_slp_4[] = R"doc()doc";

constexpr char mkd_doc_slp_5[] = R"doc()doc";

constexpr char mkd_doc_slp_DynamicsType[] = R"doc(Enum describing a type of system dynamics constraints.)doc";

constexpr char mkd_doc_slp_DynamicsType_DISCRETE[] = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

constexpr char mkd_doc_slp_DynamicsType_EXPLICIT_ODE[] = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

constexpr char mkd_doc_slp_EqualityConstraints[] =
R"doc(A vector of equality constraints of the form cₑ(x) = 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints[] =
R"doc(Concatenates multiple equality constraints.

Args:
    equality_constraints: The list of EqualityConstraints to
                          concatenate.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints_2[] =
R"doc(Concatenates multiple equality constraints.

This overload is for Python bindings only.

Args:
    equality_constraints: The list of EqualityConstraints to
                          concatenate.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints_3[] =
R"doc(Constructs an equality constraint from a left and right side.

The standard form for equality constraints is c(x) = 0. This function
takes a constraint of the form lhs = rhs and converts it to lhs - rhs
= 0.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_constraints[] = R"doc(A vector of scalar equality constraints.)doc";

constexpr char mkd_doc_slp_EqualityConstraints_operator_bool[] = R"doc(Implicit conversion operator to bool.)doc";

constexpr char mkd_doc_slp_ExitStatus[] = R"doc(Solver exit status. Negative values indicate failure.)doc";

constexpr char mkd_doc_slp_ExitStatus_CALLBACK_REQUESTED_STOP[] =
R"doc(The solver returned its solution so far after the user requested a
stop.)doc";

constexpr char mkd_doc_slp_ExitStatus_DIVERGING_ITERATES[] =
R"doc(The solver encountered diverging primal iterates xₖ and/or sₖ and gave
up.)doc";

constexpr char mkd_doc_slp_ExitStatus_FACTORIZATION_FAILED[] = R"doc(The linear system factorization failed.)doc";

constexpr char mkd_doc_slp_ExitStatus_FEASIBILITY_RESTORATION_FAILED[] =
R"doc(The solver failed to reach the desired tolerance, and feasibility
restoration failed to converge.)doc";

constexpr char mkd_doc_slp_ExitStatus_GLOBALLY_INFEASIBLE[] =
R"doc(The problem setup frontend determined the problem to have an empty
feasible region.)doc";

constexpr char mkd_doc_slp_ExitStatus_LINE_SEARCH_FAILED[] =
R"doc(The backtracking line search failed, and the problem isn't locally
infeasible.)doc";

constexpr char mkd_doc_slp_ExitStatus_LOCALLY_INFEASIBLE[] =
R"doc(The solver determined the problem to be locally infeasible and gave
up.)doc";

constexpr char mkd_doc_slp_ExitStatus_MAX_ITERATIONS_EXCEEDED[] =
R"doc(The solver returned its solution so far after exceeding the maximum
number of iterations.)doc";

constexpr char mkd_doc_slp_ExitStatus_NONFINITE_INITIAL_GUESS[] =
R"doc(The solver encountered nonfinite initial cost, constraints, or
derivatives and gave up.)doc";

constexpr char mkd_doc_slp_ExitStatus_SUCCESS[] = R"doc(Solved the problem to the desired tolerance.)doc";

constexpr char mkd_doc_slp_ExitStatus_TIMEOUT[] =
R"doc(The solver returned its solution so far after exceeding the maximum
elapsed wall clock time.)doc";

constexpr char mkd_doc_slp_ExitStatus_TOO_FEW_DOFS[] = R"doc(The solver determined the problem to be overconstrained and gave up.)doc";

constexpr char mkd_doc_slp_ExpressionType[] =
R"doc(Expression type.

Used for autodiff caching.)doc";

constexpr char mkd_doc_slp_ExpressionType_CONSTANT[] = R"doc(The expression is a constant.)doc";

constexpr char mkd_doc_slp_ExpressionType_LINEAR[] = R"doc(The expression is composed of linear and lower-order operators.)doc";

constexpr char mkd_doc_slp_ExpressionType_NONE[] = R"doc(There is no expression.)doc";

constexpr char mkd_doc_slp_ExpressionType_NONLINEAR[] = R"doc(The expression is composed of nonlinear and lower-order operators.)doc";

constexpr char mkd_doc_slp_ExpressionType_QUADRATIC[] = R"doc(The expression is composed of quadratic and lower-order operators.)doc";

constexpr char mkd_doc_slp_Gradient[] =
R"doc(This class calculates the gradient of a variable with respect to a
vector of variables.

The gradient is only recomputed if the variable expression is
quadratic or higher order.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Gradient_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_Gradient[] =
R"doc(Constructs a Gradient object.

Args:
    variable: Variable of which to compute the gradient.
    wrt: Variable with respect to which to compute the gradient.

)doc";

constexpr char mkd_doc_slp_Gradient_Gradient_2[] =
R"doc(Constructs a Gradient object.

Args:
    variable: Variable of which to compute the gradient.
    wrt: Vector of variables with respect to which to compute the
         gradient.

)doc";

constexpr char mkd_doc_slp_Gradient_get[] =
R"doc(Returns the gradient as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The gradient as a VariableMatrix.

)doc";

constexpr char mkd_doc_slp_Gradient_m_g[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_m_jacobian[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_value[] =
R"doc(Evaluates the gradient at wrt's value.

Returns:
    The gradient at wrt's value.

)doc";

constexpr char mkd_doc_slp_Hessian[] =
R"doc(This class calculates the Hessian of a variable with respect to a
vector of variables.

The gradient tree is cached so subsequent Hessian calculations are
faster, and the Hessian is only recomputed if the variable expression
is nonlinear.

Template Args:
    Scalar: Scalar type.
    UpLo: Which part of the Hessian to compute (Lower or Lower |
          Upper). Default is Lower | Upper.)doc";

constexpr char mkd_doc_slp_Hessian_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_Hessian[] =
R"doc(Constructs a Hessian object.

Args:
    variable: Variable of which to compute the Hessian.
    wrt: Variable with respect to which to compute the Hessian.

)doc";

constexpr char mkd_doc_slp_Hessian_Hessian_2[] =
R"doc(Constructs a Hessian object.

Args:
    variable: Variable of which to compute the Hessian.
    wrt: Vector of variables with respect to which to compute the
         Hessian.

)doc";

constexpr char mkd_doc_slp_Hessian_get[] =
R"doc(Returns the Hessian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Hessian as a VariableMatrix.

)doc";

constexpr char mkd_doc_slp_Hessian_m_H[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_cached_triplets[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_nonlinear_rows[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_output_lists[] = R"doc(List of output rows as column-node pairs)doc";

constexpr char mkd_doc_slp_Hessian_m_top_lists[] =
R"doc(List of topologically sorted graphs from parent to child, one for each
row)doc";

constexpr char mkd_doc_slp_Hessian_m_variables[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_wrt[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_value[] =
R"doc(Evaluates the Hessian at wrt's value.

Returns:
    The Hessian at wrt's value.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints[] =
R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints[] =
R"doc(Concatenates multiple inequality constraints.

Args:
    inequality_constraints: The list of InequalityConstraints to
                            concatenate.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints_2[] =
R"doc(Concatenates multiple inequality constraints.

This overload is for Python bindings only.

Args:
    inequality_constraints: The list of InequalityConstraints to
                            concatenate.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints_3[] =
R"doc(Constructs an inequality constraint from a left and right side.

The standard form for inequality constraints is c(x) ≥ 0. This
function takes a constraints of the form lhs ≥ rhs and converts it to
lhs - rhs ≥ 0.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_constraints[] = R"doc(A vector of scalar inequality constraints.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_operator_bool[] = R"doc(Implicit conversion operator to bool.)doc";

constexpr char mkd_doc_slp_IterationInfo[] =
R"doc(Solver iteration information exposed to an iteration callback.

If some decision variables lie on a manifold (e.g., from
Problem::decision_variable(so3)), x is the point in the ambient space,
and the derivatives are with respect to the manifold's tangent space
at x, so they have fewer columns than x has rows.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_IterationInfo_A_e[] = R"doc(The equality constraint Jacobian.)doc";

constexpr char mkd_doc_slp_IterationInfo_A_i[] = R"doc(The inequality constraint Jacobian.)doc";

constexpr char mkd_doc_slp_IterationInfo_H[] = R"doc(The Hessian of the Lagrangian.)doc";

constexpr char mkd_doc_slp_IterationInfo_g[] = R"doc(The gradient of the cost function.)doc";

constexpr char mkd_doc_slp_IterationInfo_iteration[] = R"doc(The solver iteration.)doc";

constexpr char mkd_doc_slp_IterationInfo_s[] = R"doc(The inequality constraint slack variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_x[] = R"doc(The decision variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_y[] = R"doc(The equality constraint dual variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_z[] = R"doc(The inequality constraint dual variables.)doc";

constexpr char mkd_doc_slp_Jacobian[] =
R"doc(This class calculates the Jacobian of a vector of variables with
respect to a vector of variables.

The Jacobian is only recomputed if the variable expression is
quadratic or higher order.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Jacobian_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian[] =
R"doc(Constructs a Jacobian object.

Args:
    variable: Variable of which to compute the Jacobian.
    wrt: Variable with respect to which to compute the Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian_2[] =
R"doc(Constructs a Jacobian object.

Args:
    variable: Variable of which to compute the Jacobian.
    wrt: Vector of variables with respect to which to compute the
         Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian_3[] =
R"doc(Constructs a Jacobian object.

Args:
    variables: Vector of variables of which to compute the Jacobian.
    wrt: Vector of variables with respect to which to compute the
         Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_get[] =
R"doc(Returns the Jacobian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Jacobian as a VariableMatrix.

)doc";

constexpr char mkd_doc_slp_Jacobian_m_J[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_m_cached_triplets[] = R"doc(Cached triplets for gradients of linear rows)doc";

constexpr char mkd_doc_slp_Jacobian_m_nonlinear_rows[] =
R"doc(List of row indices for nonlinear rows whose graients will be computed
in value())doc";

constexpr char mkd_doc_slp_Jacobian_m_output_lists[] = R"doc(List of output rows as column-node pairs)doc";

constexpr char mkd_doc_slp_Jacobian_m_top_lists[] =
R"doc(List of topologically sorted graphs from parent to child, one for each
row)doc";

constexpr char mkd_doc_slp_Jacobian_m_variables[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_m_wrt[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_value[] =
R"doc(Evaluates the Jacobian at wrt's value.

Returns:
    The Jacobian at wrt's value.

)doc";

constexpr char mkd_doc_slp_OCP[] =
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

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_OCP_2[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_OCP[] =
R"doc(Builds an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Args:
    num_states: The number of system states.
    num_inputs: The number of system inputs.
    dt: The timestep for fixed-step integration.
    num_steps: The number of control points.
    dynamics: Function representing an explicit or implicit ODE, or a
              discrete state transition function.

* Explicit: dx/dt = f(x, u, *)

 

* Implicit: f([x dx/dt]', u, *) = 0

 

* State transition: xₖ₊₁ = f(xₖ, uₖ)

 
    dynamics_type: The type of system evolution function.
    timestep_method: The timestep method.
    transcription_method: The transcription method.

)doc";

constexpr char mkd_doc_slp_OCP_OCP_2[] =
R"doc(Builds an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Args:
    num_states: The number of system states.
    num_inputs: The number of system inputs.
    dt: The timestep for fixed-step integration.
    num_steps: The number of control points.
    dynamics: Function representing an explicit or implicit ODE, or a
              discrete state transition function.

* Explicit: dx/dt = f(t, x, u, *)

 

* Implicit: f(t, [x dx/dt]', u, *) = 0

 

* State transition: xₖ₊₁ = f(t, xₖ, uₖ, dt)

 
    dynamics_type: The type of system evolution function.
    timestep_method: The timestep method.
    transcription_method: The transcription method.

)doc";

constexpr char mkd_doc_slp_OCP_U[] =
R"doc(Gets the input variables. After the problem is solved, this will
contain the inputs corresponding to the optimized trajectory.

Shaped (num_inputs)x(num_steps+1), although the last input step is
unused in the trajectory.

Returns:
    The input variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_X[] =
R"doc(Gets the state variables. After the problem is solved, this will
contain the optimized trajectory.

Shaped (num_states)x(num_steps+1).

Returns:
    The state variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_direct_collocation[] = R"doc(Applies direct collocation dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_constrain_direct_transcription[] = R"doc(Applies direct transcription dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_constrain_final_state[] =
R"doc(Constrains the final state.

Args:
    final_state: the final state to constrain to.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_initial_state[] =
R"doc(Constrains the initial state.

Args:
    initial_state: the initial state to constrain to.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_single_shooting[] = R"doc(Applies single shooting dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_dt[] =
R"doc(Gets the timestep variables. After the problem is solved, this will
contain the timesteps corresponding to the optimized trajectory.

Shaped 1x(num_steps+1), although the last timestep is unused in the
trajectory.

Returns:
    The timestep variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_final_state[] =
R"doc(Gets the final state in the trajectory.

Returns:
    The final state of the trajectory.

)doc";

constexpr char mkd_doc_slp_OCP_for_each_step[] =
R"doc(Sets the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input
VariableMatrices.

Args:
    callback: The callback f(x, u) where x is the state and u is the
              input vector.

)doc";

constexpr char mkd_doc_slp_OCP_for_each_step_2[] =
R"doc(Sets the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input
VariableMatrices.

Args:
    callback: The callback f(t, x, u, dt) where t is time, x is the
              state vector, u is the input vector, and dt is the
              timestep duration.

)doc";

constexpr char mkd_doc_slp_OCP_initial_state[] =
R"doc(Gets the initial state in the trajectory.

Returns:
    The initial state of the trajectory.

)doc";

constexpr char mkd_doc_slp_OCP_m_DT[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_U[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_X[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_dynamics[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_dynamics_type[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_num_steps[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_transcription_method[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_rk4[] =
R"doc(Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for
dt.

Args:
    f: The function to integrate. It must take two arguments x and u.
    x: The initial value of x.
    u: The value u held constant over the integration period.
    t0: The initial time.
    dt: The time over which to integrate.

)doc";

constexpr char mkd_doc_slp_OCP_set_lower_input_bound[] =
R"doc(Sets a lower bound on the input.

Args:
    lower_bound: The lower bound that inputs must always be above.
                 Must be shaped (num_inputs)x1.

)doc";

constexpr char mkd_doc_slp_OCP_set_max_timestep[] =
R"doc(Sets an upper bound on the timestep.

Args:
    max_timestep: The maximum timestep.

)doc";

constexpr char mkd_doc_slp_OCP_set_min_timestep[] =
R"doc(Sets a lower bound on the timestep.

Args:
    min_timestep: The minimum timestep.

)doc";

constexpr char mkd_doc_slp_OCP_set_upper_input_bound[] =
R"doc(Sets an upper bound on the input.

Args:
    upper_bound: The upper bound that inputs must always be below.
                 Must be shaped (num_inputs)x1.

)doc";

constexpr char mkd_doc_slp_Options[] =
R"doc(Solver options.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Options_diagnostics[] =
R"doc(Enables diagnostic output.

See https://sleipnirgroup.github.io/Sleipnir/md_usage.html#output for
more information.)doc";

constexpr char mkd_doc_slp_Options_feasible_ipm[] =
R"doc(Enables the feasible interior-point method.

When the inequality constraints are all feasible, step sizes are
reduced when necessary to prevent them becoming infeasible again. This
is useful when parts of the problem are ill-conditioned in infeasible
regions (e.g., square root of a negative value). This can slow or
prevent progress toward a solution though, so only enable it if
necessary.)doc";

constexpr char mkd_doc_slp_Options_max_iterations[] = R"doc(The maximum number of solver iterations before returning a solution.)doc";

constexpr char mkd_doc_slp_Options_timeout[] = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

constexpr char mkd_doc_slp_Options_tolerance[] =
R"doc(The solver will stop once the error is below this tolerance.

Defaults to the square root of the scalar type's machine epsilon.)doc";

constexpr char mkd_doc_slp_Pose2[] =
R"doc(A 2D pose of autodiff variables, which represents a rigid transform in
SE(2).

A pose is a translation t and a rotation R. It maps a point p in its
own frame to R p + t in the parent frame.

Poses are created by Problem::decision_variable(se2), from constants,
and from a translation and a Rotation2.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Pose2_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose2_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose2_Pose2[] =
R"doc(Constructs a constant pose.

Args:
    translation: The translation.
    angle: The rotation angle in radians.

)doc";

constexpr char mkd_doc_slp_Pose2_Pose2_2[] =
R"doc(Constructs a pose from a translation and a rotation.

Args:
    translation: The translation.
    rotation: The rotation.

)doc";

constexpr char mkd_doc_slp_Pose2_inverse[] =
R"doc(Returns the inverse pose.

It maps points in the parent frame to this pose's frame.

Returns:
    The inverse pose.

)doc";

constexpr char mkd_doc_slp_Pose2_log[] =
R"doc(Returns the twist that moves from the identity to this pose.

This is the inverse of Twist2::exp(), and the logarithm map of SE(2).
The twist's angular part is in (−π, π].

Returns:
    The twist.

)doc";

constexpr char mkd_doc_slp_Pose2_m_rotation[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose2_m_translation[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose2_relative_to[] =
R"doc(Returns this pose relative to another pose.

The result is this pose expressed in the other pose's frame, other⁻¹ *
this. It's often used as the error between a pose and a reference.

Args:
    other: The pose whose frame the result is expressed in.

Returns:
    This pose relative to the other pose.

)doc";

constexpr char mkd_doc_slp_Pose2_rotation[] =
R"doc(Returns the rotation.

Returns:
    The rotation.

)doc";

constexpr char mkd_doc_slp_Pose2_rotation_2[] =
R"doc(Returns the rotation.

Returns:
    The rotation.

)doc";

constexpr char mkd_doc_slp_Pose2_to_matrix[] =
R"doc(Returns the 3x3 homogeneous transformation matrix.

Returns:
    The 3x3 homogeneous transformation matrix.

)doc";

constexpr char mkd_doc_slp_Pose2_translation[] =
R"doc(Returns the translation.

Returns:
    The translation.

)doc";

constexpr char mkd_doc_slp_Pose2_translation_2[] =
R"doc(Returns the translation.

Returns:
    The translation.

)doc";

constexpr char mkd_doc_slp_Pose3[] =
R"doc(A 3D pose of autodiff variables, which represents a rigid transform in
SE(3).

A pose is a translation t and a rotation R. It maps a point p in its
own frame to R p + t in the parent frame.

Poses are created by Problem::decision_variable(se3), from constants,
and from a translation and a Rotation3.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Pose3_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose3_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose3_Pose3[] =
R"doc(Constructs a constant pose.

Args:
    translation: The translation.
    rotation: The rotation. It's normalized, so it must have nonzero
              norm.

)doc";

constexpr char mkd_doc_slp_Pose3_Pose3_2[] =
R"doc(Constructs a pose from a translation and a rotation.

Args:
    translation: The translation.
    rotation: The rotation.

)doc";

constexpr char mkd_doc_slp_Pose3_inverse[] =
R"doc(Returns the inverse pose.

It maps points in the parent frame to this pose's frame.

Returns:
    The inverse pose.

)doc";

constexpr char mkd_doc_slp_Pose3_log[] =
R"doc(Returns the twist that moves from the identity to this pose.

This is the inverse of Twist3::exp(), and the logarithm map of SE(3).
The twist's angular part is the rotation's to_vector(), whose angle is
in [0, π].

Returns:
    The twist.

)doc";

constexpr char mkd_doc_slp_Pose3_m_rotation[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose3_m_translation[] = R"doc()doc";

constexpr char mkd_doc_slp_Pose3_relative_to[] =
R"doc(Returns this pose relative to another pose.

The result is this pose expressed in the other pose's frame, other⁻¹ *
this. It's often used as the error between a pose and a reference.

Args:
    other: The pose whose frame the result is expressed in.

Returns:
    This pose relative to the other pose.

)doc";

constexpr char mkd_doc_slp_Pose3_rotation[] =
R"doc(Returns the rotation.

Returns:
    The rotation.

)doc";

constexpr char mkd_doc_slp_Pose3_rotation_2[] =
R"doc(Returns the rotation.

Returns:
    The rotation.

)doc";

constexpr char mkd_doc_slp_Pose3_to_matrix[] =
R"doc(Returns the 4x4 homogeneous transformation matrix.

Returns:
    The 4x4 homogeneous transformation matrix.

)doc";

constexpr char mkd_doc_slp_Pose3_translation[] =
R"doc(Returns the translation.

Returns:
    The translation.

)doc";

constexpr char mkd_doc_slp_Pose3_translation_2[] =
R"doc(Returns the translation.

Returns:
    The translation.

)doc";

constexpr char mkd_doc_slp_Problem[] =
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

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Problem_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_Problem[] = R"doc(Constructs the optimization problem.)doc";

constexpr char mkd_doc_slp_Problem_add_callback[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return void.

Args:
    callback: The callback.

)doc";

constexpr char mkd_doc_slp_Problem_add_callback_2[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return bool.

Args:
    callback: The callback. Returning true from the callback causes
              the solver to exit early with the solution it has so
              far.

)doc";

constexpr char mkd_doc_slp_Problem_add_persistent_callback[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

Language bindings should call this in the Problem constructor to
register callbacks that shouldn't be removed by clear_callbacks().
Persistent callbacks run after non-persistent callbacks.

Args:
    callback: The callback. Returning true from the callback causes
              the solver to exit early with the solution it has so
              far.

)doc";

constexpr char mkd_doc_slp_Problem_clear_callbacks[] = R"doc(Clears the registered callbacks.)doc";

constexpr char mkd_doc_slp_Problem_cost_function_type[] =
R"doc(Returns the cost function's type.

Returns:
    The cost function's type.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable[] =
R"doc(Creates a decision variable in the optimization problem.

Decision variables have an initial value of zero.

Returns:
    A decision variable in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_2[] =
R"doc(Creates a matrix of decision variables in the optimization problem.

Decision variables have an initial value of zero.

Args:
    rows: Number of matrix rows.
    cols: Number of matrix columns.

Returns:
    A matrix of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_3[] =
R"doc(Creates a unit vector of decision variables in the optimization
problem.

The solver keeps the vector on the unit sphere by stepping along the
sphere instead of through ℝⁿ, so no equality constraint is needed.

The vector has an initial value of e₁ = [1, 0, …, 0]ᵀ. Values set
elsewhere are normalized when the solve starts.

Args:
    rows: The vector's dimension. Must be at least 2.

Returns:
    A unit vector of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_4[] =
R"doc(Creates a 2D rotation decision variable in the optimization problem.

The solver keeps the rotation's complex number unit-length by stepping
along SO(2) instead of through ℝ², so no equality constraint is
needed.

The rotation has an initial value of zero radians. Values set
elsewhere are normalized when the solve starts.

Returns:
    A 2D rotation of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_5[] =
R"doc(Creates a 3D rotation decision variable in the optimization problem.

The solver keeps the quaternion unit-length by stepping along SO(3)
instead of through ℝ⁴, so no equality constraint is needed.

The rotation has an initial value of the identity. Values set
elsewhere are normalized when the solve starts.

Returns:
    A 3D rotation of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_6[] =
R"doc(Creates a 2D pose decision variable in the optimization problem.

The translation is two ordinary decision variables, and the rotation
is a 2D rotation decision variable (see decision_variable(so2)).

The pose has an initial value of the identity.

Returns:
    A 2D pose of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_decision_variable_7[] =
R"doc(Creates a 3D pose decision variable in the optimization problem.

The translation is three ordinary decision variables, and the rotation
is a 3D rotation decision variable (see decision_variable(so3)).

The pose has an initial value of the identity.

Returns:
    A 3D pose of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_equality_constraint_type[] =
R"doc(Returns the type of the highest order equality constraint.

Returns:
    The type of the highest order equality constraint.

)doc";

constexpr char mkd_doc_slp_Problem_inequality_constraint_type[] =
R"doc(Returns the type of the highest order inequality constraint.

Returns:
    The type of the highest order inequality constraint.

)doc";

constexpr char mkd_doc_slp_Problem_m_decision_variables[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_equality_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_f[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_inequality_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_iteration_callbacks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_manifold_blocks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_persistent_iteration_callbacks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_maximize[] =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    objective: The objective function to maximize. A 1x1
               VariableMatrix will implicitly convert to a Variable,
               and a non-1x1 VariableMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_maximize_2[] =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    objective: The objective function to maximize. A 1x1
               VariableMatrix will implicitly convert to a Variable,
               and a non-1x1 VariableMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_minimize[] =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    cost: The cost function to minimize. A 1x1 VariableMatrix will
          implicitly convert to a Variable, and a non-1x1
          VariableMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_minimize_2[] =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    cost: The cost function to minimize. A 1x1 VariableMatrix will
          implicitly convert to a Variable, and a non-1x1
          VariableMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_print_exit_conditions[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_print_problem_analysis[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_solve[] =
R"doc(Solves the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Args:
    options: Solver options.
    spy: Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files
         named H.spy, A_e.spy, and A_i.spy respectively during solve.
         Use tools/spy.py to plot them.

Returns:
    The solver status.

)doc";

constexpr char mkd_doc_slp_Problem_solve_ipm[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_solve_newton[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_solve_sqp[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_subject_to[] =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_2[] =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_3[] =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_4[] =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_symmetric_decision_variable[] =
R"doc(Creates a symmetric matrix of decision variables in the optimization
problem.

Variable instances are reused across the diagonal, which helps reduce
problem dimensionality.

Decision variables have an initial value of zero.

Args:
    rows: Number of matrix rows.

Returns:
    A symmetric matrix of decision variables in the optimization
    problem.

)doc";

constexpr char mkd_doc_slp_Rotation2[] =
R"doc(A 2D rotation of autodiff variables.

The rotation by θ is stored as the unit complex number cos θ + i sin
θ, so it has no discontinuity at ±π.

Rotations are created by Problem::decision_variable(so2), which keeps
them unit-length during the solve, from constant angles, and by
composition and inversion.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Rotation2_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_Rotation2[] =
R"doc(Constructs a constant rotation.

Args:
    angle: The rotation angle in radians.

)doc";

constexpr char mkd_doc_slp_Rotation2_Rotation2_2[] =
R"doc(Constructs a rotation from unit complex number components that are
already unit length.

This is for internal use. The variables aren't normalized.

Args:
    cos: The real component cos θ.
    sin: The imaginary component sin θ.

)doc";

constexpr char mkd_doc_slp_Rotation2_Rotation2_3[] =
R"doc(Constructs a rotation from an angle that depends on other variables.

This is the inverse of angle() for angles in (−π, π]. It's the
exponential map of SO(2).

Args:
    angle: The rotation angle in radians.

)doc";

constexpr char mkd_doc_slp_Rotation2_Rotation2_4[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_angle[] =
R"doc(Returns the rotation angle in radians.

The angle is in (−π, π]. It's discontinuous at ±π, so prefer cos() and
sin() in costs and constraints whose solutions may be near ±π.

Returns:
    The rotation angle in radians.

)doc";

constexpr char mkd_doc_slp_Rotation2_cos[] =
R"doc(Returns the real component cos θ.

Returns:
    The real component cos θ.

)doc";

constexpr char mkd_doc_slp_Rotation2_inverse[] =
R"doc(Returns the inverse rotation.

For a unit complex number, this is the conjugate.

Returns:
    The inverse rotation.

)doc";

constexpr char mkd_doc_slp_Rotation2_m_cos[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_m_sin[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_relative_to[] =
R"doc(Returns this rotation relative to another rotation.

The result is the rotation that takes the other rotation to this one,
other⁻¹ * this. Its angle() is the angle from the other rotation to
this one, in (−π, π], so it only wraps when the rotations are
opposite.

Args:
    other: The rotation this one is expressed relative to.

Returns:
    This rotation relative to the other rotation.

)doc";

constexpr char mkd_doc_slp_Rotation2_scalar_cos[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_scalar_sin[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation2_set_value[] =
R"doc(Sets the rotation angle.

This is only meaningful for rotations created by
Problem::decision_variable(so2).

Args:
    angle: The rotation angle in radians.

)doc";

constexpr char mkd_doc_slp_Rotation2_sin[] =
R"doc(Returns the imaginary component sin θ.

Returns:
    The imaginary component sin θ.

)doc";

constexpr char mkd_doc_slp_Rotation2_to_matrix[] =
R"doc(Returns the 2x2 rotation matrix.

Returns:
    The 2x2 rotation matrix.

)doc";

constexpr char mkd_doc_slp_Rotation2_value[] =
R"doc(Returns the rotation angle's value in radians, in (−π, π].

Returns:
    The rotation angle's value in radians.

)doc";

constexpr char mkd_doc_slp_Rotation3[] =
R"doc(A 3D rotation of autodiff variables.

The rotation is stored as a unit quaternion with components ordered
(w, x, y, z), where w is the scalar part.

Rotations are created by Problem::decision_variable(so3), which keeps
them unit-length during the solve, from constant Eigen quaternions,
and by composition and inversion.

q and −q represent the same rotation.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Rotation3_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_Rotation3[] =
R"doc(Constructs a constant rotation from a quaternion, which is normalized.

Args:
    q: The value. It must have nonzero norm.

)doc";

constexpr char mkd_doc_slp_Rotation3_Rotation3_2[] =
R"doc(Constructs a rotation from unit quaternion components that are already
unit length.

This is for internal use. The variables aren't normalized.

Args:
    w: The scalar component w.
    x: The vector component x.
    y: The vector component y.
    z: The vector component z.

)doc";

constexpr char mkd_doc_slp_Rotation3_Rotation3_3[] =
R"doc(Constructs a rotation from a rotation vector (axis times angle in
radians).

This is the inverse of to_vector() for angles in [0, π]. It's the
exponential map of SO(3). Its value and derivatives are accurate for
all rotation vectors, including zero.

Args:
    rvec: The rotation vector. Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Rotation3_Rotation3_4[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_components[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_from_rotation_vector[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_inverse[] =
R"doc(Returns the inverse rotation.

For a unit quaternion, this is the conjugate.

Returns:
    The inverse rotation.

)doc";

constexpr char mkd_doc_slp_Rotation3_m_w[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_m_x[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_m_y[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_m_z[] = R"doc()doc";

constexpr char mkd_doc_slp_Rotation3_relative_to[] =
R"doc(Returns this rotation relative to another rotation.

The result is the rotation that takes the other rotation to this one
in the other rotation's frame, other⁻¹ * this. Its to_vector() is the
rotation vector from the other rotation to this one.

Args:
    other: The rotation this one is expressed relative to.

Returns:
    This rotation relative to the other rotation.

)doc";

constexpr char mkd_doc_slp_Rotation3_set_value[] =
R"doc(Sets the rotation's value to a quaternion, which is normalized.

This is only meaningful for rotations created by
Problem::decision_variable(so3).

Args:
    q: The value. It must have nonzero norm.

)doc";

constexpr char mkd_doc_slp_Rotation3_to_matrix[] =
R"doc(Returns the 3x3 rotation matrix.

Returns:
    The 3x3 rotation matrix.

)doc";

constexpr char mkd_doc_slp_Rotation3_to_vector[] =
R"doc(Returns the rotation vector (axis times angle in radians).

The angle is in [0, π]. The rotation vector is discontinuous at π,
where the axis flips, so it's meant for rotations well below π, like
residuals between nearby rotations. Its value, gradient, and Hessian
are accurate near the identity.

Returns:
    The rotation vector.

)doc";

constexpr char mkd_doc_slp_Rotation3_value[] =
R"doc(Returns the rotation's value as a unit quaternion.

Returns:
    The quaternion's value.

)doc";

constexpr char mkd_doc_slp_Rotation3_w[] =
R"doc(Returns the scalar component w.

Returns:
    The scalar component w.

)doc";

constexpr char mkd_doc_slp_Rotation3_x[] =
R"doc(Returns the vector component x.

Returns:
    The vector component x.

)doc";

constexpr char mkd_doc_slp_Rotation3_y[] =
R"doc(Returns the vector component y.

Returns:
    The vector component y.

)doc";

constexpr char mkd_doc_slp_Rotation3_z[] =
R"doc(Returns the vector component z.

Returns:
    The vector component z.

)doc";

constexpr char mkd_doc_slp_TimestepMethod[] = R"doc(Enum describing the type of system timestep.)doc";

constexpr char mkd_doc_slp_TimestepMethod_FIXED[] = R"doc(The timestep is a fixed constant.)doc";

constexpr char mkd_doc_slp_TimestepMethod_VARIABLE[] = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

constexpr char mkd_doc_slp_TimestepMethod_VARIABLE_SINGLE[] =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod[] = R"doc(Enum describing an OCP transcription method.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_DIRECT_COLLOCATION[] =
R"doc(The trajectory is modeled as a series of cubic polynomials where the
centerpoint slope is constrained.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_DIRECT_TRANSCRIPTION[] =
R"doc(Each state is a decision variable constrained to the integrated
dynamics of the previous state.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_SINGLE_SHOOTING[] =
R"doc(States depend explicitly as a function of all previous states and all
previous inputs.)doc";

constexpr char mkd_doc_slp_Translation2[] =
R"doc(A 2D translation of autodiff variables.

A translation is a point or a displacement (x, y). It converts
implicitly to a 2x1 VariableMatrix.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Translation2_Translation2[] =
R"doc(Constructs a translation from its components.

Args:
    x: The x component.
    y: The y component.

)doc";

constexpr char mkd_doc_slp_Translation2_Translation2_2[] =
R"doc(Constructs a translation from a vector.

Args:
    vector: The vector (x, y). Must be 2x1.

)doc";

constexpr char mkd_doc_slp_Translation2_Translation2_3[] =
R"doc(Constructs a constant translation.

Args:
    vector: The vector (x, y). Must be 2x1.

)doc";

constexpr char mkd_doc_slp_Translation2_distance[] =
R"doc(Returns the distance to another translation.

The distance isn't differentiable where it's zero. For constraints and
costs that can reach zero distance, use squared_distance().

Args:
    other: The other translation.

Returns:
    The distance.

)doc";

constexpr char mkd_doc_slp_Translation2_m_x[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation2_m_y[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation2_operator_slp_VariableMatrix[] =
R"doc(Implicit conversion to a 2x1 VariableMatrix.

Returns:
    The translation as a 2x1 VariableMatrix.

)doc";

constexpr char mkd_doc_slp_Translation2_rotate_by[] =
R"doc(Returns this translation rotated by a rotation.

Args:
    rotation: The rotation.

Returns:
    The rotated translation.

)doc";

constexpr char mkd_doc_slp_Translation2_set_value[] =
R"doc(Sets the translation's value.

This is only meaningful for translations of decision variables.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Translation2_squared_distance[] =
R"doc(Returns the squared distance to another translation.

Args:
    other: The other translation.

Returns:
    The squared distance.

)doc";

constexpr char mkd_doc_slp_Translation2_transform_by[] =
R"doc(Returns this point transformed by a pose.

The result is R p + t, where p is this point, and R and t are the
pose's rotation and translation.

Args:
    pose: The pose.

Returns:
    The transformed point.

)doc";

constexpr char mkd_doc_slp_Translation2_value[] =
R"doc(Returns the translation's value.

Returns:
    The translation's value.

)doc";

constexpr char mkd_doc_slp_Translation2_x[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation2_x_2[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation2_y[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation2_y_2[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3[] =
R"doc(A 3D translation of autodiff variables.

A translation is a point or a displacement (x, y, z). It converts
implicitly to a 3x1 VariableMatrix.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Translation3_Translation3[] =
R"doc(Constructs a translation from its components.

Args:
    x: The x component.
    y: The y component.
    z: The z component.

)doc";

constexpr char mkd_doc_slp_Translation3_Translation3_2[] =
R"doc(Constructs a translation from a vector.

Args:
    vector: The vector (x, y, z). Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Translation3_Translation3_3[] =
R"doc(Constructs a constant translation.

Args:
    vector: The vector (x, y, z). Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Translation3_distance[] =
R"doc(Returns the distance to another translation.

The distance isn't differentiable where it's zero. For constraints and
costs that can reach zero distance, use squared_distance().

Args:
    other: The other translation.

Returns:
    The distance.

)doc";

constexpr char mkd_doc_slp_Translation3_m_x[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_m_y[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_m_z[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_operator_slp_VariableMatrix[] =
R"doc(Implicit conversion to a 3x1 VariableMatrix.

Returns:
    The translation as a 3x1 VariableMatrix.

)doc";

constexpr char mkd_doc_slp_Translation3_rotate_by[] =
R"doc(Returns this translation rotated by a rotation.

Args:
    rotation: The rotation.

Returns:
    The rotated translation.

)doc";

constexpr char mkd_doc_slp_Translation3_set_value[] =
R"doc(Sets the translation's value.

This is only meaningful for translations of decision variables.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Translation3_squared_distance[] =
R"doc(Returns the squared distance to another translation.

Args:
    other: The other translation.

Returns:
    The squared distance.

)doc";

constexpr char mkd_doc_slp_Translation3_transform_by[] =
R"doc(Returns this point transformed by a pose.

The result is R p + t, where p is this point, and R and t are the
pose's rotation and translation.

Args:
    pose: The pose.

Returns:
    The transformed point.

)doc";

constexpr char mkd_doc_slp_Translation3_value[] =
R"doc(Returns the translation's value.

Returns:
    The translation's value.

)doc";

constexpr char mkd_doc_slp_Translation3_x[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation3_x_2[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation3_y[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3_y_2[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3_z[] =
R"doc(Returns the z component.

Returns:
    The z component.

)doc";

constexpr char mkd_doc_slp_Translation3_z_2[] =
R"doc(Returns the z component.

Returns:
    The z component.

)doc";

constexpr char mkd_doc_slp_Twist2[] =
R"doc(A 2D twist of autodiff variables.

A twist is a change in pose along an arc of constant curvature,
expressed in the frame of the pose it starts from. It's a linear part
(dx, dy) and an angular part dθ in radians.

exp() returns the pose reached by moving along the twist from the
identity, and Pose2::log() is its inverse. A body-frame velocity times
a timestep is a twist, so pose * (velocity * dt).exp() integrates a
constant velocity for dt.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Twist2_Twist2[] =
R"doc(Constructs a twist from its linear and angular parts.

Args:
    linear: The linear part (dx, dy). Must be 2x1.
    angular: The angular part dθ in radians.

)doc";

constexpr char mkd_doc_slp_Twist2_angular[] =
R"doc(Returns the angular part dθ in radians.

Returns:
    The angular part.

)doc";

constexpr char mkd_doc_slp_Twist2_angular_2[] =
R"doc(Returns the angular part dθ in radians.

Returns:
    The angular part.

)doc";

constexpr char mkd_doc_slp_Twist2_exp[] =
R"doc(Returns the pose reached by moving along this twist from the identity.

This is the exponential map of SE(2). Its value and derivatives are
accurate for all twists, including those with zero angular part.

Returns:
    The pose.

)doc";

constexpr char mkd_doc_slp_Twist2_linear[] =
R"doc(Returns the linear part (dx, dy).

Returns:
    The linear part.

)doc";

constexpr char mkd_doc_slp_Twist2_linear_2[] =
R"doc(Returns the linear part (dx, dy).

Returns:
    The linear part.

)doc";

constexpr char mkd_doc_slp_Twist2_m_angular[] = R"doc()doc";

constexpr char mkd_doc_slp_Twist2_m_linear[] = R"doc()doc";

constexpr char mkd_doc_slp_Twist3[] =
R"doc(A 3D twist of autodiff variables.

A twist is a change in pose along a screw motion, expressed in the
frame of the pose it starts from. It's a linear part (dx, dy, dz) and
an angular part (rx, ry, rz), a rotation vector in radians.

exp() returns the pose reached by moving along the twist from the
identity, and Pose3::log() is its inverse. A body-frame velocity times
a timestep is a twist, so pose * (velocity * dt).exp() integrates a
constant velocity for dt.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Twist3_Twist3[] =
R"doc(Constructs a twist from its linear and angular parts.

Args:
    linear: The linear part (dx, dy, dz). Must be 3x1.
    angular: The angular part (rx, ry, rz) in radians. Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Twist3_angular[] =
R"doc(Returns the angular part (rx, ry, rz) in radians.

Returns:
    The angular part.

)doc";

constexpr char mkd_doc_slp_Twist3_angular_2[] =
R"doc(Returns the angular part (rx, ry, rz) in radians.

Returns:
    The angular part.

)doc";

constexpr char mkd_doc_slp_Twist3_exp[] =
R"doc(Returns the pose reached by moving along this twist from the identity.

This is the exponential map of SE(3). Its value and derivatives are
accurate for all twists, including those with zero angular part.

Returns:
    The pose.

)doc";

constexpr char mkd_doc_slp_Twist3_linear[] =
R"doc(Returns the linear part (dx, dy, dz).

Returns:
    The linear part.

)doc";

constexpr char mkd_doc_slp_Twist3_linear_2[] =
R"doc(Returns the linear part (dx, dy, dz).

Returns:
    The linear part.

)doc";

constexpr char mkd_doc_slp_Twist3_m_angular[] = R"doc()doc";

constexpr char mkd_doc_slp_Twist3_m_linear[] = R"doc()doc";

constexpr char mkd_doc_slp_UnitVector[] =
R"doc(A unit-length column vector of autodiff variables.

UnitVectors are created by Problem::decision_variable(sphere, n),
which keeps them on the unit sphere during the solve, from constant
Eigen vectors, and by operations that preserve unit length, like
negation and rotate_by(). Operations that don't preserve unit length,
like addition and scaling, return a VariableMatrix.

Elements are read-only. A UnitVector converts implicitly to a
VariableMatrix for use in general expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_UnitVector_2[] =
R"doc(A unit-length column vector of autodiff variables.

UnitVectors are created by Problem::decision_variable(sphere, n),
which keeps them on the unit sphere during the solve, from constant
Eigen vectors, and by operations that preserve unit length, like
negation and rotate_by(). Operations that don't preserve unit length,
like addition and scaling, return a VariableMatrix.

Elements are read-only. A UnitVector converts implicitly to a
VariableMatrix for use in general expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_UnitVector_T[] =
R"doc(Returns the transpose.

Returns:
    The transpose.

)doc";

constexpr char mkd_doc_slp_UnitVector_UnitVector[] =
R"doc(Constructs a constant unit vector from the normalized values.

Args:
    values: The values. They must form a column vector with nonzero
            norm.

)doc";

constexpr char mkd_doc_slp_UnitVector_UnitVector_2[] =
R"doc(Constructs a unit vector from variables that are already unit length.

This is for internal use. The variables aren't normalized.

Args:
    storage: The variables. They must form a column vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_UnitVector_3[] = R"doc()doc";

constexpr char mkd_doc_slp_UnitVector_begin[] =
R"doc(Returns an iterator to the beginning of the vector.

Returns:
    An iterator to the beginning of the vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_cols[] =
R"doc(Returns the number of columns, which is always 1.

Returns:
    The number of columns.

)doc";

constexpr char mkd_doc_slp_UnitVector_cross[] =
R"doc(Returns the cross product with another vector. Both vectors must be
3D.

Args:
    rhs: The other vector.

Returns:
    The cross product.

)doc";

constexpr char mkd_doc_slp_UnitVector_dot[] =
R"doc(Returns the dot product with another vector.

Args:
    rhs: The other vector.

Returns:
    The dot product.

)doc";

constexpr char mkd_doc_slp_UnitVector_end[] =
R"doc(Returns an iterator to the end of the vector.

Returns:
    An iterator to the end of the vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_m_storage[] = R"doc()doc";

constexpr char mkd_doc_slp_UnitVector_operator_array[] =
R"doc(Returns an element.

Args:
    row: The row of the element to return.

Returns:
    An element.

)doc";

constexpr char mkd_doc_slp_UnitVector_operator_array_2[] =
R"doc(Returns an element.

Args:
    row: The row of the element to return.
    col: The column of the element to return, which must be 0.

Returns:
    An element.

)doc";

constexpr char mkd_doc_slp_UnitVector_operator_slp_VariableMatrix[] =
R"doc(Converts the vector to a VariableMatrix.

Returns:
    The vector as a VariableMatrix.

)doc";

constexpr char mkd_doc_slp_UnitVector_rotate_by[] =
R"doc(Returns this 3D unit vector rotated by a rotation.

Args:
    q: The rotation.

Returns:
    The rotated unit vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_rotate_by_2[] =
R"doc(Returns this 2D unit vector rotated by a rotation.

Args:
    z: The rotation.

Returns:
    The rotated unit vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_rows[] =
R"doc(Returns the number of rows.

Returns:
    The number of rows.

)doc";

constexpr char mkd_doc_slp_UnitVector_set_value[] =
R"doc(Sets the vector's values to the normalized values.

This is only meaningful for UnitVectors created by
Problem::decision_variable(sphere, n).

Args:
    values: The values. They must have nonzero norm.

)doc";

constexpr char mkd_doc_slp_UnitVector_value[] =
R"doc(Returns an element's value.

Args:
    row: The row of the element whose value to return.

Returns:
    An element's value.

)doc";

constexpr char mkd_doc_slp_UnitVector_value_2[] =
R"doc(Returns the vector's values.

Returns:
    The vector's values.

)doc";

constexpr char mkd_doc_slp_Variable[] =
R"doc(An autodiff variable pointing to an expression node.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_VariableBlock[] =
R"doc(A submatrix of autodiff variables with reference semantics.

Template Args:
    Mat: The type of the matrix whose storage this class points to.)doc";

constexpr char mkd_doc_slp_VariableBlock_T[] =
R"doc(Returns the transpose of the variable matrix.

Returns:
    The transpose of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_VariableBlock[] = R"doc(Copy constructor.)doc";

constexpr char mkd_doc_slp_VariableBlock_VariableBlock_2[] = R"doc(Move constructor.)doc";

constexpr char mkd_doc_slp_VariableBlock_VariableBlock_3[] =
R"doc(Constructs a Variable block pointing to all of the given matrix.

Args:
    mat: The matrix to which to point.

)doc";

constexpr char mkd_doc_slp_VariableBlock_VariableBlock_4[] =
R"doc(Constructs a Variable block pointing to a subset of the given matrix.

Args:
    mat: The matrix to which to point.
    row_offset: The block's row offset.
    col_offset: The block's column offset.
    block_rows: The number of rows in the block.
    block_cols: The number of columns in the block.

)doc";

constexpr char mkd_doc_slp_VariableBlock_VariableBlock_5[] =
R"doc(Constructs a Variable block pointing to a subset of the given matrix.

Note that the slices are taken as is rather than adjusted.

Args:
    mat: The matrix to which to point.
    row_slice: The block's row slice.
    row_slice_length: The block's row length.
    col_slice: The block's column slice.
    col_slice_length: The block's column length.

)doc";

constexpr char mkd_doc_slp_VariableBlock_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_begin_2[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_block[] =
R"doc(Returns a block of the variable matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_block_2[] =
R"doc(Returns a block slice of the variable matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_col[] =
R"doc(Returns a column slice of the variable matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_col_2[] =
R"doc(Returns a column slice of the variable matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_const_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_const_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_crbegin[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_crend[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_end_2[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_exp[] =
R"doc(Returns the matrix exponential.

Returns:
    The matrix exponential.

)doc";

constexpr char mkd_doc_slp_VariableBlock_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_m_col_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_m_col_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_m_row_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_m_row_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array[] =
R"doc(Returns a scalar subblock at the given row and column.

Args:
    row: The scalar subblock's row.
    col: The scalar subblock's column.

Returns:
    A scalar subblock at the given row and column.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_2[] =
R"doc(Returns a scalar subblock at the given row and column.

Args:
    row: The scalar subblock's row.
    col: The scalar subblock's column.

Returns:
    A scalar subblock at the given row and column.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_3[] =
R"doc(Returns a scalar subblock at the given index.

Args:
    index: The scalar subblock's index.

Returns:
    A scalar subblock at the given index.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_4[] =
R"doc(Returns a scalar subblock at the given index.

Args:
    index: The scalar subblock's index.

Returns:
    A scalar subblock at the given index.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_5[] =
R"doc(Returns a slice of the variable matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_6[] =
R"doc(Returns a slice of the variable matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_7[] =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_array_8[] =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign[] =
R"doc(Assigns a VariableBlock to the block.

Args:
    values: VariableBlock of values.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign_2[] =
R"doc(Assigns a VariableBlock to the block.

Args:
    values: VariableBlock of values.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign_3[] =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign_4[] =
R"doc(Assigns an Eigen matrix to the block.

Args:
    values: Eigen matrix of values to assign.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign_5[] =
R"doc(Assigns a VariableMatrix to the block.

Args:
    values: VariableMatrix of values.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_assign_6[] =
R"doc(Assigns a VariableMatrix to the block.

Args:
    values: VariableMatrix of values.

Returns:
    This VariableBlock.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Variable to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_iadd_2[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Variable to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_idiv[] =
R"doc(Compound matrix division-assignment operator.

Args:
    rhs: Variable to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_imul[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Variable to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_imul_2[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Variable to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Variable to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_isub_2[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Variable to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_VariableBlock_operator_slp_Variable[] = R"doc(Implicit conversion operator from 1x1 VariableBlock to Variable.)doc";

constexpr char mkd_doc_slp_VariableBlock_rbegin[] =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_rbegin_2[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_rend[] =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_rend_2[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableBlock_row[] =
R"doc(Returns a row slice of the variable matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_row_2[] =
R"doc(Returns a row slice of the variable matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_segment[] =
R"doc(Returns a segment of the variable vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the variable vector.

)doc";

constexpr char mkd_doc_slp_VariableBlock_segment_2[] =
R"doc(Returns a segment of the variable vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the variable vector.

)doc";

constexpr char mkd_doc_slp_VariableBlock_set_value[] =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

)doc";

constexpr char mkd_doc_slp_VariableBlock_set_value_2[] =
R"doc(Sets block's internal values.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_VariableBlock_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_value[] =
R"doc(Returns an element of the variable matrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableBlock_value_2[] =
R"doc(Returns an element of the variable block.

Args:
    index: The index of the element to return.

Returns:
    An element of the variable block.

)doc";

constexpr char mkd_doc_slp_VariableBlock_value_3[] =
R"doc(Returns the contents of the variable matrix.

Returns:
    The contents of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_2[] =
R"doc(A matrix of autodiff variables.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_VariableMatrix_3[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_T[] =
R"doc(Returns the transpose of the variable matrix.

Returns:
    The transpose of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix[] = R"doc(Constructs an empty VariableMatrix.)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_2[] =
R"doc(Constructs a zero-initialized VariableMatrix column vector with the
given rows.

Args:
    rows: The number of matrix rows.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_3[] =
R"doc(Constructs a zero-initialized VariableMatrix with the given
dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_4[] =
R"doc(Constructs an empty VariableMatrix with the given dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_5[] =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

Args:
    list: The nested list of Variables.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_6[] =
R"doc(Constructs a scalar VariableMatrix from a nested list of scalars.

This overload is for Python bindings only.

Args:
    list: The nested list of Variables.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_7[] =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

This overload is for Python bindings only.

Args:
    list: The nested list of Variables.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_8[] =
R"doc(Constructs a VariableMatrix from an Eigen matrix.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_9[] =
R"doc(Constructs a VariableMatrix from an Eigen diagonal matrix.

Args:
    values: Diagonal matrix of values.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_10[] =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Args:
    variable: Variable.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_11[] =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Args:
    variable: Variable.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_12[] =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Args:
    values: VariableBlock of values.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_13[] =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Args:
    values: VariableBlock of values.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_14[] =
R"doc(Constructs a column vector wrapper around a Variable array.

Args:
    values: Variable array to wrap.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_VariableMatrix_15[] =
R"doc(Constructs a matrix wrapper around a Variable array.

Args:
    values: Variable array to wrap.
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_begin_2[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_block[] =
R"doc(Returns a block of the variable matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_block_2[] =
R"doc(Returns a block of the variable matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_col[] =
R"doc(Returns a column slice of the variable matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_col_2[] =
R"doc(Returns a column slice of the variable matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_const_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_m_it[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_const_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_constant[] =
R"doc(Returns a variable matrix filled with a constant.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.
    constant: The constant.

Returns:
    A variable matrix filled with a constant.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_crbegin[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_crend[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_end_2[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_exp[] =
R"doc(Returns the matrix exponential.

Returns:
    The matrix exponential.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_identity[] =
R"doc(Returns an identity variable matrix.

Args:
    rows: The number of matrix rows.

Returns:
    An identity variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_m_it[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_m_cols[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_m_rows[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_m_storage[] = R"doc()doc";

constexpr char mkd_doc_slp_VariableMatrix_one[] =
R"doc(Returns a variable matrix filled with ones.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

Returns:
    A variable matrix filled with ones.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_2[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_3[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_4[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_5[] =
R"doc(Returns a slice of the variable matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_6[] =
R"doc(Returns a slice of the variable matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_7[] =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_array_8[] =
R"doc(Returns a slice of the variable matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_assign[] =
R"doc(Assigns an Eigen matrix to a VariableMatrix.

Args:
    values: Eigen matrix of values.

Returns:
    This VariableMatrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_assign_2[] =
R"doc(Assigns a scalar to the matrix.

This only works for matrices with one row and one column.

Args:
    value: Value to assign.

Returns:
    This VariableMatrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Variable to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_iadd_2[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Variable to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_idiv[] =
R"doc(Compound matrix division-assignment operator.

Args:
    rhs: Variable to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_imul[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Variable to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_imul_2[] =
R"doc(Compound matrix-scalar multiplication-assignment operator.

Args:
    rhs: Variable to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Variable to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_isub_2[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Variable to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_operator_slp_Variable[] = R"doc(Implicit conversion operator from 1x1 VariableMatrix to Variable.)doc";

constexpr char mkd_doc_slp_VariableMatrix_rbegin[] =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rbegin_2[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rend[] =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rend_2[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rotate_by[] =
R"doc(Returns this 3D vector rotated by a rotation.

Args:
    q: The rotation.

Returns:
    The rotated vector.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rotate_by_2[] =
R"doc(Returns this 2D vector rotated by a rotation.

Args:
    z: The rotation.

Returns:
    The rotated vector.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_row[] =
R"doc(Returns a row slice of the variable matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_row_2[] =
R"doc(Returns a row slice of the variable matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_segment[] =
R"doc(Returns a segment of the variable vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the variable vector.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_segment_2[] =
R"doc(Returns a segment of the variable vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the variable vector.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_set_value[] =
R"doc(Sets the VariableMatrix's internal values.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_to_eigen[] =
R"doc(Converts the VariableMatrix to an Eigen matrix.

Returns:
    Eigen matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_value[] =
R"doc(Returns an element of the variable matrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_value_2[] =
R"doc(Returns an element of the variable matrix.

Args:
    index: The index of the element to return.

Returns:
    An element of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_value_3[] =
R"doc(Returns the contents of the variable matrix.

Returns:
    The contents of the variable matrix.

)doc";

constexpr char mkd_doc_slp_VariableMatrix_zero[] =
R"doc(Returns a variable matrix filled with zeros.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

Returns:
    A variable matrix filled with zeros.

)doc";

constexpr char mkd_doc_slp_Variable_Variable[] = R"doc(Constructs a linear Variable with a value of zero.)doc";

constexpr char mkd_doc_slp_Variable_Variable_2[] = R"doc(Constructs an empty Variable.)doc";

constexpr char mkd_doc_slp_Variable_Variable_3[] =
R"doc(Constructs a Variable from a scalar type.

Args:
    value: The value of the Variable.

)doc";

constexpr char mkd_doc_slp_Variable_Variable_4[] =
R"doc(Constructs a Variable from a scalar type.

Args:
    value: The value of the Variable.

)doc";

constexpr char mkd_doc_slp_Variable_Variable_5[] =
R"doc(Constructs a Variable from a floating-point type.

Args:
    value: The value of the Variable.

)doc";

constexpr char mkd_doc_slp_Variable_Variable_6[] =
R"doc(Constructs a Variable from an integral type.

Args:
    value: The value of the Variable.

)doc";

constexpr char mkd_doc_slp_Variable_Variable_7[] =
R"doc(Constructs a Variable pointing to the specified expression.

Args:
    expr: The autodiff variable.

)doc";

constexpr char mkd_doc_slp_Variable_Variable_8[] =
R"doc(Constructs a Variable pointing to the specified expression.

Args:
    expr: The autodiff variable.

)doc";

constexpr char mkd_doc_slp_Variable_expr[] = R"doc(The expression node)doc";

constexpr char mkd_doc_slp_Variable_m_graph[] =
R"doc(Used to update the value of this variable based on the values of its
dependent variables)doc";

constexpr char mkd_doc_slp_Variable_m_graph_initialized[] = R"doc(Used for lazy initialization of m_graph)doc";

constexpr char mkd_doc_slp_Variable_operator_assign[] =
R"doc(Assignment operator for scalar.

Args:
    value: The value of the Variable.

Returns:
    This variable.

)doc";

constexpr char mkd_doc_slp_Variable_operator_iadd[] =
R"doc(Variable-Variable compound addition operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_Variable_operator_idiv[] =
R"doc(Variable-Variable compound division operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_Variable_operator_imul[] =
R"doc(Variable-Variable compound multiplication operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_Variable_operator_isub[] =
R"doc(Variable-Variable compound subtraction operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_Variable_set_value[] =
R"doc(Sets Variable's internal value.

Args:
    value: The value of the Variable.

)doc";

constexpr char mkd_doc_slp_Variable_type[] =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).

Returns:
    The type of this expression.

)doc";

constexpr char mkd_doc_slp_Variable_value[] =
R"doc(Returns the value of this variable.

Returns:
    The value of this variable.

)doc";

constexpr char mkd_doc_slp_abs[] =
R"doc(abs() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_acos[] =
R"doc(acos() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_asin[] =
R"doc(asin() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan[] =
R"doc(atan() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2[] =
R"doc(atan2() for Variables.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2_2[] =
R"doc(atan2() for Variables.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2_3[] =
R"doc(atan2() for Variables.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_block[] =
R"doc(Assembles a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

Args:
    list: The nested list of blocks.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_block_2[] =
R"doc(Assembles a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Args:
    list: The nested list of blocks.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_bounds[] =
R"doc(Helper function for creating bound constraints.

Args:
    l: Lower bound.
    x: Variable to bound.
    u: Upper bound.

)doc";

constexpr char mkd_doc_slp_cbrt[] =
R"doc(cbrt() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cos[] =
R"doc(cos() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cosh[] =
R"doc(cosh() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cwise_reduce[] =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Args:
    lhs: The left-hand side of the binary operator.
    rhs: The right-hand side of the binary operator.
    binary_op: The binary operator to use for the reduce operation.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_detail_atan2_over_norm[] =
R"doc(Returns atan2(n, w)/n, where n = √n².

Near n = 0, atan2(n, w)/n loses precision, and its derivatives are
infinite at n = 0. Where w > 0 and n is small relative to w, this uses

  atan2(n, w)/n = 1/w atan(√t)/√t
                = 1/w (1 − t/3 + t²/5 − t³/7 + …)

with t = n²/w², a series in n² that's smooth at n = 0. Truncating it
after t⁷ makes its Hessian error grow like n¹⁴, and the atan2 form's
Hessian roundoff grows like ε/n³, so they cross at t ≈ ε^(2/17).

The result is infinite at n = 0 with w < 0.

Args:
    n_sq: n².
    w: w.

Returns:
    atan2(n, w)/n.

)doc";

constexpr char mkd_doc_slp_detail_cross[] =
R"doc(Returns the cross product of two 3D vectors.

Args:
    a: The first vector.
    b: The second vector.

Returns:
    a × b.

)doc";

constexpr char mkd_doc_slp_detail_gradient_tree[] = R"doc()doc";

constexpr char mkd_doc_slp_detail_gradient_tree_2[] =
R"doc(Returns the variable's gradient tree.

This function lazily allocates variables, so elements of the returned
VariableMatrix will be empty if the corresponding element of wrt had
no adjoint. Ensure Variable::expr isn't nullptr before calling member
functions.

Args:
    top_list: Topologically sorted graph from parent to child.
    wrt: Variables with respect to which to compute the gradient.

Template Args:
    Scalar: Scalar type.

Returns:
    The variable's gradient tree.

)doc";

constexpr char mkd_doc_slp_detail_is_constant[] =
R"doc(Returns whether every element of a unit vector is constant.

Args:
    v: The unit vector.

Returns:
    Whether every element is constant.

)doc";

constexpr char mkd_doc_slp_detail_sphere_log[] =
R"doc(Returns the sphere's logarithm map log_b(a) in an orthonormal basis of
b's tangent space.

log_b(a) is the tangent vector at b that points toward a, with length
equal to the angle between a and b. Its only zero is a = b.

The basis comes from a Householder reflection built around b's current
value. It's valid unless b later moves to that reflection's one
singular point, which is more than 90° from b's current value.

Args:
    a: The unit vector the logarithm points to.
    b: The unit vector whose tangent space the logarithm is in.

Returns:
    The n − 1 components of log_b(a).

)doc";

constexpr char mkd_doc_slp_detail_unit_length_t[] =
R"doc(Tag type for constructing a unit-length type from values that are
already unit length.)doc";

constexpr char mkd_doc_slp_detail_unit_length_t_unit_length_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_erf[] =
R"doc(erf() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_exp[] =
R"doc(Returns the pose reached by moving along this twist from the identity.

This is the exponential map of SE(2). Its value and derivatives are
accurate for all twists, including those with zero angular part.

Returns:
    The pose.

)doc";

constexpr char mkd_doc_slp_exp_2[] =
R"doc(Returns the pose reached by moving along this twist from the identity.

This is the exponential map of SE(3). Its value and derivatives are
accurate for all twists, including those with zero angular part.

Returns:
    The pose.

)doc";

constexpr char mkd_doc_slp_exp_3[] =
R"doc(exp() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot[] =
R"doc(hypot() for Variables.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_2[] =
R"doc(hypot() for Variables.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_3[] =
R"doc(hypot() for Variables.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_4[] =
R"doc(hypot() for Variables.

Args:
    x: The x argument.
    y: The y argument.
    z: The z argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_if_else[] =
R"doc(if_else() for Variables.

Returns t if cond(a, b) is true, otherwise f. The condition is
evaluated on the values of a and b each time the result is evaluated,
and the gradient only flows through the selected branch.

Args:
    cond: Condition evaluated on a and b.
    a: The condition's first argument.
    b: The condition's second argument.
    t: Value selected when cond(a, b) is true.
    f: Value selected when cond(a, b) is false.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_log[] =
R"doc(log() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_log10[] =
R"doc(log10() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_make_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_2[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_3[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_4[] = R"doc()doc";

constexpr char mkd_doc_slp_max[] =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_max_2[] =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_max_3[] =
R"doc(max() for Variables.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min[] =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min_2[] =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min_3[] =
R"doc(min() for Variables.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_operator_div[] =
R"doc(Binary division operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_operator_eq[] =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_eq_2[] =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_eq_3[] =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_ge[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_ge_2[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_ge_3[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_gt[] =
R"doc(Greater-than comparison operator that returns an inequality constraint
for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_le[] =
R"doc(Less-than-or-equal-to comparison operator that returns an inequality
constraint for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_lt[] =
R"doc(Less-than comparison operator that returns an inequality constraint
for two Variables.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_mul[] =
R"doc(Matrix-scalar multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_operator_mul_2[] =
R"doc(Scalar-matrix multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_pow[] =
R"doc(pow() for Variables.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_pow_2[] =
R"doc(pow() for Variables.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_pow_3[] =
R"doc(pow() for Variables.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_rotate_by[] =
R"doc(Returns this 2D vector rotated by a rotation.

Args:
    z: The rotation.

Returns:
    The rotated vector.

)doc";

constexpr char mkd_doc_slp_rotate_by_2[] =
R"doc(Returns this 2D unit vector rotated by a rotation.

Args:
    z: The rotation.

Returns:
    The rotated unit vector.

)doc";

constexpr char mkd_doc_slp_rotate_by_3[] =
R"doc(Returns this 3D vector rotated by a rotation.

Args:
    q: The rotation.

Returns:
    The rotated vector.

)doc";

constexpr char mkd_doc_slp_rotate_by_4[] =
R"doc(Returns this 3D unit vector rotated by a rotation.

Args:
    q: The rotation.

Returns:
    The rotated unit vector.

)doc";

constexpr char mkd_doc_slp_se2_t[] =
R"doc(Tag type for decision variables in the rigid transform group SE(2).

Problem::decision_variable(se2) returns a Pose2.)doc";

constexpr char mkd_doc_slp_se2_t_se2_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_se3_t[] =
R"doc(Tag type for decision variables in the rigid transform group SE(3).

Problem::decision_variable(se3) returns a Pose3.)doc";

constexpr char mkd_doc_slp_se3_t_se3_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_sign[] =
R"doc(sign() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_sin[] =
R"doc(sin() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_sinh[] =
R"doc(sinh() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_so2_t[] =
R"doc(Tag type for decision variables in the rotation group SO(2).

Problem::decision_variable(so2) returns a Rotation2.)doc";

constexpr char mkd_doc_slp_so2_t_so2_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_so3_t[] =
R"doc(Tag type for decision variables in the rotation group SO(3).

Problem::decision_variable(so3) returns a Rotation3.)doc";

constexpr char mkd_doc_slp_so3_t_so3_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_solve[] =
R"doc(Solves the VariableMatrix equation AX = B for X.

Args:
    A: The left-hand side.
    B: The right-hand side.

Template Args:
    Scalar: Scalar type.

Returns:
    The solution X.

)doc";

constexpr char mkd_doc_slp_sphere_t[] =
R"doc(Tag type for decision variables on the unit sphere.

Problem::decision_variable(sphere, n) returns a UnitVector in ℝⁿ.)doc";

constexpr char mkd_doc_slp_sphere_t_sphere_t[] = R"doc(Constructs the tag.)doc";

constexpr char mkd_doc_slp_sqrt[] =
R"doc(sqrt() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_tan[] =
R"doc(tan() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_tanh[] =
R"doc(tanh() for Variables.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_transform_by[] =
R"doc(Returns this point transformed by a pose.

The result is R p + t, where p is this point, and R and t are the
pose's rotation and translation.

Args:
    pose: The pose.

Returns:
    The transformed point.

)doc";

constexpr char mkd_doc_slp_transform_by_2[] =
R"doc(Returns this point transformed by a pose.

The result is R p + t, where p is this point, and R and t are the
pose's rotation and translation.

Args:
    pose: The pose.

Returns:
    The transformed point.

)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

