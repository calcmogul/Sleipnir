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

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3 = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_AddCost = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_IsComplex = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_IsInteger = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_IsSigned = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_MulCost = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_ReadCost = R"doc()doc";

static const char *__doc_Eigen_NumTraits_unnamed_enum_at_home_tav_git_Sleipnir_include_sleipnir_autodiff_Variable_hpp_802_3_RequireInitialization = R"doc()doc";

static const char *__doc_sleipnir_Block =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_sleipnir_Block_2 =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_sleipnir_CwiseReduce =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Parameter ``lhs``:
    The left-hand side of the binary operator.

Parameter ``rhs``:
    The right-hand side of the binary operator.

Parameter ``binary_op``:
    The binary operator to use for the reduce operation.)doc";

static const char *__doc_sleipnir_DynamicsType = R"doc(Enum describing a type of system dynamics constraints.)doc";

static const char *__doc_sleipnir_DynamicsType_kDiscrete = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

static const char *__doc_sleipnir_DynamicsType_kExplicitODE = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

static const char *__doc_sleipnir_EqualityConstraints = R"doc(A vector of equality constraints of the form cₑ(x) = 0.)doc";

static const char *__doc_sleipnir_EqualityConstraints_EqualityConstraints =
R"doc(Concatenates multiple equality constraints.

Parameter ``equality_constraints``:
    The list of EqualityConstraints to concatenate.)doc";

static const char *__doc_sleipnir_EqualityConstraints_EqualityConstraints_2 =
R"doc(Concatenates multiple equality constraints.

This overload is for Python bindings only.

Parameter ``equality_constraints``:
    The list of EqualityConstraints to concatenate.)doc";

static const char *__doc_sleipnir_EqualityConstraints_EqualityConstraints_3 =
R"doc(Constructs an equality constraint from a left and right side.

The standard form for equality constraints is c(x) = 0. This function
takes a constraint of the form lhs = rhs and converts it to lhs - rhs
= 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_EqualityConstraints_constraints = R"doc(A vector of scalar equality constraints.)doc";

static const char *__doc_sleipnir_EqualityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_sleipnir_ExpressionType =
R"doc(Expression type.

Used for autodiff caching.)doc";

static const char *__doc_sleipnir_ExpressionType_kConstant = R"doc(The expression is a constant.)doc";

static const char *__doc_sleipnir_ExpressionType_kLinear = R"doc(The expression is composed of linear and lower-order operators.)doc";

static const char *__doc_sleipnir_ExpressionType_kNone = R"doc(There is no expression.)doc";

static const char *__doc_sleipnir_ExpressionType_kNonlinear = R"doc(The expression is composed of nonlinear and lower-order operators.)doc";

static const char *__doc_sleipnir_ExpressionType_kQuadratic = R"doc(The expression is composed of quadratic and lower-order operators.)doc";

static const char *__doc_sleipnir_Gradient =
R"doc(This class calculates the gradient of a a variable with respect to a
vector of variables.

The gradient is only recomputed if the variable expression is
quadratic or higher order.)doc";

static const char *__doc_sleipnir_Gradient_Get =
R"doc(Returns the gradient as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Gradient_GetProfilers = R"doc(Returns the profiler.)doc";

static const char *__doc_sleipnir_Gradient_Gradient =
R"doc(Constructs a Gradient object.

Parameter ``variable``:
    Variable of which to compute the gradient.

Parameter ``wrt``:
    Variable with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_Gradient_Gradient_2 =
R"doc(Constructs a Gradient object.

Parameter ``variable``:
    Variable of which to compute the gradient.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_Gradient_Value = R"doc(Evaluates the gradient at wrt's value.)doc";

static const char *__doc_sleipnir_Gradient_m_g = R"doc()doc";

static const char *__doc_sleipnir_Gradient_m_jacobian = R"doc()doc";

static const char *__doc_sleipnir_Hessian =
R"doc(This class calculates the Hessian of a variable with respect to a
vector of variables.

The gradient tree is cached so subsequent Hessian calculations are
faster, and the Hessian is only recomputed if the variable expression
is nonlinear.)doc";

static const char *__doc_sleipnir_Hessian_2 = R"doc()doc";

static const char *__doc_sleipnir_Hessian_Get =
R"doc(Returns the Hessian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Hessian_GetProfilers = R"doc(Returns the profilers.)doc";

static const char *__doc_sleipnir_Hessian_Hessian =
R"doc(Constructs a Hessian object.

Parameter ``variable``:
    Variable of which to compute the Hessian.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the Hessian.)doc";

static const char *__doc_sleipnir_Hessian_Value = R"doc(Evaluates the Hessian at wrt's value.)doc";

static const char *__doc_sleipnir_Hessian_m_jacobian = R"doc()doc";

static const char *__doc_sleipnir_InequalityConstraints = R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.)doc";

static const char *__doc_sleipnir_InequalityConstraints_InequalityConstraints =
R"doc(Concatenates multiple inequality constraints.

Parameter ``inequality_constraints``:
    The list of InequalityConstraints to concatenate.)doc";

static const char *__doc_sleipnir_InequalityConstraints_InequalityConstraints_2 =
R"doc(Concatenates multiple inequality constraints.

This overload is for Python bindings only.

Parameter ``inequality_constraints``:
    The list of InequalityConstraints to concatenate.)doc";

static const char *__doc_sleipnir_InequalityConstraints_InequalityConstraints_3 =
R"doc(Constructs an inequality constraint from a left and right side.

The standard form for inequality constraints is c(x) ≥ 0. This
function takes a constraints of the form lhs ≥ rhs and converts it to
lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_InequalityConstraints_constraints = R"doc(A vector of scalar inequality constraints.)doc";

static const char *__doc_sleipnir_InequalityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_sleipnir_InteriorPoint =
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

Parameter ``config``:
    Configuration options for the solver.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Parameter ``status``:
    The solver status.)doc";

static const char *__doc_sleipnir_Jacobian =
R"doc(This class calculates the Jacobian of a vector of variables with
respect to a vector of variables.

The Jacobian is only recomputed if the variable expression is
quadratic or higher order.)doc";

static const char *__doc_sleipnir_Jacobian_2 = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_Get =
R"doc(Returns the Jacobian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Jacobian_GetProfilers = R"doc(Returns the profilers.)doc";

static const char *__doc_sleipnir_Jacobian_Jacobian =
R"doc(Constructs a Jacobian object.

Parameter ``variables``:
    Vector of variables of which to compute the Jacobian.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the Jacobian.)doc";

static const char *__doc_sleipnir_Jacobian_Value = R"doc(Evaluates the Jacobian at wrt's value.)doc";

static const char *__doc_sleipnir_Jacobian_m_J = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_cachedTriplets = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_graphs = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_nonlinearRows = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_profilers = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_variables = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_wrt = R"doc()doc";

static const char *__doc_sleipnir_MakeConstraints =
R"doc(Make a list of constraints.

The standard form for equality constraints is c(x) = 0, and the
standard form for inequality constraints is c(x) ≥ 0. This function
takes constraints of the form lhs = rhs or lhs ≥ rhs and converts them
to lhs - rhs = 0 or lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_Multistart =
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

static const char *__doc_sleipnir_MultistartResult =
R"doc(The result of a multistart solve.

Template parameter ``DecisionVariables``:
    The type containing the decision variable initial guess.)doc";

static const char *__doc_sleipnir_MultistartResult_status = R"doc()doc";

static const char *__doc_sleipnir_MultistartResult_variables = R"doc()doc";

static const char *__doc_sleipnir_Newton =
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

Parameter ``config``:
    Configuration options for the solver.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Parameter ``status``:
    The solver status.)doc";

static const char *__doc_sleipnir_OCPSolver =
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

static const char *__doc_sleipnir_OCPSolver_ConstrainDirectCollocation = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainDirectTranscription = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainFinalState =
R"doc(Utility function to constrain the final state.

Parameter ``final_state``:
    the final state to constrain to.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainInitialState =
R"doc(Utility function to constrain the initial state.

Parameter ``initial_state``:
    the initial state to constrain to.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainSingleShooting = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_DT =
R"doc(Get the timestep variables. After the problem is solved, this will
contain the timesteps corresponding to the optimized trajectory.

Shaped 1x(numSteps+1), although the last timestep is unused in the
trajectory.

Returns:
    The timestep variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_FinalState =
R"doc(Convenience function to get the final state in the trajectory.

Returns:
    The final state of the trajectory.)doc";

static const char *__doc_sleipnir_OCPSolver_ForEachStep =
R"doc(Set the constraint evaluation function. This function is called
`numSteps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``callback``:
    The callback f(x, u) where x is the state and u is the input
    vector.)doc";

static const char *__doc_sleipnir_OCPSolver_ForEachStep_2 =
R"doc(Set the constraint evaluation function. This function is called
`numSteps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``callback``:
    The callback f(t, x, u, dt) where t is time, x is the state
    vector, u is the input vector, and dt is the timestep duration.)doc";

static const char *__doc_sleipnir_OCPSolver_InitialState =
R"doc(Convenience function to get the initial state in the trajectory.

Returns:
    The initial state of the trajectory.)doc";

static const char *__doc_sleipnir_OCPSolver_OCPSolver =
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

static const char *__doc_sleipnir_OCPSolver_OCPSolver_2 =
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

static const char *__doc_sleipnir_OCPSolver_SetLowerInputBound =
R"doc(Convenience function to set a lower bound on the input.

Parameter ``lower_bound``:
    The lower bound that inputs must always be above. Must be shaped
    (numInputs)x1.)doc";

static const char *__doc_sleipnir_OCPSolver_SetMaxTimestep =
R"doc(Convenience function to set an upper bound on the timestep.

Parameter ``max_timestep``:
    The maximum timestep.)doc";

static const char *__doc_sleipnir_OCPSolver_SetMinTimestep =
R"doc(Convenience function to set a lower bound on the timestep.

Parameter ``min_timestep``:
    The minimum timestep.)doc";

static const char *__doc_sleipnir_OCPSolver_SetUpperInputBound =
R"doc(Convenience function to set an upper bound on the input.

Parameter ``upper_bound``:
    The upper bound that inputs must always be below. Must be shaped
    (numInputs)x1.)doc";

static const char *__doc_sleipnir_OCPSolver_U =
R"doc(Get the input variables. After the problem is solved, this will
contain the inputs corresponding to the optimized trajectory.

Shaped (numInputs)x(numSteps+1), although the last input step is
unused in the trajectory.

Returns:
    The input variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_X =
R"doc(Get the state variables. After the problem is solved, this will
contain the optimized trajectory.

Shaped (numStates)x(numSteps+1).

Returns:
    The state variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_m_DT = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_U = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_X = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dt = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dynamicsFunction = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dynamicsType = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numInputs = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numStates = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numSteps = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_timestepMethod = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_transcriptionMethod = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem =
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

static const char *__doc_sleipnir_OptimizationProblem_AddCallback =
R"doc(Adds a callback to be called at each solver iteration.

The callback for this overload should return void.

Parameter ``callback``:
    The callback.)doc";

static const char *__doc_sleipnir_OptimizationProblem_AddCallback_2 =
R"doc(Adds a callback to be called at each solver iteration.

The callback for this overload should return bool.

Parameter ``callback``:
    The callback. Returning true from the callback causes the solver
    to exit early with the solution it has so far.)doc";

static const char *__doc_sleipnir_OptimizationProblem_ClearCallbacks = R"doc(Clears the registered callbacks.)doc";

static const char *__doc_sleipnir_OptimizationProblem_DecisionVariable = R"doc(Create a decision variable in the optimization problem.)doc";

static const char *__doc_sleipnir_OptimizationProblem_DecisionVariable_2 =
R"doc(Create a matrix of decision variables in the optimization problem.

Parameter ``rows``:
    Number of matrix rows.

Parameter ``cols``:
    Number of matrix columns.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Maximize =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Maximize_2 =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Minimize =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Minimize_2 =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_OptimizationProblem = R"doc(Construct the optimization problem.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Solve =
R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``config``:
    Configuration options for the solver.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_2 =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_3 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_4 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SymmetricDecisionVariable =
R"doc(Create a symmetric matrix of decision variables in the optimization
problem.

Variable instances are reused across the diagonal, which helps reduce
problem dimensionality.

Parameter ``rows``:
    Number of matrix rows.)doc";

static const char *__doc_sleipnir_OptimizationProblem_m_callbacks = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_decisionVariables = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_equalityConstraints = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_f = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_inequalityConstraints = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_status = R"doc()doc";

static const char *__doc_sleipnir_RK4 =
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

static const char *__doc_sleipnir_SQP =
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

Parameter ``config``:
    Configuration options for the solver.

Parameter ``x``:
    The initial guess and output location for the decision variables.

Parameter ``status``:
    The solver status.)doc";

static const char *__doc_sleipnir_Slice = R"doc()doc";

static const char *__doc_sleipnir_Slice_Adjust =
R"doc(Adjusts start and end slice indices assuming a sequence of the
specified length.

Parameter ``length``:
    The sequence length.

Returns:
    The slice length.)doc";

static const char *__doc_sleipnir_Slice_Slice = R"doc(Constructs a Slice.)doc";

static const char *__doc_sleipnir_Slice_Slice_2 =
R"doc(Constructs a slice.

Parameter ``stop``:
    Slice stop index (exclusive).)doc";

static const char *__doc_sleipnir_Slice_Slice_3 =
R"doc(Constructs a slice.

Parameter ``start``:
    Slice start index (inclusive).

Parameter ``stop``:
    Slice stop index (exclusive).)doc";

static const char *__doc_sleipnir_Slice_Slice_4 =
R"doc(Constructs a slice.

Parameter ``start``:
    Slice start index (inclusive).

Parameter ``stop``:
    Slice stop index (exclusive).

Parameter ``step``:
    Slice step.)doc";

static const char *__doc_sleipnir_Slice_start = R"doc(Start index (inclusive).)doc";

static const char *__doc_sleipnir_Slice_step = R"doc(Step.)doc";

static const char *__doc_sleipnir_Slice_stop = R"doc(Stop index (exclusive).)doc";

static const char *__doc_sleipnir_Solve =
R"doc(Solves the VariableMatrix equation AX = B for X.

Parameter ``A``:
    The left-hand side.

Parameter ``B``:
    The right-hand side.

Returns:
    The solution X.)doc";

static const char *__doc_sleipnir_SolverConfig = R"doc(Solver configuration.)doc";

static const char *__doc_sleipnir_SolverConfig_acceptableTolerance =
R"doc(The solver will stop once the error is below this tolerance for
`acceptableIterations` iterations. This is useful in cases where the
solver might not be able to achieve the desired level of accuracy due
to floating-point round-off.)doc";

static const char *__doc_sleipnir_SolverConfig_diagnostics =
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
iterate (sᵀz)</td> </tr> <tr> <td>reg</td> <td>Iteration matrix
regularization</td> </tr> <tr> <td>primal α</td> <td>Primal step
size</td> </tr> <tr> <td>dual α</td> <td>Dual step size</td> </tr>
<tr> <td>↩</td> <td>Number of line search backtracks</td> </tr>
</table>)doc";

static const char *__doc_sleipnir_SolverConfig_feasibleIPM =
R"doc(Enables the feasible interior-point method. When the inequality
constraints are all feasible, step sizes are reduced when necessary to
prevent them becoming infeasible again. This is useful when parts of
the problem are ill-conditioned in infeasible regions (e.g., square
root of a negative value). This can slow or prevent progress toward a
solution though, so only enable it if necessary.)doc";

static const char *__doc_sleipnir_SolverConfig_maxAcceptableIterations =
R"doc(The solver will stop once the error is below `acceptableTolerance` for
this many iterations.)doc";

static const char *__doc_sleipnir_SolverConfig_maxIterations = R"doc(The maximum number of solver iterations before returning a solution.)doc";

static const char *__doc_sleipnir_SolverConfig_spy =
R"doc(Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named
H.spy, A_e.spy, and A_i.spy respectively during solve.

Use tools/spy.py to plot them.)doc";

static const char *__doc_sleipnir_SolverConfig_timeout = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

static const char *__doc_sleipnir_SolverConfig_tolerance = R"doc(The solver will stop once the error is below this tolerance.)doc";

static const char *__doc_sleipnir_SolverExitCondition = R"doc(Solver exit condition.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kCallbackRequestedStop =
R"doc(The solver returned its solution so far after the user requested a
stop.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kDivergingIterates =
R"doc(The solver encountered diverging primal iterates xₖ and/or sₖ and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kFactorizationFailed = R"doc(The linear system factorization failed.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kLineSearchFailed =
R"doc(The backtracking line search failed, and the problem isn't locally
infeasible.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kLocallyInfeasible =
R"doc(The solver determined the problem to be locally infeasible and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kMaxIterationsExceeded =
R"doc(The solver returned its solution so far after exceeding the maximum
number of iterations.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kNonfiniteInitialCostOrConstraints =
R"doc(The solver encountered nonfinite initial cost or constraints and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kSolvedToAcceptableTolerance =
R"doc(Solved the problem to an acceptable tolerance, but not the desired
one.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kSuccess = R"doc(Solved the problem to the desired tolerance.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kTimeout =
R"doc(The solver returned its solution so far after exceeding the maximum
elapsed wall clock time.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kTooFewDOFs = R"doc(The solver determined the problem to be overconstrained and gave up.)doc";

static const char *__doc_sleipnir_SolverIterationInfo = R"doc(Solver iteration information exposed to a user callback.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_A_e = R"doc(The equality constraint Jacobian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_A_i = R"doc(The inequality constraint Jacobian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_H = R"doc(The Hessian of the Lagrangian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_g = R"doc(The gradient of the cost function.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_iteration = R"doc(The solver iteration.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_s = R"doc(The inequality constraint slack variables.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_x = R"doc(The decision variables.)doc";

static const char *__doc_sleipnir_SolverStatus =
R"doc(Return value of OptimizationProblem::Solve() containing the cost
function and constraint types and solver's exit condition.)doc";

static const char *__doc_sleipnir_SolverStatus_cost = R"doc(The solution's cost.)doc";

static const char *__doc_sleipnir_SolverStatus_costFunctionType = R"doc(The cost function type detected by the solver.)doc";

static const char *__doc_sleipnir_SolverStatus_equalityConstraintType = R"doc(The equality constraint type detected by the solver.)doc";

static const char *__doc_sleipnir_SolverStatus_exitCondition = R"doc(The solver's exit condition.)doc";

static const char *__doc_sleipnir_SolverStatus_inequalityConstraintType = R"doc(The inequality constraint type detected by the solver.)doc";

static const char *__doc_sleipnir_TimestepMethod = R"doc(Enum describing the type of system timestep.)doc";

static const char *__doc_sleipnir_TimestepMethod_kFixed = R"doc(The timestep is a fixed constant.)doc";

static const char *__doc_sleipnir_TimestepMethod_kVariable = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

static const char *__doc_sleipnir_TimestepMethod_kVariableSingle =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

static const char *__doc_sleipnir_ToMessage =
R"doc(Returns user-readable message corresponding to the exit condition.

Parameter ``exit_condition``:
    Solver exit condition.)doc";

static const char *__doc_sleipnir_TranscriptionMethod = R"doc(Enum describing an OCP transcription method.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kDirectCollocation =
R"doc(The trajectory is modeled as a series of cubic polynomials where the
centerpoint slope is constrained.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kDirectTranscription =
R"doc(Each state is a decision variable constrained to the integrated
dynamics of the previous state.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kSingleShooting =
R"doc(States depend explicitly as a function of all previous states and all
previous inputs.)doc";

static const char *__doc_sleipnir_Variable = R"doc(An autodiff variable pointing to an expression node.)doc";

static const char *__doc_sleipnir_VariableBlock =
R"doc(A submatrix of autodiff variables with reference semantics.

Template parameter ``Mat``:
    The type of the matrix whose storage this class points to.)doc";

static const char *__doc_sleipnir_VariableBlock_Block =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableBlock_Block_2 =
R"doc(Returns a block slice of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableBlock_Col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Cols = R"doc(Returns number of columns in the matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_CwiseTransform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unary_op``:
    The unary operator to use for the transform operation.)doc";

static const char *__doc_sleipnir_VariableBlock_Row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Rows = R"doc(Returns number of rows in the matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_Segment =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableBlock_Segment_2 =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableBlock_SetValue =
R"doc(Assigns a double to the block.

This only works for blocks with one row and one column.

Parameter ``value``:
    Value to assign.)doc";

static const char *__doc_sleipnir_VariableBlock_SetValue_2 =
R"doc(Sets block's internal values.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_sleipnir_VariableBlock_T = R"doc(Returns the transpose of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_Value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.)doc";

static const char *__doc_sleipnir_VariableBlock_Value_2 =
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.)doc";

static const char *__doc_sleipnir_VariableBlock_Value_3 = R"doc(Returns the contents of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_3 =
R"doc(Constructs a Variable block pointing to all of the given matrix.

Parameter ``mat``:
    The matrix to which to point.)doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_4 =
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

static const char *__doc_sleipnir_VariableBlock_VariableBlock_5 =
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

static const char *__doc_sleipnir_VariableBlock_begin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_begin_2 = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_cbegin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_cend = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_const_iterator_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_m_index = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_end = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_end_2 = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_iterator_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_m_index = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_colSlice = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_colSliceLength = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_rowSlice = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_rowSliceLength = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_operator_array =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_2 =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_3 =
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_4 =
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_5 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_6 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_7 =
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
    The column slice length.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_array_8 =
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
    The column slice length.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign =
R"doc(Assigns a VariableBlock to the block.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_2 =
R"doc(Assigns a VariableBlock to the block.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_3 =
R"doc(Assigns a double to the block.

This only works for blocks with one row and one column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_4 =
R"doc(Assigns an Eigen matrix to the block.

Parameter ``values``:
    Eigen matrix of values to assign.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_5 =
R"doc(Assigns a VariableMatrix to the block.

Parameter ``values``:
    VariableMatrix of values.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_6 =
R"doc(Assigns a VariableMatrix to the block.

Parameter ``values``:
    VariableMatrix of values.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_iadd =
R"doc(Compound addition-assignment operator.

Parameter ``rhs``:
    Variable to add.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_idiv =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_idiv_2 =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_imul =
R"doc(Compound matrix multiplication-assignment operator.

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_imul_2 =
R"doc(Compound matrix multiplication-assignment operator (only enabled when
lhs is a scalar).

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_isub =
R"doc(Compound subtraction-assignment operator.

Parameter ``rhs``:
    Variable to subtract.)doc";

static const char *__doc_sleipnir_VariableBlock_size = R"doc(Returns number of elements in matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix = R"doc(A matrix of autodiff variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_Block =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableMatrix_Block_2 =
R"doc(Returns a block of the variable matrix.

Parameter ``row_offset``:
    The row offset of the block selection.

Parameter ``col_offset``:
    The column offset of the block selection.

Parameter ``block_rows``:
    The number of rows in the block selection.

Parameter ``block_cols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableMatrix_Col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Cols = R"doc(Returns number of columns in the matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_CwiseTransform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unary_op``:
    The unary operator to use for the transform operation.)doc";

static const char *__doc_sleipnir_VariableMatrix_Ones =
R"doc(Returns a variable matrix filled with ones.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_Row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Rows = R"doc(Returns number of rows in the matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_Segment =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableMatrix_Segment_2 =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableMatrix_SetValue =
R"doc(Sets the VariableMatrix's internal values.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_T = R"doc(Returns the transpose of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value_2 =
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value_3 = R"doc(Returns the contents of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix = R"doc(Constructs an empty VariableMatrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_2 =
R"doc(Constructs a VariableMatrix column vector with the given rows.

Parameter ``rows``:
    The number of matrix rows.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_3 =
R"doc(Constructs a zero-initialized VariableMatrix with the given
dimensions.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_4 =
R"doc(Constructs an empty VariableMatrix with the given dimensions.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_5 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_6 =
R"doc(Constructs a scalar VariableMatrix from a nested list of doubles.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_7 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_8 =
R"doc(Constructs a VariableMatrix from an Eigen matrix.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_9 =
R"doc(Constructs a VariableMatrix from an Eigen diagonal matrix.

Parameter ``values``:
    Diagonal matrix of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_10 =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Parameter ``variable``:
    Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_11 =
R"doc(Constructs a scalar VariableMatrix from a Variable.

Parameter ``variable``:
    Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_12 =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_13 =
R"doc(Constructs a VariableMatrix from a VariableBlock.

Parameter ``values``:
    VariableBlock of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_14 =
R"doc(Constructs a column vector wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_15 =
R"doc(Constructs a matrix wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_Zero =
R"doc(Returns a variable matrix filled with zeroes.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_begin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_begin_2 = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_cbegin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_cend = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_const_iterator_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_m_it = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_empty_t = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_end = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_end_2 = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_iterator_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_m_it = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_cols = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_rows = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_storage = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_operator_Variable = R"doc(Implicit conversion operator from 1x1 VariableMatrix to Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_2 =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_3 =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_4 =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_5 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_6 =
R"doc(Returns a slice of the variable matrix.

Parameter ``row_slice``:
    The row slice.

Parameter ``col_slice``:
    The column slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_7 =
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
    The column slice length.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_array_8 =
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
    The column slice length.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_assign =
R"doc(Assigns an Eigen matrix to a VariableMatrix.

Parameter ``values``:
    Eigen matrix of values.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_iadd =
R"doc(Compound addition-assignment operator.

Parameter ``rhs``:
    Variable to add.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_idiv =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_imul =
R"doc(Compound matrix multiplication-assignment operator.

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_isub =
R"doc(Compound subtraction-assignment operator.

Parameter ``rhs``:
    Variable to subtract.)doc";

static const char *__doc_sleipnir_VariableMatrix_size = R"doc(Returns number of elements in matrix.)doc";

static const char *__doc_sleipnir_Variable_SetValue =
R"doc(Sets Variable's internal value.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_Type =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).)doc";

static const char *__doc_sleipnir_Variable_Value = R"doc(Returns the value of this variable.)doc";

static const char *__doc_sleipnir_Variable_Variable = R"doc(Constructs a linear Variable with a value of zero.)doc";

static const char *__doc_sleipnir_Variable_Variable_2 = R"doc(Constructs an empty Variable.)doc";

static const char *__doc_sleipnir_Variable_Variable_3 = R"doc()doc";

static const char *__doc_sleipnir_Variable_Variable_4 = R"doc()doc";

static const char *__doc_sleipnir_Variable_Variable_5 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_sleipnir_Variable_Variable_6 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_sleipnir_Variable_expr = R"doc(The expression node.)doc";

static const char *__doc_sleipnir_Variable_m_graph =
R"doc(Updates the value of this variable based on the values of its
dependent variables)doc";

static const char *__doc_sleipnir_Variable_operator_assign =
R"doc(Assignment operator for double.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_operator_iadd =
R"doc(Variable-Variable compound addition operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_idiv =
R"doc(Variable-Variable compound division operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_imul =
R"doc(Variable-Variable compound multiplication operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_isub =
R"doc(Variable-Variable compound subtraction operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_abs =
R"doc(std::abs() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_acos =
R"doc(std::acos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_asin =
R"doc(std::asin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_atan =
R"doc(std::atan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_atan2 =
R"doc(std::atan2() for Variables.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_sleipnir_cos =
R"doc(std::cos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_cosh =
R"doc(std::cosh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_AbsExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_AbsExpression_AbsExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_AbsExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_AbsExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_AbsExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_AbsExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_AcosExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_AcosExpression_AcosExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_AcosExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_AcosExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_AcosExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_AcosExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph =
R"doc(This class is an adaptor type that performs value updates of an
expression's adjoint graph.)doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_AdjointExpressionGraph =
R"doc(Generates the adjoint graph for the given expression.

Parameter ``root``:
    The root node of the expression.)doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_AppendAdjointTriplets =
R"doc(Updates the adjoints in the expression graph (computes the gradient)
then appends the adjoints of wrt to the sparse matrix triplets.

Parameter ``triplets``:
    The sparse matrix triplets.

Parameter ``row``:
    The row of wrt.)doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_GenerateGradientTree =
R"doc(Returns the variable's gradient tree.

This function lazily allocates variables, so elements of the returned
VariableMatrix will be empty if the corresponding element of wrt had
no adjoint. Ensure Variable::expr isn't nullptr before calling member
functions.

Parameter ``wrt``:
    Variables with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_UpdateValues =
R"doc(Update the values of all nodes in this adjoint graph based on the
values of their dependent nodes.)doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_m_colList = R"doc()doc";

static const char *__doc_sleipnir_detail_AdjointExpressionGraph_m_topList = R"doc()doc";

static const char *__doc_sleipnir_detail_AsinExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_AsinExpression_AsinExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_AsinExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_AsinExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_AsinExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_AsinExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_Atan2Expression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_Atan2Expression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_Atan2Expression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_AtanExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_AtanExpression_AtanExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_AtanExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_AtanExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_AtanExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_AtanExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_BinaryMinusExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryMinusExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_BinaryPlusExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_BinaryPlusExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_ConstExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_ConstExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_ConstExpression_ConstExpression = R"doc(Constructs a constant expression with a value of zero.)doc";

static const char *__doc_sleipnir_detail_ConstExpression_ConstExpression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_sleipnir_detail_ConstExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_ConstExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_CosExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_CosExpression_CosExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_CosExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_CosExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_CosExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_CosExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_CoshExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_CoshExpression_CoshExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_CoshExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_CoshExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_CoshExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_CoshExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_DecisionVariableExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_DecisionVariableExpression_DecisionVariableExpression = R"doc(Constructs a decision variable expression with a value of zero.)doc";

static const char *__doc_sleipnir_detail_DecisionVariableExpression_DecisionVariableExpression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_sleipnir_detail_DecisionVariableExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_DecisionVariableExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_DivExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_DivExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_DivExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_ErfExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_ErfExpression_ErfExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_ErfExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_ErfExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_ErfExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_ErfExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpExpression_ExpExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_ExpExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_Expression = R"doc(An autodiff expression node.)doc";

static const char *__doc_sleipnir_detail_Expression_2 = R"doc(An autodiff expression node.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression = R"doc(Constructs a constant expression with a value of zero.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_3 =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_4 =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_Expression_GradExprL =
R"doc(Returns Expression adjoint of the left child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_GradExprR =
R"doc(Returns Expression adjoint of the right child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_GradL =
R"doc(Returns double adjoint of the left child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_GradR =
R"doc(Returns double adjoint of the right child expression.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.

Parameter ``parent_adjoint``:
    Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_IsConstant =
R"doc(Returns true if the expression is the given constant.

Parameter ``constant``:
    The constant.)doc";

static const char *__doc_sleipnir_detail_Expression_Type =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).)doc";

static const char *__doc_sleipnir_detail_Expression_Value =
R"doc(Either nullary operator with no arguments, unary operator with one
argument, or binary operator with two arguments. This operator is used
to update the node's value.

Parameter ``lhs``:
    Left argument to binary operator.

Parameter ``rhs``:
    Right argument to binary operator.)doc";

static const char *__doc_sleipnir_detail_Expression_adjoint = R"doc(The adjoint of the expression node used during autodiff.)doc";

static const char *__doc_sleipnir_detail_Expression_adjointExpr =
R"doc(The adjoint of the expression node used during gradient expression
tree generation.)doc";

static const char *__doc_sleipnir_detail_Expression_args = R"doc(Expression arguments.)doc";

static const char *__doc_sleipnir_detail_Expression_col = R"doc(This expression's column in a Jacobian, or -1 otherwise.)doc";

static const char *__doc_sleipnir_detail_Expression_incomingEdges = R"doc(Counts incoming edges for this node.)doc";

static const char *__doc_sleipnir_detail_Expression_refCount = R"doc(Reference count for intrusive shared pointer.)doc";

static const char *__doc_sleipnir_detail_Expression_value = R"doc(The value of the expression node.)doc";

static const char *__doc_sleipnir_detail_HypotExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_HypotExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_HypotExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_HypotExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrDecRefCount =
R"doc(Refcount decrement for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrDecRefCount_2 =
R"doc(Refcount decrement for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrIncRefCount =
R"doc(Refcount increment for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrIncRefCount_2 =
R"doc(Refcount increment for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_Log10Expression = R"doc()doc";

static const char *__doc_sleipnir_detail_Log10Expression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_Log10Expression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_Log10Expression_Log10Expression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_Log10Expression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_Log10Expression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_LogExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_LogExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_LogExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_LogExpression_LogExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_LogExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_LogExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_MakeExpressionPtr =
R"doc(Creates an intrusive shared pointer to an expression from the global
pool allocator.

Template parameter ``T``:
    The derived expression type.

Parameter ``args``:
    Constructor arguments for Expression.)doc";

static const char *__doc_sleipnir_detail_MultExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_MultExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_MultExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_MultExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_GradExprR = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_GradR = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_PowExpression =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_PowExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_PowExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_SignExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_SignExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_SignExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_SignExpression_SignExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_SignExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_SignExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_SinExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_SinExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_SinExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_SinExpression_SinExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_SinExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_SinExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_SinhExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_SinhExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_SinhExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_SinhExpression_SinhExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_SinhExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_SinhExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_SqrtExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_SqrtExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_SqrtExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_SqrtExpression_SqrtExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_SqrtExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_SqrtExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_TanExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_TanExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_TanExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_TanExpression_TanExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_TanExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_TanExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_TanhExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_TanhExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_TanhExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_TanhExpression_TanhExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_TanhExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_TanhExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_TopologicalSort =
R"doc(Generate a topological sort of an expression graph from parent to
child.

https://en.wikipedia.org/wiki/Topological_sorting

Parameter ``root``:
    The root node of the expression.)doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression = R"doc()doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_GradExprL = R"doc()doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_GradL = R"doc()doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_Type = R"doc()doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_UnaryMinusExpression =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_UnaryMinusExpression_Value = R"doc()doc";

static const char *__doc_sleipnir_detail_UpdateAdjoints =
R"doc(Updates the adjoints in the expression graph (computes the gradient).

Parameter ``list``:
    Topological sort of graph from parent to child.)doc";

static const char *__doc_sleipnir_detail_UpdateValues =
R"doc(Update the values of all nodes in this graph based on the values of
their dependent nodes.

Parameter ``list``:
    Topological sort of graph from parent to child.)doc";

static const char *__doc_sleipnir_detail_abs =
R"doc(std::abs() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_acos =
R"doc(std::acos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_asin =
R"doc(std::asin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_atan =
R"doc(std::atan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_atan2 =
R"doc(std::atan2() for Expressions.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_sleipnir_detail_cos =
R"doc(std::cos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_cosh =
R"doc(std::cosh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_erf =
R"doc(std::erf() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_exp =
R"doc(std::exp() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_exp_2 =
R"doc(std::exp() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_hypot =
R"doc(std::hypot() for Expressions.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_sleipnir_detail_hypot_2 =
R"doc(std::hypot() for Expressions.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_sleipnir_detail_log =
R"doc(std::log() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_log10 =
R"doc(std::log10() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_pow =
R"doc(std::pow() for Expressions.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_sleipnir_detail_pow_2 =
R"doc(std::pow() for Expressions.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_sleipnir_detail_sign =
R"doc(sign() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sin =
R"doc(std::sin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sin_2 =
R"doc(std::sin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sinh =
R"doc(std::sinh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sinh_2 =
R"doc(std::sinh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sqrt =
R"doc(std::sqrt() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sqrt_2 =
R"doc(std::sqrt() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_tan =
R"doc(std::tan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_tanh =
R"doc(std::tanh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_erf =
R"doc(std::erf() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_exp =
R"doc(std::exp() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_hypot =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_sleipnir_hypot_2 =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.

Parameter ``z``:
    The z argument.)doc";

static const char *__doc_sleipnir_log =
R"doc(std::log() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_log10 =
R"doc(std::log10() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_operator_eq =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_ge =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_gt =
R"doc(Greater-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_le =
R"doc(Less-than-or-equal-to comparison operator that returns an inequality
constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_lt =
R"doc(Less-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_pow =
R"doc(std::pow() for Variables.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_sleipnir_sign =
R"doc(sign() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_sin =
R"doc(std::sin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_sinh =
R"doc(std::sinh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_slicing_none_t = R"doc()doc";

static const char *__doc_sleipnir_sqrt =
R"doc(std::sqrt() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_tan =
R"doc(std::tan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_tanh =
R"doc(std::tanh() for Variables.

Parameter ``x``:
    The argument.)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

