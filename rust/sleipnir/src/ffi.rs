// Copyright (c) Sleipnir contributors

//! Bridge to the C++ library. The recorded expression graph is rebuilt as a
//! C++ `slp::Problem` held by an opaque handle, which later solves reuse until
//! the problem's structure changes. Each solve copies the decision variable
//! and parameter values in and the final iterate back out.

use nalgebra_sparse::CscMatrix;

use crate::variable::{Binary, Condition, Node, Unary};

#[cxx::bridge(namespace = "sleipnir_rs")]
mod bridge {
    #[repr(u8)]
    enum NodeKind {
        Decision,
        Parameter,
        Constant,
        Neg,
        Abs,
        Acos,
        Asin,
        Atan,
        Cbrt,
        Cos,
        Cosh,
        Erf,
        Exp,
        Log,
        Log10,
        Sign,
        Sin,
        Sinh,
        Sqrt,
        Tan,
        Tanh,
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Atan2,
        Hypot,
        Max,
        Min,
        IfElse,
    }

    #[repr(u8)]
    enum ConditionKind {
        Less,
        LessEqual,
        Greater,
        GreaterEqual,
        Equal,
        NotEqual,
    }

    /// Expression graph node. `operands` index earlier nodes, and `value` is
    /// the constant, initial decision variable, or parameter value.
    struct NodeF64 {
        kind: NodeKind,
        condition: ConditionKind,
        operands: [usize; 4],
        value: f64,
    }

    struct NodeF32 {
        kind: NodeKind,
        condition: ConditionKind,
        operands: [usize; 4],
        value: f32,
    }

    /// Mirrors `slp::Options` plus `solve()`'s spy flag. `timeout` is in
    /// seconds.
    struct OptionsF64 {
        tolerance: f64,
        max_iterations: i32,
        timeout: f64,
        feasible_ipm: bool,
        diagnostics: bool,
        spy: bool,
    }

    struct OptionsF32 {
        tolerance: f32,
        max_iterations: i32,
        timeout: f64,
        feasible_ipm: bool,
        diagnostics: bool,
        spy: bool,
    }

    /// Compressed sparse column matrix, as Eigen stores it.
    struct SparseF64 {
        rows: usize,
        cols: usize,
        col_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    }

    struct SparseF32 {
        rows: usize,
        cols: usize,
        col_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f32>,
    }

    // `unsafe` because C++ can't check the lifetimes; C++ only calls the
    // callbacks during the solve call they were passed to
    extern "Rust" {
        type CallbackF64<'a>;
        type CallbackF32<'a>;

        /// Returns true to stop solving.
        #[allow(clippy::too_many_arguments)]
        unsafe fn call<'a>(
            self: &mut CallbackF64<'a>,
            iteration: i32,
            x: &[f64],
            s: &[f64],
            y: &[f64],
            z: &[f64],
            g: SparseF64,
            h: SparseF64,
            a_e: SparseF64,
            a_i: SparseF64,
        ) -> bool;

        /// Returns true to stop solving.
        #[allow(clippy::too_many_arguments)]
        unsafe fn call<'a>(
            self: &mut CallbackF32<'a>,
            iteration: i32,
            x: &[f32],
            s: &[f32],
            y: &[f32],
            z: &[f32],
            g: SparseF32,
            h: SparseF32,
            a_e: SparseF32,
            a_i: SparseF32,
        ) -> bool;
    }

    unsafe extern "C++" {
        include!("sleipnir/cpp/bridge.hpp");

        /// C++ problem built from an expression graph.
        type ProblemF64;
        type ProblemF32;

        /// Builds a problem. A negative `cost` means there's no cost
        /// function.
        fn build_problem_f64(
            nodes: &[NodeF64],
            cost: isize,
            equality_constraints: &[usize],
            inequality_constraints: &[usize],
        ) -> Result<UniquePtr<ProblemF64>>;

        fn build_problem_f32(
            nodes: &[NodeF32],
            cost: isize,
            equality_constraints: &[usize],
            inequality_constraints: &[usize],
        ) -> Result<UniquePtr<ProblemF32>>;

        /// Sets the decision variable and parameter values, solves, writes
        /// the final iterate to `values`, and returns the `slp::ExitStatus`
        /// value.
        #[allow(clippy::too_many_arguments)]
        fn solve_problem_f64<'a>(
            problem: Pin<&mut ProblemF64>,
            decision_values: &[f64],
            parameter_values: &[f64],
            options: &OptionsF64,
            has_callback: bool,
            callback: &mut CallbackF64<'a>,
            values: &mut [f64],
        ) -> Result<i32>;

        #[allow(clippy::too_many_arguments)]
        fn solve_problem_f32<'a>(
            problem: Pin<&mut ProblemF32>,
            decision_values: &[f32],
            parameter_values: &[f32],
            options: &OptionsF32,
            has_callback: bool,
            callback: &mut CallbackF32<'a>,
            values: &mut [f32],
        ) -> Result<i32>;
    }
}

// SAFETY: slp::Problem has no thread affinity, and the handle is only used
// through &mut
unsafe impl Send for bridge::ProblemF64 {}
unsafe impl Send for bridge::ProblemF32 {}

/// Solver options in the form the bridge consumes.
pub struct RawOptions<S> {
    pub tolerance: S,
    pub max_iterations: i32,
    pub timeout: f64,
    pub feasible_ipm: bool,
    pub diagnostics: bool,
    pub spy: bool,
}

/// Solver state at the start of an iteration.
pub struct RawIteration<'a, S> {
    pub iteration: i32,
    pub x: &'a [S],
    pub s: &'a [S],
    pub y: &'a [S],
    pub z: &'a [S],
    pub g: CscMatrix<S>,
    pub h: CscMatrix<S>,
    pub a_e: CscMatrix<S>,
    pub a_i: CscMatrix<S>,
}

/// Iteration callback. Returns true to stop solving.
pub type Callback<'a, S> = &'a mut dyn FnMut(&RawIteration<'_, S>) -> bool;

/// Per-scalar-type entry point into the bridge.
pub trait Backend: Sized {
    /// Built C++ problem.
    #[doc(hidden)]
    type Handle: Send;

    /// Builds a C++ problem from the expression graph.
    #[doc(hidden)]
    fn build(
        nodes: &[Node<Self>],
        cost: Option<usize>,
        equality_constraints: &[usize],
        inequality_constraints: &[usize],
    ) -> Result<Self::Handle, cxx::Exception>;

    /// Solves a built problem, writes the final iterate to `values`, and
    /// returns the `slp::ExitStatus` value.
    #[doc(hidden)]
    fn solve(
        handle: &mut Self::Handle,
        decision_values: &[Self],
        parameter_values: &[Self],
        options: &RawOptions<Self>,
        callback: Option<Callback<'_, Self>>,
        values: &mut [Self],
    ) -> Result<i32, cxx::Exception>;
}

fn encode<S: Copy>(
    node: &Node<S>,
    zero: S,
) -> (bridge::NodeKind, bridge::ConditionKind, [usize; 4], S) {
    use bridge::{ConditionKind as C, NodeKind as K};

    match *node {
        Node::Decision(value) => (K::Decision, C::Less, [0; 4], value),
        Node::Parameter(value) => (K::Parameter, C::Less, [0; 4], value),
        Node::Constant(value) => (K::Constant, C::Less, [0; 4], value),
        Node::Unary(op, a) => {
            let kind = match op {
                Unary::Neg => K::Neg,
                Unary::Abs => K::Abs,
                Unary::Acos => K::Acos,
                Unary::Asin => K::Asin,
                Unary::Atan => K::Atan,
                Unary::Cbrt => K::Cbrt,
                Unary::Cos => K::Cos,
                Unary::Cosh => K::Cosh,
                Unary::Erf => K::Erf,
                Unary::Exp => K::Exp,
                Unary::Ln => K::Log,
                Unary::Log10 => K::Log10,
                Unary::Sign => K::Sign,
                Unary::Sin => K::Sin,
                Unary::Sinh => K::Sinh,
                Unary::Sqrt => K::Sqrt,
                Unary::Tan => K::Tan,
                Unary::Tanh => K::Tanh,
            };
            (kind, C::Less, [a, 0, 0, 0], zero)
        }
        Node::Binary(op, a, b) => {
            let kind = match op {
                Binary::Add => K::Add,
                Binary::Sub => K::Sub,
                Binary::Mul => K::Mul,
                Binary::Div => K::Div,
                Binary::Pow => K::Pow,
                Binary::Atan2 => K::Atan2,
                Binary::Hypot => K::Hypot,
                Binary::Max => K::Max,
                Binary::Min => K::Min,
            };
            (kind, C::Less, [a, b, 0, 0], zero)
        }
        Node::IfElse(condition, operands) => {
            let condition = match condition {
                Condition::Less => C::Less,
                Condition::LessEqual => C::LessEqual,
                Condition::Greater => C::Greater,
                Condition::GreaterEqual => C::GreaterEqual,
                Condition::Equal => C::Equal,
                Condition::NotEqual => C::NotEqual,
            };
            (K::IfElse, condition, operands, zero)
        }
    }
}

macro_rules! backend {
    (
        $S:ident, $Node:ident, $Options:ident, $Sparse:ident, $Callback:ident,
        $Problem:ident, $build:ident, $solve:ident
    ) => {
        pub struct $Callback<'a>(Option<Callback<'a, $S>>);

        impl $Callback<'_> {
            #[allow(clippy::too_many_arguments)]
            fn call(
                &mut self,
                iteration: i32,
                x: &[$S],
                s: &[$S],
                y: &[$S],
                z: &[$S],
                g: bridge::$Sparse,
                h: bridge::$Sparse,
                a_e: bridge::$Sparse,
                a_i: bridge::$Sparse,
            ) -> bool {
                let Some(callback) = self.0.as_mut() else {
                    return false;
                };
                let csc = |m: bridge::$Sparse| {
                    CscMatrix::try_from_csc_data(
                        m.rows,
                        m.cols,
                        m.col_offsets,
                        m.row_indices,
                        m.values,
                    )
                    .expect("Eigen sparse matrices are valid CSC data")
                };
                callback(&RawIteration {
                    iteration,
                    x,
                    s,
                    y,
                    z,
                    g: csc(g),
                    h: csc(h),
                    a_e: csc(a_e),
                    a_i: csc(a_i),
                })
            }
        }

        impl Backend for $S {
            type Handle = cxx::UniquePtr<bridge::$Problem>;

            fn build(
                nodes: &[Node<Self>],
                cost: Option<usize>,
                equality_constraints: &[usize],
                inequality_constraints: &[usize],
            ) -> Result<Self::Handle, cxx::Exception> {
                let nodes: Vec<_> = nodes
                    .iter()
                    .map(|node| {
                        let (kind, condition, operands, value) = encode(node, 0.0);
                        bridge::$Node {
                            kind,
                            condition,
                            operands,
                            value,
                        }
                    })
                    .collect();
                bridge::$build(
                    &nodes,
                    cost.map_or(-1, |cost| cost as isize),
                    equality_constraints,
                    inequality_constraints,
                )
            }

            fn solve(
                handle: &mut Self::Handle,
                decision_values: &[Self],
                parameter_values: &[Self],
                options: &RawOptions<Self>,
                callback: Option<Callback<'_, Self>>,
                values: &mut [Self],
            ) -> Result<i32, cxx::Exception> {
                let options = bridge::$Options {
                    tolerance: options.tolerance,
                    max_iterations: options.max_iterations,
                    timeout: options.timeout,
                    feasible_ipm: options.feasible_ipm,
                    diagnostics: options.diagnostics,
                    spy: options.spy,
                };
                let has_callback = callback.is_some();
                bridge::$solve(
                    handle.pin_mut(),
                    decision_values,
                    parameter_values,
                    &options,
                    has_callback,
                    &mut $Callback(callback),
                    values,
                )
            }
        }
    };
}

backend!(
    f64,
    NodeF64,
    OptionsF64,
    SparseF64,
    CallbackF64,
    ProblemF64,
    build_problem_f64,
    solve_problem_f64
);
backend!(
    f32,
    NodeF32,
    OptionsF32,
    SparseF32,
    CallbackF32,
    ProblemF32,
    build_problem_f32,
    solve_problem_f32
);
