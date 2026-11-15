// Copyright (c) Sleipnir contributors

use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::Scalar;

/// How an expression depends on the decision variables, ordered from
/// simplest to most general.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpressionType {
    /// No expression, e.g., the cost of a problem without one.
    None,
    /// Doesn't depend on the decision variables.
    Constant,
    /// Linear in the decision variables.
    Linear,
    /// Quadratic in the decision variables.
    Quadratic,
    /// Any other dependence on the decision variables.
    Nonlinear,
}

impl fmt::Display for ExpressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::Nonlinear => "nonlinear",
        })
    }
}

/// Comparison that [`if_else`] evaluates on its first two arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Condition {
    /// a < b
    Less,
    /// a ≤ b
    LessEqual,
    /// a > b
    Greater,
    /// a ≥ b
    GreaterEqual,
    /// a = b
    Equal,
    /// a ≠ b
    NotEqual,
}

impl Condition {
    /// Evaluates the comparison.
    pub fn test<S: Scalar>(self, a: S, b: S) -> bool {
        match self {
            Self::Less => a < b,
            Self::LessEqual => a <= b,
            Self::Greater => a > b,
            Self::GreaterEqual => a >= b,
            Self::Equal => a == b,
            Self::NotEqual => a != b,
        }
    }
}

/// Unary expression graph operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
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
    Ln,
    Log10,
    Sign,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
}

/// Binary expression graph operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
    Hypot,
    Max,
    Min,
}

/// Expression graph node. Operands always precede the nodes that use them.
#[derive(Clone, Copy, Debug)]
pub enum Node<S> {
    Decision(S),
    /// Leaf whose value can be set but that the solver doesn't optimize.
    Parameter(S),
    Constant(S),
    Unary(Unary, usize),
    Binary(Binary, usize, usize),
    /// `t` if `condition(a, b)`, otherwise `f`, with operands `[a, b, t, f]`.
    IfElse(Condition, [usize; 4]),
}

impl Unary {
    pub(crate) fn apply<S: Scalar>(self, x: S) -> S {
        match self {
            Self::Neg => -x,
            Self::Abs => x.abs(),
            Self::Acos => x.acos(),
            Self::Asin => x.asin(),
            Self::Atan => x.atan(),
            Self::Cbrt => x.cbrt(),
            Self::Cos => x.cos(),
            Self::Cosh => x.cosh(),
            Self::Erf => x.erf(),
            Self::Exp => x.exp(),
            Self::Ln => x.ln(),
            Self::Log10 => x.log10(),
            Self::Sign => {
                if x < S::ZERO {
                    -S::ONE
                } else if x == S::ZERO {
                    S::ZERO
                } else {
                    S::ONE
                }
            }
            Self::Sin => x.sin(),
            Self::Sinh => x.sinh(),
            Self::Sqrt => x.sqrt(),
            Self::Tan => x.tan(),
            Self::Tanh => x.tanh(),
        }
    }
}

impl Binary {
    pub(crate) fn apply<S: Scalar>(self, a: S, b: S) -> S {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => a.powf(b),
            Self::Atan2 => a.atan2(b),
            Self::Hypot => a.hypot(b),
            Self::Max => a.max(b),
            Self::Min => a.min(b),
        }
    }
}

impl<S: Scalar> Node<S> {
    /// Evaluates the node from its operands' values.
    fn eval(&self, values: &[S]) -> S {
        match *self {
            Node::Decision(value) | Node::Parameter(value) | Node::Constant(value) => value,
            Node::Unary(op, a) => op.apply(values[a]),
            Node::Binary(op, a, b) => op.apply(values[a], values[b]),
            Node::IfElse(condition, [a, b, t, f]) => {
                if condition.test(values[a], values[b]) {
                    values[t]
                } else {
                    values[f]
                }
            }
        }
    }

    /// Returns the node's expression type from its operands' types, following
    /// the C++ library's rules.
    fn expression_type(&self, types: &[ExpressionType]) -> ExpressionType {
        use ExpressionType::*;

        match *self {
            Node::Decision(_) | Node::Parameter(_) => Linear,
            Node::Constant(_) => Constant,
            Node::Unary(Unary::Neg, a) => types[a],
            Node::Unary(..) | Node::IfElse(..) => Nonlinear,
            Node::Binary(Binary::Add | Binary::Sub, a, b) => types[a].max(types[b]),
            Node::Binary(Binary::Mul, a, b) => match (types[a], types[b]) {
                (Constant, other) | (other, Constant) => other,
                (Linear, Linear) => Quadratic,
                _ => Nonlinear,
            },
            Node::Binary(Binary::Div, a, b) if types[b] == Constant => types[a],
            Node::Binary(..) => Nonlinear,
        }
    }
}

/// Append-only storage for a problem's expression graph, with cached values.
#[derive(Debug)]
pub(crate) struct Arena<S> {
    nodes: RefCell<Vec<Node<S>>>,
    types: RefCell<Vec<ExpressionType>>,
    /// Values of the first `values.len()` nodes. Setting a leaf's value
    /// truncates this to the leaf, since only later nodes depend on it.
    values: RefCell<Vec<S>>,
}

impl<S> Default for Arena<S> {
    fn default() -> Self {
        Self {
            nodes: RefCell::default(),
            types: RefCell::default(),
            values: RefCell::default(),
        }
    }
}

impl<S: Scalar> Arena<S> {
    pub(crate) fn push(&self, node: Node<S>) -> usize {
        let mut types = self.types.borrow_mut();
        let node_type = node.expression_type(&types);
        types.push(node_type);
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(node);
        nodes.len() - 1
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub(crate) fn nodes(&self) -> std::cell::Ref<'_, Vec<Node<S>>> {
        self.nodes.borrow()
    }

    pub(crate) fn node(&self, id: usize) -> Node<S> {
        self.nodes.borrow()[id]
    }

    pub(crate) fn expression_type(&self, id: usize) -> ExpressionType {
        self.types.borrow()[id]
    }

    /// Sets a decision variable's or parameter's value.
    #[track_caller]
    pub(crate) fn set_leaf_value(&self, id: usize, value: S) {
        match &mut self.nodes.borrow_mut()[id] {
            Node::Decision(x) | Node::Parameter(x) => *x = value,
            _ => panic!("only decision variables and parameters have settable values"),
        }
        self.values.borrow_mut().truncate(id);
    }

    /// Returns node `id`'s value, evaluating uncached nodes up to it.
    pub(crate) fn value(&self, id: usize) -> S {
        let mut values = self.values.borrow_mut();
        if values.len() <= id {
            let nodes = self.nodes.borrow();
            for node in &nodes[values.len()..=id] {
                let value = node.eval(&values);
                values.push(value);
            }
        }
        values[id]
    }

    /// Evaluates node `id` with the kth decision variable taking the value
    /// `x[k]`, without touching the cache.
    pub(crate) fn value_at(&self, id: usize, x: &[S]) -> S {
        let nodes = self.nodes.borrow();
        let mut values = Vec::with_capacity(id + 1);
        let mut decision_index = 0;
        for node in &nodes[..=id] {
            let value = match *node {
                Node::Decision(_) => {
                    decision_index += 1;
                    x[decision_index - 1]
                }
                ref node => node.eval(&values),
            };
            values.push(value);
        }
        values[id]
    }
}

/// Constant, decision variable, or expression of decision variables.
///
/// Handles are `Copy`. A constant (e.g., from `Variable::from(2.0)`) needs no
/// [`Problem`](crate::Problem), and arithmetic on constants folds to a
/// constant. Any other handle borrows the problem that owns the expression
/// graph, and arithmetic on it records new expressions in that graph.
/// Combining handles from different problems panics.
///
/// Like the C++ library, arithmetic prunes trivial operations, e.g., `x * 0.0`
/// is the constant zero and `x * 1.0` is `x`.
#[derive(Clone, Copy)]
pub struct Variable<'p, S: Scalar>(pub(crate) Repr<'p, S>);

#[derive(Clone, Copy)]
pub(crate) enum Repr<'p, S: Scalar> {
    Constant(S),
    Node(&'p Arena<S>, usize),
}

impl<'p, S: Scalar> Variable<'p, S> {
    pub(crate) fn new(arena: &'p Arena<S>, node: Node<S>) -> Self {
        Self(Repr::Node(arena, arena.push(node)))
    }

    /// Returns the expression's value at the current decision variable values.
    pub fn value(&self) -> S {
        match self.0 {
            Repr::Constant(value) => value,
            Repr::Node(arena, id) => arena.value(id),
        }
    }

    /// Returns how the expression depends on the decision variables.
    pub fn expression_type(&self) -> ExpressionType {
        match self.0 {
            Repr::Constant(_) => ExpressionType::Constant,
            Repr::Node(arena, id) => arena.expression_type(id),
        }
    }

    /// Returns true if this is a constant, i.e., it doesn't belong to a
    /// problem.
    pub fn is_constant(&self) -> bool {
        matches!(self.0, Repr::Constant(_))
    }

    fn constant_value(&self) -> Option<S> {
        match self.0 {
            Repr::Constant(value) => Some(value),
            Repr::Node(..) => None,
        }
    }

    /// Returns this expression's node in `arena`, adding a constant node if
    /// it's a constant.
    ///
    /// # Panics
    ///
    /// Panics if the expression belongs to another arena.
    #[track_caller]
    pub(crate) fn node_in(self, arena: &Arena<S>) -> usize {
        match self.0 {
            Repr::Constant(value) => arena.push(Node::Constant(value)),
            Repr::Node(owner, id) => {
                assert!(
                    std::ptr::eq(owner, arena),
                    "variable belongs to a different problem"
                );
                id
            }
        }
    }

    fn unary(self, op: Unary) -> Self {
        match self.0 {
            Repr::Constant(value) => Self(Repr::Constant(op.apply(value))),
            Repr::Node(arena, id) => Self::new(arena, Node::Unary(op, id)),
        }
    }

    /// Returns the arena shared by the non-constant operands.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different problems.
    #[track_caller]
    fn common_arena(operands: &[Self]) -> Option<&'p Arena<S>> {
        let mut common = None;
        for operand in operands {
            if let Repr::Node(arena, _) = operand.0 {
                match common {
                    None => common = Some(arena),
                    Some(common) => assert!(
                        std::ptr::eq(common, arena),
                        "variables belong to different problems"
                    ),
                }
            }
        }
        common
    }

    #[track_caller]
    fn binary(self, rhs: Self, op: Binary) -> Self {
        let Some(arena) = Self::common_arena(&[self, rhs]) else {
            let (a, b) = (
                self.constant_value().unwrap(),
                rhs.constant_value().unwrap(),
            );
            return Self(Repr::Constant(op.apply(a, b)));
        };

        // Prune trivial operations like the C++ library does
        let (zero, one) = (Some(S::ZERO), Some(S::ONE));
        let (a, b) = (self.constant_value(), rhs.constant_value());
        match op {
            Binary::Add if a == zero => return rhs,
            Binary::Add | Binary::Sub if b == zero => return self,
            Binary::Sub if a == zero => return -rhs,
            Binary::Mul if a == zero || b == one => return self,
            Binary::Mul if b == zero || a == one => return rhs,
            Binary::Div if a == zero || b == one => return self,
            Binary::Pow if a == zero || a == one || b == one => return self,
            Binary::Pow if b == zero => return S::ONE.into(),
            Binary::Pow if b == Some(S::ONE + S::ONE) => return self * self,
            Binary::Pow if b == Some(S::ONE / (S::ONE + S::ONE)) => return self.sqrt(),
            _ => {}
        }

        let node = Node::Binary(op, self.node_in(arena), rhs.node_in(arena));
        Self::new(arena, node)
    }
}

impl<'p, S: Scalar> std::iter::Sum for Variable<'p, S> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, x| sum + x)
            .unwrap_or_else(|| S::ZERO.into())
    }
}

impl<'a, 'p, S: Scalar> std::iter::Sum<&'a Self> for Variable<'p, S> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl<'p, S: Scalar> std::iter::Product for Variable<'p, S> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|product, x| product * x)
            .unwrap_or_else(|| S::ONE.into())
    }
}

impl<'a, 'p, S: Scalar> std::iter::Product<&'a Self> for Variable<'p, S> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl<S: Scalar> From<S> for Variable<'_, S> {
    fn from(value: S) -> Self {
        Self(Repr::Constant(value))
    }
}

/// Returns `t` if `condition(a, b)` holds, otherwise `f`.
///
/// The condition is evaluated on the values of `a` and `b` each time the
/// result is evaluated, and derivatives only flow through the selected
/// branch.
///
/// ```
/// use sleipnir::{Condition, Problem, if_else};
///
/// let problem: Problem = Problem::new();
/// let x = problem.decision_variable();
/// let relu = if_else(Condition::Greater, x, 0.0, x, 0.0);
///
/// problem.set_initial(x, -2.0);
/// assert_eq!(relu.value(), 0.0);
/// problem.set_initial(x, 3.0);
/// assert_eq!(relu.value(), 3.0);
/// ```
#[track_caller]
pub fn if_else<'p, S: Scalar>(
    condition: Condition,
    a: impl Into<Variable<'p, S>>,
    b: impl Into<Variable<'p, S>>,
    t: impl Into<Variable<'p, S>>,
    f: impl Into<Variable<'p, S>>,
) -> Variable<'p, S> {
    let (a, b, t, f) = (a.into(), b.into(), t.into(), f.into());

    // Prune like the C++ library does
    if let (Some(a), Some(b)) = (a.constant_value(), b.constant_value()) {
        return if condition.test(a, b) { t } else { f };
    }
    if let (Some(t_value), Some(f_value)) = (t.constant_value(), f.constant_value())
        && t_value == f_value
    {
        return t;
    }

    let arena = Variable::common_arena(&[a, b, t, f]).unwrap();
    let operands = [a, b, t, f].map(|x| x.node_in(arena));
    Variable::new(arena, Node::IfElse(condition, operands))
}

macro_rules! unary_fns {
    ($($(#[$doc:meta])* $name:ident => $op:ident,)*) => {
        impl<S: Scalar> Variable<'_, S> {
            $(
                $(#[$doc])*
                pub fn $name(self) -> Self {
                    self.unary(Unary::$op)
                }
            )*
        }
    };
}

unary_fns! {
    /// Absolute value.
    abs => Abs,
    /// Arccosine.
    acos => Acos,
    /// Arcsine.
    asin => Asin,
    /// Arctangent.
    atan => Atan,
    /// Cube root.
    cbrt => Cbrt,
    /// Cosine.
    cos => Cos,
    /// Hyperbolic cosine.
    cosh => Cosh,
    /// Error function.
    erf => Erf,
    /// Exponential, eˣ.
    exp => Exp,
    /// Natural logarithm.
    ln => Ln,
    /// Base-10 logarithm.
    log10 => Log10,
    /// Sign: −1, 0, or 1. NaN maps to 1, as in the C++ library.
    sign => Sign,
    /// Sine.
    sin => Sin,
    /// Hyperbolic sine.
    sinh => Sinh,
    /// Square root.
    sqrt => Sqrt,
    /// Tangent.
    tan => Tan,
    /// Hyperbolic tangent.
    tanh => Tanh,
}

impl<S: Scalar> Variable<'_, S> {
    /// Raises `self` to a variable power.
    #[track_caller]
    pub fn pow(self, exponent: Self) -> Self {
        self.binary(exponent, Binary::Pow)
    }

    /// Raises `self` to a constant power.
    pub fn powf(self, exponent: S) -> Self {
        self.binary(exponent.into(), Binary::Pow)
    }

    /// Four-quadrant arctangent of `self` (y) and `x`.
    #[track_caller]
    pub fn atan2(self, x: Self) -> Self {
        self.binary(x, Binary::Atan2)
    }

    /// √(self² + other²).
    #[track_caller]
    pub fn hypot(self, other: Self) -> Self {
        self.binary(other, Binary::Hypot)
    }

    /// Maximum of `self` and `other`.
    #[track_caller]
    pub fn max(self, other: Self) -> Self {
        self.binary(other, Binary::Max)
    }

    /// Minimum of `self` and `other`.
    #[track_caller]
    pub fn min(self, other: Self) -> Self {
        self.binary(other, Binary::Min)
    }
}

impl<S: Scalar> fmt::Debug for Variable<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Repr::Constant(value) => f.debug_tuple("Variable::Constant").field(&value).finish(),
            Repr::Node(_, id) => f
                .debug_struct("Variable")
                .field("id", &id)
                .field("value", &self.value())
                .finish(),
        }
    }
}

impl<'p, S: Scalar> Neg for Variable<'p, S> {
    type Output = Self;

    fn neg(self) -> Self {
        self.unary(Unary::Neg)
    }
}

macro_rules! binary_op {
    ($Trait:ident, $method:ident, $AssignTrait:ident, $assign_method:ident, $Node:ident) => {
        impl<'p, S: Scalar> $Trait for Variable<'p, S> {
            type Output = Self;

            #[track_caller]
            fn $method(self, rhs: Self) -> Self {
                self.binary(rhs, Binary::$Node)
            }
        }

        impl<'p, S: Scalar> $Trait<S> for Variable<'p, S> {
            type Output = Self;

            fn $method(self, rhs: S) -> Self {
                self.binary(rhs.into(), Binary::$Node)
            }
        }

        impl<'p, S: Scalar> $AssignTrait for Variable<'p, S> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: Self) {
                *self = self.binary(rhs, Binary::$Node);
            }
        }

        impl<'p, S: Scalar> $AssignTrait<S> for Variable<'p, S> {
            fn $assign_method(&mut self, rhs: S) {
                *self = self.binary(rhs.into(), Binary::$Node);
            }
        }

        binary_op!(@scalar_lhs f32, $Trait, $method, $Node);
        binary_op!(@scalar_lhs f64, $Trait, $method, $Node);
    };
    (@scalar_lhs $S:ty, $Trait:ident, $method:ident, $Node:ident) => {
        impl<'p> $Trait<Variable<'p, $S>> for $S {
            type Output = Variable<'p, $S>;

            fn $method(self, rhs: Variable<'p, $S>) -> Variable<'p, $S> {
                Variable::from(self).binary(rhs, Binary::$Node)
            }
        }
    };
}

binary_op!(Add, add, AddAssign, add_assign, Add);
binary_op!(Sub, sub, SubAssign, sub_assign, Sub);
binary_op!(Mul, mul, MulAssign, mul_assign, Mul);
binary_op!(Div, div, DivAssign, div_assign, Div);
