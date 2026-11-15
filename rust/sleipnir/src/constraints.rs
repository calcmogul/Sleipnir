// Copyright (c) Sleipnir contributors

use crate::matrix_ops::to_matrix;
use crate::variable::Repr;
use crate::{Scalar, Variable, VariableBlock, VariableMatrix};

/// Equality constraints cᵢ(x) = 0.
#[derive(Clone, Debug)]
pub struct EqualityConstraints<'p, S: Scalar> {
    residuals: Vec<Variable<'p, S>>,
}

/// Inequality constraints cᵢ(x) ≥ 0.
#[derive(Clone, Debug)]
pub struct InequalityConstraints<'p, S: Scalar> {
    residuals: Vec<Variable<'p, S>>,
}

impl<'p, S: Scalar> EqualityConstraints<'p, S> {
    /// Constraint functions cᵢ(x), constrained to equal zero.
    pub fn residuals(&self) -> &[Variable<'p, S>] {
        &self.residuals
    }
}

impl<'p, S: Scalar> InequalityConstraints<'p, S> {
    /// Constraint functions cᵢ(x), constrained to be nonnegative.
    pub fn residuals(&self) -> &[Variable<'p, S>] {
        &self.residuals
    }
}

/// Equality or inequality constraints, as accepted by
/// [`Problem::subject_to`](crate::Problem::subject_to).
#[derive(Clone, Debug)]
pub enum Constraints<'p, S: Scalar> {
    /// Equality constraints.
    Equality(EqualityConstraints<'p, S>),
    /// Inequality constraints.
    Inequality(InequalityConstraints<'p, S>),
}

impl<'p, S: Scalar> From<EqualityConstraints<'p, S>> for Constraints<'p, S> {
    fn from(constraints: EqualityConstraints<'p, S>) -> Self {
        Self::Equality(constraints)
    }
}

impl<'p, S: Scalar> From<InequalityConstraints<'p, S>> for Constraints<'p, S> {
    fn from(constraints: InequalityConstraints<'p, S>) -> Self {
        Self::Inequality(constraints)
    }
}

/// Constraint-building comparisons between [`ConstraintOperand`]s.
///
/// The [`constrain!`](crate::constrain) macro expands to these methods.
/// Matrix operands are compared elementwise, and scalar or [`Variable`]
/// operands are broadcast against matrix operands. At least one side must
/// be something other than a plain scalar.
#[diagnostic::on_unimplemented(
    message = "can't constrain `{Self}` against `{Rhs}`",
    label = "no constraint between these operand types",
    note = "operands are `Variable`s, `VariableMatrix`es, nested arrays, nalgebra matrices, and floats of the problem's scalar type, and at least one side must not be a plain float",
    note = "scalars must be floats of the problem's scalar type, e.g., `0.0`, not `0`"
)]
pub trait Compare<'p, S: Scalar, Rhs: ?Sized = Self> {
    #[doc(hidden)]
    fn __operands<'a>(&'a self, rhs: &'a Rhs) -> (Operand<'a, 'p, S>, Operand<'a, 'p, S>);

    /// Constrains `self == rhs`.
    #[track_caller]
    fn equals(&self, rhs: &Rhs) -> EqualityConstraints<'p, S> {
        let (lhs, rhs) = self.__operands(rhs);
        EqualityConstraints {
            residuals: difference(lhs, rhs),
        }
    }

    /// Constrains `self <= rhs`.
    #[track_caller]
    fn leq(&self, rhs: &Rhs) -> InequalityConstraints<'p, S> {
        let (lhs, rhs) = self.__operands(rhs);
        InequalityConstraints {
            residuals: difference(rhs, lhs),
        }
    }

    /// Constrains `self >= rhs`.
    #[track_caller]
    fn geq(&self, rhs: &Rhs) -> InequalityConstraints<'p, S> {
        let (lhs, rhs) = self.__operands(rhs);
        InequalityConstraints {
            residuals: difference(lhs, rhs),
        }
    }
}

#[doc(hidden)]
pub enum Operand<'a, 'p, S: Scalar> {
    Variable(Variable<'p, S>),
    Matrix(&'a VariableMatrix<'p, S>),
    /// Array or nalgebra matrix converted to a `VariableMatrix`.
    Owned(VariableMatrix<'p, S>),
}

/// Returns lhs − rhs elementwise, broadcasting variables against matrices.
#[track_caller]
fn difference<'p, S: Scalar>(
    lhs: Operand<'_, 'p, S>,
    rhs: Operand<'_, 'p, S>,
) -> Vec<Variable<'p, S>> {
    use Operand::*;

    let (lhs, rhs) = match (lhs, rhs) {
        (Owned(a), b) => return difference(Matrix(&a), b),
        (a, Owned(b)) => return difference(a, Matrix(&b)),
        operands => operands,
    };
    match (lhs, rhs) {
        (Variable(a), Variable(b)) => vec![a - b],
        (Matrix(a), Matrix(b)) => {
            assert!(
                (a.rows(), a.cols()) == (b.rows(), b.cols()),
                "constraint shape mismatch: {}×{} vs {}×{}",
                a.rows(),
                a.cols(),
                b.rows(),
                b.cols()
            );
            a.iter().zip(b.iter()).map(|(&a, &b)| a - b).collect()
        }
        (Matrix(a), Variable(b)) => {
            let b = share(b, a);
            a.iter().map(|&a| a - b).collect()
        }
        (Variable(a), Matrix(b)) => {
            let a = share(a, b);
            b.iter().map(|&b| a - b).collect()
        }
        (Owned(_), _) | (_, Owned(_)) => unreachable!("owned operands are borrowed above"),
    }
}

/// Adds a constant to the graph of the matrix's first non-constant element,
/// so broadcasting it adds one node instead of one per element.
fn share<'p, S: Scalar>(x: Variable<'p, S>, matrix: &VariableMatrix<'p, S>) -> Variable<'p, S> {
    // Zero stays a constant so `a - 0` prunes to `a`
    if !x.is_constant() || x.value() == S::ZERO {
        return x;
    }
    matrix
        .iter()
        .find_map(|element| match element.0 {
            Repr::Node(arena, _) => Some(Variable(Repr::Node(arena, x.node_in(arena)))),
            Repr::Constant(_) => None,
        })
        .unwrap_or(x)
}

/// Value that can be one side of a constraint: a float of the problem's
/// scalar type, a [`Variable`], a [`VariableMatrix`], a [`VariableBlock`], a
/// nested array
/// (`[[T; C]; R]`), an nalgebra matrix, or a
/// reference to any of these.
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be a constraint operand",
    label = "not a constraint operand",
    note = "operands are `Variable`s, `VariableMatrix`es, nested arrays, nalgebra matrices, and floats of the problem's scalar type",
    note = "scalars must be floats of the problem's scalar type, e.g., `0.0`, not `0`"
)]
pub trait ConstraintOperand<'p, S: Scalar> {
    #[doc(hidden)]
    type Kind;

    #[doc(hidden)]
    fn __operand(&self) -> Operand<'_, 'p, S>;
}

#[doc(hidden)]
pub mod kind {
    /// Plain scalar operand.
    pub struct Scalar;

    /// Operand that isn't a plain scalar.
    pub struct Expression;

    /// Operand kind pairs a constraint can compare.
    #[diagnostic::on_unimplemented(
        message = "can't constrain a scalar against a scalar",
        label = "both sides are plain scalars",
        note = "at least one side must be a `Variable`, `VariableMatrix`, or matrix"
    )]
    pub trait Comparable {}

    impl Comparable for (Scalar, Expression) {}
    impl Comparable for (Expression, Scalar) {}
    impl Comparable for (Expression, Expression) {}
}

// Concrete impls, since a generic `impl for S` would overlap with the `&T`
// impl
macro_rules! scalar_operand {
    ($($S:ty),*) => {$(
        impl<'p> ConstraintOperand<'p, $S> for $S {
            type Kind = kind::Scalar;

            fn __operand(&self) -> Operand<'_, 'p, $S> {
                Operand::Variable((*self).into())
            }
        }
    )*};
}

scalar_operand!(f32, f64);

impl<'p, S: Scalar> ConstraintOperand<'p, S> for Variable<'p, S> {
    type Kind = kind::Expression;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        Operand::Variable(*self)
    }
}

impl<'p, S: Scalar> ConstraintOperand<'p, S> for VariableMatrix<'p, S> {
    type Kind = kind::Expression;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        Operand::Matrix(self)
    }
}

impl<'p, S: Scalar> ConstraintOperand<'p, S> for VariableBlock<'_, 'p, S> {
    type Kind = kind::Expression;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        Operand::Owned(self.to_matrix())
    }
}

impl<'p, S, T, const R: usize, const C: usize> ConstraintOperand<'p, S> for [[T; C]; R]
where
    S: Scalar,
    T: Copy + Into<Variable<'p, S>>,
{
    type Kind = kind::Expression;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        Operand::Owned(to_matrix(self))
    }
}

impl<'p, S, R, C, St> ConstraintOperand<'p, S> for nalgebra::Matrix<S, R, C, St>
where
    S: Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<S, R, C>,
{
    type Kind = kind::Expression;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        Operand::Owned(to_matrix(self))
    }
}

impl<'p, S: Scalar, T: ConstraintOperand<'p, S> + ?Sized> ConstraintOperand<'p, S> for &T {
    type Kind = T::Kind;

    fn __operand(&self) -> Operand<'_, 'p, S> {
        (**self).__operand()
    }
}

impl<'p, S, L, R> Compare<'p, S, R> for L
where
    S: Scalar,
    L: ConstraintOperand<'p, S> + ?Sized,
    R: ConstraintOperand<'p, S> + ?Sized,
    (L::Kind, R::Kind): kind::Comparable,
{
    fn __operands<'a>(&'a self, rhs: &'a R) -> (Operand<'a, 'p, S>, Operand<'a, 'p, S>) {
        (self.__operand(), rhs.__operand())
    }
}

/// Concatenates the links of a chained inequality like `a <= b <= c`.
#[doc(hidden)]
pub fn chain<'p, S: Scalar, const N: usize>(
    parts: [InequalityConstraints<'p, S>; N],
) -> InequalityConstraints<'p, S> {
    InequalityConstraints {
        residuals: parts.into_iter().flat_map(|part| part.residuals).collect(),
    }
}

#[cfg(test)]
mod tests {
    use crate::{Problem, constrain};

    #[test]
    fn broadcast_constant_adds_one_node() {
        let problem: Problem = Problem::new();
        let y = problem.decision_variable_matrix(100, 1);

        // One shared constant node plus one subtraction per element
        let before = problem.node_count();
        let _ = constrain!(y >= 1.0);
        assert_eq!(problem.node_count() - before, 101);

        // y − 0 prunes to y
        let before = problem.node_count();
        let c = constrain!(y >= 0.0);
        assert_eq!(problem.node_count(), before);
        assert_eq!(c.residuals().len(), 100);
    }
}
