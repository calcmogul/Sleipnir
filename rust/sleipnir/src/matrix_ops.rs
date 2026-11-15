// Copyright (c) Sleipnir contributors

//! Arithmetic operators between [`VariableMatrix`]es, scalars,
//! [`Variable`]s, nested arrays, and nalgebra
//! matrices.
//!
//! `+` and `-` are elementwise, and `*` between matrices is the matrix
//! product. At least one operand of each matrix operator is a
//! [`VariableMatrix`] or [`VariableBlock`].

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{Scalar, Variable, VariableBlock, VariableMatrix};

/// Matrix-shaped operand.
pub(crate) trait MatrixLike<'p, S: Scalar> {
    fn shape(&self) -> (usize, usize);

    fn element(&self, row: usize, col: usize) -> Variable<'p, S>;
}

impl<'p, S: Scalar> MatrixLike<'p, S> for VariableMatrix<'p, S> {
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn element(&self, row: usize, col: usize) -> Variable<'p, S> {
        self[(row, col)]
    }
}

impl<'p, S, T, const R: usize, const C: usize> MatrixLike<'p, S> for [[T; C]; R]
where
    S: Scalar,
    T: Copy + Into<Variable<'p, S>>,
{
    fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    fn element(&self, row: usize, col: usize) -> Variable<'p, S> {
        self[row][col].into()
    }
}

impl<'p, S, R, C, St> MatrixLike<'p, S> for nalgebra::Matrix<S, R, C, St>
where
    S: Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<S, R, C>,
{
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn element(&self, row: usize, col: usize) -> Variable<'p, S> {
        self[(row, col)].into()
    }
}

impl<'p, S: Scalar> MatrixLike<'p, S> for VariableBlock<'_, 'p, S> {
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn element(&self, row: usize, col: usize) -> Variable<'p, S> {
        self[(row, col)]
    }
}

impl<'p, S: Scalar, M: MatrixLike<'p, S>> MatrixLike<'p, S> for &M {
    fn shape(&self) -> (usize, usize) {
        (**self).shape()
    }

    fn element(&self, row: usize, col: usize) -> Variable<'p, S> {
        (**self).element(row, col)
    }
}

/// Copies a matrix-shaped operand into a [`VariableMatrix`].
pub(crate) fn to_matrix<'p, S: Scalar>(m: &impl MatrixLike<'p, S>) -> VariableMatrix<'p, S> {
    let (rows, cols) = m.shape();
    VariableMatrix::from_fn(rows, cols, |row, col| m.element(row, col))
}

#[track_caller]
fn elementwise<'p, S: Scalar>(
    a: &impl MatrixLike<'p, S>,
    b: &impl MatrixLike<'p, S>,
    f: impl Fn(Variable<'p, S>, Variable<'p, S>) -> Variable<'p, S>,
) -> VariableMatrix<'p, S> {
    let ((a_rows, a_cols), (b_rows, b_cols)) = (a.shape(), b.shape());
    assert!(
        (a_rows, a_cols) == (b_rows, b_cols),
        "matrix shape mismatch: {a_rows}×{a_cols} vs {b_rows}×{b_cols}"
    );
    VariableMatrix::from_fn(a_rows, a_cols, |row, col| {
        f(a.element(row, col), b.element(row, col))
    })
}

#[track_caller]
fn matmul<'p, S: Scalar>(
    a: &impl MatrixLike<'p, S>,
    b: &impl MatrixLike<'p, S>,
) -> VariableMatrix<'p, S> {
    let ((a_rows, a_cols), (b_rows, b_cols)) = (a.shape(), b.shape());
    assert!(
        a_cols == b_rows,
        "matrix shape mismatch: {a_rows}×{a_cols} * {b_rows}×{b_cols}"
    );
    VariableMatrix::from_fn(a_rows, b_cols, |row, col| {
        (0..a_cols)
            .map(|k| a.element(row, k) * b.element(k, col))
            .sum()
    })
}

/// Implements `+`, `-`, and `*` for one operand pair. Each operand comes
/// with its extra impl generics and where-clause bounds, each with a
/// trailing comma.
macro_rules! matrix_ops {
    ([$($lg:tt)*] [$($rg:tt)*] [$($lb:tt)*] [$($rb:tt)*] $Lhs:ty, $Rhs:ty) => {
        impl<'p, S: Scalar, $($lg)* $($rg)*> Add<$Rhs> for $Lhs where $($lb)* $($rb)* {
            type Output = VariableMatrix<'p, S>;

            #[track_caller]
            fn add(self, rhs: $Rhs) -> VariableMatrix<'p, S> {
                elementwise(&self, &rhs, |a, b| a + b)
            }
        }

        impl<'p, S: Scalar, $($lg)* $($rg)*> Sub<$Rhs> for $Lhs where $($lb)* $($rb)* {
            type Output = VariableMatrix<'p, S>;

            #[track_caller]
            fn sub(self, rhs: $Rhs) -> VariableMatrix<'p, S> {
                elementwise(&self, &rhs, |a, b| a - b)
            }
        }

        impl<'p, S: Scalar, $($lg)* $($rg)*> Mul<$Rhs> for $Lhs where $($lb)* $($rb)* {
            type Output = VariableMatrix<'p, S>;

            /// Matrix product.
            #[track_caller]
            fn mul(self, rhs: $Rhs) -> VariableMatrix<'p, S> {
                matmul(&self, &rhs)
            }
        }
    };
}

/// Implements `+`, `-`, and `*` for every pair of a left operand from the
/// first list and a right operand from the second. Operands are written as
/// `{[generics] [bounds] Type}`.
macro_rules! matrix_ops_cross {
    ([$($lhs:tt)*] $rhs:tt) => {
        $(matrix_ops_cross!(@row $lhs $rhs);)*
    };
    (@row {$lg:tt $lb:tt $Lhs:ty} [$({$rg:tt $rb:tt $Rhs:ty})*]) => {
        $(matrix_ops!($lg $rg $lb $rb $Lhs, $Rhs);)*
    };
}

/// Implements `+=`, `-=`, and `*=` (matrix product) on `VariableMatrix` for
/// each right operand.
macro_rules! matrix_assign_ops {
    ($({[$($generics:tt)*] [$($bounds:tt)*] $Rhs:ty})*) => {$(
        impl<'p, S: Scalar, $($generics)*> AddAssign<$Rhs> for VariableMatrix<'p, S>
        where
            $($bounds)*
        {
            #[track_caller]
            fn add_assign(&mut self, rhs: $Rhs) {
                *self = elementwise(&*self, &rhs, |a, b| a + b);
            }
        }

        impl<'p, S: Scalar, $($generics)*> SubAssign<$Rhs> for VariableMatrix<'p, S>
        where
            $($bounds)*
        {
            #[track_caller]
            fn sub_assign(&mut self, rhs: $Rhs) {
                *self = elementwise(&*self, &rhs, |a, b| a - b);
            }
        }

        impl<'p, S: Scalar, $($generics)*> MulAssign<$Rhs> for VariableMatrix<'p, S>
        where
            $($bounds)*
        {
            #[track_caller]
            fn mul_assign(&mut self, rhs: $Rhs) {
                *self = matmul(&*self, &rhs);
            }
        }
    )*};
}

// Operands with variables: matrices and views, owned and borrowed
matrix_ops_cross!(
    [
        {[] [] VariableMatrix<'p, S>}
        {[] [] &VariableMatrix<'p, S>}
        {[] [] VariableBlock<'_, 'p, S>}
        {[] [] &VariableBlock<'_, 'p, S>}
    ]
    [
        {[] [] VariableMatrix<'p, S>}
        {[] [] &VariableMatrix<'p, S>}
        {[] [] VariableBlock<'_, 'p, S>}
        {[] [] &VariableBlock<'_, 'p, S>}
        {[T2, const R2: usize, const C2: usize,] [T2: Copy + Into<Variable<'p, S>>,] [[T2; C2]; R2]}
        {[T2, const R2: usize, const C2: usize,] [T2: Copy + Into<Variable<'p, S>>,] &[[T2; C2]; R2]}
    ]
);

// Nested arrays on the left
matrix_ops_cross!(
    [
        {[T1, const R1: usize, const C1: usize,] [T1: Copy + Into<Variable<'p, S>>,] [[T1; C1]; R1]}
        {[T1, const R1: usize, const C1: usize,] [T1: Copy + Into<Variable<'p, S>>,] &[[T1; C1]; R1]}
    ]
    [
        {[] [] VariableMatrix<'p, S>}
        {[] [] &VariableMatrix<'p, S>}
        {[] [] VariableBlock<'_, 'p, S>}
        {[] [] &VariableBlock<'_, 'p, S>}
    ]
);

matrix_assign_ops!(
    {[] [] VariableMatrix<'p, S>}
    {[] [] &VariableMatrix<'p, S>}
    {[] [] VariableBlock<'_, 'p, S>}
    {[] [] &VariableBlock<'_, 'p, S>}
    {[T, const R: usize, const C: usize,] [T: Copy + Into<Variable<'p, S>>,] [[T; C]; R]}
    {[T, const R: usize, const C: usize,] [T: Copy + Into<Variable<'p, S>>,] &[[T; C]; R]}
);

mod nalgebra_ops {
    use super::*;
    use nalgebra::{Dim, Matrix, RawStorage};

    matrix_ops_cross!(
        [
            {[] [] VariableMatrix<'p, S>}
            {[] [] &VariableMatrix<'p, S>}
            {[] [] VariableBlock<'_, 'p, S>}
            {[] [] &VariableBlock<'_, 'p, S>}
        ]
        [
            {[R2: Dim, C2: Dim, St2: RawStorage<S, R2, C2>,] [] Matrix<S, R2, C2, St2>}
            {[R2: Dim, C2: Dim, St2: RawStorage<S, R2, C2>,] [] &Matrix<S, R2, C2, St2>}
        ]
    );

    matrix_ops_cross!(
        [
            {[R1: Dim, C1: Dim, St1: RawStorage<S, R1, C1>,] [] Matrix<S, R1, C1, St1>}
            {[R1: Dim, C1: Dim, St1: RawStorage<S, R1, C1>,] [] &Matrix<S, R1, C1, St1>}
        ]
        [
            {[] [] VariableMatrix<'p, S>}
            {[] [] &VariableMatrix<'p, S>}
            {[] [] VariableBlock<'_, 'p, S>}
            {[] [] &VariableBlock<'_, 'p, S>}
        ]
    );

    matrix_assign_ops!(
        {[R: Dim, C: Dim, St: RawStorage<S, R, C>,] [] Matrix<S, R, C, St>}
        {[R: Dim, C: Dim, St: RawStorage<S, R, C>,] [] &Matrix<S, R, C, St>}
    );
}

/// Implements matrix-scalar `*` and `/`, scalar-matrix `*`, and `*=` and `/=`
/// for scalar and [`Variable`] factors.
macro_rules! scalar_ops {
    ($($Matrix:ty),*) => {$(
        impl<'p, S: Scalar> Mul<S> for $Matrix {
            type Output = VariableMatrix<'p, S>;

            fn mul(self, rhs: S) -> VariableMatrix<'p, S> {
                self.map(|x| x * rhs)
            }
        }

        impl<'p, S: Scalar> Mul<Variable<'p, S>> for $Matrix {
            type Output = VariableMatrix<'p, S>;

            #[track_caller]
            fn mul(self, rhs: Variable<'p, S>) -> VariableMatrix<'p, S> {
                self.map(|x| x * rhs)
            }
        }

        impl<'p, S: Scalar> Mul<$Matrix> for Variable<'p, S> {
            type Output = VariableMatrix<'p, S>;

            #[track_caller]
            fn mul(self, rhs: $Matrix) -> VariableMatrix<'p, S> {
                rhs.map(|x| self * x)
            }
        }

        impl<'p, S: Scalar> Div<S> for $Matrix {
            type Output = VariableMatrix<'p, S>;

            fn div(self, rhs: S) -> VariableMatrix<'p, S> {
                self.map(|x| x / rhs)
            }
        }

        impl<'p, S: Scalar> Div<Variable<'p, S>> for $Matrix {
            type Output = VariableMatrix<'p, S>;

            #[track_caller]
            fn div(self, rhs: Variable<'p, S>) -> VariableMatrix<'p, S> {
                self.map(|x| x / rhs)
            }
        }
    )*};
}

scalar_ops!(
    VariableMatrix<'p, S>,
    &VariableMatrix<'p, S>,
    VariableBlock<'_, 'p, S>,
    &VariableBlock<'_, 'p, S>
);

macro_rules! scalar_lhs_ops {
    ($($S:ty),*) => {$(
        impl<'p> Mul<VariableMatrix<'p, $S>> for $S {
            type Output = VariableMatrix<'p, $S>;

            fn mul(self, rhs: VariableMatrix<'p, $S>) -> VariableMatrix<'p, $S> {
                rhs.map(|x| self * x)
            }
        }

        impl<'p> Mul<&VariableMatrix<'p, $S>> for $S {
            type Output = VariableMatrix<'p, $S>;

            fn mul(self, rhs: &VariableMatrix<'p, $S>) -> VariableMatrix<'p, $S> {
                rhs.map(|x| self * x)
            }
        }

        impl<'p> Mul<VariableBlock<'_, 'p, $S>> for $S {
            type Output = VariableMatrix<'p, $S>;

            fn mul(self, rhs: VariableBlock<'_, 'p, $S>) -> VariableMatrix<'p, $S> {
                rhs.map(|x| self * x)
            }
        }

        impl<'p> Mul<&VariableBlock<'_, 'p, $S>> for $S {
            type Output = VariableMatrix<'p, $S>;

            fn mul(self, rhs: &VariableBlock<'_, 'p, $S>) -> VariableMatrix<'p, $S> {
                rhs.map(|x| self * x)
            }
        }
    )*};
}

scalar_lhs_ops!(f32, f64);

impl<'p, S: Scalar> MulAssign<S> for VariableMatrix<'p, S> {
    fn mul_assign(&mut self, rhs: S) {
        *self = &*self * rhs;
    }
}

impl<'p, S: Scalar> MulAssign<Variable<'p, S>> for VariableMatrix<'p, S> {
    #[track_caller]
    fn mul_assign(&mut self, rhs: Variable<'p, S>) {
        *self = &*self * rhs;
    }
}

impl<'p, S: Scalar> DivAssign<S> for VariableMatrix<'p, S> {
    fn div_assign(&mut self, rhs: S) {
        *self = &*self / rhs;
    }
}

impl<'p, S: Scalar> DivAssign<Variable<'p, S>> for VariableMatrix<'p, S> {
    #[track_caller]
    fn div_assign(&mut self, rhs: Variable<'p, S>) {
        *self = &*self / rhs;
    }
}
