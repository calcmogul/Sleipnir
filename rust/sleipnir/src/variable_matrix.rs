// Copyright (c) Sleipnir contributors

use std::ops::{Index, IndexMut, Neg};

use crate::{Scalar, Variable};

/// Row-major matrix of [`Variable`]s.
#[derive(Clone, Debug)]
pub struct VariableMatrix<'p, S: Scalar> {
    rows: usize,
    cols: usize,
    data: Vec<Variable<'p, S>>,
}

impl<'p, S: Scalar> VariableMatrix<'p, S> {
    /// Builds a matrix whose element (row, col) is `f(row, col)`.
    pub fn from_fn(
        rows: usize,
        cols: usize,
        mut f: impl FnMut(usize, usize) -> Variable<'p, S>,
    ) -> Self {
        let data = (0..rows)
            .flat_map(|row| (0..cols).map(move |col| (row, col)))
            .map(|(row, col)| f(row, col))
            .collect();
        Self { rows, cols, data }
    }

    /// Builds a matrix with every element equal to `value`.
    pub fn from_element(rows: usize, cols: usize, value: impl Into<Variable<'p, S>>) -> Self {
        let value = value.into();
        Self::from_fn(rows, cols, |_, _| value)
    }

    /// Builds a matrix from elements in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    #[track_caller]
    pub fn from_row_slice<T: Copy + Into<Variable<'p, S>>>(
        rows: usize,
        cols: usize,
        data: &[T],
    ) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "{} elements can't fill a {rows}×{cols} matrix",
            data.len()
        );
        Self {
            rows,
            cols,
            data: data.iter().map(|&x| x.into()).collect(),
        }
    }

    /// Builds a matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_element(rows, cols, S::ZERO)
    }

    /// Builds a matrix of ones.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::from_element(rows, cols, S::ONE)
    }

    /// Builds an n×n identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |row, col| {
            if row == col { S::ONE } else { S::ZERO }.into()
        })
    }

    /// Concatenates blocks into one matrix. Blocks in a row must have the
    /// same number of rows, and block rows must have the same number of
    /// columns.
    ///
    /// ```
    /// use sleipnir::VariableMatrix;
    ///
    /// let a = VariableMatrix::<f64>::identity(2);
    /// let b = VariableMatrix::zeros(2, 1);
    /// let c = VariableMatrix::ones(1, 3);
    /// let m = VariableMatrix::block(&[&[&a, &b], &[&c]]);
    /// assert_eq!(m.values(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the block shapes don't line up.
    #[track_caller]
    pub fn block(blocks: &[&[&Self]]) -> Self {
        let cols = blocks
            .first()
            .map_or(0, |row| row.iter().map(|block| block.cols).sum());
        let mut rows = 0;
        let mut data = Vec::new();
        for (i, block_row) in blocks.iter().enumerate() {
            let height = block_row.first().map_or(0, |block| block.rows);
            assert!(
                block_row.iter().all(|block| block.rows == height),
                "blocks in block row {i} have different row counts"
            );
            assert_eq!(
                block_row.iter().map(|block| block.cols).sum::<usize>(),
                cols,
                "block row {i} has a different column count than block row 0"
            );
            for row in 0..height {
                for block in *block_row {
                    data.extend_from_slice(&block.data[row * block.cols..(row + 1) * block.cols]);
                }
            }
            rows += height;
        }
        Self { rows, cols, data }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Elements in row-major order.
    pub fn as_slice(&self) -> &[Variable<'p, S>] {
        &self.data
    }

    /// Iterates over elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, Variable<'p, S>> {
        self.data.iter()
    }

    /// Returns the elements' values in row-major order.
    pub fn values(&self) -> Vec<S> {
        self.data.iter().map(Variable::value).collect()
    }

    /// Returns the element values as an nalgebra matrix.
    pub fn value_matrix(&self) -> nalgebra::DMatrix<S> {
        nalgebra::DMatrix::from_row_slice(self.rows, self.cols, &self.values())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |row, col| self[(col, row)])
    }

    /// Sum of the elements.
    pub fn sum(&self) -> Variable<'p, S> {
        self.data.iter().sum()
    }

    /// Sum of the elementwise products, i.e., the dot product for vectors.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    pub fn dot(&self, rhs: &Self) -> Variable<'p, S> {
        self.component_mul(rhs).sum()
    }

    /// Applies `f` to each element.
    pub fn map(&self, mut f: impl FnMut(Variable<'p, S>) -> Variable<'p, S>) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Applies `f` to each pair of corresponding elements.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    pub fn zip_map(
        &self,
        rhs: &Self,
        mut f: impl FnMut(Variable<'p, S>, Variable<'p, S>) -> Variable<'p, S>,
    ) -> Self {
        assert!(
            (self.rows, self.cols) == (rhs.rows, rhs.cols),
            "matrix shape mismatch: {}×{} vs {}×{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    /// Elementwise product.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    pub fn component_mul(&self, rhs: &Self) -> Self {
        self.zip_map(rhs, |a, b| a * b)
    }

    /// Elementwise quotient.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    pub fn component_div(&self, rhs: &Self) -> Self {
        self.zip_map(rhs, |a, b| a / b)
    }
}

impl<'a, 'p, S: Scalar> IntoIterator for &'a VariableMatrix<'p, S> {
    type Item = &'a Variable<'p, S>;
    type IntoIter = std::slice::Iter<'a, Variable<'p, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'p, S: Scalar> IntoIterator for VariableMatrix<'p, S> {
    type Item = Variable<'p, S>;
    type IntoIter = std::vec::IntoIter<Variable<'p, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<S, R, C, St> From<&nalgebra::Matrix<S, R, C, St>> for VariableMatrix<'_, S>
where
    S: Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<S, R, C>,
{
    fn from(matrix: &nalgebra::Matrix<S, R, C, St>) -> Self {
        Self::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
            matrix[(row, col)].into()
        })
    }
}

impl<'p, S: Scalar, T: Into<Variable<'p, S>>, const R: usize, const C: usize> From<[[T; C]; R]>
    for VariableMatrix<'p, S>
{
    /// Builds a matrix from rows, e.g., `[[1.0, 2.0], [3.0, 4.0]]`.
    fn from(rows: [[T; C]; R]) -> Self {
        Self {
            rows: R,
            cols: C,
            data: rows.into_iter().flatten().map(Into::into).collect(),
        }
    }
}

impl<'p, S: Scalar> From<Variable<'p, S>> for VariableMatrix<'p, S> {
    /// Builds a 1×1 matrix.
    fn from(x: Variable<'p, S>) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![x],
        }
    }
}

impl<'p, S: Scalar, T: Into<Variable<'p, S>>> FromIterator<T> for VariableMatrix<'p, S> {
    /// Builds a column vector.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<_> = iter.into_iter().map(Into::into).collect();
        Self {
            rows: data.len(),
            cols: 1,
            data,
        }
    }
}

impl<'p, S: Scalar> Index<usize> for VariableMatrix<'p, S> {
    type Output = Variable<'p, S>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<S: Scalar> IndexMut<usize> for VariableMatrix<'_, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'p, S: Scalar> Index<(usize, usize)> for VariableMatrix<'p, S> {
    type Output = Variable<'p, S>;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        &self.data[row * self.cols + col]
    }
}

impl<S: Scalar> IndexMut<(usize, usize)> for VariableMatrix<'_, S> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        &mut self.data[row * self.cols + col]
    }
}

impl<'p, S: Scalar> Neg for &VariableMatrix<'p, S> {
    type Output = VariableMatrix<'p, S>;

    fn neg(self) -> Self::Output {
        VariableMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| -x).collect(),
        }
    }
}

impl<'p, S: Scalar> Neg for VariableMatrix<'p, S> {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}
