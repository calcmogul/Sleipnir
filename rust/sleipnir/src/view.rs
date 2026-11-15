// Copyright (c) Sleipnir contributors

//! Borrowed views into a [`VariableMatrix`].

use std::fmt;
use std::ops::{
    Bound, Index, IndexMut, Neg, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::{Scalar, Variable, VariableMatrix};

/// Strided selection of rows or columns, like `start..end` with a step.
///
/// A positive step selects every `step`th index from the start of the
/// range. A negative step selects every `|step|`th index from the end of the
/// range, in reverse.
///
/// ```
/// use sleipnir::{Slice, VariableMatrix};
///
/// let m = VariableMatrix::<f64>::from([[0.0, 1.0, 2.0, 3.0, 4.0]]);
/// assert_eq!(m.view(0, Slice::new(..).step(2)).values(), [0.0, 2.0, 4.0]);
/// assert_eq!(m.view(0, Slice::new(1..).step(-2)).values(), [4.0, 2.0]);
/// assert_eq!(m.view(0, Slice::new(..).step(-1)).values(), [4.0, 3.0, 2.0, 1.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    start: Bound<usize>,
    end: Bound<usize>,
    step: isize,
}

impl Slice {
    /// Selects every index in `range`.
    pub fn new(range: impl RangeBounds<usize>) -> Self {
        Self {
            start: range.start_bound().cloned(),
            end: range.end_bound().cloned(),
            step: 1,
        }
    }

    /// Sets the step.
    ///
    /// # Panics
    ///
    /// Panics if `step` is zero.
    #[track_caller]
    pub fn step(self, step: isize) -> Self {
        assert!(step != 0, "slice step can't be zero");
        Self { step, ..self }
    }
}

/// Indices of a dimension selected by a view: `first + i * step` for `i` in
/// `0..len`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    first: usize,
    len: usize,
    step: isize,
}

impl Selection {
    fn all(len: usize) -> Self {
        Self {
            first: 0,
            len,
            step: 1,
        }
    }

    fn index(&self, i: usize) -> usize {
        (self.first as isize + i as isize * self.step) as usize
    }

    /// Selects `inner`, which indexes this selection, from the underlying
    /// dimension.
    fn compose(&self, inner: Self) -> Self {
        Self {
            first: if inner.len == 0 {
                0
            } else {
                self.index(inner.first)
            },
            len: inner.len,
            step: self.step * inner.step,
        }
    }
}

/// Row or column selection for [`VariableMatrix::view`]: a `usize`, any
/// `std` range of `usize`, or a [`Slice`].
pub trait ViewRange {
    #[doc(hidden)]
    fn __select(self, dim: usize, axis: &str) -> Selection;
}

impl ViewRange for usize {
    #[track_caller]
    fn __select(self, dim: usize, axis: &str) -> Selection {
        assert!(self < dim, "{axis} {self} out of bounds for {dim} {axis}s");
        Selection {
            first: self,
            len: 1,
            step: 1,
        }
    }
}

impl ViewRange for Slice {
    #[track_caller]
    fn __select(self, dim: usize, axis: &str) -> Selection {
        let start = match self.start {
            Bound::Included(start) => start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end = match self.end {
            Bound::Included(end) => end + 1,
            Bound::Excluded(end) => end,
            Bound::Unbounded => dim,
        };
        assert!(
            start <= end && end <= dim,
            "{axis} range {start}..{end} out of bounds for {dim} {axis}s"
        );
        let len = (end - start).div_ceil(self.step.unsigned_abs());
        let first = if self.step > 0 || len == 0 {
            start
        } else {
            end - 1
        };
        Selection {
            first,
            len,
            step: self.step,
        }
    }
}

macro_rules! range_view_range {
    ($($Range:ty),*) => {$(
        impl ViewRange for $Range {
            #[track_caller]
            fn __select(self, dim: usize, axis: &str) -> Selection {
                Slice::new(self).__select(dim, axis)
            }
        }
    )*};
}

range_view_range!(
    Range<usize>,
    RangeInclusive<usize>,
    RangeFrom<usize>,
    RangeTo<usize>,
    RangeToInclusive<usize>,
    RangeFull
);

/// Borrowed view of selected rows and columns of a [`VariableMatrix`].
///
/// Views can be used anywhere a `VariableMatrix` operand can. Methods that
/// compute new matrices return owned [`VariableMatrix`]es.
#[derive(Clone, Copy)]
pub struct VariableBlock<'a, 'p, S: Scalar> {
    matrix: &'a VariableMatrix<'p, S>,
    rows: Selection,
    cols: Selection,
}

/// Mutable view of selected rows and columns of a [`VariableMatrix`].
pub struct VariableBlockMut<'a, 'p, S: Scalar> {
    matrix: &'a mut VariableMatrix<'p, S>,
    rows: Selection,
    cols: Selection,
}

impl<'p, S: Scalar> VariableMatrix<'p, S> {
    /// View of the selected rows and columns.
    ///
    /// ```
    /// use sleipnir::VariableMatrix;
    ///
    /// let m = VariableMatrix::<f64>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// assert_eq!(m.view(.., 1..).values(), [2.0, 3.0, 5.0, 6.0]);
    /// assert_eq!(m.view(1, ..=1).values(), [4.0, 5.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a selection is out of bounds.
    #[track_caller]
    pub fn view(&self, rows: impl ViewRange, cols: impl ViewRange) -> VariableBlock<'_, 'p, S> {
        VariableBlock {
            rows: rows.__select(self.rows(), "row"),
            cols: cols.__select(self.cols(), "column"),
            matrix: self,
        }
    }

    /// Mutable view of the selected rows and columns.
    ///
    /// # Panics
    ///
    /// Panics if a selection is out of bounds.
    #[track_caller]
    pub fn view_mut(
        &mut self,
        rows: impl ViewRange,
        cols: impl ViewRange,
    ) -> VariableBlockMut<'_, 'p, S> {
        VariableBlockMut {
            rows: rows.__select(self.rows(), "row"),
            cols: cols.__select(self.cols(), "column"),
            matrix: self,
        }
    }

    /// View of row `row`.
    #[track_caller]
    pub fn row(&self, row: usize) -> VariableBlock<'_, 'p, S> {
        self.view(row, ..)
    }

    /// Mutable view of row `row`.
    #[track_caller]
    pub fn row_mut(&mut self, row: usize) -> VariableBlockMut<'_, 'p, S> {
        self.view_mut(row, ..)
    }

    /// View of column `col`.
    #[track_caller]
    pub fn column(&self, col: usize) -> VariableBlock<'_, 'p, S> {
        self.view(.., col)
    }

    /// Mutable view of column `col`.
    #[track_caller]
    pub fn column_mut(&mut self, col: usize) -> VariableBlockMut<'_, 'p, S> {
        self.view_mut(.., col)
    }

    /// View of the whole matrix.
    pub fn as_view(&self) -> VariableBlock<'_, 'p, S> {
        VariableBlock {
            rows: Selection::all(self.rows()),
            cols: Selection::all(self.cols()),
            matrix: self,
        }
    }
}

impl<'a, 'p, S: Scalar> VariableBlock<'a, 'p, S> {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows.len
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols.len
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.rows.len * self.cols.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over elements in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &'a Variable<'p, S>> + use<'a, 'p, S> {
        let (matrix, rows, cols) = (self.matrix, self.rows, self.cols);
        (0..rows.len).flat_map(move |row| {
            (0..cols.len).map(move |col| &matrix[(rows.index(row), cols.index(col))])
        })
    }

    /// Returns the elements' values in row-major order.
    pub fn values(&self) -> Vec<S> {
        self.iter().map(Variable::value).collect()
    }

    /// Copies the view into a new matrix.
    pub fn to_matrix(&self) -> VariableMatrix<'p, S> {
        VariableMatrix::from_fn(self.rows(), self.cols(), |row, col| self[(row, col)])
    }

    /// Transpose.
    pub fn transpose(&self) -> VariableMatrix<'p, S> {
        VariableMatrix::from_fn(self.cols(), self.rows(), |row, col| self[(col, row)])
    }

    /// Sum of the elements.
    pub fn sum(&self) -> Variable<'p, S> {
        self.iter().sum()
    }

    /// Applies `f` to each element.
    pub fn map(
        &self,
        mut f: impl FnMut(Variable<'p, S>) -> Variable<'p, S>,
    ) -> VariableMatrix<'p, S> {
        VariableMatrix::from_fn(self.rows(), self.cols(), |row, col| f(self[(row, col)]))
    }

    /// View of the selected rows and columns of this view.
    ///
    /// # Panics
    ///
    /// Panics if a selection is out of bounds.
    #[track_caller]
    pub fn view(&self, rows: impl ViewRange, cols: impl ViewRange) -> VariableBlock<'a, 'p, S> {
        VariableBlock {
            matrix: self.matrix,
            rows: self.rows.compose(rows.__select(self.rows(), "row")),
            cols: self.cols.compose(cols.__select(self.cols(), "column")),
        }
    }

    /// View of row `row`.
    #[track_caller]
    pub fn row(&self, row: usize) -> VariableBlock<'a, 'p, S> {
        self.view(row, ..)
    }

    /// View of column `col`.
    #[track_caller]
    pub fn column(&self, col: usize) -> VariableBlock<'a, 'p, S> {
        self.view(.., col)
    }
}

impl<'p, S: Scalar> VariableBlockMut<'_, 'p, S> {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows.len
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols.len
    }

    /// Read-only view of the same elements.
    pub fn as_view(&self) -> VariableBlock<'_, 'p, S> {
        VariableBlock {
            matrix: self.matrix,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Returns the elements' values in row-major order.
    pub fn values(&self) -> Vec<S> {
        self.as_view().values()
    }

    /// Copies the view into a new matrix.
    pub fn to_matrix(&self) -> VariableMatrix<'p, S> {
        self.as_view().to_matrix()
    }

    /// Mutable view of the selected rows and columns of this view.
    ///
    /// # Panics
    ///
    /// Panics if a selection is out of bounds.
    #[track_caller]
    pub fn view_mut(
        &mut self,
        rows: impl ViewRange,
        cols: impl ViewRange,
    ) -> VariableBlockMut<'_, 'p, S> {
        VariableBlockMut {
            rows: self.rows.compose(rows.__select(self.rows(), "row")),
            cols: self.cols.compose(cols.__select(self.cols(), "column")),
            matrix: self.matrix,
        }
    }

    /// Overwrites the viewed elements with `values`, e.g., a matrix, view,
    /// nested array, or nalgebra matrix.
    ///
    /// ```
    /// use sleipnir::VariableMatrix;
    ///
    /// let mut m = VariableMatrix::<f64>::zeros(2, 3);
    /// m.column_mut(1).copy_from([[1.0], [2.0]]);
    /// assert_eq!(m.values(), [0.0, 1.0, 0.0, 0.0, 2.0, 0.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    pub fn copy_from(&mut self, values: impl Into<VariableMatrix<'p, S>>) {
        let values = values.into();
        assert!(
            (self.rows(), self.cols()) == (values.rows(), values.cols()),
            "matrix shape mismatch: {}×{} vs {}×{}",
            self.rows(),
            self.cols(),
            values.rows(),
            values.cols()
        );
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                self[(row, col)] = values[(row, col)];
            }
        }
    }

    /// Sets every viewed element to `value`.
    pub fn fill(&mut self, value: impl Into<Variable<'p, S>>) {
        let value = value.into();
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                self[(row, col)] = value;
            }
        }
    }
}

impl<'a, 'p, S: Scalar> Index<(usize, usize)> for VariableBlock<'a, 'p, S> {
    type Output = Variable<'p, S>;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(
            row < self.rows() && col < self.cols(),
            "index out of bounds"
        );
        &self.matrix[(self.rows.index(row), self.cols.index(col))]
    }
}

impl<'a, 'p, S: Scalar> Index<usize> for VariableBlock<'a, 'p, S> {
    type Output = Variable<'p, S>;

    /// Element `index` in row-major order.
    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len(), "index out of bounds");
        &self[(index / self.cols(), index % self.cols())]
    }
}

impl<'p, S: Scalar> Index<(usize, usize)> for VariableBlockMut<'_, 'p, S> {
    type Output = Variable<'p, S>;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(
            row < self.rows() && col < self.cols(),
            "index out of bounds"
        );
        &self.matrix[(self.rows.index(row), self.cols.index(col))]
    }
}

impl<S: Scalar> IndexMut<(usize, usize)> for VariableBlockMut<'_, '_, S> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(
            row < self.rows() && col < self.cols(),
            "index out of bounds"
        );
        let (row, col) = (self.rows.index(row), self.cols.index(col));
        &mut self.matrix[(row, col)]
    }
}

impl<'p, S: Scalar> Index<usize> for VariableBlockMut<'_, 'p, S> {
    type Output = Variable<'p, S>;

    /// Element `index` in row-major order.
    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows() * self.cols(), "index out of bounds");
        &self[(index / self.cols(), index % self.cols())]
    }
}

impl<S: Scalar> IndexMut<usize> for VariableBlockMut<'_, '_, S> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rows() * self.cols(), "index out of bounds");
        let cols = self.cols();
        &mut self[(index / cols, index % cols)]
    }
}

impl<'p, S: Scalar> From<VariableBlock<'_, 'p, S>> for VariableMatrix<'p, S> {
    fn from(view: VariableBlock<'_, 'p, S>) -> Self {
        view.to_matrix()
    }
}

impl<'p, S: Scalar> From<&VariableBlock<'_, 'p, S>> for VariableMatrix<'p, S> {
    fn from(view: &VariableBlock<'_, 'p, S>) -> Self {
        view.to_matrix()
    }
}

impl<'p, S: Scalar> From<&VariableMatrix<'p, S>> for VariableMatrix<'p, S> {
    fn from(matrix: &VariableMatrix<'p, S>) -> Self {
        matrix.clone()
    }
}

impl<'p, S: Scalar> Neg for VariableBlock<'_, 'p, S> {
    type Output = VariableMatrix<'p, S>;

    fn neg(self) -> VariableMatrix<'p, S> {
        self.map(|x| -x)
    }
}

impl<'p, S: Scalar> Neg for &VariableBlock<'_, 'p, S> {
    type Output = VariableMatrix<'p, S>;

    fn neg(self) -> VariableMatrix<'p, S> {
        self.map(|x| -x)
    }
}

impl<S: Scalar> fmt::Debug for VariableBlock<'_, '_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VariableBlock")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("elements", &self.iter().collect::<Vec<_>>())
            .finish()
    }
}

impl<S: Scalar> fmt::Debug for VariableBlockMut<'_, '_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VariableBlockMut")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("elements", &self.as_view().iter().collect::<Vec<_>>())
            .finish()
    }
}
