// Copyright (c) Sleipnir contributors

use crate::{Scalar, Variable, VariableMatrix};

impl<'p, S: Scalar> VariableMatrix<'p, S> {
    /// Solves AX = B for X, where A is `self`.
    ///
    /// Square matrices up to 4×4 use the closed-form inverse adj(A)/det(A).
    /// Other matrices use Householder QR, which gives the least-squares
    /// solution if A has more rows than columns. QR chooses each reflection's
    /// sign from the values when it's called, like Eigen's QR does with
    /// autodiff scalars, so the recorded expressions stay valid but can lose
    /// accuracy if the values later change sign.
    ///
    /// ```
    /// use sleipnir::VariableMatrix;
    ///
    /// let a = VariableMatrix::<f64>::from([[2.0, 1.0], [1.0, 3.0]]);
    /// let x = a.solve([[3.0], [5.0]]);
    /// for (x, expected) in x.values().into_iter().zip([0.8, 1.4]) {
    ///     assert!((x - expected).abs() < 1e-12);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if A and B have different row counts, or A has fewer rows than
    /// columns.
    #[track_caller]
    pub fn solve(&self, b: impl Into<VariableMatrix<'p, S>>) -> VariableMatrix<'p, S> {
        let b = b.into();
        let (m, n) = (self.rows(), self.cols());
        assert!(
            m == b.rows(),
            "solve needs A and B to have the same number of rows, got {m}×{n} and {}×{}",
            b.rows(),
            b.cols()
        );
        assert!(
            m >= n,
            "solve needs A to have at least as many rows as columns, got {m}×{n}"
        );

        match n {
            0 => VariableMatrix::zeros(0, b.cols()),
            1 if m == 1 => b.map(|x| x / self[0]),
            2..=4 if m == n => {
                let adj = adjugate(self);
                let det: Variable<'p, S> = (0..n).map(|j| self[(0, j)] * adj[(j, 0)]).sum();
                &(&adj / det) * &b
            }
            _ => householder_solve(self, &b),
        }
    }
}

type Rows<'p, S> = Vec<Vec<Variable<'p, S>>>;

fn to_rows<'p, S: Scalar>(m: &VariableMatrix<'p, S>) -> Rows<'p, S> {
    (0..m.rows())
        .map(|row| (0..m.cols()).map(|col| m[(row, col)]).collect())
        .collect()
}

/// Determinant by Laplace expansion along the first row.
fn determinant<'p, S: Scalar>(a: &Rows<'p, S>) -> Variable<'p, S> {
    match a.len() {
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        n => (0..n)
            .map(|j| {
                let term = a[0][j] * determinant(&minor(a, 0, j));
                if j % 2 == 0 { term } else { -term }
            })
            .sum(),
    }
}

/// Matrix without row `row` and column `col`.
fn minor<'p, S: Scalar>(a: &Rows<'p, S>, row: usize, col: usize) -> Rows<'p, S> {
    a.iter()
        .enumerate()
        .filter(|&(i, _)| i != row)
        .map(|(_, r)| {
            r.iter()
                .enumerate()
                .filter(|&(j, _)| j != col)
                .map(|(_, &x)| x)
                .collect()
        })
        .collect()
}

/// Adjugate (transposed cofactor matrix) of a square matrix.
fn adjugate<'p, S: Scalar>(a: &VariableMatrix<'p, S>) -> VariableMatrix<'p, S> {
    let rows = to_rows(a);
    VariableMatrix::from_fn(a.rows(), a.cols(), |row, col| {
        let cofactor = determinant(&minor(&rows, col, row));
        if (row + col) % 2 == 0 {
            cofactor
        } else {
            -cofactor
        }
    })
}

/// Solves AX = B in the least-squares sense with Householder QR.
fn householder_solve<'p, S: Scalar>(
    a: &VariableMatrix<'p, S>,
    b: &VariableMatrix<'p, S>,
) -> VariableMatrix<'p, S> {
    let (m, n, p) = (a.rows(), a.cols(), b.cols());
    let mut r = to_rows(a);
    let mut qtb = to_rows(b);

    for k in 0..n {
        // Reflect x = R[k.., k] onto αe₁ with v = x − αe₁, choosing the sign
        // of α opposite x₀ to avoid cancellation
        let norm = (k..m)
            .map(|i| r[i][k] * r[i][k])
            .sum::<Variable<'p, S>>()
            .sqrt();
        let x0 = r[k][k];
        let alpha = if x0.value() >= S::ZERO { -norm } else { norm };
        let mut v: Vec<_> = (k..m).map(|i| r[i][k]).collect();
        v[0] = x0 - alpha;
        let vtv: Variable<'p, S> = v.iter().map(|&vi| vi * vi).sum();
        if vtv.value() == S::ZERO {
            // The column below the diagonal is already zero
            continue;
        }

        // Apply H = I − 2vvᵀ/(vᵀv) to the remaining columns of R and to QᵀB
        let two = S::ONE + S::ONE;
        let reflect = |column: &mut [Variable<'p, S>]| {
            let dot: Variable<'p, S> = v.iter().zip(&*column).map(|(&vi, &xi)| vi * xi).sum();
            let scale = dot * two / vtv;
            for (x, &vi) in column.iter_mut().zip(&v) {
                *x -= scale * vi;
            }
        };
        for j in k + 1..n {
            let mut column: Vec<_> = (k..m).map(|i| r[i][j]).collect();
            reflect(&mut column);
            for (i, x) in (k..m).zip(column) {
                r[i][j] = x;
            }
        }
        for j in 0..p {
            let mut column: Vec<_> = (k..m).map(|i| qtb[i][j]).collect();
            reflect(&mut column);
            for (i, x) in (k..m).zip(column) {
                qtb[i][j] = x;
            }
        }
        r[k][k] = alpha;
        for row in r.iter_mut().skip(k + 1) {
            row[k] = S::ZERO.into();
        }
    }

    // Back-substitute RX = QᵀB
    let mut x: Rows<'p, S> = vec![vec![S::ZERO.into(); p]; n];
    for j in (0..n).rev() {
        for c in 0..p {
            let rhs = (j + 1..n).fold(qtb[j][c], |sum, l| sum - r[j][l] * x[l][c]);
            x[j][c] = rhs / r[j][j];
        }
    }
    VariableMatrix::from_fn(n, p, |row, col| x[row][col])
}
