use crate::{Array1, Array2, Float};

#[derive(Debug)]
pub enum LinearSolver {
    GaussSeidel {
        max_iters: usize,
        abs_error_tol: Float,
        rel_error_tol: Float,
    },
    Direct,
}
impl LinearSolver {
    pub fn solve(
        &self,
        matrix: sprs::CsMatView<Float>,
        x: Array1,
        rhs: Array1,
    ) -> Result<Array1, LinearSolveError> {
        match self {
            LinearSolver::GaussSeidel {
                max_iters,
                rel_error_tol,
                abs_error_tol,
            } => solve_linear_system_gauss_seidel(
                matrix,
                x,
                rhs,
                *max_iters,
                *rel_error_tol,
                *abs_error_tol,
            ),
            LinearSolver::Direct => solve_linear_system_direct(matrix.to_dense(), rhs),
        }
    }
}

#[derive(Debug)]
pub enum LinearSolveError {
    NotDiagonallyDominant {
        row_index: usize,
        abs_diag: Float,
        sum_abs_row: Float,
    },
    MaxItersReached {
        rel_error: Option<Float>,
        abs_error: Float,
        rel_error_tol: Float,
        abs_error_tol: Float,
        iters: usize,
    },
    SingularMatrix,
}

/// Use the Gauss-Seidel method to solve a diagonally dominant linear system
///
/// Ripped off from `sprs`, which in turn was ripped off from:
/// https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method#Algorithm
pub fn solve_linear_system_gauss_seidel(
    matrix: sprs::CsMatView<Float>,
    mut x: Array1,
    rhs: Array1,
    max_iters: usize,
    rel_error_tol: Float,
    abs_error_tol: Float,
) -> Result<Array1, LinearSolveError> {
    assert!(matrix.rows() == matrix.cols());
    assert!(matrix.rows() == x.shape()[0]);
    assert!(matrix.is_csr());

    let compute_rel_and_abs_error = |x: &Array1| {
        use ndarray_linalg::Norm;

        let abs_error = (&matrix * x - &rhs).norm() / (x.len() as Float).sqrt();
        let rhs_norm = rhs.norm();
        (
            if rhs_norm < 1e-14 {
                None
            } else {
                Some(abs_error / rhs_norm)
            },
            abs_error,
        )
    };

    let (mut rel_error, mut abs_error) = compute_rel_and_abs_error(&x);
    for _ in 0..max_iters {
        if rel_error
            .map(|rel_error| rel_error < rel_error_tol)
            .unwrap_or(true)
            && abs_error < abs_error_tol
        {
            return Ok(x);
        }

        for (row_index, vec) in matrix.outer_iterator().enumerate() {
            let mut sigma = 0.;
            let mut sum_abs_row = 0.;
            let mut diag = None;
            for (col_index, &val) in vec.iter() {
                if row_index != col_index {
                    sigma += val * x[[col_index]];
                    sum_abs_row += val.abs();
                } else {
                    if diag.is_some() {
                        panic!("More than one diagonal in a row");
                    }
                    diag = Some(val);
                }
            }
            let diag = diag.expect("Gauss-Seidel requires a non-zero diagonal");
            if (diag.abs() - sum_abs_row) / diag.abs() < -1e-6 {
                return Err(LinearSolveError::NotDiagonallyDominant {
                    row_index,
                    abs_diag: diag.abs(),
                    sum_abs_row,
                });
            }
            let current_rhs = rhs[[row_index]];
            x[[row_index]] = (current_rhs - sigma) / diag;
        }

        (rel_error, abs_error) = compute_rel_and_abs_error(&x);
    }
    Err(LinearSolveError::MaxItersReached {
        rel_error,
        abs_error,
        rel_error_tol,
        abs_error_tol,
        iters: max_iters,
    })
}

pub fn solve_linear_system_direct(matrix: Array2, rhs: Array1) -> Result<Array1, LinearSolveError> {
    use ndarray_linalg::Solve;

    matrix
        .solve(&rhs)
        .map_err(|_| LinearSolveError::SingularMatrix)
}
