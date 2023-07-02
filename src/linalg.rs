use crate::{Array1, Array2, Float};

#[derive(Debug)]
pub enum LinearSolveError {
    NotDiagonallyDominant {
        row_index: usize,
        abs_diag: Float,
        sum_abs_row: Float,
    },
    MaxItersReached {
        error: Float,
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
    error_tolerance: Float,
) -> Result<Array1, LinearSolveError> {
    assert!(matrix.rows() == matrix.cols());
    assert!(matrix.rows() == x.shape()[0]);
    assert!(matrix.is_csr());

    let compute_error = |x: &Array1| {
        use ndarray_linalg::Norm;

        let rhs_norm = rhs.norm();
        if rhs_norm < 1e-14 {
            0.
        } else {
            (&matrix * x - &rhs).norm() / rhs_norm / (x.len() as Float).sqrt()
        }
    };

    let mut error = compute_error(&x);
    for _ in 0..max_iters {
        if error < error_tolerance {
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
            if diag.abs() + 1e-8 < sum_abs_row {
                return Err(LinearSolveError::NotDiagonallyDominant {
                    row_index,
                    abs_diag: diag.abs(),
                    sum_abs_row,
                });
            }
            let current_rhs = rhs[[row_index]];
            x[[row_index]] = (current_rhs - sigma) / diag;
        }

        error = compute_error(&x);
    }
    Err(LinearSolveError::MaxItersReached {
        error,
        iters: max_iters,
    })
}

pub fn solve_linear_system_direct(matrix: Array2, rhs: Array1) -> Result<Array1, LinearSolveError> {
    use ndarray_linalg::Solve;

    matrix
        .solve(&rhs)
        .map_err(|_| LinearSolveError::SingularMatrix)
}
