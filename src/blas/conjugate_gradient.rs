use anyhow::{bail, Result};

use crate::blas::matrix_vector_operation;
use crate::blas::vector_operation;
use crate::blas::CsrMatrix;
use crate::utility::rayon_pool::build_pool;

/// Parameters for the conjugate gradient solver.
#[derive(Debug, Clone)]
pub struct ConjugateGradientParameters {
    /// Convergence threshold (residual squared)
    pub residual_tol: f64,
    /// Maximum iterations
    pub max_step: usize,
}

#[derive(Debug, Clone)]
pub struct ConjugateGradientLog {
    pub elapsed_secs: f64,
    pub step_num: usize,
}

/// Conjugate gradient method for solving (A + shift*I) * x = y.
///
/// - `a`: The coefficient matrix (must be symmetric positive definite after shift)
/// - `y`: The right-hand side vector
/// - `result_vec`: The solution vector x (initial guess on input, solution on output)
/// - `shift`: Scalar shift applied to diagonal (shift * I)
/// - `param`: Solver parameters
/// - `num_threads`: Number of threads for parallel computation
///
/// Returns `ConjugateGradientLog` containing elapsed time and step count.
pub fn conjugate_gradient(
    a: &CsrMatrix,
    y: &[f64],
    result_vec: &mut [f64],
    shift: f64,
    param: &ConjugateGradientParameters,
    num_threads: usize,
) -> Result<ConjugateGradientLog> {
    let start_time = std::time::Instant::now();
    // Check input matrix
    if a.row_dim != a.col_dim || a.row_dim == 0 {
        bail!(
            "Conjugate gradient: input matrix must be square and non-empty (row={}, col={})",
            a.row_dim,
            a.col_dim
        );
    }

    let n = a.row_dim;

    if y.len() != n || result_vec.len() != n {
        bail!(
            "Conjugate gradient: dimension mismatch (y={}, result_vec={}, matrix={})",
            y.len(),
            result_vec.len(),
            n
        );
    }

    let pool = build_pool(num_threads)?;

    // If y is zero vector, x = 0 is the solution
    let y_norm = vector_operation::norm2(&pool, y)?;
    if y_norm < 1e-30 {
        vector_operation::zero(&pool, result_vec)?;
        return Ok(ConjugateGradientLog {
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            step_num: 0,
        });
    }

    let residual_tol = param.residual_tol;
    let max_step = param.max_step;

    let mut r = vec![0.0; n]; // residual
    let mut p = vec![0.0; n]; // search direction
    let mut ap = vec![0.0; n]; // A * p

    // Normalize initial guess
    let x_norm = vector_operation::norm2(&pool, result_vec)?;
    vector_operation::normalize(&pool, result_vec, x_norm)?;

    // r = y - (A + shift*I) * result_vec
    // First compute r = (A + shift*I) * result_vec, then r = y + (-1)*r = y - r
    matrix_vector_operation::csr_matvec(&pool, &mut r, a, result_vec, shift)?;
    vector_operation::xpay(&pool, &mut r, -1.0, y)?;

    // p = r
    vector_operation::copy(&pool, &r, &mut p)?;

    let mut step_num = 0;

    for step in 1..max_step {
        // ap = (A + shift*I) * p
        matrix_vector_operation::csr_matvec(&pool, &mut ap, a, &p, shift)?;

        // inner_prod = <r, r>
        let inner_prod = vector_operation::dot(&pool, &r, &r)?;

        // alpha = <r, r> / <p, ap>
        let p_ap = vector_operation::dot(&pool, &p, &ap)?;
        if p_ap.abs() < 1e-30 {
            bail!("Conjugate gradient: <p, A*p> is too small, matrix may not be positive definite");
        }
        let alpha = inner_prod / p_ap;

        // result_vec += alpha * p
        // r -= alpha * ap
        vector_operation::axpy_inplace(&pool, result_vec, alpha, &p)?;
        vector_operation::axpy_inplace(&pool, &mut r, -alpha, &ap)?;

        // residual_error = <r, r>
        let residual_error = vector_operation::dot(&pool, &r, &r)?;

        if residual_error < residual_tol {
            step_num = step;
            break;
        }

        // beta = <r_new, r_new> / <r_old, r_old>
        let beta = residual_error / inner_prod;

        // p = r + beta * p
        vector_operation::xpay(&pool, &mut p, beta, &r)?;
    }

    if step_num == 0 {
        eprintln!(
            "Warning: Conjugate gradient not converged after {} iterations",
            max_step
        );
    }

    Ok(ConjugateGradientLog {
        elapsed_secs: start_time.elapsed().as_secs_f64(),
        step_num,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_cg_identity() {
        // Identity matrix: I * x = y  =>  x = y
        let a = CsrMatrix {
            row_dim: 3,
            col_dim: 3,
            rows: vec![0, 1, 2, 3],
            cols: vec![0, 1, 2],
            vals: vec![1.0, 1.0, 1.0],
        };

        let y = vec![1.0, 2.0, 3.0];
        let mut result_vec = vec![0.1, 0.1, 0.1];

        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let log = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1).unwrap();

        assert!(log.step_num > 0);
        for i in 0..3 {
            assert!(
                (result_vec[i] - y[i]).abs() < TOL,
                "result_vec[{}] = {}, expected {}",
                i,
                result_vec[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_cg_diagonal() {
        // Diagonal matrix: diag(2, 3, 4) * x = [2, 6, 12]  =>  x = [1, 2, 3]
        let a = CsrMatrix {
            row_dim: 3,
            col_dim: 3,
            rows: vec![0, 1, 2, 3],
            cols: vec![0, 1, 2],
            vals: vec![2.0, 3.0, 4.0],
        };

        let y = vec![2.0, 6.0, 12.0];
        let mut result_vec = vec![0.5, 0.5, 0.5];

        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let log = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1).unwrap();

        assert!(log.step_num > 0);
        let expected = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert!(
                (result_vec[i] - expected[i]).abs() < TOL,
                "result_vec[{}] = {}, expected {}",
                i,
                result_vec[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_cg_symmetric_positive_definite() {
        // Symmetric positive definite matrix:
        // [[4, 1], [1, 3]] * x = [1, 2]
        // Solution: x = [1/11, 7/11] ≈ [0.0909, 0.6364]
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![4.0, 1.0, 1.0, 3.0],
        };

        let y = vec![1.0, 2.0];
        let mut result_vec = vec![1.0, 1.0]; // non-zero initial guess

        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let log = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1).unwrap();

        assert!(log.step_num > 0);

        // Verify A * x = y
        let pool = build_pool(1).unwrap();
        let mut result = vec![0.0; 2];
        matrix_vector_operation::csr_matvec(&pool, &mut result, &a, &result_vec, 0.0).unwrap();

        for i in 0..2 {
            assert!(
                (result[i] - y[i]).abs() < TOL,
                "A*x[{}] = {}, expected {}",
                i,
                result[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_cg_zero_rhs() {
        // A * x = 0  =>  x = 0
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![4.0, 1.0, 1.0, 3.0],
        };

        let y = vec![0.0, 0.0];
        let mut result_vec = vec![1.0, 2.0];

        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let log = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1).unwrap();

        assert_eq!(log.step_num, 0);
        assert!((result_vec[0]).abs() < TOL);
        assert!((result_vec[1]).abs() < TOL);
    }

    #[test]
    fn test_cg_error_non_square() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let y = vec![1.0, 2.0];
        let mut result_vec = vec![0.0, 0.0];
        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let result = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_cg_error_dimension_mismatch() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let y = vec![1.0, 2.0, 3.0]; // wrong size
        let mut result_vec = vec![0.0, 0.0];
        let param = ConjugateGradientParameters {
            residual_tol: 1e-12,
            max_step: 100,
        };

        let result = conjugate_gradient(&a, &y, &mut result_vec, 0.0, &param, 1);
        assert!(result.is_err());
    }
}
