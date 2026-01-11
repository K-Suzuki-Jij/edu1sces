use anyhow::{bail, Result};

use crate::blas::conjugate_gradient::{
    conjugate_gradient, ConjugateGradientLog, ConjugateGradientParameters,
};
use crate::blas::matrix_vector_operation;
use crate::blas::vector_operation;
use crate::blas::CsrMatrix;
use crate::utility::rayon_pool::build_pool;

/// Parameters for the inverse iteration solver.
#[derive(Debug, Clone)]
pub struct InverseIterationParameters {
    /// Diagonal shift to ensure positive definiteness
    pub diag_add: f64,
    /// Convergence threshold for eigenvector residual
    pub eigenvec_tol: f64,
    /// Maximum inverse iteration steps
    pub max_step: usize,
    /// Parameters for the inner CG solver
    pub cg_params: ConjugateGradientParameters,
}

#[derive(Debug, Clone)]
pub struct InverseIterationLog {
    pub elapsed_secs: f64,
    pub initial_residual_error: f64,
    pub residual_errors: Vec<f64>,
    pub cg_logs: Vec<ConjugateGradientLog>,
}

/// Inverse iteration method to refine an eigenvector.
///
/// Given an approximate eigenpair (eigen_val, eigen_vec), this method
/// improves the eigenvector by solving (M - λI + diag_add*I)^{-1} * v = v_new
/// repeatedly until convergence.
///
/// - `m`: The matrix (must be symmetric)
/// - `eigen_vec`: The eigenvector to refine (modified in place)
/// - `eigen_val`: The approximate eigenvalue
/// - `param`: Solver parameters
/// - `num_threads`: Number of threads for parallel computation
///
/// Returns `InverseIterationLog` containing elapsed time, initial residual, and per-step logs.
pub fn inverse_iteration(
    m: &CsrMatrix,
    eigen_vec: &mut [f64],
    eigen_val: f64,
    param: &InverseIterationParameters,
    num_threads: usize,
) -> Result<InverseIterationLog> {
    let start_time = std::time::Instant::now();
    // Check input matrix
    if m.row_dim != m.col_dim || m.row_dim == 0 {
        bail!(
            "Inverse iteration: input matrix must be square and non-empty (row={}, col={})",
            m.row_dim,
            m.col_dim
        );
    }

    let n = m.row_dim;

    if eigen_vec.len() != n {
        bail!(
            "Inverse iteration: dimension mismatch (eigen_vec={}, matrix={})",
            eigen_vec.len(),
            n
        );
    }

    let pool = build_pool(num_threads)?;

    let diag_add = param.diag_add;
    let eigenvec_tol = param.eigenvec_tol;
    let max_step = param.max_step;

    // shift = diag_add - eigen_val
    // We solve (M + shift*I) * x_new = x_old
    let shift = diag_add - eigen_val;

    let mut improved_eigen_vec = eigen_vec.to_vec();

    let mut residual_errors = Vec::new();
    let mut cg_logs = Vec::new();

    // Compute initial residual error before any optimization
    let initial_residual_error =
        matrix_vector_operation::eigenpair_residual_norm1(&pool, m, eigen_vec, eigen_val)?;

    if initial_residual_error < eigenvec_tol {
        return Ok(InverseIterationLog {
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            initial_residual_error,
            residual_errors,
            cg_logs,
        });
    }

    for step in 0..max_step {
        // Solve (M + shift*I) * improved_eigen_vec = eigen_vec
        let cg_log = conjugate_gradient(
            m,
            eigen_vec,
            &mut improved_eigen_vec,
            shift,
            &param.cg_params,
            num_threads,
        )?;

        cg_logs.push(cg_log);

        // Normalize
        let norm = vector_operation::norm2(&pool, &improved_eigen_vec)?;
        vector_operation::normalize(&pool, &mut improved_eigen_vec, norm)?;

        // Copy improved_eigen_vec to eigen_vec
        vector_operation::copy(&pool, &improved_eigen_vec, eigen_vec)?;

        // Compute residual error after this step
        let residual_error =
            matrix_vector_operation::eigenpair_residual_norm1(&pool, m, eigen_vec, eigen_val)?;
        residual_errors.push(residual_error);

        if residual_error < eigenvec_tol {
            break;
        }

        if step == max_step - 1 {
            eprintln!(
                "Warning: Inverse iteration not converged after {} iterations, error={}",
                max_step, residual_error
            );
        }
    }

    Ok(InverseIterationLog {
        elapsed_secs: start_time.elapsed().as_secs_f64(),
        initial_residual_error,
        residual_errors,
        cg_logs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_inverse_iteration_diagonal() {
        // Diagonal matrix: diag(1, 2, 3)
        // Eigenvalues: 1, 2, 3
        // Eigenvector for eigenvalue 1: [1, 0, 0]
        let m = CsrMatrix {
            row_dim: 3,
            col_dim: 3,
            rows: vec![0, 1, 2, 3],
            cols: vec![0, 1, 2],
            vals: vec![1.0, 2.0, 3.0],
        };

        // Start with approximate eigenvector
        let mut eigen_vec = vec![0.9, 0.1, 0.1];
        let eigen_val = 1.0;

        let param = InverseIterationParameters {
            diag_add: 1e-07,
            eigenvec_tol: 1e-8,
            max_step: 100,
            cg_params: ConjugateGradientParameters {
                residual_tol: 1e-12,
                max_step: 100,
            },
        };

        let log = inverse_iteration(&m, &mut eigen_vec, eigen_val, &param, 1).unwrap();

        assert!(log.cg_logs.len() < 100);

        // Check eigenvector is close to [1, 0, 0] or [-1, 0, 0]
        assert!(eigen_vec[0].abs() > 1.0 - TOL);
        assert!(eigen_vec[1].abs() < TOL);
        assert!(eigen_vec[2].abs() < TOL);
    }

    #[test]
    fn test_inverse_iteration_symmetric_2x2() {
        // Symmetric matrix: [[2, 1], [1, 2]]
        // Eigenvalues: 1, 3
        // Eigenvector for eigenvalue 1: [1, -1] / sqrt(2)
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![2.0, 1.0, 1.0, 2.0],
        };

        // Start with approximate eigenvector (normalized)
        let mut eigen_vec = vec![0.8, -0.6];
        let norm = vector_operation::norm2(&build_pool(1).unwrap(), &eigen_vec).unwrap();
        vector_operation::normalize(&build_pool(1).unwrap(), &mut eigen_vec, norm).unwrap();
        let eigen_val = 1.0;

        let param = InverseIterationParameters {
            diag_add: 1e-07,
            eigenvec_tol: 1e-8,
            max_step: 100,
            cg_params: ConjugateGradientParameters {
                residual_tol: 1e-12,
                max_step: 100,
            },
        };

        let log = inverse_iteration(&m, &mut eigen_vec, eigen_val, &param, 1).unwrap();

        assert!(log.cg_logs.len() < 100);

        // Check M * v = lambda * v
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; 2];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &eigen_vec, 0.0).unwrap();

        for i in 0..2 {
            assert!(
                (mv[i] - eigen_val * eigen_vec[i]).abs() < TOL,
                "M*v[{}] = {}, expected {}",
                i,
                mv[i],
                eigen_val * eigen_vec[i]
            );
        }
    }

    #[test]
    fn test_inverse_iteration_error_non_square() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let mut eigen_vec = vec![1.0, 0.0];
        let param = InverseIterationParameters {
            diag_add: 1e-07,
            eigenvec_tol: 1e-8,
            max_step: 100,
            cg_params: ConjugateGradientParameters {
                residual_tol: 1e-12,
                max_step: 100,
            },
        };

        let result = inverse_iteration(&m, &mut eigen_vec, 1.0, &param, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_inverse_iteration_error_dimension_mismatch() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let mut eigen_vec = vec![1.0, 0.0, 0.0]; // wrong size
        let param = InverseIterationParameters {
            diag_add: 1e-07,
            eigenvec_tol: 1e-8,
            max_step: 100,
            cg_params: ConjugateGradientParameters {
                residual_tol: 1e-12,
                max_step: 100,
            },
        };

        let result = inverse_iteration(&m, &mut eigen_vec, 1.0, &param, 1);
        assert!(result.is_err());
    }
}
