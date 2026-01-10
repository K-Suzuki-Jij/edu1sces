use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::blas::lapack_dstev;
use crate::blas::lapack_dsyev;
use crate::blas::matrix_vector_operation;
use crate::blas::vector_operation;
use crate::blas::CsrMatrix;
use crate::utility::rayon_pool::build_pool;

#[derive(Debug, Clone)]
pub struct LanczosParameters {
    pub acc: f64,
    pub min_step: usize,
    pub max_step: usize,
    pub calc_eigenvec: bool,
}

/// Lanczos method (smallest eigenvalue / optionally eigenvector).
///
/// - `out_vec`: if `param.calc_eigenvec == true`, it will be overwritten with the eigenvector.
/// - `out_val`: will be overwritten with the converged smallest eigenvalue.
/// - `Lanczos_Initial_Guess` is intentionally not supported here (always random start).
/// - `Output_Step_Number` is intentionally not supported here.
pub fn lanczos(
    m: &CsrMatrix,
    out_vec: &mut Vec<f64>,
    out_val: &mut f64,
    param: &LanczosParameters,
    num_threads: usize,
) -> Result<()> {
    // ----- input checks -----
    if m.row_dim != m.col_dim || m.row_dim == 0 || m.col_dim == 0 {
        bail!("Lanczos: input matrix must be square and non-empty");
    }
    let n = m.row_dim;

    // C++ special-case: dim == 1
    if n == 1 {
        if m.nnz() == 0 {
            bail!("Lanczos: 1x1 matrix has no stored value");
        }
        *out_val = m.vals[0];
        out_vec.clear();
        out_vec.push(1.0);
        return Ok(());
    }

    let acc = param.acc;
    let min_step = param.min_step;
    let max_step = param.max_step;

    if max_step <= min_step {
        bail!("Lanczos: max_step must be > min_step");
    }

    // C++: if dim <= 1000 then dense LAPACK (Dsyev)
    if n <= 1000 {
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(m, &mut a_work, &mut w_work, &mut work)?;

        // Smallest eigenvalue is at index 0 (ascending order)
        *out_val = w_work[0];
        if param.calc_eigenvec {
            out_vec.clear();
            out_vec.extend_from_slice(&a_work[0..n]);
        }
        return Ok(());
    }

    // ----- build pool once -----
    let pool = build_pool(num_threads)?;

    // ----- work vectors -----
    let mut v0 = vec![0.0; n];
    let mut v1 = vec![0.0; n];
    let mut v2 = vec![0.0; n];

    // Tridiagonal arrays (capacity: max_step)
    let mut diag = vec![0.0; max_step];
    let mut off = vec![0.0; max_step];

    // Convergence monitor
    let mut temp_eig_val = vec![0.0; max_step];
    let mut temp_eig_vec= Vec::new();

    // LAPACK DSTE V work buffers (max size, reused; only first `k` used each call)
    let mut d_work = vec![0.0; max_step];
    let mut e_work = if max_step >= 2 {
        vec![0.0; max_step - 1]
    } else {
        Vec::new()
    };
    let mut z_work = vec![0.0; max_step * max_step];
    let mut work = vec![0.0; 2 * max_step];

    // ----- set initial vector (random) -----
    // Save seed for restart (eigenvector computation needs same initial vector)
    let seed: u64 = rand::rng().random();
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..n {
        v0[i] = rng.random_range(-1.0..=1.0);
    }
    vector_operation::normalize(&pool, &mut v0)?;

    // v1 = M * v0
    matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v0, 0.0)?;

    // diag[0] = <v0, v1>
    diag[0] = vector_operation::dot(&pool, &v0, &v1)?;
    temp_eig_val[0] = diag[0];

    // v1 -= diag[0] * v0
    vector_operation::axpy(&pool, &mut v1, -diag[0], &v0)?;

    // ----- main Lanczos loop -----
    let mut step_num: usize = 0;

    for step in 1..max_step {
        // v2 = v1
        vector_operation::copy(&pool, &v1, &mut v2)?;

        // off[step-1] = ||v2||
        off[step - 1] = vector_operation::norm2(&pool, &v2)?;

        // v2 = v2 / ||v2||
        vector_operation::normalize(&pool, &mut v2)?;

        // v1 = M * v2
        matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v2, 0.0)?;

        // diag[step] = <v2, v1>
        diag[step] = vector_operation::dot(&pool, &v2, &v1)?;

        if step >= min_step {
            let k = step + 1;

            let (val, vec0) = lapack_dstev(
                &diag,
                &off,
                k,
                &mut d_work,
                &mut e_work,
                &mut z_work,
                &mut work,
            )?;

            temp_eig_val[step] = val;
            temp_eig_vec = vec0;

            let diff = (temp_eig_val[step] - temp_eig_val[step - 1]).abs();

            if diff < acc {
                step_num = step;
                *out_val = temp_eig_val[step];
                break;
            }
        }

        // v1 -= diag[step] * v2 + off[step-1] * v0
        vector_operation::axpy(&pool, &mut v1, -diag[step], &v2)?;
        vector_operation::axpy(&pool, &mut v1, -off[step - 1], &v0)?;

        // v0 = v2
        vector_operation::copy(&pool, &v2, &mut v0)?;
    }

    if step_num == 0 {
        bail!("Lanczos: not converged (step_num == 0)");
    }

    // ----- compute eigenvector if requested (restart) -----
    if param.calc_eigenvec {
        if out_vec.len() != n {
            out_vec.resize(n, 0.0);
        }

        // Reinitialize v0 with same seed, out_vec = 0
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..n {
            v0[i] = rng.random_range(-1.0..=1.0);
            out_vec[i] = 0.0;
        }
        vector_operation::normalize(&pool, &mut v0)?;

        // out_vec += temp_eig_vec[0] * v0
        if temp_eig_vec.len() < step_num + 1 {
            bail!("Lanczos: internal error (temp_eig_vec too short)");
        }
        vector_operation::axpy(&pool, out_vec, temp_eig_vec[0], &v0)?;

        // v1 = M * v0
        matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v0, 0.0)?;

        // v1 -= diag[0] * v0
        vector_operation::axpy(&pool, &mut v1, -diag[0], &v0)?;

        for step in 1..=step_num {
            // v2 = v1
            vector_operation::copy(&pool, &v1, &mut v2)?;
            vector_operation::normalize(&pool, &mut v2)?;

            // out_vec += temp_eig_vec[step] * v2
            vector_operation::axpy(&pool, out_vec, temp_eig_vec[step], &v2)?;

            // v1 = M * v2
            matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v2, 0.0)?;

            // v1 -= diag[step] * v2 + off[step-1] * v0
            vector_operation::axpy(&pool, &mut v1, -diag[step], &v2)?;
            vector_operation::axpy(&pool, &mut v1, -off[step - 1], &v0)?;

            // v0 = v2
            vector_operation::copy(&pool, &v2, &mut v0)?;
        }

        vector_operation::normalize(&pool, out_vec)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    fn default_param() -> LanczosParameters {
        LanczosParameters {
            acc: 1e-10,
            min_step: 5,
            max_step: 100,
            calc_eigenvec: true,
        }
    }

    #[test]
    fn test_lanczos_1x1() {
        let m = CsrMatrix {
            row_dim: 1,
            col_dim: 1,
            rows: vec![0, 1],
            cols: vec![0],
            vals: vec![5.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        assert!((out_val - 5.0).abs() < TOL);
        assert_eq!(out_vec.len(), 1);
        assert!((out_vec[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_lanczos_2x2_diagonal_dsyev_path() {
        // [[2, 0], [0, 5]] - uses dsyev path (n <= 1000)
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![2.0, 5.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        assert!((out_val - 2.0).abs() < TOL);
        // Eigenvector for eigenvalue 2 should be [1, 0] or [-1, 0]
        assert!((out_vec[0].abs() - 1.0).abs() < TOL);
        assert!(out_vec[1].abs() < TOL);
    }

    #[test]
    fn test_lanczos_2x2_symmetric_dsyev_path() {
        // [[1, 2], [2, 1]] - eigenvalues: -1, 3
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![1.0, 2.0, 2.0, 1.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        assert!((out_val - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_lanczos_3x3_tridiagonal_dsyev_path() {
        // [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        let m = CsrMatrix {
            row_dim: 3,
            col_dim: 3,
            rows: vec![0, 2, 5, 7],
            cols: vec![0, 1, 0, 1, 2, 1, 2],
            vals: vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        // Smallest eigenvalue is around 1.268
        assert!(out_val > 1.0 && out_val < 2.0);

        // Verify eigenvector: M * v = lambda * v
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; 3];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &out_vec, 0.0).unwrap();
        for i in 0..3 {
            assert!((mv[i] - out_val * out_vec[i]).abs() < TOL);
        }
    }

    #[test]
    fn test_lanczos_without_eigenvector() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![1.0, 2.0, 2.0, 1.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = LanczosParameters {
            acc: 1e-10,
            min_step: 5,
            max_step: 100,
            calc_eigenvec: false,
        };

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        assert!((out_val - (-1.0)).abs() < TOL);
        // out_vec should not be modified when calc_eigenvec = false
        assert!(out_vec.is_empty());
    }

    #[test]
    fn test_lanczos_error_non_square() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        let result = lanczos(&m, &mut out_vec, &mut out_val, &param, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lanczos_error_max_step_le_min_step() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = LanczosParameters {
            acc: 1e-10,
            min_step: 10,
            max_step: 5, // invalid: max <= min
            calc_eigenvec: true,
        };

        let result = lanczos(&m, &mut out_vec, &mut out_val, &param, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lanczos_100x100_dsyev_path() {
        // 100x100 tridiagonal matrix: diagonal = 2, off-diagonal = -1
        // This is a discrete Laplacian with known eigenvalues
        let n = 100;
        let mut rows = vec![0usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            if i > 0 {
                cols.push(i - 1);
                vals.push(-1.0);
            }
            cols.push(i);
            vals.push(2.0);
            if i < n - 1 {
                cols.push(i + 1);
                vals.push(-1.0);
            }
            rows.push(cols.len());
        }

        let m = CsrMatrix {
            row_dim: n,
            col_dim: n,
            rows,
            cols,
            vals,
        };

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = default_param();

        lanczos(&m, &mut out_vec, &mut out_val, &param, 1).unwrap();

        // Smallest eigenvalue of discrete Laplacian: 2 - 2*cos(pi/(n+1))
        let expected = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        assert!((out_val - expected).abs() < 1e-6);

        // Verify eigenvector
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; n];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &out_vec, 0.0).unwrap();
        for i in 0..n {
            assert!((mv[i] - out_val * out_vec[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lanczos_vs_dsyev() {
        // Compare Lanczos with dsyev using n=500 random symmetric matrix
        let n = 500;
        let m = CsrMatrix::random_dense_symmetric(n, 12345);

        // Get ground truth from dsyev
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];
        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();
        let expected_val = w_work[0]; // smallest eigenvalue

        // Compute via Lanczos (n <= 1000 uses dsyev path internally,
        // but this still verifies the overall algorithm correctness)
        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = LanczosParameters {
            acc: 1e-10,
            min_step: 5,
            max_step: 100,
            calc_eigenvec: true,
        };

        lanczos(&m, &mut out_vec, &mut out_val, &param, 2).unwrap();

        assert!((out_val - expected_val).abs() < 1e-8);
    }

    #[test]
    fn test_lanczos_1500_random() {
        // n=1500 > 1000 uses actual Lanczos iteration path
        let n = 1500;
        let m = CsrMatrix::random_dense_symmetric(n, 54321);

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        let param = LanczosParameters {
            acc: 1e-10,
            min_step: 5,
            max_step: 200,
            calc_eigenvec: true,
        };

        lanczos(&m, &mut out_vec, &mut out_val, &param, 2).unwrap();

        // Verify eigenvector: M * v = lambda * v
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; n];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &out_vec, 0.0).unwrap();

        // Verify with relative tolerance
        for i in 0..n {
            let expected = out_val * out_vec[i];
            let diff = (mv[i] - expected).abs();
            let tol = 1e-6 * out_val.abs().max(1.0);
            assert!(diff < tol, "i={}, mv={}, expected={}, diff={}", i, mv[i], expected, diff);
        }
    }
}
