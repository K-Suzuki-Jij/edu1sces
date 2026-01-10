use anyhow::{bail, Result};
use rand::Rng;

use crate::blas::lapack_dstev;
use crate::blas::matrix_vector_operation;
use crate::blas::vector_operation;
use crate::blas::CsrMatrix;
use crate::utility::rayon_pool::build_pool;

#[derive(Debug, Clone)]
pub struct DiagParam {
    pub diag_acc: f64,
    pub diag_min_step: usize,
    pub diag_max_step: usize,
    pub calc_vec: bool,
}

/// Lanczos method (smallest eigenvalue / optionally eigenvector).
///
/// - `out_vec`: if `param.calc_vec == true`, it will be overwritten with the eigenvector.
/// - `out_val`: will be overwritten with the converged smallest eigenvalue.
/// - `Lanczos_Initial_Guess` is intentionally not supported here (always random start).
/// - `Output_Step_Number` is intentionally not supported here.
pub fn lanczos(
    m: &CsrMatrix,
    out_vec: &mut Vec<f64>,
    out_val: &mut f64,
    param: &DiagParam,
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

    let acc = param.diag_acc;
    let min_step = param.diag_min_step;
    let max_step = param.diag_max_step;

    if max_step <= min_step {
        bail!("Lanczos: max_step must be > min_step");
    }

    // C++: if dim <= 1000 then dense LAPACK (Dsyev). Not implemented here.
    if n <= 1000 {
        bail!("Lanczos: dense diagonalization (Dsyev) is not implemented");
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
    let mut temp_eig_vec: Vec<f64> = Vec::new();

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
    {
        let mut rng = rand::rng();
        for i in 0..n {
            v0[i] = rng.random_range(-1.0..=1.0);
        }
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
            // C++ Lapack_Dstev(step+1, Diag, Off_Diag, Temp_Eigen_Vec, Temp_Eigen_Val[step])
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

            if (temp_eig_val[step] - temp_eig_val[step - 1]).abs() < acc {
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
    if param.calc_vec {
        if out_vec.len() != n {
            out_vec.resize(n, 0.0);
        }

        // Reinitialize v0 random, out_vec = 0
        {
            let mut rng = rand::rng();
            for i in 0..n {
                v0[i] = rng.random_range(-1.0..=1.0);
                out_vec[i] = 0.0;
            }
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
