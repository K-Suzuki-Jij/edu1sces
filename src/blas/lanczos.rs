use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::blas::lapack_dstev;
use crate::blas::lapack_dsyev;
use crate::blas::matrix_vector_operation;
use crate::blas::vector_operation;
use crate::blas::CsrMatrix;
use crate::utility::py_log::py_print_overwrite;
use crate::utility::rayon_pool::build_pool;

#[derive(Debug, Clone)]
pub struct LanczosParameters {
    pub acc: f64,
    pub min_step: usize,
    pub max_step: usize,
    pub calc_eigenvec: bool,
    /// If true, print progress to stderr (overwrites same line with \r)
    pub output_log: bool,
}

#[derive(Debug, Clone)]
#[pyo3::pyclass(get_all)]
pub struct LanczosLog {
    pub elapsed_secs: f64,
    pub step_num: usize,
}

/// Lanczos method (smallest eigenvalue / optionally eigenvector).
///
/// - `out_vec`: if `param.calc_eigenvec == true`, it will be overwritten with the eigenvector.
/// - `out_val`: will be overwritten with the converged smallest eigenvalue.
/// - `known_eigenvecs`: known eigenvectors to orthogonalize against. Empty for ground state.
/// - Returns `LanczosLog` containing elapsed time and step count.
pub fn lanczos(
    m: &CsrMatrix,
    out_vec: &mut Vec<f64>,
    out_val: &mut f64,
    param: &LanczosParameters,
    known_eigenvecs: &[&[f64]],
    num_threads: usize,
) -> Result<LanczosLog> {
    let start_time = std::time::Instant::now();
    // ----- input checks -----
    if m.row_dim != m.col_dim || m.row_dim == 0 || m.col_dim == 0 {
        bail!("Lanczos: input matrix must be square and non-empty");
    }
    let n = m.row_dim;

    // special-case: dim == 1
    if n == 1 {
        if m.nnz() == 0 {
            bail!("Lanczos: 1x1 matrix has no stored value");
        }
        *out_val = m.vals[0];
        out_vec.clear();
        out_vec.push(1.0);
        return Ok(LanczosLog {
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            step_num: 0,
        });
    }

    let acc = param.acc;
    let min_step = param.min_step;
    let max_step = param.max_step;

    if max_step <= min_step {
        bail!("Lanczos: max_step must be > min_step");
    }

    // if dim <= 1000 then dense LAPACK (Dsyev)
    if n <= 1000 {
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(m, &mut a_work, &mut w_work, &mut work)?;

        // Eigenvalue index: 0 for ground state, k for k-th excited state
        let eig_idx = known_eigenvecs.len();
        if eig_idx >= n {
            bail!(
                "Lanczos: requested eigenvalue index {} >= matrix dimension {}",
                eig_idx,
                n
            );
        }
        *out_val = w_work[eig_idx];
        if param.calc_eigenvec {
            out_vec.clear();
            out_vec.extend_from_slice(&a_work[eig_idx * n..(eig_idx + 1) * n]);
        }
        return Ok(LanczosLog {
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            step_num: 0,
        });
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
    let mut temp_eig_vec = Vec::new();

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
    let mut norm = 0.0;
    for i in 0..n {
        v0[i] = rng.random_range(-1.0..=1.0);
        norm += v0[i] * v0[i];
    }
    vector_operation::normalize(&pool, &mut v0, norm.sqrt())?;

    // Orthogonalize against known eigenvectors (for excited states)
    if !known_eigenvecs.is_empty() {
        vector_operation::orthogonalize(&pool, &mut v0, known_eigenvecs)?;
        let norm = vector_operation::norm2(&pool, &v0)?;
        vector_operation::normalize(&pool, &mut v0, norm)?;
    }

    // v1 = M * v0
    matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v0, 0.0)?;

    // diag[0] = <v0, v1>
    diag[0] = vector_operation::dot(&pool, &v0, &v1)?;
    temp_eig_val[0] = diag[0];

    // v1 -= diag[0] * v0
    vector_operation::axpy_inplace(&pool, &mut v1, -diag[0], &v0)?;

    // ----- main Lanczos loop -----
    let mut step_num: usize = 0;

    for step in 1..max_step {
        // v2 = v1
        vector_operation::copy(&pool, &v1, &mut v2)?;

        // off[step-1] = ||v2||, v2 = v2 / ||v2||
        off[step - 1] = vector_operation::norm2(&pool, &v2)?;
        vector_operation::normalize(&pool, &mut v2, off[step - 1])?;

        // Orthogonalize against known eigenvectors (for excited states)
        if !known_eigenvecs.is_empty() {
            vector_operation::orthogonalize(&pool, &mut v2, known_eigenvecs)?;
            let norm = vector_operation::norm2(&pool, &v2)?;
            vector_operation::normalize(&pool, &mut v2, norm)?;
        }

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

            if param.output_log {
                py_print_overwrite(&format!("Lanczos_Step[{}]={:.1e}", step, diff));
            }

            if diff < acc {
                step_num = step;
                *out_val = temp_eig_val[step];
                break;
            }
        }

        // v1 -= diag[step] * v2 + off[step-1] * v0
        vector_operation::axpbz_inplace(&pool, &mut v1, -diag[step], &v2, -off[step - 1], &v0)?;

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
        let mut norm = 0.0;
        for i in 0..n {
            v0[i] = rng.random_range(-1.0..=1.0);
            norm += v0[i] * v0[i];
            out_vec[i] = 0.0;
        }
        vector_operation::normalize(&pool, &mut v0, norm.sqrt())?;

        // Orthogonalize against known eigenvectors (for excited states)
        if !known_eigenvecs.is_empty() {
            vector_operation::orthogonalize(&pool, &mut v0, known_eigenvecs)?;
            let norm = vector_operation::norm2(&pool, &v0)?;
            vector_operation::normalize(&pool, &mut v0, norm)?;
        }

        // out_vec += temp_eig_vec[0] * v0
        if temp_eig_vec.len() < step_num + 1 {
            bail!("Lanczos: internal error (temp_eig_vec too short)");
        }
        vector_operation::axpy_inplace(&pool, out_vec, temp_eig_vec[0], &v0)?;

        // v1 = M * v0
        matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v0, 0.0)?;

        // v1 -= diag[0] * v0
        vector_operation::axpy_inplace(&pool, &mut v1, -diag[0], &v0)?;

        for step in 1..=step_num {
            if param.output_log {
                py_print_overwrite(&format!("Lanczos_Vec_Step:{}/{}", step, step_num));
            }

            // v2 = v1 / ||v1|| (copy + normalize in one pass)
            let norm_sq: f64 = pool.install(|| {
                v2.par_iter_mut()
                    .zip(v1.par_iter())
                    .map(|(dst, src)| {
                        *dst = *src;
                        (*src) * (*src)
                    })
                    .sum()
            });
            vector_operation::normalize(&pool, &mut v2, norm_sq.sqrt())?;

            // Orthogonalize against known eigenvectors (for excited states)
            if !known_eigenvecs.is_empty() {
                vector_operation::orthogonalize(&pool, &mut v2, known_eigenvecs)?;
                let norm = vector_operation::norm2(&pool, &v2)?;
                vector_operation::normalize(&pool, &mut v2, norm)?;
            }

            // out_vec += temp_eig_vec[step] * v2
            vector_operation::axpy_inplace(&pool, out_vec, temp_eig_vec[step], &v2)?;

            // v1 = M * v2
            matrix_vector_operation::csr_matvec(&pool, &mut v1, m, &v2, 0.0)?;

            // v1 -= diag[step] * v2 + off[step-1] * v0
            vector_operation::axpbz_inplace(&pool, &mut v1, -diag[step], &v2, -off[step - 1], &v0)?;

            // v0 = v2
            vector_operation::copy(&pool, &v2, &mut v0)?;
        }

        let norm = vector_operation::norm2(&pool, out_vec)?;
        vector_operation::normalize(&pool, out_vec, norm)?;
    }

    let elapsed_secs = start_time.elapsed().as_secs_f64();

    Ok(LanczosLog {
        elapsed_secs,
        step_num,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    fn default_param() -> LanczosParameters {
        LanczosParameters {
            acc: 1e-10,
            min_step: 5,
            max_step: 200,
            calc_eigenvec: true,
            output_log: false,
        }
    }

    #[test]
    fn test_lanczos_dsyev_path() {
        // n=500 uses dsyev path (n <= 1000)
        let n = 500;
        let m = CsrMatrix::random_dense_symmetric(n, 12345);

        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];
        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        lanczos(&m, &mut out_vec, &mut out_val, &default_param(), &[], 1).unwrap();
        assert!((out_val - w_work[0]).abs() < TOL);
    }

    #[test]
    fn test_lanczos_iteration_path() {
        // n=1500 > 1000 uses Lanczos iteration
        let n = 1500;
        let m = CsrMatrix::random_dense_symmetric(n, 54321);

        let mut out_vec = Vec::new();
        let mut out_val = 0.0;
        lanczos(&m, &mut out_vec, &mut out_val, &default_param(), &[], 2).unwrap();

        // Verify: M * v = lambda * v
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; n];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &out_vec, 0.0).unwrap();
        for i in 0..n {
            let diff = (mv[i] - out_val * out_vec[i]).abs();
            assert!(diff < 1e-6 * out_val.abs().max(1.0));
        }
    }

    #[test]
    fn test_lanczos_excited_dsyev_path() {
        // n=500 uses dsyev path
        let n = 500;
        let m = CsrMatrix::random_dense_symmetric(n, 11111);

        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];
        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        // Ground state
        let mut gs_vec = Vec::new();
        let mut gs_val = 0.0;
        lanczos(&m, &mut gs_vec, &mut gs_val, &default_param(), &[], 1).unwrap();
        assert!((gs_val - w_work[0]).abs() < TOL);

        // First excited state
        let known = [gs_vec.as_slice()];
        let mut ex1_vec = Vec::new();
        let mut ex1_val = 0.0;
        lanczos(&m, &mut ex1_vec, &mut ex1_val, &default_param(), &known, 1).unwrap();
        assert!((ex1_val - w_work[1]).abs() < TOL);
    }

    #[test]
    fn test_lanczos_excited_iteration_path() {
        // n=1500 > 1000 uses Lanczos iteration
        let n = 1500;
        let m = CsrMatrix::random_dense_symmetric(n, 22222);

        // Ground state
        let mut gs_vec = Vec::new();
        let mut gs_val = 0.0;
        lanczos(&m, &mut gs_vec, &mut gs_val, &default_param(), &[], 2).unwrap();

        // First excited state
        let known = [gs_vec.as_slice()];
        let mut ex1_vec = Vec::new();
        let mut ex1_val = 0.0;
        lanczos(&m, &mut ex1_vec, &mut ex1_val, &default_param(), &known, 2).unwrap();

        // ex1_val > gs_val
        assert!(ex1_val > gs_val + 1e-10);

        // Verify: M * v = lambda * v
        let pool = build_pool(1).unwrap();
        let mut mv = vec![0.0; n];
        matrix_vector_operation::csr_matvec(&pool, &mut mv, &m, &ex1_vec, 0.0).unwrap();
        for i in 0..n {
            let diff = (mv[i] - ex1_val * ex1_vec[i]).abs();
            assert!(diff < 1e-6 * ex1_val.abs().max(1.0));
        }

        // Orthogonality: <gs_vec, ex1_vec> ≈ 0
        let dot: f64 = gs_vec.iter().zip(ex1_vec.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-10);
    }
}
