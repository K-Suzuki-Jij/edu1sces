use anyhow::{bail, Result};
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::blas::CsrMatrix;

/// y = (A + shift * I) x
/// A is a general CSR matrix (not assumed symmetric).
///
/// # Arguments
/// * `pool` - The thread pool for parallel execution
/// * `y` - Output vector (length = row_dim)
/// * `m` - CSR matrix (must be square)
/// * `x` - Input vector (length = col_dim)
/// * `shift` - Scalar shift applied to diagonal (shift * I)
pub fn csr_matvec(
    pool: &ThreadPool,
    y: &mut [f64],
    m: &CsrMatrix,
    x: &[f64],
    shift: f64,
) -> Result<()> {
    if m.row_dim != m.col_dim {
        bail!("matrix must be square for (A + shift*I) x");
    }
    if x.len() != m.col_dim {
        bail!("dimension mismatch: x.len() != m.col_dim");
    }
    if y.len() != m.row_dim {
        bail!("dimension mismatch: y.len() != m.row_dim");
    }
    if m.rows.len() != m.row_dim + 1 || m.rows[m.row_dim] != m.nnz() {
        bail!("invalid CSR structure (rows)");
    }

    // Store pointers as usize (usize is Sync) to pass to parallel closure
    let rows_ptr = m.rows.as_ptr() as usize;
    let cols_ptr = m.cols.as_ptr() as usize;
    let vals_ptr = m.vals.as_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;

    pool.install(|| {
        // Use into_par_iter() with zip for dynamic scheduling
        // This allows Rayon to distribute work more dynamically
        (0..m.row_dim)
            .into_par_iter()
            .zip(y.par_iter_mut())
            .for_each(|(i, yi)| {
                unsafe {
                    let rows = rows_ptr as *const usize;
                    let cols = cols_ptr as *const usize;
                    let vals = vals_ptr as *const f64;
                    let x = x_ptr as *const f64;

                    let row_start = *rows.add(i);
                    let row_end = *rows.add(i + 1);

                    // Use 4-way unrolling with independent accumulators
                    // to reduce data dependency and improve ILP
                    let mut sum0 = 0.0;
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;

                    let len = row_end - row_start;
                    let unroll_end = row_start + (len / 4) * 4;

                    let mut p = row_start;
                    while p < unroll_end {
                        sum0 += *vals.add(p) * *x.add(*cols.add(p));
                        sum1 += *vals.add(p + 1) * *x.add(*cols.add(p + 1));
                        sum2 += *vals.add(p + 2) * *x.add(*cols.add(p + 2));
                        sum3 += *vals.add(p + 3) * *x.add(*cols.add(p + 3));
                        p += 4;
                    }

                    let mut sum = sum0 + sum1 + sum2 + sum3;
                    while p < row_end {
                        sum += *vals.add(p) * *x.add(*cols.add(p));
                        p += 1;
                    }

                    *yi = sum + shift * *x.add(i);
                }
            });
    });

    Ok(())
}

/// Compute L1 norm of eigenpair residual: ||A*x - lambda*x||_1
/// This is computed in a single pass for efficiency.
///
/// # Arguments
/// * `pool` - The thread pool for parallel execution
/// * `m` - CSR matrix (must be square)
/// * `x` - Eigenvector
/// * `lambda` - Eigenvalue
///
/// Returns the L1 norm of (A*x - lambda*x)
pub fn eigenpair_residual_norm1(
    pool: &ThreadPool,
    m: &CsrMatrix,
    x: &[f64],
    lambda: f64,
) -> Result<f64> {
    if m.row_dim != m.col_dim {
        bail!("matrix must be square for eigenpair residual");
    }
    if x.len() != m.col_dim {
        bail!("dimension mismatch: x.len() != m.col_dim");
    }
    if m.rows.len() != m.row_dim + 1 || m.rows[m.row_dim] != m.nnz() {
        bail!("invalid CSR structure (rows)");
    }

    let rows_ptr = m.rows.as_ptr() as usize;
    let cols_ptr = m.cols.as_ptr() as usize;
    let vals_ptr = m.vals.as_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;

    let norm = pool.install(|| {
        (0..m.row_dim)
            .into_par_iter()
            .map(|i| {
                unsafe {
                    let rows = rows_ptr as *const usize;
                    let cols = cols_ptr as *const usize;
                    let vals = vals_ptr as *const f64;
                    let x = x_ptr as *const f64;

                    let row_start = *rows.add(i);
                    let row_end = *rows.add(i + 1);

                    // Use 4-way unrolling with independent accumulators
                    // to reduce data dependency and improve ILP
                    let mut sum0 = 0.0;
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;

                    let len = row_end - row_start;
                    let unroll_end = row_start + (len / 4) * 4;

                    let mut p = row_start;
                    while p < unroll_end {
                        sum0 += *vals.add(p) * *x.add(*cols.add(p));
                        sum1 += *vals.add(p + 1) * *x.add(*cols.add(p + 1));
                        sum2 += *vals.add(p + 2) * *x.add(*cols.add(p + 2));
                        sum3 += *vals.add(p + 3) * *x.add(*cols.add(p + 3));
                        p += 4;
                    }

                    let mut sum = sum0 + sum1 + sum2 + sum3;
                    while p < row_end {
                        sum += *vals.add(p) * *x.add(*cols.add(p));
                        p += 1;
                    }

                    // |A*x[i] - lambda*x[i]|
                    (sum - lambda * *x.add(i)).abs()
                }
            })
            .sum::<f64>()
    });

    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::rayon_pool::build_pool;

    #[test]
    fn test_csr_matvec_identity() {
        // Identity matrix: y = (I + 0*I) x = x
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        csr_matvec(&pool, &mut y, &m, &x, 0.0).unwrap();

        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csr_matvec_with_shift() {
        // Zero matrix with shift: y = (0 + 2*I) x = 2*x
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        csr_matvec(&pool, &mut y, &m, &x, 2.0).unwrap();

        assert_eq!(y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_csr_matvec_general() {
        // General matrix:
        // A = [[1, 2, 0],
        //      [0, 3, 4],
        //      [5, 0, 6]]
        // x = [1, 2, 3]
        // A*x = [1*1 + 2*2, 3*2 + 4*3, 5*1 + 6*3] = [5, 18, 23]
        // With shift=1: y = A*x + 1*x = [5+1, 18+2, 23+3] = [6, 20, 26]
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[
            vec![1.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0],
            vec![5.0, 0.0, 6.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        csr_matvec(&pool, &mut y, &m, &x, 1.0).unwrap();

        assert_eq!(y, vec![6.0, 20.0, 26.0]);
    }

    #[test]
    fn test_csr_matvec_no_shift() {
        // Same matrix, no shift
        // A*x = [5, 18, 23]
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[
            vec![1.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0],
            vec![5.0, 0.0, 6.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        csr_matvec(&pool, &mut y, &m, &x, 0.0).unwrap();

        assert_eq!(y, vec![5.0, 18.0, 23.0]);
    }

    #[test]
    fn test_csr_matvec_single_element() {
        // 1x1 matrix
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[vec![5.0]]);
        let x = vec![3.0];
        let mut y = vec![0.0; 1];

        csr_matvec(&pool, &mut y, &m, &x, 2.0).unwrap();

        // y = 5*3 + 2*3 = 15 + 6 = 21
        assert_eq!(y, vec![21.0]);
    }

    #[test]
    fn test_csr_matvec_dimension_mismatch_x() {
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = vec![1.0, 2.0, 3.0]; // wrong size
        let mut y = vec![0.0; 2];

        let result = csr_matvec(&pool, &mut y, &m, &x, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_matvec_dimension_mismatch_y() {
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix::from_dense(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3]; // wrong size

        let result = csr_matvec(&pool, &mut y, &m, &x, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_matvec_non_square_error() {
        // Non-square matrix should fail
        let pool = build_pool(2).unwrap();
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 2, 3],
            cols: vec![0, 1, 2],
            vals: vec![1.0, 2.0, 3.0],
        };
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        let result = csr_matvec(&pool, &mut y, &m, &x, 0.0);
        assert!(result.is_err());
    }
}
