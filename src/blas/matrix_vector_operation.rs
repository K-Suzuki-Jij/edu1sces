use anyhow::{bail, Result};
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::blas::CsrMatrix;

/// y = (A + shift * I) x
/// A is symmetric and stored in CSR format with LOWER triangle only.
/// Diagonal entries may be missing (treated as zero).
///
/// Requirements:
/// - m is square
/// - x.len() == y.len() == n
/// - y_locals.len() == pool.current_num_threads()
/// - y_locals[t].len() == n for all t
///
/// This matches the C++ structure:
///  1) parallel over rows i: accumulate into thread-local buffers y_locals[tid]
///  2) parallel over k: y[k] = shift*x[k] + sum_t y_locals[t][k]; then reset y_locals[t][k]=0
pub fn csr_sym_lower_matvec(
    pool: &ThreadPool,
    m: &CsrMatrix,
    x: &[f64],
    shift: f64,
    y: &mut [f64],
    y_locals: &mut [Vec<f64>],
) -> Result<()> {
    if m.row_dim != m.col_dim {
        bail!("matrix must be square");
    }
    let n = m.row_dim;

    if x.len() != n {
        bail!("dimension mismatch: x.len() != n");
    }
    if y.len() != n {
        bail!("dimension mismatch: y.len() != n");
    }
    if m.rows.len() != n + 1 || m.rows[n] != m.nnz() {
        bail!("invalid CSR structure (rows)");
    }

    let num_threads = pool.current_num_threads();
    if y_locals.len() != num_threads {
        bail!("dimension mismatch: y_locals.len() != pool.current_num_threads()");
    }
    for t in 0..num_threads {
        if y_locals[t].len() != n {
            bail!("dimension mismatch: y_locals[t].len() != n");
        }
    }

    // Share thread-local buffers with parallel closures WITHOUT introducing a new wrapper type:
    // store raw pointers as usize (usize is Sync), then cast back inside unsafe blocks.
    let locals_ptrs: Vec<usize> = y_locals
        .iter_mut()
        .map(|v| v.as_mut_ptr() as usize)
        .collect();

    pool.install(|| -> Result<()> {
        // ---------- phase 1: main accumulate (parallel over rows i) ----------
        //
        // Each rayon worker uses its own tid and writes ONLY into y_locals[tid][*].
        // That matches the C++ Temp_V[thread_num][*] update pattern.
        (0..n).into_par_iter().try_for_each(|i| -> Result<()> {
            let tid = rayon::current_thread_index()
                .ok_or_else(|| anyhow::anyhow!("rayon thread index not available"))?;

            if tid >= num_threads {
                bail!("internal error: tid out of range");
            }

            let y_local_ptr = locals_ptrs[tid] as *mut f64;

            let xi = x[i];

            let start = m.rows[i];
            let end = m.rows[i + 1];

            for p in start..end {
                let j = m.cols[p];
                let a = m.vals[p];

                if j < i {
                    // lower off-diagonal contributes to both i and j
                    unsafe {
                        *y_local_ptr.add(i) += a * x[j];
                        *y_local_ptr.add(j) += a * xi;
                    }
                } else if j == i {
                    // diagonal (if present)
                    unsafe {
                        *y_local_ptr.add(i) += a * xi;
                    }
                } else {
                    bail!("upper element found in lower-only CSR");
                }
            }

            Ok(())
        })?;

        // ---------- phase 2: reduction + reset (parallel over k) ----------
        //
        // y[k] = shift*x[k] + sum_t y_locals[t][k]
        // and reset y_locals[t][k] = 0 in the same loop (no separate clear loop).
        y.par_iter_mut().enumerate().for_each(|(k, yk)| {
            let mut acc = shift * x[k];

            // Sum all thread-local contributions for this k, then reset them.
            for t in 0..num_threads {
                let ptr = locals_ptrs[t] as *mut f64;
                unsafe {
                    acc += *ptr.add(k);
                    *ptr.add(k) = 0.0;
                }
            }

            *yk = acc;
        });

        Ok(())
    })
}

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

    pool.install(|| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = 0.0;

            let row_start = m.rows[i];
            let row_end = m.rows[i + 1];
            for p in row_start..row_end {
                sum += m.vals[p] * x[m.cols[p]];
            }

            // shift * I part (square matrix ensured above)
            *yi = sum + shift * x[i];
        });
    });

    Ok(())
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

    /// Create a CSR matrix storing only the lower triangle of a symmetric dense matrix.
    fn symmetric_lower_to_csr(dense: &[Vec<f64>]) -> CsrMatrix {
        let n = dense.len();
        let mut rows = vec![0usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            for j in 0..=i {
                let val = dense[i][j];
                if val != 0.0 {
                    cols.push(j);
                    vals.push(val);
                }
            }
            rows.push(cols.len());
        }

        CsrMatrix {
            row_dim: n,
            col_dim: n,
            rows,
            cols,
            vals,
        }
    }

    #[test]
    fn test_csr_sym_lower_matvec_identity() {
        // Identity matrix (symmetric): y = (I + 0*I) x = x
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals).unwrap();

        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_with_shift() {
        // Zero matrix with shift: y = (0 + 2*I) x = 2*x
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 2.0, &mut y, &mut y_locals).unwrap();

        assert_eq!(y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_symmetric() {
        // Symmetric matrix:
        // A = [[1, 2, 3],
        //      [2, 4, 5],
        //      [3, 5, 6]]
        // Lower triangle stored:
        // row 0: (0,0)=1
        // row 1: (1,0)=2, (1,1)=4
        // row 2: (2,0)=3, (2,1)=5, (2,2)=6
        //
        // x = [1, 2, 3]
        // A*x = [1*1 + 2*2 + 3*3, 2*1 + 4*2 + 5*3, 3*1 + 5*2 + 6*3]
        //     = [1 + 4 + 9, 2 + 8 + 15, 3 + 10 + 18]
        //     = [14, 25, 31]
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 5.0],
            vec![3.0, 5.0, 6.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals).unwrap();

        assert_eq!(y, vec![14.0, 25.0, 31.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_with_shift_general() {
        // Same symmetric matrix with shift=1
        // y = A*x + 1*x = [14+1, 25+2, 31+3] = [15, 27, 34]
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 5.0],
            vec![3.0, 5.0, 6.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 1.0, &mut y, &mut y_locals).unwrap();

        assert_eq!(y, vec![15.0, 27.0, 34.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_diagonal_only() {
        // Diagonal matrix (no off-diagonal elements)
        // A = [[2, 0, 0],
        //      [0, 3, 0],
        //      [0, 0, 4]]
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals).unwrap();

        // y = [2*1, 3*2, 4*3] = [2, 6, 12]
        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_off_diagonal_only() {
        // Matrix with only off-diagonal elements (diagonal = 0)
        // A = [[0, 1, 2],
        //      [1, 0, 3],
        //      [2, 3, 0]]
        let pool = build_pool(2).unwrap();
        let n = 3;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ]);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals).unwrap();

        // y = [0*1 + 1*2 + 2*3, 1*1 + 0*2 + 3*3, 2*1 + 3*2 + 0*3]
        //   = [0 + 2 + 6, 1 + 0 + 9, 2 + 6 + 0]
        //   = [8, 10, 8]
        assert_eq!(y, vec![8.0, 10.0, 8.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_single_element() {
        // 1x1 matrix
        let pool = build_pool(2).unwrap();
        let n = 1;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[vec![5.0]]);
        let x = vec![3.0];
        let mut y = vec![0.0; n];

        csr_sym_lower_matvec(&pool, &m, &x, 2.0, &mut y, &mut y_locals).unwrap();

        // y = 5*3 + 2*3 = 15 + 6 = 21
        assert_eq!(y, vec![21.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_reusable_y_locals() {
        // Test that y_locals is properly reset after each call
        let pool = build_pool(2).unwrap();
        let n = 2;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[vec![1.0, 2.0], vec![2.0, 3.0]]);
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; n];

        // First call
        csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals).unwrap();
        assert_eq!(y, vec![3.0, 5.0]); // [1+2, 2+3]

        // Second call with same y_locals (should still work correctly)
        let x2 = vec![2.0, 3.0];
        csr_sym_lower_matvec(&pool, &m, &x2, 0.0, &mut y, &mut y_locals).unwrap();
        // A*x2 = [[1,2],[2,3]] * [2,3] = [1*2+2*3, 2*2+3*3] = [8, 13]
        assert_eq!(y, vec![8.0, 13.0]);
    }

    #[test]
    fn test_csr_sym_lower_matvec_dimension_mismatch_x() {
        let pool = build_pool(2).unwrap();
        let n = 2;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = vec![1.0, 2.0, 3.0]; // wrong size
        let mut y = vec![0.0; n];

        let result = csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_sym_lower_matvec_dimension_mismatch_y() {
        let pool = build_pool(2).unwrap();
        let n = 2;
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();

        let m = symmetric_lower_to_csr(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3]; // wrong size

        let result = csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_sym_lower_matvec_non_square_error() {
        let pool = build_pool(2).unwrap();
        let num_threads = pool.current_num_threads();
        let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; 2]).collect();

        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 2.0],
        };
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        let result = csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_sym_lower_matvec_y_locals_wrong_count() {
        let pool = build_pool(2).unwrap();
        let n = 2;
        // Intentionally wrong number of y_locals
        let mut y_locals: Vec<Vec<f64>> = vec![vec![0.0; n]]; // only 1, but need num_threads

        let m = symmetric_lower_to_csr(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; n];

        let result = csr_sym_lower_matvec(&pool, &m, &x, 0.0, &mut y, &mut y_locals);
        assert!(result.is_err());
    }
}
