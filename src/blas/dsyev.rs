use anyhow::{bail, Result};

use crate::blas::CsrMatrix;

/// Diagonalize symmetric matrix using LAPACK dsyev.
/// Returns all eigenvalues (ascending order) and eigenvectors.
///
/// - `m`: Input CSR matrix (must be square and symmetric)
/// - `a_work`: Work buffer for dense matrix (len >= n*n), overwritten with eigenvectors (column-major)
/// - `w_work`: Work buffer for eigenvalues (len >= n)
/// - `work`: LAPACK work buffer (len >= 3*n)
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i*n..(i+1)*n] is the i-th eigenvector.
pub fn lapack_dsyev(
    m: &CsrMatrix,
    a_work: &mut [f64],
    w_work: &mut [f64],
    work: &mut [f64],
) -> Result<()> {
    if m.row_dim != m.col_dim || m.row_dim == 0 {
        bail!("dsyev: input matrix must be square and non-empty");
    }
    let n = m.row_dim;

    if a_work.len() < n * n {
        bail!("dsyev: a_work too small");
    }
    if w_work.len() < n {
        bail!("dsyev: w_work too small");
    }
    if work.len() < 3 * n {
        bail!("dsyev: work too small");
    }

    // Initialize dense matrix to zero
    for i in 0..n * n {
        a_work[i] = 0.0;
    }

    // Convert CSR to dense (column-major for LAPACK)
    // Also symmetrize: A[i,j] = A[j,i] = M[i,j]
    for i in 0..n {
        for k in m.rows[i]..m.rows[i + 1] {
            let j = m.cols[k];
            let val = m.vals[k];
            // Column-major: A[i,j] = a_work[i + j*n]
            a_work[i + j * n] = val;
            a_work[j + i * n] = val;
        }
    }

    let mut info: i32 = 0;
    let lwork = work.len() as i32;

    unsafe {
        lapack::dsyev(
            b'V',        // compute eigenvalues and eigenvectors
            b'L',        // lower triangle stored
            n as i32,    // matrix order
            a_work,      // matrix A (overwritten with eigenvectors)
            n as i32,    // leading dimension
            w_work,      // eigenvalues output
            work,        // workspace
            lwork,       // workspace size
            &mut info,   // info
        );
    }

    if info != 0 {
        bail!("dsyev failed: info={}", info);
    }

    // a_work now contains eigenvectors (column-major)
    // w_work now contains eigenvalues in ascending order
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} to be close to {} (diff={})",
            a,
            b,
            (a - b).abs()
        );
    }

    fn assert_normalized(v: &[f64], tol: f64) {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_close(norm, 1.0, tol);
    }

    #[test]
    fn test_dsyev_1x1() {
        let m = CsrMatrix {
            row_dim: 1,
            col_dim: 1,
            rows: vec![0, 1],
            cols: vec![0],
            vals: vec![5.0],
        };

        let mut a_work = vec![0.0; 1];
        let mut w_work = vec![0.0; 1];
        let mut work = vec![0.0; 3];

        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        assert_close(w_work[0], 5.0, TOL);
        assert_close(a_work[0].abs(), 1.0, TOL);
    }

    #[test]
    fn test_dsyev_2x2_diagonal() {
        // [[2, 0], [0, 5]]
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![2.0, 5.0],
        };

        let n = 2;
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        // Eigenvalues in ascending order
        assert_close(w_work[0], 2.0, TOL);
        assert_close(w_work[1], 5.0, TOL);

        // First eigenvector (for eigenvalue 2)
        let v0 = &a_work[0..n];
        assert_normalized(v0, TOL);
        assert_close(v0[0].abs(), 1.0, TOL);
        assert_close(v0[1].abs(), 0.0, TOL);
    }

    #[test]
    fn test_dsyev_2x2_symmetric() {
        // [[1, 2], [2, 1]]
        // Eigenvalues: -1, 3
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![1.0, 2.0, 2.0, 1.0],
        };

        let n = 2;
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        assert_close(w_work[0], -1.0, TOL);
        assert_close(w_work[1], 3.0, TOL);

        let v0 = &a_work[0..n];
        assert_normalized(v0, TOL);
    }

    #[test]
    fn test_dsyev_all_eigenvectors() {
        // [[1, 2], [2, 1]]
        // Eigenvalues: -1, 3
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 4],
            cols: vec![0, 1, 0, 1],
            vals: vec![1.0, 2.0, 2.0, 1.0],
        };

        let n = 2;
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        // Both eigenvalues
        assert_close(w_work[0], -1.0, TOL);
        assert_close(w_work[1], 3.0, TOL);

        // Both eigenvectors normalized
        let v0 = &a_work[0..n];
        let v1 = &a_work[n..2 * n];
        assert_normalized(v0, TOL);
        assert_normalized(v1, TOL);

        // Eigenvectors orthogonal
        let dot: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        assert_close(dot, 0.0, TOL);
    }

    #[test]
    fn test_dsyev_3x3() {
        // [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        let m = CsrMatrix {
            row_dim: 3,
            col_dim: 3,
            rows: vec![0, 2, 5, 7],
            cols: vec![0, 1, 0, 1, 2, 1, 2],
            vals: vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        };

        let n = 3;
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];

        lapack_dsyev(&m, &mut a_work, &mut w_work, &mut work).unwrap();

        // All eigenvectors normalized
        for i in 0..n {
            let v = &a_work[i * n..(i + 1) * n];
            assert_normalized(v, TOL);
        }

        // Smallest eigenvalue should be around 1.268
        assert!(w_work[0] > 1.0 && w_work[0] < 2.0);
    }
}
