use anyhow::{bail, Result};

/// Smallest eigenpair of symmetric tridiagonal matrix.
/// diag: d[0..n], offdiag: e[0..n-1]
pub fn lapack_dstev(
    d_in: &[f64],
    e_in: &[f64],
    n: usize,
    d_work: &mut [f64], // len >= n
    e_work: &mut [f64], // len >= n-1 (or 0 if n<2)
    z_work: &mut [f64], // len >= n*n
    work: &mut [f64],   // len >= 2*n
) -> Result<(f64, Vec<f64>)> {
    if n == 0 {
        bail!("n must be positive");
    }
    if d_in.len() < n || d_work.len() < n {
        bail!("d buffer too small");
    }
    if n >= 2 && (e_in.len() < n - 1 || e_work.len() < n - 1) {
        bail!("e buffer too small");
    }
    if z_work.len() < n * n {
        bail!("z_work.len() < n*n");
    }
    if work.len() < 2 * n {
        bail!("work.len() < 2*n");
    }

    // copy input → LAPACK work buffers
    d_work[..n].copy_from_slice(&d_in[..n]);
    if n >= 2 {
        e_work[..(n - 1)].copy_from_slice(&e_in[..(n - 1)]);
    }

    let ldz = n as i32;
    let mut info: i32 = 0;

    unsafe {
        lapack::dstev(b'V', n as i32, d_work, e_work, z_work, ldz, work, &mut info);
    }

    if info != 0 {
        bail!("dstev failed: info={}", info);
    }

    Ok((d_work[0], z_work[..n].to_vec()))
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

    fn assert_vec_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() < tol,
                "at index {}: expected {} to be close to {} (diff={})",
                i,
                ai,
                bi,
                (ai - bi).abs()
            );
        }
    }

    fn assert_normalized(v: &[f64], tol: f64) {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_close(norm, 1.0, tol);
    }

    #[test]
    fn test_dstev_1x1() {
        // 1x1 matrix: [5]
        // eigenvalue = 5, eigenvector = [1]
        let d_in = [5.0];
        let e_in: [f64; 0] = [];
        let n = 1;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; 1]; // not used but needs to exist
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_close(eigenvalue, 5.0, TOL);
        assert_eq!(eigenvector.len(), 1);
        assert_close(eigenvector[0].abs(), 1.0, TOL);
    }

    #[test]
    fn test_dstev_2x2_diagonal() {
        // Diagonal 2x2 matrix: [[2, 0], [0, 5]]
        // d = [2, 5], e = [0]
        // Smallest eigenvalue = 2, eigenvector = [1, 0]
        let d_in = [2.0, 5.0];
        let e_in = [0.0];
        let n = 2;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_close(eigenvalue, 2.0, TOL);
        assert_normalized(&eigenvector, TOL);
        // eigenvector should be [±1, 0]
        assert_close(eigenvector[0].abs(), 1.0, TOL);
        assert_close(eigenvector[1].abs(), 0.0, TOL);
    }

    #[test]
    fn test_dstev_2x2_symmetric() {
        // 2x2 symmetric tridiagonal: [[1, 2], [2, 1]]
        // d = [1, 1], e = [2]
        // Eigenvalues: 1 ± 2 = {-1, 3}
        // Smallest eigenvalue = -1
        // Eigenvector for -1: [1, -1]/sqrt(2)
        let d_in = [1.0, 1.0];
        let e_in = [2.0];
        let n = 2;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_close(eigenvalue, -1.0, TOL);
        assert_normalized(&eigenvector, TOL);
        // eigenvector components should have equal magnitude, opposite signs
        assert_close(eigenvector[0].abs(), eigenvector[1].abs(), TOL);
        assert!(eigenvector[0] * eigenvector[1] < 0.0);
    }

    #[test]
    fn test_dstev_3x3_identity() {
        // Identity matrix (tridiagonal form): d = [1, 1, 1], e = [0, 0]
        // All eigenvalues = 1
        let d_in = [1.0, 1.0, 1.0];
        let e_in = [0.0, 0.0];
        let n = 3;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_close(eigenvalue, 1.0, TOL);
        assert_normalized(&eigenvector, TOL);
    }

    #[test]
    fn test_dstev_3x3_general() {
        // 3x3 symmetric tridiagonal:
        // [[2, 1, 0],
        //  [1, 3, 1],
        //  [0, 1, 2]]
        // d = [2, 3, 2], e = [1, 1]
        // Eigenvalues can be computed analytically or verified numerically
        let d_in = [2.0, 3.0, 2.0];
        let e_in = [1.0, 1.0];
        let n = 3;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        // Verify eigenvector is normalized
        assert_normalized(&eigenvector, TOL);

        // Verify A*v = lambda*v by reconstructing the matrix-vector product
        // T*v where T is tridiagonal with d on diagonal, e on off-diagonal
        let mut av = vec![0.0; n];
        av[0] = d_in[0] * eigenvector[0] + e_in[0] * eigenvector[1];
        for i in 1..n - 1 {
            av[i] = e_in[i - 1] * eigenvector[i - 1]
                + d_in[i] * eigenvector[i]
                + e_in[i] * eigenvector[i + 1];
        }
        av[n - 1] = e_in[n - 2] * eigenvector[n - 2] + d_in[n - 1] * eigenvector[n - 1];

        // Check A*v = lambda*v
        let lambda_v: Vec<f64> = eigenvector.iter().map(|&x| eigenvalue * x).collect();
        assert_vec_close(&av, &lambda_v, TOL);

        // The smallest eigenvalue for this matrix is approximately 1.268
        assert!(eigenvalue > 1.0 && eigenvalue < 2.0);
    }

    #[test]
    fn test_dstev_negative_eigenvalue() {
        // Matrix with negative eigenvalue:
        // [[-1, 0.5], [0.5, -1]]
        // d = [-1, -1], e = [0.5]
        // Eigenvalues: -1 ± 0.5 = {-1.5, -0.5}
        let d_in = [-1.0, -1.0];
        let e_in = [0.5];
        let n = 2;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        let (eigenvalue, eigenvector) = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_close(eigenvalue, -1.5, TOL);
        assert_normalized(&eigenvector, TOL);
    }

    #[test]
    fn test_dstev_zero_error() {
        let d_in: [f64; 0] = [];
        let e_in: [f64; 0] = [];
        let n = 0;

        let mut d_work = vec![];
        let mut e_work = vec![];
        let mut z_work = vec![];
        let mut work = vec![];

        let result = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dstev_buffer_too_small_d() {
        let d_in = [1.0]; // only 1 element
        let e_in = [0.0];
        let n = 2; // but n = 2

        let mut d_work = vec![0.0; 1]; // too small
        let mut e_work = vec![0.0; 1];
        let mut z_work = vec![0.0; 4];
        let mut work = vec![0.0; 4];

        let result = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dstev_buffer_too_small_z() {
        let d_in = [1.0, 2.0];
        let e_in = [0.5];
        let n = 2;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; 3]; // too small, needs n*n = 4
        let mut work = vec![0.0; 2 * n];

        let result = lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dstev_preserves_input() {
        // Verify that input arrays are not modified
        let d_in = [2.0, 3.0, 2.0];
        let e_in = [1.0, 1.0];
        let d_in_copy = d_in;
        let e_in_copy = e_in;
        let n = 3;

        let mut d_work = vec![0.0; n];
        let mut e_work = vec![0.0; n - 1];
        let mut z_work = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];

        lapack_dstev(
            &d_in,
            &e_in,
            n,
            &mut d_work,
            &mut e_work,
            &mut z_work,
            &mut work,
        )
        .unwrap();

        assert_eq!(d_in, d_in_copy);
        assert_eq!(e_in, e_in_copy);
    }
}
