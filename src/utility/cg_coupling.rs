//! Eigenvalue calculation for spin-spin interactions in coupled basis.

/// Eigenvalue of S_i · S_j in the ij-coupled basis.
///
/// λ(K) = (1/2)[K(K+1) - S_i(S_i+1) - S_j(S_j+1)]
///
/// # Arguments
/// * `two_si` - 2*S_i
/// * `two_sj` - 2*S_j
/// * `two_k` - 2*K (result of S_i ⊗ S_j coupling)
///
/// # Returns
/// The eigenvalue λ(K)
pub fn eigenvalue_si_sj(two_si: i32, two_sj: i32, two_k: i32) -> f64 {
    // S(S+1) = (2S/2)((2S+2)/2) = 2S(2S+2)/4 = (2S)(2S+2)/4
    let si_sq = (two_si * (two_si + 2)) as f64 / 4.0;
    let sj_sq = (two_sj * (two_sj + 2)) as f64 / 4.0;
    let k_sq = (two_k * (two_k + 2)) as f64 / 4.0;

    0.5 * (k_sq - si_sq - sj_sq)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_eigenvalue_si_sj() {
        // Two spin-1/2: S_i · S_j eigenvalues
        // K=0 (singlet): λ = (1/2)[0 - 3/4 - 3/4] = -3/4
        // K=1 (triplet): λ = (1/2)[2 - 3/4 - 3/4] = 1/4
        assert!((eigenvalue_si_sj(1, 1, 0) - (-0.75)).abs() < TOL);
        assert!((eigenvalue_si_sj(1, 1, 2) - 0.25).abs() < TOL);

        // Spin-1/2 and spin-1:
        // K=1/2: λ = (1/2)[3/4 - 3/4 - 2] = -1
        // K=3/2: λ = (1/2)[15/4 - 3/4 - 2] = 1/2
        assert!((eigenvalue_si_sj(1, 2, 1) - (-1.0)).abs() < TOL);
        assert!((eigenvalue_si_sj(1, 2, 3) - 0.5).abs() < TOL);
    }
}
