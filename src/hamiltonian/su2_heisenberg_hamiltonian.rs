//! Hamiltonian construction for SU(2) symmetric Heisenberg model.
//!
//! Builds the Hamiltonian matrix in the left-coupled SU(2) basis where
//! basis states are labeled by intermediate total spin quantum numbers.
//!
//! Uses 6j symbols for efficient computation of matrix elements, avoiding
//! explicit summation over magnetic quantum numbers.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::model::su2_heisenberg::SU2HeisenbergModel;
use crate::utility::cg_coupling::eigenvalue_si_sj;
use crate::utility::rayon_pool::with_pool;
use crate::utility::wigner::WignerSymbols;

/// Compute the matrix element ⟨bra| S_i · S_j |ket⟩ in the left-coupled basis.
///
/// Uses 6j symbols to compute the matrix element directly, without explicit
/// summation over magnetic quantum numbers.
///
/// The formula is based on Wigner-Eckart theorem and recoupling theory:
/// ⟨J'| S_i · S_j |J⟩ = Σ_K λ(K) × U(bra, K) × U(ket, K)
///
/// where K is the coupling result of S_i ⊗ S_j, λ(K) is the eigenvalue,
/// and U(state, K) is the transformation coefficient computed using 6j symbols.
///
/// IMPORTANT: Selection rules require that quantum numbers outside the [i, j] range
/// must be identical between bra and ket:
/// - J_k for k < i must match
/// - J_k for k > j must match
fn compute_si_sj_element(
    wigner: &WignerSymbols,
    two_s_list: &[i32],
    bra: &[u8],
    ket: &[u8],
    site_i: usize,
    site_j: usize,
) -> f64 {
    let n = two_s_list.len();

    // Selection rule: quantum numbers outside [i, j) range must match
    // J_k for k < i must be the same
    for k in 0..site_i {
        if bra[k] != ket[k] {
            return 0.0;
        }
    }
    // J_k for k >= j must be the same (including J_j itself!)
    // This is because the transformation to (S_i ⊗ S_j)_K preserves J_j = (J_{j-1} ⊗ S_j)
    // after recoupling (S_i and S_j couple to K, but J_j is preserved).
    for k in site_j..n {
        if bra[k] != ket[k] {
            return 0.0;
        }
    }

    let two_si = two_s_list[site_i];
    let two_sj = two_s_list[site_j];

    // K ranges from |S_i - S_j| to S_i + S_j
    let two_k_min = (two_si - two_sj).abs();
    let two_k_max = two_si + two_sj;

    let mut result = 0.0;

    let mut two_k = two_k_min;
    while two_k <= two_k_max {
        let eigenvalue = eigenvalue_si_sj(two_si, two_sj, two_k);

        // Compute transformation coefficients using 6j symbols
        let coeff_bra =
            compute_transformation_coeff_6j(wigner, two_s_list, bra, site_i, site_j, two_k);
        if coeff_bra.abs() < 1e-15 {
            two_k += 2;
            continue;
        }

        let coeff_ket =
            compute_transformation_coeff_6j(wigner, two_s_list, ket, site_i, site_j, two_k);
        if coeff_ket.abs() < 1e-15 {
            two_k += 2;
            continue;
        }

        result += eigenvalue * coeff_bra * coeff_ket;
        two_k += 2;
    }

    result
}

/// Compute the transformation coefficient using 6j symbols.
///
/// This computes the coefficient for transforming from the left-coupled basis
/// to a basis where S_i and S_j are coupled to K first.
///
/// The recoupling is done step by step using 6j symbols.
/// For sites i < j, we need to "move" S_j next to S_i in the coupling order.
///
/// In the left-coupled basis:
///   |..., J_{i-1}, J_i, J_{i+1}, ..., J_{j-1}, J_j, ...; S_total⟩
/// where J_k = (J_{k-1} ⊗ S_k)
///
/// We recouple to get:
///   |..., J_{i-1}, (S_i ⊗ S_j)_K, ...; S_total⟩
///
/// The transformation involves a product of 6j symbols for each intermediate step.
fn compute_transformation_coeff_6j(
    wigner: &WignerSymbols,
    two_s_list: &[i32],
    left_state: &[u8],
    site_i: usize,
    site_j: usize,
    two_k: i32,
) -> f64 {
    debug_assert!(site_i < site_j);
    let n = two_s_list.len();
    debug_assert!(site_j < n);

    // Special case: adjacent sites (i, i+1)
    //
    // In the left-coupled basis, the state is:
    //   |J_0, J_1, ..., J_{n-1}; M⟩
    // where J_k is the total spin after coupling sites 0, 1, ..., k.
    // J_0 = S_0 (trivially), and J_k = (J_{k-1} ⊗ S_k).
    //
    // For adjacent sites (i, i+1), we want to recouple:
    //   ((J_{i-1} ⊗ S_i)_{J_i} ⊗ S_{i+1})_{J_{i+1}} → (J_{i-1} ⊗ (S_i ⊗ S_{i+1})_K)_{J_{i+1}}
    //
    // Note: left_state[k] = 2*J_k, so:
    //   left_state[i-1] = 2*J_{i-1} (for i > 0)
    //   left_state[i] = 2*J_i
    //   left_state[i+1] = 2*J_{i+1}
    //
    // Coefficient = (-1)^{J_{i-1} + S_i + S_{i+1} + J_{i+1}} × sqrt((2J_i+1)(2K+1))
    //               × { J_{i-1}  S_i     J_i    }
    //                 { S_{i+1}  J_{i+1}  K     }
    if site_j == site_i + 1 {
        // Handle the case i = 0 specially
        if site_i == 0 {
            // For i=0, j=1:
            // The left-coupled state has J_1 = (S_0 ⊗ S_1), stored in left_state[1].
            // We want K = (S_0 ⊗ S_1), so the coefficient is δ_{J_1, K} = 1 if they match.
            let two_j_1 = left_state[1] as i32; // J_1 = (S_0 ⊗ S_1)
            return if two_j_1 == two_k { 1.0 } else { 0.0 };
        }

        // General case: i > 0
        let two_j_i_minus_1 = left_state[site_i - 1] as i32; // J_{i-1}
        let two_j_i = left_state[site_i] as i32; // J_i = (J_{i-1} ⊗ S_i)
        let two_j_i_plus_1 = left_state[site_i + 1] as i32; // J_{i+1} = (J_i ⊗ S_{i+1})
        let two_s_i = two_s_list[site_i];
        let two_s_i_plus_1 = two_s_list[site_i + 1];

        // Phase: (-1)^{J_{i-1} + S_i + S_{i+1} + J_{i+1}}
        let phase_exp = (two_j_i_minus_1 + two_s_i + two_s_i_plus_1 + two_j_i_plus_1) / 2;
        let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

        // Dimension factors: sqrt((2J_i + 1)(2K + 1))
        let dim_factor = (((two_j_i + 1) * (two_k + 1)) as f64).sqrt();

        // 6j symbol
        let sixj = wigner.wigner_6j(
            two_j_i_minus_1,
            two_s_i,
            two_j_i,
            two_s_i_plus_1,
            two_j_i_plus_1,
            two_k,
        );

        return phase * dim_factor * sixj;
    }

    // General case: non-adjacent sites
    // We need to perform a series of recouplings to move S_j next to S_i.
    //
    // Strategy: Sum over intermediate quantum numbers.
    // For each k from i+1 to j-1, we "pass" S_j through the coupling with S_k.
    //
    // This involves a product of 6j symbols and a sum over intermediate spins.
    compute_transformation_coeff_6j_general(wigner, two_s_list, left_state, site_i, site_j, two_k)
}

/// General transformation coefficient for non-adjacent sites using 6j symbols.
///
/// For non-adjacent sites i < j, we compute the transformation coefficient
/// that relates the left-coupled basis to the ij-coupled basis where S_i and S_j
/// are first coupled to K.
///
/// Uses a chain of 6j recouplings to "bubble" S_j towards S_i, tracking the
/// ordering of spins after each swap.
fn compute_transformation_coeff_6j_general(
    wigner: &WignerSymbols,
    two_s_list: &[i32],
    left_state: &[u8],
    site_i: usize,
    site_j: usize,
    two_k: i32,
) -> f64 {
    // Initialize spin_order: spin_order[k] = original site index at position k
    let n = two_s_list.len();
    let spin_order: Vec<usize> = (0..n).collect();

    compute_bubble_with_order(
        wigner,
        two_s_list,
        left_state,
        &spin_order,
        site_i,
        site_j, // current position of S_j
        two_k,
    )
}

/// Recursive function to compute transformation coefficient by bubbling S_j towards S_i.
///
/// Uses a series of 6j recouplings to "swap" S_j through intermediate spins until
/// it is adjacent to S_i.
///
/// The recoupling identity used is:
///   ((a ⊗ b)_c ⊗ d)_e = Σ_f (-1)^{b+c+d+f} √((2c+1)(2f+1)) {a b c; e d f} ((a ⊗ d)_f ⊗ b)_e
///
/// This swaps b and d in the coupling order.
fn compute_bubble_with_order(
    wigner: &WignerSymbols,
    two_s_list: &[i32],
    left_state: &[u8],
    spin_order: &[usize], // spin_order[k] = original site index at position k
    site_i: usize,
    current_pos: usize, // Current position of S_j in the coupling
    two_k: i32,
) -> f64 {
    // Get the original site indices for the spins at relevant positions
    let orig_site_at_i = spin_order[site_i];
    let orig_site_at_current = spin_order[current_pos];

    let two_s_i = two_s_list[orig_site_at_i];
    let two_s_j = two_s_list[orig_site_at_current]; // The spin we're moving

    // Base case: S_j is at position i+1 (adjacent to S_i)
    if current_pos == site_i + 1 {
        if site_i == 0 {
            // For i=0: check if the coupling (S_i ⊗ S_j) at position 1 equals K
            let two_j_1 = left_state[1] as i32;
            return if two_j_1 == two_k { 1.0 } else { 0.0 };
        }

        // For i > 0: use 6j symbol
        let two_j_i_minus_1 = left_state[site_i - 1] as i32;
        let two_j_i = left_state[site_i] as i32;
        let two_j_i_plus_1 = left_state[site_i + 1] as i32;

        let phase_exp = (two_j_i_minus_1 + two_s_i + two_s_j + two_j_i_plus_1) / 2;
        let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };
        let dim_factor = (((two_j_i + 1) * (two_k + 1)) as f64).sqrt();
        let sixj = wigner.wigner_6j(
            two_j_i_minus_1,
            two_s_i,
            two_j_i,
            two_s_j,
            two_j_i_plus_1,
            two_k,
        );

        return phase * dim_factor * sixj;
    }

    // Recursive case: swap S_j (at position p) with the spin at position p-1
    let p = current_pos;
    let orig_site_at_p_minus_1 = spin_order[p - 1];
    let two_s_p_minus_1 = two_s_list[orig_site_at_p_minus_1];

    let two_j_p = left_state[p] as i32;
    let two_j_p_minus_1 = left_state[p - 1] as i32;
    let two_j_p_minus_2 = if p >= 2 {
        left_state[p - 2] as i32
    } else {
        // p = 1: J_{-1} doesn't exist, use S at position 0
        two_s_list[spin_order[0]]
    };

    // Recoupling identity for swapping positions:
    // ((a ⊗ b)_c ⊗ d)_e = Σ_f (-1)^{b+c+d+f} √((2c+1)(2f+1)) {a b c; e d f} ((a ⊗ d)_f ⊗ b)_e
    //
    // Here: a=J_{p-2}, b=S_{p-1}, c=J_{p-1}, d=S_j, e=J_p
    // After swap: ((J_{p-2} ⊗ S_j)_f ⊗ S_{p-1})_{J_p}
    //
    // 6j symbol: {a b c; e d f} = {J_{p-2}, S_{p-1}, J_{p-1}; J_p, S_j, f}
    //
    // f must satisfy triangles: (a, d, f) = (J_{p-2}, S_j, f) and (b, e, f) = (S_{p-1}, J_p, f)
    let range1_min = (two_j_p_minus_2 - two_s_j).abs();
    let range1_max = two_j_p_minus_2 + two_s_j;
    let range2_min = (two_s_p_minus_1 - two_j_p).abs();
    let range2_max = two_s_p_minus_1 + two_j_p;

    let two_j_prime_min = range1_min.max(range2_min);
    let two_j_prime_max = range1_max.min(range2_max);

    // Parity check: f must have the same parity as both (a + d) and (b + e)
    let parity1 = (two_j_p_minus_2 + two_s_j) % 2;
    let parity2 = (two_s_p_minus_1 + two_j_p) % 2;
    if parity1 != parity2 {
        return 0.0;
    }

    let two_f_start = if two_j_prime_min % 2 == parity1 {
        two_j_prime_min
    } else {
        two_j_prime_min + 1
    };

    let mut total = 0.0;

    // Sum over all valid intermediate spin quantum numbers f
    let mut two_f = two_f_start;
    while two_f <= two_j_prime_max {
        // 6j coefficient for this swap: {a b c; e d f}
        // = {J_{p-2}, S_{p-1}, J_{p-1}; J_p, S_j, f}
        //
        // Phase: (-1)^{b + c + d + f} = (-1)^{S_{p-1} + J_{p-1} + S_j + f}
        let phase_exp = (two_s_p_minus_1 + two_j_p_minus_1 + two_s_j + two_f) / 2;
        let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };
        let dim_factor = (((two_j_p_minus_1 + 1) * (two_f + 1)) as f64).sqrt();
        let sixj = wigner.wigner_6j(
            two_j_p_minus_2,
            two_s_p_minus_1,
            two_j_p_minus_1,
            two_j_p,
            two_s_j,
            two_f,
        );

        if sixj.abs() < 1e-15 {
            two_f += 2;
            continue;
        }

        // Create modified state and spin order for recursion
        let mut modified_state = left_state.to_vec();
        modified_state[p - 1] = two_f as u8;

        let mut modified_order = spin_order.to_vec();
        modified_order.swap(p - 1, p);

        // Recurse: S_j is now at position p-1
        let recursive_coeff = compute_bubble_with_order(
            wigner,
            two_s_list,
            &modified_state,
            &modified_order,
            site_i,
            p - 1,
            two_k,
        );

        total += phase * dim_factor * sixj * recursive_coeff;
        two_f += 2;
    }

    total
}

/// Build the Hamiltonian matrix for SU(2) Heisenberg model.
///
/// # Arguments
/// * `model` - The SU(2) Heisenberg model
/// * `total_s` - The total spin quantum number S
/// * `num_threads` - Number of threads for parallel computation
///
/// # Returns
/// The Hamiltonian matrix in CSR format
pub fn make_su2_heisenberg_hamiltonian(
    model: &SU2HeisenbergModel,
    total_s: f64,
    num_threads: usize,
) -> Result<CsrMatrix> {
    let two_s_f = 2.0 * total_s;
    let two_s_total = two_s_f.round() as i32;

    if (two_s_f - (two_s_total as f64)).abs() > 1e-12 {
        bail!("total_s must be integer or half-integer (got {})", total_s);
    }

    let (basis, _inverse) = model.build_basis(total_s)?;
    let dim = basis.len();

    if dim == 0 {
        bail!(
            "Target Hilbert space has zero dimension for total_s = {}",
            total_s
        );
    }

    // Precompute max two_j for Wigner symbols
    let max_two_j: i32 = model.two_s_list.iter().sum::<i32>() + 10;
    let wigner = WignerSymbols::new(max_two_j as usize);

    // Convert two_s_list to i32 vec for convenience
    let two_s_list: Vec<i32> = model.two_s_list.clone();

    // Collect exchange interactions
    let interactions: Vec<((usize, usize), f64)> =
        model.exchange.iter().map(|(&k, &v)| (k, v)).collect();

    with_pool(num_threads, || -> Result<CsrMatrix> {
        // Pass 1: Count non-zeros per row (parallel)
        let row_nnz: Vec<usize> = basis
            .par_iter()
            .map(|bra| {
                let mut count = 0;
                for ket in &basis {
                    let mut val = 0.0;
                    for &((i, j), coeff) in &interactions {
                        let (site_i, site_j) = if i < j { (i, j) } else { (j, i) };
                        val += coeff
                            * compute_si_sj_element(&wigner, &two_s_list, bra, ket, site_i, site_j);
                    }
                    if val.abs() > MATRIX_ZERO_EPS {
                        count += 1;
                    }
                }
                count
            })
            .collect();

        // Prefix sum
        let mut out = CsrMatrix::new();
        out.row_dim = dim;
        out.col_dim = dim;
        out.rows = vec![0; dim + 1];
        for i in 0..dim {
            out.rows[i + 1] = out.rows[i] + row_nnz[i];
        }

        let nnz = out.rows[dim];
        out.cols = vec![0; nnz];
        out.vals = vec![0.0; nnz];

        // Prepare per-row slices
        let mut cols_rem = out.cols.as_mut_slice();
        let mut vals_rem = out.vals.as_mut_slice();

        let mut row_slices: Vec<(&mut [usize], &mut [f64])> = Vec::with_capacity(dim);
        for row in 0..dim {
            let len = row_nnz[row];
            let (c_head, c_tail) = cols_rem.split_at_mut(len);
            let (v_head, v_tail) = vals_rem.split_at_mut(len);
            row_slices.push((c_head, v_head));
            cols_rem = c_tail;
            vals_rem = v_tail;
        }

        // Pass 2: Fill values (parallel)
        row_slices
            .into_par_iter()
            .enumerate()
            .for_each(|(row, (row_cols, row_vals))| {
                let bra = &basis[row];
                let mut entries: Vec<(usize, f64)> = Vec::with_capacity(row_cols.len());

                for (col, ket) in basis.iter().enumerate() {
                    let mut val = 0.0;
                    for &((i, j), coeff) in &interactions {
                        let (site_i, site_j) = if i < j { (i, j) } else { (j, i) };
                        val += coeff
                            * compute_si_sj_element(&wigner, &two_s_list, bra, ket, site_i, site_j);
                    }
                    if val.abs() > MATRIX_ZERO_EPS {
                        entries.push((col, val));
                    }
                }

                entries.sort_unstable_by_key(|&(col, _)| col);

                for (k, (col, v)) in entries.into_iter().enumerate() {
                    row_cols[k] = col;
                    row_vals[k] = v;
                }
            });

        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_two_spin_half_singlet() {
        // Two spin-1/2 in singlet (S=0)
        // H = J * S_0 · S_1
        // In singlet sector, there's only one state.
        // ⟨singlet| S_0 · S_1 |singlet⟩ = -3/4

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            exchange,
        };

        let h = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();

        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        assert!(
            (h.vals[0] - (-0.75)).abs() < TOL,
            "Expected -0.75, got {}",
            h.vals[0]
        );
    }

    #[test]
    fn test_two_spin_half_triplet() {
        // Two spin-1/2 in triplet (S=1)
        // ⟨triplet| S_0 · S_1 |triplet⟩ = 1/4

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            exchange,
        };

        let h = make_su2_heisenberg_hamiltonian(&model, 1.0, 1).unwrap();

        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        assert!(
            (h.vals[0] - 0.25).abs() < TOL,
            "Expected 0.25, got {}",
            h.vals[0]
        );
    }

    #[test]
    fn test_three_spin_half_chain() {
        // Three spin-1/2 with J=1 for (0,1) and (1,2)
        // H = S_0·S_1 + S_1·S_2
        //
        // S=1/2 sector has 2 basis states:
        // |1⟩ = [1, 0, 1]: J_1=1/2, J_2=0, J_3=1/2 (sites 0,1 form singlet)
        // |2⟩ = [1, 2, 1]: J_1=1/2, J_2=1, J_3=1/2 (sites 0,1 form triplet)
        //
        // The Hamiltonian should be a 2x2 matrix.

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        exchange.insert((1, 2), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 1, 1],
            exchange,
        };

        let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();

        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert!(h.check().is_ok());
        assert!(
            h.is_symmetric(TOL).unwrap(),
            "Hamiltonian should be symmetric"
        );

        // Print matrix for debugging
        eprintln!("H matrix:");
        eprintln!("  nnz = {}", h.nnz());
        for row in 0..h.row_dim {
            for col in h.rows[row]..h.rows[row + 1] {
                eprintln!("  H[{},{}] = {}", row, h.cols[col], h.vals[col]);
            }
        }
    }

    #[test]
    fn test_three_spin_half_max_spin() {
        // Three spin-1/2 in S=3/2 sector
        // Only one state: [1, 2, 3]
        // H = S_0·S_1 + S_1·S_2
        // In this fully polarized state, all pairs are in triplet configuration
        // ⟨S=3/2| S_i·S_j |S=3/2⟩ = 1/4 for each pair
        // Total: 1/4 + 1/4 = 1/2

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        exchange.insert((1, 2), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 1, 1],
            exchange,
        };

        let h = make_su2_heisenberg_hamiltonian(&model, 1.5, 1).unwrap();

        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        assert!(
            (h.vals[0] - 0.5).abs() < TOL,
            "Expected 0.5, got {}",
            h.vals[0]
        );
    }

    #[test]
    fn test_si_sj_element_three_sites() {
        use crate::utility::wigner::WignerSymbols;

        let wigner = WignerSymbols::new(50);
        let two_s_list = vec![1i32, 1, 1];

        // Left-coupled state for S_total = 3/2: [1, 2, 3]
        let left_state = vec![1u8, 2, 3];

        // Compute ⟨S=3/2 | S_0·S_1 | S=3/2⟩
        // In the maximally polarized state, S_0 and S_1 are in the triplet (K=1)
        // So ⟨S_0·S_1⟩ = 1/4
        let val = compute_si_sj_element(&wigner, &two_s_list, &left_state, &left_state, 0, 1);
        assert!((val - 0.25).abs() < 1e-10, "Expected 0.25, got {}", val);
    }

    #[test]
    fn test_si_sj_element_non_adjacent() {
        use crate::utility::wigner::WignerSymbols;

        let wigner = WignerSymbols::new(50);
        let two_s_list = vec![1i32, 1, 1];

        // Left-coupled state for S_total = 3/2: [1, 2, 3]
        // J_0 = S_0 = 1/2, J_1 = 1, J_2 = 3/2
        let left_state = vec![1u8, 2, 3];

        // Compute ⟨S=3/2 | S_0·S_2 | S=3/2⟩ (non-adjacent sites 0 and 2)
        // In the maximally polarized state (S=3/2), all spins are aligned.
        // By symmetry, ⟨S_0·S_2⟩ should also be 1/4.
        let val = compute_si_sj_element(&wigner, &two_s_list, &left_state, &left_state, 0, 2);
        assert!((val - 0.25).abs() < 1e-10, "Expected 0.25, got {}", val);
    }

    #[test]
    fn test_four_site_hamiltonian() {
        // Four spin-1/2 with both adjacent and non-adjacent interactions
        // H = J * (S_0·S_1 + S_1·S_2 + S_2·S_3 + S_0·S_2)
        // Test that it builds correctly

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        exchange.insert((1, 2), 1.0);
        exchange.insert((2, 3), 1.0);
        exchange.insert((0, 2), 0.5); // Non-adjacent interaction

        let model = SU2HeisenbergModel {
            num_sites: 4,
            two_s_list: vec![1, 1, 1, 1],
            exchange,
        };

        // S=0 sector has 2 basis states
        let h = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert!(h.check().is_ok());
        assert!(
            h.is_symmetric(TOL).unwrap(),
            "Hamiltonian should be symmetric"
        );
    }

    #[test]
    fn test_si_sj_element_distant_sites() {
        // Test (0, 3) pair - three intermediate sites
        use crate::utility::wigner::WignerSymbols;

        let wigner = WignerSymbols::new(50);
        let two_s_list = vec![1i32, 1, 1, 1]; // Four spin-1/2

        // Left-coupled state for S_total = 2 (fully polarized):
        // J_0 = 1/2, J_1 = 1, J_2 = 3/2, J_3 = 2
        let left_state = vec![1u8, 2, 3, 4];

        // In the fully polarized state, all spins are aligned.
        // ⟨S=2 | S_0·S_3 | S=2⟩ = 1/4 (same as any pair)
        let val = compute_si_sj_element(&wigner, &two_s_list, &left_state, &left_state, 0, 3);
        assert!((val - 0.25).abs() < 1e-10, "Expected 0.25, got {}", val);

        // Also test (1, 3) pair
        let val_13 = compute_si_sj_element(&wigner, &two_s_list, &left_state, &left_state, 1, 3);
        assert!(
            (val_13 - 0.25).abs() < 1e-10,
            "Expected 0.25 for (1,3), got {}",
            val_13
        );
    }

    #[test]
    fn test_long_range_hamiltonian() {
        // Five spin-1/2 with long-range interaction (0, 4)
        // H = J * S_0·S_4

        let mut exchange = HashMap::new();
        exchange.insert((0, 4), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 5,
            two_s_list: vec![1, 1, 1, 1, 1],
            exchange,
        };

        // S=5/2 sector has 1 basis state (fully polarized)
        let h = make_su2_heisenberg_hamiltonian(&model, 2.5, 1).unwrap();
        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        // ⟨S=5/2 | S_0·S_4 | S=5/2⟩ = 1/4
        assert!(
            (h.vals[0] - 0.25).abs() < TOL,
            "Expected 0.25, got {}",
            h.vals[0]
        );
        assert!(h.check().is_ok());

        // S=1/2 sector should also work
        let h2 = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
        assert!(h2.check().is_ok());
        assert!(
            h2.is_symmetric(TOL).unwrap(),
            "Hamiltonian should be symmetric"
        );
    }

    /// Test: 2-site systems with various spin combinations
    /// Compare SU(2) Hamiltonian matrix elements with analytically known values.
    ///
    /// For H = S_1 · S_2, the eigenvalue in the total spin S sector is:
    ///   ⟨S|S_1·S_2|S⟩ = (S(S+1) - S_1(S_1+1) - S_2(S_2+1)) / 2
    ///
    /// This is the fundamental test for the SU(2) matrix element calculation.
    mod two_site_direct_comparison {
        use super::*;

        /// Two spin-1/2 particles (S_1 = S_2 = 1/2)
        /// S = 0 (singlet): ⟨S_1·S_2⟩ = (0 - 3/4 - 3/4)/2 = -3/4
        /// S = 1 (triplet): ⟨S_1·S_2⟩ = (2 - 3/4 - 3/4)/2 = +1/4
        #[test]
        fn test_two_spin_half() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 2,
                two_s_list: vec![1, 1], // 2*S = 1 for spin-1/2
                exchange,
            };

            // S = 0 sector (singlet)
            let h0 = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
            assert_eq!(h0.row_dim, 1, "S=0 sector should have dim=1");
            let expected_s0 = -0.75; // -3/4
            assert!(
                (h0.vals[0] - expected_s0).abs() < TOL,
                "S=0: expected {}, got {}",
                expected_s0,
                h0.vals[0]
            );

            // S = 1 sector (triplet)
            let h1 = make_su2_heisenberg_hamiltonian(&model, 1.0, 1).unwrap();
            assert_eq!(h1.row_dim, 1, "S=1 sector should have dim=1");
            let expected_s1 = 0.25; // +1/4
            assert!(
                (h1.vals[0] - expected_s1).abs() < TOL,
                "S=1: expected {}, got {}",
                expected_s1,
                h1.vals[0]
            );
        }

        /// Spin-1/2 and spin-1 (S_1 = 1/2, S_2 = 1)
        /// S = 1/2: ⟨S_1·S_2⟩ = (3/4 - 3/4 - 2)/2 = -1
        /// S = 3/2: ⟨S_1·S_2⟩ = (15/4 - 3/4 - 2)/2 = +1/2
        #[test]
        fn test_spin_half_and_spin_one() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 2,
                two_s_list: vec![1, 2], // 2*S_1 = 1 (spin-1/2), 2*S_2 = 2 (spin-1)
                exchange,
            };

            // S = 1/2 sector
            let h_half = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h_half.row_dim, 1, "S=1/2 sector should have dim=1");
            let expected_half = -1.0;
            assert!(
                (h_half.vals[0] - expected_half).abs() < TOL,
                "S=1/2: expected {}, got {}",
                expected_half,
                h_half.vals[0]
            );

            // S = 3/2 sector
            let h_three_half = make_su2_heisenberg_hamiltonian(&model, 1.5, 1).unwrap();
            assert_eq!(h_three_half.row_dim, 1, "S=3/2 sector should have dim=1");
            let expected_three_half = 0.5;
            assert!(
                (h_three_half.vals[0] - expected_three_half).abs() < TOL,
                "S=3/2: expected {}, got {}",
                expected_three_half,
                h_three_half.vals[0]
            );
        }

        /// Two spin-1 particles (S_1 = S_2 = 1)
        /// S = 0: ⟨S_1·S_2⟩ = (0 - 2 - 2)/2 = -2
        /// S = 1: ⟨S_1·S_2⟩ = (2 - 2 - 2)/2 = -1
        /// S = 2: ⟨S_1·S_2⟩ = (6 - 2 - 2)/2 = +1
        #[test]
        fn test_two_spin_one() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 2,
                two_s_list: vec![2, 2], // 2*S = 2 for spin-1
                exchange,
            };

            // S = 0 sector
            let h0 = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
            assert_eq!(h0.row_dim, 1, "S=0 sector should have dim=1");
            let expected_s0 = -2.0;
            assert!(
                (h0.vals[0] - expected_s0).abs() < TOL,
                "S=0: expected {}, got {}",
                expected_s0,
                h0.vals[0]
            );

            // S = 1 sector
            let h1 = make_su2_heisenberg_hamiltonian(&model, 1.0, 1).unwrap();
            assert_eq!(h1.row_dim, 1, "S=1 sector should have dim=1");
            let expected_s1 = -1.0;
            assert!(
                (h1.vals[0] - expected_s1).abs() < TOL,
                "S=1: expected {}, got {}",
                expected_s1,
                h1.vals[0]
            );

            // S = 2 sector
            let h2 = make_su2_heisenberg_hamiltonian(&model, 2.0, 1).unwrap();
            assert_eq!(h2.row_dim, 1, "S=2 sector should have dim=1");
            let expected_s2 = 1.0;
            assert!(
                (h2.vals[0] - expected_s2).abs() < TOL,
                "S=2: expected {}, got {}",
                expected_s2,
                h2.vals[0]
            );
        }

        /// Mixed: spin-1 and spin-3/2 (S_1 = 1, S_2 = 3/2)
        /// S_1(S_1+1) = 2, S_2(S_2+1) = 15/4
        /// S = 1/2: ⟨S_1·S_2⟩ = (3/4 - 2 - 15/4)/2 = (3/4 - 8/4 - 15/4)/2 = -20/8 = -5/2
        /// S = 3/2: ⟨S_1·S_2⟩ = (15/4 - 2 - 15/4)/2 = -1
        /// S = 5/2: ⟨S_1·S_2⟩ = (35/4 - 2 - 15/4)/2 = (35/4 - 8/4 - 15/4)/2 = 12/8 = 3/2
        #[test]
        fn test_spin_one_and_spin_three_half() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 2,
                two_s_list: vec![2, 3], // 2*S_1 = 2 (spin-1), 2*S_2 = 3 (spin-3/2)
                exchange,
            };

            // S = 1/2 sector
            // S(S+1) = 3/4, S_1(S_1+1) = 2, S_2(S_2+1) = 15/4
            // ⟨S_1·S_2⟩ = (3/4 - 2 - 15/4)/2 = -5/2
            let h_half = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h_half.row_dim, 1, "S=1/2 sector should have dim=1");
            let expected_half = -2.5; // -5/2
            assert!(
                (h_half.vals[0] - expected_half).abs() < TOL,
                "S=1/2: expected {}, got {}",
                expected_half,
                h_half.vals[0]
            );

            // S = 3/2 sector
            let h_three_half = make_su2_heisenberg_hamiltonian(&model, 1.5, 1).unwrap();
            assert_eq!(h_three_half.row_dim, 1, "S=3/2 sector should have dim=1");
            // S(S+1) = 3/2 * 5/2 = 15/4, S_1(S_1+1) = 2, S_2(S_2+1) = 15/4
            // ⟨S_1·S_2⟩ = (15/4 - 2 - 15/4)/2 = -1
            let expected_three_half = -1.0;
            assert!(
                (h_three_half.vals[0] - expected_three_half).abs() < TOL,
                "S=3/2: expected {}, got {}",
                expected_three_half,
                h_three_half.vals[0]
            );

            // S = 5/2 sector
            let h_five_half = make_su2_heisenberg_hamiltonian(&model, 2.5, 1).unwrap();
            assert_eq!(h_five_half.row_dim, 1, "S=5/2 sector should have dim=1");
            let expected_five_half = 1.5;
            assert!(
                (h_five_half.vals[0] - expected_five_half).abs() < TOL,
                "S=5/2: expected {}, got {}",
                expected_five_half,
                h_five_half.vals[0]
            );
        }

        /// Two spin-3/2 particles (S_1 = S_2 = 3/2)
        /// S = 0: ⟨S_1·S_2⟩ = (0 - 15/4 - 15/4)/2 = -15/4
        /// S = 1: ⟨S_1·S_2⟩ = (2 - 15/4 - 15/4)/2 = -11/4
        /// S = 2: ⟨S_1·S_2⟩ = (6 - 15/4 - 15/4)/2 = -3/4
        /// S = 3: ⟨S_1·S_2⟩ = (12 - 15/4 - 15/4)/2 = +9/4
        #[test]
        fn test_two_spin_three_half() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 2,
                two_s_list: vec![3, 3], // 2*S = 3 for spin-3/2
                exchange,
            };

            // S = 0 sector
            let h0 = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
            assert_eq!(h0.row_dim, 1, "S=0 sector should have dim=1");
            let expected_s0 = -15.0 / 4.0;
            assert!(
                (h0.vals[0] - expected_s0).abs() < TOL,
                "S=0: expected {}, got {}",
                expected_s0,
                h0.vals[0]
            );

            // S = 1 sector
            let h1 = make_su2_heisenberg_hamiltonian(&model, 1.0, 1).unwrap();
            assert_eq!(h1.row_dim, 1, "S=1 sector should have dim=1");
            let expected_s1 = -11.0 / 4.0;
            assert!(
                (h1.vals[0] - expected_s1).abs() < TOL,
                "S=1: expected {}, got {}",
                expected_s1,
                h1.vals[0]
            );

            // S = 2 sector
            let h2 = make_su2_heisenberg_hamiltonian(&model, 2.0, 1).unwrap();
            assert_eq!(h2.row_dim, 1, "S=2 sector should have dim=1");
            let expected_s2 = -3.0 / 4.0;
            assert!(
                (h2.vals[0] - expected_s2).abs() < TOL,
                "S=2: expected {}, got {}",
                expected_s2,
                h2.vals[0]
            );

            // S = 3 sector
            let h3 = make_su2_heisenberg_hamiltonian(&model, 3.0, 1).unwrap();
            assert_eq!(h3.row_dim, 1, "S=3 sector should have dim=1");
            let expected_s3 = 9.0 / 4.0;
            assert!(
                (h3.vals[0] - expected_s3).abs() < TOL,
                "S=3: expected {}, got {}",
                expected_s3,
                h3.vals[0]
            );
        }
    }

    /// Test: 3-site systems to check non-adjacent site calculations
    /// These tests verify the transformation coefficient calculations for S_0·S_2.
    mod three_site_tests {
        use super::*;

        /// Three spin-1/2 in the S=3/2 (fully polarized) sector.
        /// There is only one state: all spins aligned.
        /// H = J01 * S_0·S_1 + J12 * S_1·S_2 + J02 * S_0·S_2
        ///
        /// In the fully polarized state, all pairs are "triplet-like":
        /// ⟨S_i·S_j⟩ = 1/4 for any pair of spin-1/2.
        #[test]
        fn test_three_spin_half_max_spin() {
            // Test with only nearest-neighbor interactions
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);
            exchange.insert((1, 2), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 1.5, 1).unwrap();
            assert_eq!(h.row_dim, 1, "S=3/2 sector should have dim=1");
            // ⟨S_0·S_1⟩ + ⟨S_1·S_2⟩ = 1/4 + 1/4 = 1/2
            let expected = 0.5;
            assert!(
                (h.vals[0] - expected).abs() < TOL,
                "3-site S=3/2 (NN only): expected {}, got {}",
                expected,
                h.vals[0]
            );

            // Now test with non-adjacent interaction S_0·S_2
            let mut exchange_with_02 = HashMap::new();
            exchange_with_02.insert((0, 1), 1.0);
            exchange_with_02.insert((1, 2), 1.0);
            exchange_with_02.insert((0, 2), 1.0);

            let model_with_02 = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange: exchange_with_02,
            };

            let h_with_02 = make_su2_heisenberg_hamiltonian(&model_with_02, 1.5, 1).unwrap();
            // ⟨S_0·S_1⟩ + ⟨S_1·S_2⟩ + ⟨S_0·S_2⟩ = 1/4 + 1/4 + 1/4 = 3/4
            let expected_with_02 = 0.75;
            assert!(
                (h_with_02.vals[0] - expected_with_02).abs() < TOL,
                "3-site S=3/2 (with S_0·S_2): expected {}, got {}",
                expected_with_02,
                h_with_02.vals[0]
            );
        }

        /// Three spin-1/2 in the S=1/2 sector.
        /// There are 2 basis states:
        /// |1⟩: [1, 0, 1] - sites 0,1 in singlet (J_1=0), total S=1/2
        /// |2⟩: [1, 2, 1] - sites 0,1 in triplet (J_1=1), total S=1/2
        ///
        /// For H = S_0·S_1 only:
        /// H|1⟩ = -3/4 |1⟩  (singlet eigenvalue)
        /// H|2⟩ = +1/4 |2⟩  (triplet eigenvalue)
        /// So H is diagonal with eigenvalues -3/4 and +1/4.
        #[test]
        fn test_three_spin_half_s01_only() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h.row_dim, 2, "S=1/2 sector should have dim=2");

            // Get the full 2x2 matrix
            let mut mat = [[0.0; 2]; 2];
            for row in 0..2 {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    mat[row][col] = h.vals[idx];
                }
            }

            eprintln!("H(S_0·S_1 only) = [{:.4}, {:.4}]", mat[0][0], mat[0][1]);
            eprintln!("                 [{:.4}, {:.4}]", mat[1][0], mat[1][1]);

            // Should be diagonal with eigenvalues -3/4 and +1/4
            // (order depends on basis ordering)
            assert!(
                mat[0][1].abs() < TOL && mat[1][0].abs() < TOL,
                "H(S_0·S_1) should be diagonal, but off-diagonals are: [{}, {}]",
                mat[0][1],
                mat[1][0]
            );

            // Check that diagonal elements are -3/4 and +1/4
            let diag_sum = mat[0][0] + mat[1][1];
            let diag_prod = mat[0][0] * mat[1][1];
            let expected_sum = -0.75 + 0.25; // = -0.5
            let expected_prod = -0.75 * 0.25; // = -0.1875

            assert!(
                (diag_sum - expected_sum).abs() < TOL,
                "Trace should be {}, got {}",
                expected_sum,
                diag_sum
            );
            assert!(
                (diag_prod - expected_prod).abs() < TOL,
                "Det should be {}, got {}",
                expected_prod,
                diag_prod
            );
        }

        /// Three spin-1/2 with S_1·S_2 interaction only.
        /// H = S_1·S_2
        ///
        /// For basis states:
        /// |1⟩: [1, 0, 1] - J_1=0, J_2=1/2
        /// |2⟩: [1, 2, 1] - J_1=1, J_2=1/2
        ///
        /// The eigenvalues should be -3/4 and +1/4, same as S_0·S_1.
        /// But the matrix is NOT diagonal in this basis.
        #[test]
        fn test_three_spin_half_s12_only() {
            let mut exchange = HashMap::new();
            exchange.insert((1, 2), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h.row_dim, 2, "S=1/2 sector should have dim=2");

            let mut mat = [[0.0; 2]; 2];
            for row in 0..2 {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    mat[row][col] = h.vals[idx];
                }
            }

            eprintln!("H(S_1·S_2 only) = [{:.4}, {:.4}]", mat[0][0], mat[0][1]);
            eprintln!("                 [{:.4}, {:.4}]", mat[1][0], mat[1][1]);

            // Check eigenvalues
            let trace = mat[0][0] + mat[1][1];
            let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
            let discriminant = trace * trace - 4.0 * det;
            let sqrt_d = discriminant.sqrt();
            let lambda1 = (trace + sqrt_d) / 2.0;
            let lambda2 = (trace - sqrt_d) / 2.0;

            eprintln!("S_1·S_2 eigenvalues: {:.4}, {:.4}", lambda1, lambda2);

            // Expected: -3/4 and +1/4
            let (computed_min, computed_max) = if lambda1 < lambda2 {
                (lambda1, lambda2)
            } else {
                (lambda2, lambda1)
            };

            assert!(
                (computed_min - (-0.75)).abs() < TOL,
                "S_1·S_2 smaller eigenvalue should be -0.75, got {}",
                computed_min
            );
            assert!(
                (computed_max - 0.25).abs() < TOL,
                "S_1·S_2 larger eigenvalue should be 0.25, got {}",
                computed_max
            );
        }

        /// Three spin-1/2 with S_0·S_2 (non-adjacent) interaction only.
        /// H = S_0·S_2
        ///
        /// This specifically tests the non-adjacent recoupling code.
        #[test]
        fn test_three_spin_half_s02_only() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 2), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h.row_dim, 2, "S=1/2 sector should have dim=2");

            let mut mat = [[0.0; 2]; 2];
            for row in 0..2 {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    mat[row][col] = h.vals[idx];
                }
            }

            eprintln!("H(S_0·S_2 only) = [{:.4}, {:.4}]", mat[0][0], mat[0][1]);
            eprintln!("                 [{:.4}, {:.4}]", mat[1][0], mat[1][1]);

            // The eigenvalues should be -3/4 and +1/4 (same as for S_0·S_1 or S_1·S_2)
            // because of SU(2) symmetry - all pairs are equivalent.
            let trace = mat[0][0] + mat[1][1];
            let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

            // Eigenvalues from quadratic formula: λ = (trace ± sqrt(trace² - 4*det)) / 2
            let discriminant = trace * trace - 4.0 * det;
            assert!(discriminant >= 0.0, "Discriminant should be non-negative");
            let sqrt_d = discriminant.sqrt();
            let lambda1 = (trace + sqrt_d) / 2.0;
            let lambda2 = (trace - sqrt_d) / 2.0;

            eprintln!("Eigenvalues: {:.4}, {:.4}", lambda1, lambda2);

            // Expected eigenvalues: -3/4 and +1/4
            let expected_eigs = [-0.75, 0.25];
            let computed_eigs = if lambda1 > lambda2 {
                [lambda2, lambda1]
            } else {
                [lambda1, lambda2]
            };

            assert!(
                (computed_eigs[0] - expected_eigs[0]).abs() < TOL,
                "Smaller eigenvalue should be {}, got {}",
                expected_eigs[0],
                computed_eigs[0]
            );
            assert!(
                (computed_eigs[1] - expected_eigs[1]).abs() < TOL,
                "Larger eigenvalue should be {}, got {}",
                expected_eigs[1],
                computed_eigs[1]
            );
        }

        /// 5-site Heisenberg chain: H = Σ S_i·S_{i+1}
        /// Known exact eigenvalues for S=1/2 sector (from full diagonalization):
        ///   E = -1.92788625, -1.20710678, -0.66081859, 0.20710678, 0.58870484
        #[test]
        fn test_five_spin_half_chain_eigenvalues() {
            use lapack::dsyev;

            let mut exchange = HashMap::new();
            for i in 0..4 {
                exchange.insert((i, i + 1), 1.0);
            }

            let model = SU2HeisenbergModel {
                num_sites: 5,
                two_s_list: vec![1, 1, 1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            let dim = h.row_dim;
            assert_eq!(dim, 5, "S=1/2 sector should have dim=5");

            let mut dense = vec![0.0f64; dim * dim];
            for row in 0..dim {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    dense[row + col * dim] = h.vals[idx];
                }
            }

            let mut eigenvalues = vec![0.0f64; dim];
            let lwork = 3 * dim;
            let mut work = vec![0.0f64; lwork];
            let mut info = 0;
            let n = dim as i32;

            unsafe {
                dsyev(
                    b'N',
                    b'U',
                    n,
                    &mut dense,
                    n,
                    &mut eigenvalues,
                    &mut work,
                    lwork as i32,
                    &mut info,
                );
            }

            assert_eq!(info, 0, "LAPACK dsyev failed");

            eprintln!("5-site S=1/2 eigenvalues:");
            for (i, &e) in eigenvalues.iter().enumerate() {
                eprintln!("  E[{}] = {:.8}", i, e);
            }

            let expected = [
                -1.92788625,
                -1.20710678,
                -0.66081859,
                0.20710678,
                0.58870484,
            ];
            for (i, &exp) in expected.iter().enumerate() {
                assert!(
                    (eigenvalues[i] - exp).abs() < 1e-6,
                    "Eigenvalue {} should be {}, got {}",
                    i,
                    exp,
                    eigenvalues[i]
                );
            }
        }

        /// Test non-adjacent S_0·S_2 on 5-site system
        /// Eigenvalues should be -0.75 and +0.25 only
        #[test]
        fn test_five_site_non_adjacent_s02() {
            use crate::utility::wigner::WignerSymbols;

            let wigner = WignerSymbols::new(50);
            let two_s_list = vec![1i32; 5];

            let mut exchange = HashMap::new();
            exchange.insert((0, 2), 1.0); // Non-adjacent!

            let model = SU2HeisenbergModel {
                num_sites: 5,
                two_s_list: vec![1; 5],
                exchange,
            };

            let (basis, _) = model.build_basis(0.5).unwrap();
            eprintln!("5-site S=1/2 basis for S_0·S_2 test (dim={}):", basis.len());
            for (i, state) in basis.iter().enumerate() {
                eprintln!("  |{}⟩ = {:?}", i, state);
            }

            // Build the matrix manually
            let dim = basis.len();
            let mut matrix = vec![vec![0.0; dim]; dim];
            for (i, bra) in basis.iter().enumerate() {
                for (j, ket) in basis.iter().enumerate() {
                    let val = compute_si_sj_element(&wigner, &two_s_list, bra, ket, 0, 2);
                    matrix[i][j] = val;
                }
            }

            eprintln!("\nS_0·S_2 matrix:");
            for row in &matrix {
                let row_str: Vec<String> = row.iter().map(|v| format!("{:8.4}", v)).collect();
                eprintln!("  [{}]", row_str.join(", "));
            }

            // Check trace (should be sum of diagonal elements)
            let trace: f64 = (0..dim).map(|i| matrix[i][i]).sum();
            eprintln!("\nTrace = {}", trace);

            // Convert to dense array for eigenvalue check
            let mut dense = vec![0.0f64; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    dense[i + j * dim] = matrix[i][j];
                }
            }

            // Get eigenvalues using LAPACK
            use lapack::dsyev;
            let mut eigenvalues = vec![0.0f64; dim];
            let lwork = 3 * dim;
            let mut work = vec![0.0f64; lwork];
            let mut info = 0;
            let n = dim as i32;

            unsafe {
                dsyev(
                    b'N',
                    b'U',
                    n,
                    &mut dense,
                    n,
                    &mut eigenvalues,
                    &mut work,
                    lwork as i32,
                    &mut info,
                );
            }
            assert_eq!(info, 0, "LAPACK dsyev failed");

            eprintln!("S_0·S_2 eigenvalues:");
            for (i, &e) in eigenvalues.iter().enumerate() {
                eprintln!("  E[{}] = {:.8}", i, e);
            }

            // Check that eigenvalues are only -0.75 and +0.25
            for (i, &e) in eigenvalues.iter().enumerate() {
                let is_singlet = (e - (-0.75)).abs() < 1e-6;
                let is_triplet = (e - 0.25).abs() < 1e-6;
                assert!(
                    is_singlet || is_triplet,
                    "Eigenvalue {} = {} should be either -0.75 or +0.25",
                    i,
                    e
                );
            }
        }

        /// 6-site Heisenberg chain: H = Σ S_i·S_{i+1}
        /// Known exact eigenvalues for S=0 sector (from full diagonalization):
        ///   E = -2.49357713, -1.18522692, -0.75, -0.10608802, 0.78489207
        #[test]
        fn test_six_spin_half_chain_eigenvalues() {
            use lapack::dsyev;

            let mut exchange = HashMap::new();
            for i in 0..5 {
                exchange.insert((i, i + 1), 1.0);
            }

            let model = SU2HeisenbergModel {
                num_sites: 6,
                two_s_list: vec![1, 1, 1, 1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
            let dim = h.row_dim;
            assert_eq!(dim, 5, "S=0 sector should have dim=5");

            // Convert sparse to dense for LAPACK
            let mut dense = vec![0.0f64; dim * dim];
            for row in 0..dim {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    dense[row + col * dim] = h.vals[idx];
                }
            }

            // LAPACK diagonalization
            let mut eigenvalues = vec![0.0f64; dim];
            let mut work = vec![0.0f64; 3 * dim];
            let mut info = 0;
            let n = dim as i32;
            let lda = n;
            let lwork = work.len() as i32;

            unsafe {
                dsyev(
                    b'N', // Don't compute eigenvectors
                    b'U', // Upper triangle
                    n,
                    &mut dense,
                    lda,
                    &mut eigenvalues,
                    &mut work,
                    lwork,
                    &mut info,
                );
            }

            assert_eq!(info, 0, "LAPACK dsyev failed with info={}", info);

            eprintln!("6-site S=0 eigenvalues:");
            for (i, &e) in eigenvalues.iter().enumerate() {
                eprintln!("  E[{}] = {:.8}", i, e);
            }

            // Expected eigenvalues from full diagonalization
            let expected = [-2.49357713, -1.18522692, -0.75, -0.10608802, 0.78489207];
            for (i, &exp) in expected.iter().enumerate() {
                assert!(
                    (eigenvalues[i] - exp).abs() < 1e-6,
                    "Eigenvalue {} should be {}, got {}",
                    i,
                    exp,
                    eigenvalues[i]
                );
            }
        }

        /// 4-site Heisenberg chain: H = S_0·S_1 + S_1·S_2 + S_2·S_3
        /// Known exact eigenvalues for S=0 sector (from full diagonalization):
        ///   E = -1.616025... (ground state)
        ///   E = 0.116025...  (excited state)
        /// These values are (−1 ± √5)/2 − 1/4
        #[test]
        fn test_four_spin_half_chain_eigenvalues() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);
            exchange.insert((1, 2), 1.0);
            exchange.insert((2, 3), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 4,
                two_s_list: vec![1, 1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.0, 1).unwrap();
            assert_eq!(h.row_dim, 2, "S=0 sector should have dim=2");

            let mut mat = [[0.0; 2]; 2];
            for row in 0..2 {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    mat[row][col] = h.vals[idx];
                }
            }

            eprintln!("4-site H(S=0) = [{:.4}, {:.4}]", mat[0][0], mat[0][1]);
            eprintln!("               [{:.4}, {:.4}]", mat[1][0], mat[1][1]);

            let trace = mat[0][0] + mat[1][1];
            let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
            let discriminant = trace * trace - 4.0 * det;
            let sqrt_d = discriminant.sqrt();
            let lambda1 = (trace + sqrt_d) / 2.0;
            let lambda2 = (trace - sqrt_d) / 2.0;

            eprintln!("Eigenvalues: {:.6}, {:.6}", lambda1, lambda2);

            // Expected eigenvalues from full diagonalization
            let expected_min = -1.616025403784; // (−1 − √5)/2 − 1/4
            let expected_max = 0.116025403784; // (−1 + √5)/2 − 1/4
            let (computed_min, computed_max) = if lambda1 < lambda2 {
                (lambda1, lambda2)
            } else {
                (lambda2, lambda1)
            };

            assert!(
                (computed_min - expected_min).abs() < 1e-6,
                "Ground state energy should be {}, got {}",
                expected_min,
                computed_min
            );
            assert!(
                (computed_max - expected_max).abs() < 1e-6,
                "Excited state energy should be {}, got {}",
                expected_max,
                computed_max
            );
        }

        /// Full 3-site Heisenberg chain: H = S_0·S_1 + S_1·S_2
        /// Known exact eigenvalues for S=1/2 sector: -1 and 0
        /// (verified by full diagonalization: S=1/2 sector has E=-1, 0;
        ///  S=3/2 sector has E=0.5)
        #[test]
        fn test_three_spin_half_chain_eigenvalues() {
            let mut exchange = HashMap::new();
            exchange.insert((0, 1), 1.0);
            exchange.insert((1, 2), 1.0);

            let model = SU2HeisenbergModel {
                num_sites: 3,
                two_s_list: vec![1, 1, 1],
                exchange,
            };

            let h = make_su2_heisenberg_hamiltonian(&model, 0.5, 1).unwrap();
            assert_eq!(h.row_dim, 2);

            let mut mat = [[0.0; 2]; 2];
            for row in 0..2 {
                for idx in h.rows[row]..h.rows[row + 1] {
                    let col = h.cols[idx];
                    mat[row][col] = h.vals[idx];
                }
            }

            eprintln!(
                "H(S_0·S_1 + S_1·S_2) = [{:.4}, {:.4}]",
                mat[0][0], mat[0][1]
            );
            eprintln!("                      [{:.4}, {:.4}]", mat[1][0], mat[1][1]);

            let trace = mat[0][0] + mat[1][1];
            let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
            let discriminant = trace * trace - 4.0 * det;
            let sqrt_d = discriminant.sqrt();
            let lambda1 = (trace + sqrt_d) / 2.0;
            let lambda2 = (trace - sqrt_d) / 2.0;

            eprintln!("Eigenvalues: {:.4}, {:.4}", lambda1, lambda2);

            // Expected: -1 and 0 (verified by full diagonalization)
            let expected_min = -1.0;
            let expected_max = 0.0;
            let (computed_min, computed_max) = if lambda1 < lambda2 {
                (lambda1, lambda2)
            } else {
                (lambda2, lambda1)
            };

            assert!(
                (computed_min - expected_min).abs() < TOL,
                "Ground state energy should be {}, got {}",
                expected_min,
                computed_min
            );
            assert!(
                (computed_max - expected_max).abs() < TOL,
                "Excited state energy should be {}, got {}",
                expected_max,
                computed_max
            );
        }
    }
}
