//! SU(2) symmetric Heisenberg Hamiltonian construction using 6j symbols.

use ahash::{AHashMap, AHashSet};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::model::su2_heisenberg::SU2HeisenbergModel;
use crate::utility::cg_coupling::eigenvalue_si_sj;
use crate::utility::rayon_pool::with_pool;
use crate::utility::wigner::WignerSymbols;

/// Type alias for 6j symbol key: (j1, j2, j3, j4, j5, j6)
type SixJKey = (i32, i32, i32, i32, i32, i32);

/// Collect all 6j symbol keys needed for the Hamiltonian construction.
fn collect_required_6j_keys(
    basis: &[Vec<u8>],
    two_s_list: &[i32],
    interactions: &[((usize, usize), f64)],
) -> AHashSet<SixJKey> {
    let mut keys = AHashSet::new();

    for state in basis {
        for &((site_i, site_j), _) in interactions {
            collect_6j_keys_for_state(two_s_list, state, site_i, site_j, &mut keys);
        }
    }

    keys
}

/// Collect 6j keys needed for computing transformation coefficient for a single state.
fn collect_6j_keys_for_state(
    two_s_list: &[i32],
    left_state: &[u8],
    site_i: usize,
    site_j: usize,
    keys: &mut AHashSet<SixJKey>,
) {
    let two_si = two_s_list[site_i];
    let two_sj = two_s_list[site_j];

    // Iterate over all possible K values
    let mut two_k = (two_si - two_sj).abs();
    while two_k <= two_si + two_sj {
        if site_j == site_i + 1 {
            // Adjacent case
            if site_i > 0 {
                let (two_j_im1, two_j_i, two_j_ip1) = (
                    left_state[site_i - 1] as i32,
                    left_state[site_i] as i32,
                    left_state[site_i + 1] as i32,
                );
                let two_s_i = two_s_list[site_i];
                let two_s_ip1 = two_s_list[site_i + 1];
                keys.insert((two_j_im1, two_s_i, two_j_i, two_s_ip1, two_j_ip1, two_k));
            }
        } else {
            // Non-adjacent case: collect recursively
            let spin_order: Vec<usize> = (0..two_s_list.len()).collect();
            collect_bubble_6j_keys(
                two_s_list,
                left_state,
                &spin_order,
                site_i,
                site_j,
                two_k,
                keys,
            );
        }
        two_k += 2;
    }
}

/// Recursively collect 6j keys for bubble recoupling.
fn collect_bubble_6j_keys(
    two_s_list: &[i32],
    left_state: &[u8],
    spin_order: &[usize],
    site_i: usize,
    current_pos: usize,
    two_k: i32,
    keys: &mut AHashSet<SixJKey>,
) {
    let two_s_i = two_s_list[spin_order[site_i]];
    let two_s_j = two_s_list[spin_order[current_pos]];

    // Base case
    if current_pos == site_i + 1 {
        if site_i > 0 {
            let (two_j_im1, two_j_i, two_j_ip1) = (
                left_state[site_i - 1] as i32,
                left_state[site_i] as i32,
                left_state[site_i + 1] as i32,
            );
            keys.insert((two_j_im1, two_s_i, two_j_i, two_s_j, two_j_ip1, two_k));
        }
        return;
    }

    // Recursive case
    let p = current_pos;
    let two_s_pm1 = two_s_list[spin_order[p - 1]];
    let two_j_p = left_state[p] as i32;
    let two_j_pm1 = left_state[p - 1] as i32;
    let two_j_pm2 = if p >= 2 {
        left_state[p - 2] as i32
    } else {
        two_s_list[spin_order[0]]
    };

    let two_f_min = (two_j_pm2 - two_s_j).abs().max((two_s_pm1 - two_j_p).abs());
    let two_f_max = (two_j_pm2 + two_s_j).min(two_s_pm1 + two_j_p);

    let parity = (two_j_pm2 + two_s_j) % 2;
    if parity != (two_s_pm1 + two_j_p) % 2 {
        return;
    }

    let mut two_f = if two_f_min % 2 == parity {
        two_f_min
    } else {
        two_f_min + 1
    };

    while two_f <= two_f_max {
        // Add this 6j key
        keys.insert((two_j_pm2, two_s_pm1, two_j_pm1, two_j_p, two_s_j, two_f));

        // Recurse with modified state
        let mut modified_state = left_state.to_vec();
        modified_state[p - 1] = two_f as u8;
        let mut modified_order = spin_order.to_vec();
        modified_order.swap(p - 1, p);

        collect_bubble_6j_keys(
            two_s_list,
            &modified_state,
            &modified_order,
            site_i,
            p - 1,
            two_k,
            keys,
        );

        two_f += 2;
    }
}

/// Precompute all required 6j symbols in parallel.
fn precompute_6j_table(keys: AHashSet<SixJKey>, max_two_j: usize) -> AHashMap<SixJKey, f64> {
    let keys_vec: Vec<_> = keys.into_iter().collect();

    let entries: Vec<_> = keys_vec
        .par_iter()
        .map_init(
            || WignerSymbols::new(max_two_j),
            |wigner, &(j1, j2, j3, j4, j5, j6)| {
                let val = wigner.wigner_6j(j1, j2, j3, j4, j5, j6);
                ((j1, j2, j3, j4, j5, j6), val)
            },
        )
        .collect();

    entries.into_iter().collect()
}

/// Look up a 6j symbol from precomputed table. Returns 0.0 if not found.
#[inline]
fn lookup_6j(table: &AHashMap<SixJKey, f64>, j1: i32, j2: i32, j3: i32, j4: i32, j5: i32, j6: i32) -> f64 {
    *table.get(&(j1, j2, j3, j4, j5, j6)).unwrap_or(&0.0)
}

/// Compute the transformation coefficient using precomputed 6j table.
fn compute_transformation_coeff_with_table(
    sixj_table: &AHashMap<SixJKey, f64>,
    two_s_list: &[i32],
    left_state: &[u8],
    site_i: usize,
    site_j: usize,
    two_k: i32,
) -> f64 {
    debug_assert!(site_i < site_j && site_j < two_s_list.len());

    // Adjacent sites (i, i+1): direct 6j recoupling
    if site_j == site_i + 1 {
        if site_i == 0 {
            return if left_state[1] as i32 == two_k {
                1.0
            } else {
                0.0
            };
        }
        let (two_j_im1, two_j_i, two_j_ip1) = (
            left_state[site_i - 1] as i32,
            left_state[site_i] as i32,
            left_state[site_i + 1] as i32,
        );
        let (two_s_i, two_s_ip1) = (two_s_list[site_i], two_s_list[site_i + 1]);

        let phase = if (two_j_im1 + two_s_i + two_s_ip1 + two_j_ip1) / 2 % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let dim_factor = (((two_j_i + 1) * (two_k + 1)) as f64).sqrt();
        let sixj = lookup_6j(sixj_table, two_j_im1, two_s_i, two_j_i, two_s_ip1, two_j_ip1, two_k);
        return phase * dim_factor * sixj;
    }

    // Non-adjacent sites: bubble S_j towards S_i
    let spin_order: Vec<usize> = (0..two_s_list.len()).collect();
    compute_bubble_with_table(
        sixj_table,
        two_s_list,
        left_state,
        &spin_order,
        site_i,
        site_j,
        two_k,
    )
}

/// Recursively bubble S_j towards S_i using precomputed 6j table.
fn compute_bubble_with_table(
    sixj_table: &AHashMap<SixJKey, f64>,
    two_s_list: &[i32],
    left_state: &[u8],
    spin_order: &[usize],
    site_i: usize,
    current_pos: usize,
    two_k: i32,
) -> f64 {
    let two_s_i = two_s_list[spin_order[site_i]];
    let two_s_j = two_s_list[spin_order[current_pos]];

    // Base case: S_j is adjacent to S_i
    if current_pos == site_i + 1 {
        if site_i == 0 {
            return if left_state[1] as i32 == two_k {
                1.0
            } else {
                0.0
            };
        }
        let (two_j_im1, two_j_i, two_j_ip1) = (
            left_state[site_i - 1] as i32,
            left_state[site_i] as i32,
            left_state[site_i + 1] as i32,
        );
        let phase = if (two_j_im1 + two_s_i + two_s_j + two_j_ip1) / 2 % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let dim_factor = (((two_j_i + 1) * (two_k + 1)) as f64).sqrt();
        let sixj = lookup_6j(sixj_table, two_j_im1, two_s_i, two_j_i, two_s_j, two_j_ip1, two_k);
        return phase * dim_factor * sixj;
    }

    // Recursive case: swap S_j with the spin at position p-1
    let p = current_pos;
    let two_s_pm1 = two_s_list[spin_order[p - 1]];
    let two_j_p = left_state[p] as i32;
    let two_j_pm1 = left_state[p - 1] as i32;
    let two_j_pm2 = if p >= 2 {
        left_state[p - 2] as i32
    } else {
        two_s_list[spin_order[0]]
    };

    // Triangle inequalities for intermediate spin f
    let two_f_min = (two_j_pm2 - two_s_j).abs().max((two_s_pm1 - two_j_p).abs());
    let two_f_max = (two_j_pm2 + two_s_j).min(two_s_pm1 + two_j_p);

    // Parity check
    let parity = (two_j_pm2 + two_s_j) % 2;
    if parity != (two_s_pm1 + two_j_p) % 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut two_f = if two_f_min % 2 == parity {
        two_f_min
    } else {
        two_f_min + 1
    };
    while two_f <= two_f_max {
        let phase = if (two_s_pm1 + two_j_pm1 + two_s_j + two_f) / 2 % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let dim_factor = (((two_j_pm1 + 1) * (two_f + 1)) as f64).sqrt();
        let sixj = lookup_6j(sixj_table, two_j_pm2, two_s_pm1, two_j_pm1, two_j_p, two_s_j, two_f);

        if sixj.abs() >= 1e-15 {
            let mut modified_state = left_state.to_vec();
            modified_state[p - 1] = two_f as u8;
            let mut modified_order = spin_order.to_vec();
            modified_order.swap(p - 1, p);

            total += phase
                * dim_factor
                * sixj
                * compute_bubble_with_table(
                    sixj_table,
                    two_s_list,
                    &modified_state,
                    &modified_order,
                    site_i,
                    p - 1,
                    two_k,
                );
        }
        two_f += 2;
    }
    total
}

/// Compute ⟨bra|S_i·S_j|ket⟩ matrix element using precomputed 6j table.
fn compute_si_sj_element_with_table(
    sixj_table: &AHashMap<SixJKey, f64>,
    two_s_list: &[i32],
    bra: &[u8],
    ket: &[u8],
    site_i: usize,
    site_j: usize,
) -> f64 {
    let (two_si, two_sj) = (two_s_list[site_i], two_s_list[site_j]);
    let mut result = 0.0;
    let mut two_k = (two_si - two_sj).abs();
    while two_k <= two_si + two_sj {
        let coeff_bra =
            compute_transformation_coeff_with_table(sixj_table, two_s_list, bra, site_i, site_j, two_k);
        if coeff_bra.abs() >= 1e-15 {
            let coeff_ket =
                compute_transformation_coeff_with_table(sixj_table, two_s_list, ket, site_i, site_j, two_k);
            if coeff_ket.abs() >= 1e-15 {
                result += eigenvalue_si_sj(two_si, two_sj, two_k) * coeff_bra * coeff_ket;
            }
        }
        two_k += 2;
    }
    result
}

/// Hash of the fixed parts of a basis state for selection rule grouping.
fn extract_selection_key_hash(state: &[u8], site_i: usize, site_j: usize) -> u64 {
    let mut hasher = ahash::AHasher::default();
    state[0..site_i].hash(&mut hasher);
    state[site_j..].hash(&mut hasher);
    hasher.finish()
}

/// Group basis indices by selection key for an interaction (site_i, site_j).
fn build_selection_groups(
    basis: &[Vec<u8>],
    site_i: usize,
    site_j: usize,
) -> AHashMap<u64, Vec<usize>> {
    let mut groups: AHashMap<u64, Vec<usize>> = AHashMap::new();
    for (idx, state) in basis.iter().enumerate() {
        groups
            .entry(extract_selection_key_hash(state, site_i, site_j))
            .or_default()
            .push(idx);
    }
    groups
}

/// Build the Hamiltonian matrix for SU(2) Heisenberg model in CSR format.
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

    let basis = model.build_basis(total_s)?;
    let dim = basis.len();

    if dim == 0 {
        bail!(
            "Target Hilbert space has zero dimension for total_s = {}",
            total_s
        );
    }

    let max_two_j = (model.two_s_list.iter().sum::<i32>() + 10) as usize;
    let two_s_list = model.two_s_list.clone();
    let interactions: Vec<_> = model.exchange.iter().map(|(&k, &v)| (k, v)).collect();
    let diagonal_shift = model.diagonal_shift;

    // Precompute all required 6j symbols
    let sixj_keys = collect_required_6j_keys(&basis, &two_s_list, &interactions);
    let sixj_table = with_pool(num_threads, || Ok(precompute_6j_table(sixj_keys, max_two_j)))?;

    let (basis_ref, interactions_ref, two_s_list_ref, sixj_table_ref) =
        (&basis, &interactions, &two_s_list, &sixj_table);

    with_pool(num_threads, move || -> Result<CsrMatrix> {
        let interaction_groups: Vec<_> = interactions_ref
            .par_iter()
            .map(|&((site_i, site_j), _)| build_selection_groups(basis_ref, site_i, site_j))
            .collect();

        let compute_row = |row_idx: usize, bra: &[u8]| -> Vec<(usize, f64)> {
            let mut contributions: AHashMap<usize, f64> = AHashMap::new();
            if diagonal_shift.abs() > 1e-15 {
                contributions.insert(row_idx, diagonal_shift);
            }
            for (int_idx, &((site_i, site_j), coeff)) in interactions_ref.iter().enumerate() {
                let key = extract_selection_key_hash(bra, site_i, site_j);
                if let Some(group) = interaction_groups[int_idx].get(&key) {
                    for &col_idx in group {
                        let elem = compute_si_sj_element_with_table(
                            sixj_table_ref,
                            two_s_list_ref,
                            bra,
                            &basis_ref[col_idx],
                            site_i,
                            site_j,
                        );
                        if elem.abs() > 1e-15 {
                            *contributions.entry(col_idx).or_insert(0.0) += coeff * elem;
                        }
                    }
                }
            }
            let mut entries: Vec<_> = contributions
                .into_iter()
                .filter(|(_, v)| v.abs() > MATRIX_ZERO_EPS)
                .collect();
            entries.sort_unstable_by_key(|(col, _)| *col);
            entries
        };

        // Phase 1: count nnz per row
        let row_nnz: Vec<_> = basis_ref
            .par_iter()
            .enumerate()
            .map(|(i, bra)| compute_row(i, bra).len())
            .collect();

        // Prefix sum for CSR rows
        let mut rows = Vec::with_capacity(dim + 1);
        rows.push(0);
        for &nnz in &row_nnz {
            rows.push(rows.last().unwrap() + nnz);
        }
        let total_nnz = *rows.last().unwrap();
        let cols = vec![0usize; total_nnz];
        let vals = vec![0.0; total_nnz];

        // Phase 2: fill cols/vals
        basis_ref.par_iter().enumerate().for_each(|(row_idx, bra)| {
            let entries = compute_row(row_idx, bra);
            let start = rows[row_idx];
            unsafe {
                let (cols_ptr, vals_ptr) =
                    (cols.as_ptr() as *mut usize, vals.as_ptr() as *mut f64);
                for (i, (col, val)) in entries.into_iter().enumerate() {
                    *cols_ptr.add(start + i) = col;
                    *vals_ptr.add(start + i) = val;
                }
            }
        });

        Ok(CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows,
            cols,
            vals,
        })
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
            diagonal_shift: 0.0,
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
            diagonal_shift: 0.0,
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
            diagonal_shift: 0.0,
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
            diagonal_shift: 0.0,
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
            diagonal_shift: 0.0,
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
    fn test_long_range_hamiltonian() {
        // Five spin-1/2 with long-range interaction (0, 4)
        // H = J * S_0·S_4

        let mut exchange = HashMap::new();
        exchange.insert((0, 4), 1.0);

        let model = SU2HeisenbergModel {
            num_sites: 5,
            two_s_list: vec![1, 1, 1, 1, 1],
            exchange,
            diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
                diagonal_shift: 0.0,
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
