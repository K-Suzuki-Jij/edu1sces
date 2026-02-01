use ahash::AHashMap;
use anyhow::{bail, Result};

use crate::basis::SU2HeisenbergBasis;
use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::model::SU2HeisenbergModel;
use crate::utility::rayon_pool::with_pool;
use rayon::prelude::*;
use wigner_symbols::Wigner6j;

fn phase_from_half_integer(sum_two: i32) -> f64 {
    if ((sum_two / 2) & 1) != 0 {
        -1.0
    } else {
        1.0
    }
}

fn swap_at(
    state_map: &AHashMap<Vec<u8>, f64>,
    order: &[usize],
    pos: usize,
    two_s_list: &[i32],
    zero_eps: f64,
) -> AHashMap<Vec<u8>, f64> {
    let mut next: AHashMap<Vec<u8>, f64> = AHashMap::new();

    for (state, &coeff) in state_map.iter() {
        if coeff.abs() < zero_eps {
            continue;
        }

        if pos == 0 {
            let a = two_s_list[order[0]];
            let b = two_s_list[order[1]];
            let j1 = state[1] as i32;

            let phase = phase_from_half_integer(a + b - j1);
            let mut new_state = state.clone();
            new_state[0] = b as u8;

            let entry = next.entry(new_state).or_insert(0.0);
            *entry += coeff * phase;
            continue;
        }

        let x = state[pos - 1] as i32;
        let j_k = state[pos] as i32;
        let j_k1 = state[pos + 1] as i32;

        let a = two_s_list[order[pos]];
        let b = two_s_list[order[pos + 1]];

        let min_j = (x - b).abs();
        let max_j = x + b;

        // j_kp parity must match x+b.
        let mut j_kp = min_j;
        if ((j_kp + x + b) & 1) != 0 {
            j_kp += 1;
        }

        // t = 2*j_ab (coupling of a and b)
        let min_t = (a - b).abs();
        let max_t = a + b;
        let mut t_start = min_t;
        if ((t_start + a + b) & 1) != 0 {
            t_start += 1;
        }

        while j_kp <= max_j {
            let mut sum = 0.0;
            let mut t = t_start;
            while t <= max_t {
                let sixj1 = Wigner6j {
                    tj1: x,
                    tj2: a,
                    tj3: j_k,
                    tj4: b,
                    tj5: j_k1,
                    tj6: t,
                }
                .value();
                let s1 = f64::from(sixj1);
                if s1.abs() >= zero_eps {
                    let sixj2 = Wigner6j {
                        tj1: x,
                        tj2: b,
                        tj3: j_kp,
                        tj4: a,
                        tj5: j_k1,
                        tj6: t,
                    }
                    .value();
                    let s2 = f64::from(sixj2);
                    if s2.abs() >= zero_eps {
                        let phase = phase_from_half_integer(a + b - t);
                        let weight = (t + 1) as f64;
                        sum += phase * weight * s1 * s2;
                    }
                }

                t += 2;
            }

            if sum.abs() >= zero_eps {
                let norm = ((j_k + 1) as f64 * (j_kp + 1) as f64).sqrt();
                let c = norm * sum;
                if c.abs() >= zero_eps {
                    let mut new_state = state.clone();
                    new_state[pos] = j_kp as u8;
                    let entry = next.entry(new_state).or_insert(0.0);
                    *entry += coeff * c;
                }
            }

            j_kp += 2;
        }
    }

    next
}

fn apply_pair_operator(
    alpha: &[u8],
    i: usize,
    j: usize,
    two_s_list: &[i32],
    zero_eps: f64,
) -> AHashMap<Vec<u8>, f64> {
    let mut out: AHashMap<Vec<u8>, f64> = AHashMap::new();
    if i == j {
        return out;
    }

    let (i, j) = if i < j { (i, j) } else { (j, i) };
    let two_si = two_s_list[i];
    let two_sj = two_s_list[j];
    let n = alpha.len();

    let mut order: Vec<usize> = (0..n).collect();
    let mut state_map: AHashMap<Vec<u8>, f64> = AHashMap::new();
    state_map.insert(alpha.to_vec(), 1.0);

    let mut swaps: Vec<usize> = Vec::new();

    let mut pos_i = order.iter().position(|&v| v == i).unwrap();
    while pos_i > 0 {
        let pos = pos_i - 1;
        state_map = swap_at(&state_map, &order, pos, two_s_list, zero_eps);
        order.swap(pos, pos + 1);
        swaps.push(pos);
        pos_i -= 1;
    }

    let mut pos_j = order.iter().position(|&v| v == j).unwrap();
    while pos_j > 1 {
        let pos = pos_j - 1;
        state_map = swap_at(&state_map, &order, pos, two_s_list, zero_eps);
        order.swap(pos, pos + 1);
        swaps.push(pos);
        pos_j -= 1;
    }

    let mut diag_map: AHashMap<Vec<u8>, f64> = AHashMap::new();
    for (state, &coeff) in state_map.iter() {
        let two_k = state[1] as i32;
        let eig = 0.125
            * ((two_k * (two_k + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
        if eig.abs() < zero_eps {
            continue;
        }
        let entry = diag_map.entry(state.clone()).or_insert(0.0);
        *entry += coeff * eig;
    }

    for &pos in swaps.iter().rev() {
        diag_map = swap_at(&diag_map, &order, pos, two_s_list, zero_eps);
        order.swap(pos, pos + 1);
    }

    for (state, &coeff) in diag_map.iter() {
        if coeff.abs() < zero_eps {
            continue;
        }
        let entry = out.entry(state.clone()).or_insert(0.0);
        *entry += coeff;
    }

    out
}

pub fn make_su2_heisenberg_hamiltonian(
    basis: &SU2HeisenbergBasis,
    model: &SU2HeisenbergModel,
    num_threads: usize,
) -> Result<CsrMatrix> {
    let dim = basis.dim();
    if dim == 0 {
        bail!("Target Hilbert space has zero dimension.");
    }

    let n = basis.num_sites();
    if n == 0 {
        bail!("The system size is zero.");
    }

    let zero_eps = MATRIX_ZERO_EPS;

    with_pool(num_threads, || {
        let mut row_nnz = vec![0usize; dim];

        row_nnz
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| {
                let alpha = &basis.basis[row];
                let mut acc: AHashMap<usize, f64> = AHashMap::new();

                for (&(i, j), &val) in model.exchange.iter() {
                    let pair_map = apply_pair_operator(alpha, i, j, &basis.two_s_list, zero_eps);
                    for (beta, v) in pair_map.into_iter() {
                        if v.abs() < zero_eps {
                            continue;
                        }
                        if let Some(&col) = basis.inverse_basis.get(&beta) {
                            let entry = acc.entry(col).or_insert(0.0);
                            *entry += val * v;
                        }
                    }
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    let entry = acc.entry(row).or_insert(0.0);
                    *entry += model.diagonal_shift;
                }

                *slot = acc.values().filter(|v| v.abs() >= zero_eps).count();
            });

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

        row_slices
            .into_par_iter()
            .enumerate()
            .for_each(|(row, (row_cols, row_vals))| {
                let alpha = &basis.basis[row];
                let mut acc: AHashMap<usize, f64> = AHashMap::new();

                for (&(i, j), &val) in model.exchange.iter() {
                    let pair_map = apply_pair_operator(alpha, i, j, &basis.two_s_list, zero_eps);
                    for (beta, v) in pair_map.into_iter() {
                        if v.abs() < zero_eps {
                            continue;
                        }
                        if let Some(&col) = basis.inverse_basis.get(&beta) {
                            let entry = acc.entry(col).or_insert(0.0);
                            *entry += val * v;
                        }
                    }
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    let entry = acc.entry(row).or_insert(0.0);
                    *entry += model.diagonal_shift;
                }

                let mut entries: Vec<(usize, f64)> = acc
                    .into_iter()
                    .filter(|(_, v)| v.abs() >= zero_eps)
                    .collect();
                entries.sort_unstable_by_key(|&(c, _)| c);

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

    use crate::blas::lapack_dsyev;
    use crate::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
    use crate::model::{HeisenbergModel, QuantumModel};

    #[test]
    fn su2_ham_two_spin_half_singlet_triplet() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        let basis_singlet = model.build_basis(0.0).unwrap();
        let h_singlet = make_su2_heisenberg_hamiltonian(&basis_singlet, &model, 1).unwrap();
        assert_eq!(h_singlet.row_dim, 1);
        assert_eq!(h_singlet.col_dim, 1);
        assert!((h_singlet.vals[0] - (-0.75)).abs() < 1e-10);

        let basis_triplet = model.build_basis(1.0).unwrap();
        let h_triplet = make_su2_heisenberg_hamiltonian(&basis_triplet, &model, 1).unwrap();
        assert_eq!(h_triplet.row_dim, 1);
        assert_eq!(h_triplet.col_dim, 1);
        assert!((h_triplet.vals[0] - 0.25).abs() < 1e-10);
    }

    fn get_ground_state_energy(h: &CsrMatrix) -> f64 {
        let n = h.row_dim;
        if n == 0 {
            return f64::INFINITY;
        }
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];
        lapack_dsyev(h, &mut a_work, &mut w_work, &mut work).unwrap();
        w_work[0]
    }

    #[test]
    fn compare_su2_u1_six_sites_spin_half_long_range() {
        let n = 6;
        let spin = 0.5;
        let two_s = 1;

        // Long-range interactions with various distances
        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();

        // All pairs with different coupling strengths
        let pairs = [
            ((0, 1), 1.0),
            ((1, 2), 0.9),
            ((2, 3), 1.1),
            ((3, 4), 0.8),
            ((4, 5), 1.2),
            ((0, 2), 0.5),
            ((1, 3), 0.4),
            ((2, 4), 0.6),
            ((3, 5), 0.5),
            ((0, 3), 0.3),
            ((1, 4), 0.35),
            ((2, 5), 0.25),
            ((0, 4), 0.2),
            ((1, 5), 0.15),
            ((0, 5), 0.1),
        ];
        for ((i, j), val) in pairs {
            exchange_su2.insert((i, j), val);
            exchange_xy.insert((i, j), val);
            exchange_z.insert((i, j), val);
        }

        // SU(2) model
        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();

        // U(1) model
        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        // U(1): ground state from Sz=0 sector
        let u1_basis = u1_model.build_basis(&[0]).unwrap();
        let u1_ham = make_heisenberg_hamiltonian(&u1_basis, &u1_model, 1).unwrap();
        let u1_gs = get_ground_state_energy(&u1_ham);

        // SU(2): ground state from S=0 sector
        let su2_basis = su2_model.build_basis(0.0).unwrap();
        let su2_ham = make_su2_heisenberg_hamiltonian(&su2_basis, &su2_model, 1).unwrap();
        let su2_gs = get_ground_state_energy(&su2_ham);

        assert!(
            (u1_gs - su2_gs).abs() < 1e-8,
            "Ground state mismatch: U(1)={}, SU(2)={}",
            u1_gs,
            su2_gs
        );
    }

    #[test]
    fn compare_su2_u1_six_sites_spin_one_long_range() {
        let n = 6;
        let spin = 1.0;
        let two_s = 2;

        // Long-range interactions with various distances (all pairs)
        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();

        // All pairs with different coupling strengths
        let pairs = [
            ((0, 1), 1.0),
            ((1, 2), 0.9),
            ((2, 3), 1.1),
            ((3, 4), 0.8),
            ((4, 5), 1.2),
            ((0, 2), 0.5),
            ((1, 3), 0.4),
            ((2, 4), 0.6),
            ((3, 5), 0.5),
            ((0, 3), 0.3),
            ((1, 4), 0.35),
            ((2, 5), 0.25),
            ((0, 4), 0.2),
            ((1, 5), 0.15),
            ((0, 5), 0.1),
        ];
        for ((i, j), val) in pairs {
            exchange_su2.insert((i, j), val);
            exchange_xy.insert((i, j), val);
            exchange_z.insert((i, j), val);
        }

        // SU(2) model
        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();

        // U(1) model
        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        // U(1): ground state from Sz=0 sector
        let u1_basis = u1_model.build_basis(&[0]).unwrap();
        let u1_ham = make_heisenberg_hamiltonian(&u1_basis, &u1_model, 1).unwrap();
        let u1_gs = get_ground_state_energy(&u1_ham);

        // SU(2): ground state from S=0 sector
        let su2_basis = su2_model.build_basis(0.0).unwrap();
        let su2_ham = make_su2_heisenberg_hamiltonian(&su2_basis, &su2_model, 1).unwrap();
        let su2_gs = get_ground_state_energy(&su2_ham);

        assert!(
            (u1_gs - su2_gs).abs() < 1e-8,
            "Ground state mismatch: U(1)={}, SU(2)={}",
            u1_gs,
            su2_gs
        );
    }
}
