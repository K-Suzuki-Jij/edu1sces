use ahash::AHashMap;
use anyhow::{bail, Result};
use std::sync::Arc;

use crate::basis::SU2HeisenbergBasis;
use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::model::SU2HeisenbergModel;
use crate::utility::rayon_pool::with_pool;
use crate::utility::sixj_table::{get_cached_sixj_table, Sixj6Table};
use rayon::prelude::*;
use wigner_symbols::Wigner6j;

#[derive(Clone, Copy, Debug)]
struct SwapStep {
    pos: usize,
    a: i32,
    b: i32,
}

#[derive(Clone, Debug)]
struct PairPlan {
    two_si: i32,
    two_sj: i32,
    /// Position where the adjacent pair matrix is applied (j-1 in original indices)
    adjacent_pos: usize,
    forward: Vec<SwapStep>,
    reverse: Vec<SwapStep>,
}

struct PairInfo {
    site_i: usize,
    site_j: usize,
    val: f64,
    pos_j: usize,
}

fn swap_coeffs(
    x: i32,
    j_k: i32,
    j_k1: i32,
    a: i32,
    b: i32,
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    zero_eps: f64,
) -> Vec<(i32, f64)> {
    let mut out = Vec::new();
    let min_j = (x - b).abs();
    let max_j = x + b;

    let mut j_kp = min_j;
    if ((j_kp + x + b) & 1) != 0 {
        j_kp += 1;
    }

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
            let s1 = if let Some((table1, _)) = tables {
                table1.get(x, j_k, j_k1, t)
            } else {
                f64::from(
                    Wigner6j {
                        tj1: x,
                        tj2: a,
                        tj3: j_k,
                        tj4: b,
                        tj5: j_k1,
                        tj6: t,
                    }
                    .value(),
                )
            };
            if s1.abs() >= zero_eps {
                let s2 = if let Some((_, table2)) = tables {
                    table2.get(x, j_kp, j_k1, t)
                } else {
                    f64::from(
                        Wigner6j {
                            tj1: x,
                            tj2: b,
                            tj3: j_kp,
                            tj4: a,
                            tj5: j_k1,
                            tj6: t,
                        }
                        .value(),
                    )
                };
                if s2.abs() >= zero_eps {
                    let phase = 1.0 - 2.0 * (((a + b - t) / 2) & 1) as f64;
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
                out.push((j_kp, c));
            }
        }

        j_kp += 2;
    }

    out
}

/// Build a plan to compute S_i · S_j matrix elements.
fn build_pair_plan(
    two_s_list: &[i32],
    coupling_order: &[usize],
    site_to_pos: &[usize],
    site_i: usize,
    site_j: usize,
) -> PairPlan {
    let n = coupling_order.len();

    let pos_i = site_to_pos[site_i];
    let pos_j = site_to_pos[site_j];
    let (i, j) = if pos_i < pos_j {
        (pos_i, pos_j)
    } else {
        (pos_j, pos_i)
    };

    let mut order = (0..n).collect::<Vec<_>>();
    let mut forward = Vec::new();

    let mut cur_pos = i;
    while cur_pos < j - 1 {
        let pos = cur_pos;
        let a = two_s_list[coupling_order[order[pos]]];
        let b = two_s_list[coupling_order[order[pos + 1]]];
        forward.push(SwapStep { pos, a, b });
        order.swap(pos, pos + 1);
        cur_pos += 1;
    }

    let mut reverse = Vec::with_capacity(forward.len());
    for step in forward.iter().rev() {
        reverse.push(SwapStep {
            pos: step.pos,
            a: step.b,
            b: step.a,
        });
    }

    PairPlan {
        two_si: two_s_list[site_i],
        two_sj: two_s_list[site_j],
        adjacent_pos: j - 1,
        forward,
        reverse,
    }
}

/// Compute adjacent pair coefficients and output new prefixes with coefficients
fn compute_adjacent_transitions(
    prefix: &[u8],
    adjacent_pos: usize,
    two_si: i32,
    two_sj: i32,
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    zero_eps: f64,
    out: &mut Vec<(Vec<u8>, f64)>,
) {
    let min_k = (two_si - two_sj).abs();
    let max_k = two_si + two_sj;

    if adjacent_pos == 0 {
        let j2 = prefix[1] as i32;
        let eig = 0.125 * ((j2 * (j2 + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
        if eig.abs() >= zero_eps {
            out.push((prefix.to_vec(), eig));
        }
        return;
    }

    let j_km1 = prefix[adjacent_pos - 1] as i32;
    let j_k = prefix[adjacent_pos] as i32;
    let j_k1 = prefix[adjacent_pos + 1] as i32;

    let min_jk_prime = (j_km1 - two_si).abs().max((j_k1 - two_sj).abs());
    let max_jk_prime = (j_km1 + two_si).min(j_k1 + two_sj);

    let mut jk_prime = min_jk_prime;
    if ((jk_prime + j_km1 + two_si) & 1) != 0 {
        jk_prime += 1;
    }

    while jk_prime <= max_jk_prime {
        let mut sum = 0.0;
        let mut k = min_k;
        if ((k + two_si + two_sj) & 1) != 0 {
            k += 1;
        }
        while k <= max_k {
            let eig_k =
                0.125 * ((k * (k + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
            let sixj1 = if let Some((table1, _)) = tables {
                table1.get(j_km1, j_k, j_k1, k)
            } else {
                f64::from(
                    Wigner6j {
                        tj1: j_km1,
                        tj2: two_si,
                        tj3: j_k,
                        tj4: two_sj,
                        tj5: j_k1,
                        tj6: k,
                    }
                    .value(),
                )
            };
            if sixj1.abs() < zero_eps {
                k += 2;
                continue;
            }
            let sixj2 = if let Some((table1, _)) = tables {
                table1.get(j_km1, jk_prime, j_k1, k)
            } else {
                f64::from(
                    Wigner6j {
                        tj1: j_km1,
                        tj2: two_si,
                        tj3: jk_prime,
                        tj4: two_sj,
                        tj5: j_k1,
                        tj6: k,
                    }
                    .value(),
                )
            };
            if sixj2.abs() >= zero_eps {
                let weight = (k + 1) as f64;
                sum += weight * eig_k * sixj1 * sixj2;
            }
            k += 2;
        }

        if sum.abs() >= zero_eps {
            let norm = ((j_k + 1) as f64 * (jk_prime + 1) as f64).sqrt();
            let coeff = norm * sum;
            if coeff.abs() >= zero_eps {
                let mut new_prefix = prefix.to_vec();
                new_prefix[adjacent_pos] = jk_prime as u8;
                out.push((new_prefix, coeff));
            }
        }
        jk_prime += 2;
    }
}

/// Compute swap coefficients and output new prefixes with coefficients
fn compute_swap_transitions(
    prefix: &[u8],
    pos: usize,
    a: i32,
    b: i32,
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    zero_eps: f64,
    out: &mut Vec<(Vec<u8>, f64)>,
) {
    if pos == 0 {
        let j1 = prefix[1] as i32;
        let phase = 1.0 - 2.0 * (((a + b - j1) / 2) & 1) as f64;
        if phase.abs() >= zero_eps {
            let mut new_prefix = prefix.to_vec();
            new_prefix[0] = b as u8;
            out.push((new_prefix, phase));
        }
        return;
    }

    let x = prefix[pos - 1] as i32;
    let j_k = prefix[pos] as i32;
    let j_k1 = prefix[pos + 1] as i32;
    let coeffs = swap_coeffs(x, j_k, j_k1, a, b, tables, zero_eps);

    for &(j_kp, c) in coeffs.iter() {
        if c.abs() >= zero_eps {
            let mut new_prefix = prefix.to_vec();
            new_prefix[pos] = j_kp as u8;
            out.push((new_prefix, c));
        }
    }
}

/// Compute transitions for a single row and accumulate to final col indices
fn compute_row_transitions_and_accumulate(
    state: &[u8],
    pos_j: usize,
    plan: &PairPlan,
    inverse_basis: &AHashMap<Vec<u8>, usize>,
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    val: f64,
    acc: &mut AHashMap<usize, f64>,
    prefix_acc: &mut AHashMap<Vec<u8>, f64>,
    buf_a: &mut Vec<(Vec<u8>, f64)>,
    buf_b: &mut Vec<(Vec<u8>, f64)>,
    full_state_buf: &mut Vec<u8>,
    zero_eps: f64,
) {
    let prefix = &state[..=pos_j];
    let suffix = &state[pos_j + 1..];
    let adjacent_pos = plan.adjacent_pos;

    // Fast path: adjacent pairs have no swaps - avoid prefix allocation
    if plan.forward.is_empty() {
        let two_si = plan.two_si;
        let two_sj = plan.two_sj;
        let min_k = (two_si - two_sj).abs();
        let max_k = two_si + two_sj;

        if adjacent_pos == 0 {
            // Special case: positions 0, 1
            let j2 = prefix[1] as i32;
            let eig =
                0.125 * ((j2 * (j2 + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
            if eig.abs() >= zero_eps {
                // Diagonal element - same state
                if let Some(&col) = inverse_basis.get(state) {
                    *acc.entry(col).or_insert(0.0) += val * eig;
                }
            }
            return;
        }

        let j_km1 = prefix[adjacent_pos - 1] as i32;
        let j_k = prefix[adjacent_pos] as i32;
        let j_k1 = prefix[adjacent_pos + 1] as i32;

        let min_jk_prime = (j_km1 - two_si).abs().max((j_k1 - two_sj).abs());
        let max_jk_prime = (j_km1 + two_si).min(j_k1 + two_sj);

        let mut jk_prime = min_jk_prime;
        if ((jk_prime + j_km1 + two_si) & 1) != 0 {
            jk_prime += 1;
        }

        // Reuse full_state_buf for all transitions
        full_state_buf.clear();
        full_state_buf.extend_from_slice(state);

        while jk_prime <= max_jk_prime {
            let mut sum = 0.0;
            let mut k = min_k;
            if ((k + two_si + two_sj) & 1) != 0 {
                k += 1;
            }
            while k <= max_k {
                let eig_k =
                    0.125 * ((k * (k + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
                let sixj1 = if let Some((table1, _)) = tables {
                    table1.get(j_km1, j_k, j_k1, k)
                } else {
                    f64::from(
                        Wigner6j {
                            tj1: j_km1,
                            tj2: two_si,
                            tj3: j_k,
                            tj4: two_sj,
                            tj5: j_k1,
                            tj6: k,
                        }
                        .value(),
                    )
                };
                if sixj1.abs() < zero_eps {
                    k += 2;
                    continue;
                }
                let sixj2 = if let Some((table1, _)) = tables {
                    table1.get(j_km1, jk_prime, j_k1, k)
                } else {
                    f64::from(
                        Wigner6j {
                            tj1: j_km1,
                            tj2: two_si,
                            tj3: jk_prime,
                            tj4: two_sj,
                            tj5: j_k1,
                            tj6: k,
                        }
                        .value(),
                    )
                };
                if sixj2.abs() >= zero_eps {
                    let weight = (k + 1) as f64;
                    sum += weight * eig_k * sixj1 * sixj2;
                }
                k += 2;
            }

            if sum.abs() >= zero_eps {
                let norm = ((j_k + 1) as f64 * (jk_prime + 1) as f64).sqrt();
                let coeff = norm * sum;
                if coeff.abs() >= zero_eps {
                    full_state_buf[adjacent_pos] = jk_prime as u8;
                    if let Some(&col) = inverse_basis.get(full_state_buf.as_slice()) {
                        *acc.entry(col).or_insert(0.0) += val * coeff;
                    }
                }
            }
            jk_prime += 2;
        }
        // Restore original value
        full_state_buf[adjacent_pos] = j_k as u8;
        return;
    }

    // Slow path: non-adjacent pairs require swap chain
    buf_a.clear();
    buf_a.push((prefix.to_vec(), 1.0));

    // Slow path: non-adjacent pairs require swap chain
    prefix_acc.clear();

    // Apply forward swaps
    for step in plan.forward.iter() {
        for (p, coeff) in buf_a.drain(..) {
            if coeff.abs() < zero_eps {
                continue;
            }
            buf_b.clear();
            compute_swap_transitions(&p, step.pos, step.a, step.b, tables, zero_eps, buf_b);
            for (new_p, w) in buf_b.drain(..) {
                let v = coeff * w;
                if v.abs() >= zero_eps {
                    *prefix_acc.entry(new_p).or_insert(0.0) += v;
                }
            }
        }
        buf_a.clear();
        for (p, v) in prefix_acc.drain() {
            if v.abs() >= zero_eps {
                buf_a.push((p, v));
            }
        }
        if buf_a.is_empty() {
            return;
        }
    }

    // Apply adjacent pair
    for (p, coeff) in buf_a.drain(..) {
        if coeff.abs() < zero_eps {
            continue;
        }
        buf_b.clear();
        compute_adjacent_transitions(
            &p,
            plan.adjacent_pos,
            plan.two_si,
            plan.two_sj,
            tables,
            zero_eps,
            buf_b,
        );
        for (new_p, w) in buf_b.drain(..) {
            let v = coeff * w;
            if v.abs() >= zero_eps {
                *prefix_acc.entry(new_p).or_insert(0.0) += v;
            }
        }
    }
    buf_a.clear();
    for (p, v) in prefix_acc.drain() {
        if v.abs() >= zero_eps {
            buf_a.push((p, v));
        }
    }
    if buf_a.is_empty() {
        return;
    }

    // Apply reverse swaps
    for step in plan.reverse.iter() {
        for (p, coeff) in buf_a.drain(..) {
            if coeff.abs() < zero_eps {
                continue;
            }
            buf_b.clear();
            compute_swap_transitions(&p, step.pos, step.a, step.b, tables, zero_eps, buf_b);
            for (new_p, w) in buf_b.drain(..) {
                let v = coeff * w;
                if v.abs() >= zero_eps {
                    *prefix_acc.entry(new_p).or_insert(0.0) += v;
                }
            }
        }
        buf_a.clear();
        for (p, v) in prefix_acc.drain() {
            if v.abs() >= zero_eps {
                buf_a.push((p, v));
            }
        }
        if buf_a.is_empty() {
            return;
        }
    }

    // Accumulate final results
    for (p, v) in buf_a.drain(..) {
        if v.abs() >= zero_eps {
            full_state_buf.clear();
            full_state_buf.extend_from_slice(&p);
            full_state_buf.extend_from_slice(suffix);
            if let Some(&col) = inverse_basis.get(full_state_buf.as_slice()) {
                *acc.entry(col).or_insert(0.0) += val * v;
            }
        }
    }
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
    let site_to_pos = &basis.site_to_pos;

    with_pool(num_threads, || {
        let exchange: Vec<_> = model.exchange.iter().map(|(&k, &v)| (k, v)).collect();

        // Check if all spins are uniform (for 6j table optimization)
        let is_uniform_spin = {
            let first_s = basis.two_s_list.first().copied().unwrap_or(0);
            basis.two_s_list.iter().all(|&s| s == first_s)
        };

        // Compute max_two_j for 6j table bounds
        let max_in_basis = basis
            .basis
            .iter()
            .flat_map(|s| s.iter().map(|&v| v as i32))
            .max()
            .unwrap_or(0);
        let max_two_s = basis.two_s_list.iter().copied().max().unwrap_or(0);
        let max_two_j = max_in_basis + max_two_s;

        // Pre-build 6j tables for uniform spin case (cached)
        let tables: Option<(Arc<Sixj6Table>, Arc<Sixj6Table>)> = if is_uniform_spin && max_two_s > 0
        {
            let a = max_two_s;
            let b = max_two_s;
            Some((
                get_cached_sixj_table(a, b, max_two_j),
                get_cached_sixj_table(b, a, max_two_j),
            ))
        } else {
            None
        };
        let tables_ref = tables.as_ref().map(|(t1, t2)| (t1.as_ref(), t2.as_ref()));

        // Build pair info list
        let pair_infos: Vec<PairInfo> = exchange
            .iter()
            .map(|&((site_i, site_j), val)| {
                let pos_j = site_to_pos[site_i].max(site_to_pos[site_j]);
                PairInfo {
                    site_i,
                    site_j,
                    val,
                    pos_j,
                }
            })
            .collect();

        // Build plans for each pair
        let plans: Vec<PairPlan> = pair_infos
            .iter()
            .map(|info| {
                build_pair_plan(
                    &basis.two_s_list,
                    &basis.coupling_order,
                    &site_to_pos,
                    info.site_i,
                    info.site_j,
                )
            })
            .collect();

        let mut row_nnz = vec![0usize; dim];

        row_nnz.par_iter_mut().enumerate().for_each_init(
            || {
                let acc = AHashMap::new();
                let prefix_acc = AHashMap::new();
                let buf_a = Vec::new();
                let buf_b = Vec::new();
                let full_state_buf = Vec::new();
                (acc, prefix_acc, buf_a, buf_b, full_state_buf)
            },
            |(acc, prefix_acc, buf_a, buf_b, full_state_buf), (row, slot)| {
                acc.clear();

                let state = &basis.basis[row];
                for (info, plan) in pair_infos.iter().zip(plans.iter()) {
                    compute_row_transitions_and_accumulate(
                        state,
                        info.pos_j,
                        plan,
                        &basis.inverse_basis,
                        tables_ref,
                        info.val,
                        acc,
                        prefix_acc,
                        buf_a,
                        buf_b,
                        full_state_buf,
                        zero_eps,
                    );
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    *acc.entry(row).or_insert(0.0) += model.diagonal_shift;
                }

                let mut count = 0usize;
                for (_, v) in acc.iter() {
                    if v.abs() >= zero_eps {
                        count += 1;
                    }
                }
                *slot = count;
            },
        );

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
        let mut row_slices = Vec::with_capacity(dim);
        for row in 0..dim {
            let len = row_nnz[row];
            let (c_head, c_tail) = cols_rem.split_at_mut(len);
            let (v_head, v_tail) = vals_rem.split_at_mut(len);
            row_slices.push((c_head, v_head));
            cols_rem = c_tail;
            vals_rem = v_tail;
        }
        drop(row_nnz);

        row_slices.into_par_iter().enumerate().for_each_init(
            || {
                let acc = AHashMap::new();
                let prefix_acc = AHashMap::new();
                let buf_a = Vec::new();
                let buf_b = Vec::new();
                let full_state_buf = Vec::new();
                let sorted_cols = Vec::new();
                (acc, prefix_acc, buf_a, buf_b, full_state_buf, sorted_cols)
            },
            |(acc, prefix_acc, buf_a, buf_b, full_state_buf, sorted_cols),
             (row, (row_cols, row_vals))| {
                acc.clear();

                let state = &basis.basis[row];
                for (info, plan) in pair_infos.iter().zip(plans.iter()) {
                    compute_row_transitions_and_accumulate(
                        state,
                        info.pos_j,
                        plan,
                        &basis.inverse_basis,
                        tables_ref,
                        info.val,
                        acc,
                        prefix_acc,
                        buf_a,
                        buf_b,
                        full_state_buf,
                        zero_eps,
                    );
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    *acc.entry(row).or_insert(0.0) += model.diagonal_shift;
                }

                sorted_cols.clear();
                for (&col, &v) in acc.iter() {
                    if v.abs() >= zero_eps {
                        sorted_cols.push(col);
                    }
                }
                sorted_cols.sort_unstable();

                let mut k = 0usize;
                for &col in sorted_cols.iter() {
                    let v = *acc.get(&col).unwrap();
                    row_cols[k] = col;
                    row_vals[k] = v;
                    k += 1;
                }
            },
        );

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

        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();

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

        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();

        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        let u1_basis = u1_model.build_basis(&[0]).unwrap();
        let u1_ham = make_heisenberg_hamiltonian(&u1_basis, &u1_model, 1).unwrap();
        let u1_gs = get_ground_state_energy(&u1_ham);

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

        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();

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

        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();

        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        let u1_basis = u1_model.build_basis(&[0]).unwrap();
        let u1_ham = make_heisenberg_hamiltonian(&u1_basis, &u1_model, 1).unwrap();
        let u1_gs = get_ground_state_energy(&u1_ham);

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

    fn get_all_eigenvalues(h: &CsrMatrix) -> Vec<f64> {
        let n = h.row_dim;
        if n == 0 {
            return Vec::new();
        }
        let mut a_work = vec![0.0; n * n];
        let mut w_work = vec![0.0; n];
        let mut work = vec![0.0; 3 * n];
        lapack_dsyev(h, &mut a_work, &mut w_work, &mut work).unwrap();
        w_work
    }

    #[test]
    fn compare_all_eigenvalues_six_sites_spin_half() {
        let n = 6;
        let spin = 0.5;
        let two_s = 1;

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

        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        for ((i, j), val) in pairs {
            exchange_su2.insert((i, j), val);
            exchange_xy.insert((i, j), val);
            exchange_z.insert((i, j), val);
        }

        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();
        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        let mut u1_all_eigs = Vec::new();
        for sz2 in (-(n as i32)..=(n as i32)).step_by(2) {
            let basis = u1_model.build_basis(&[sz2]).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_heisenberg_hamiltonian(&basis, &u1_model, 1).unwrap();
            u1_all_eigs.extend(get_all_eigenvalues(&ham));
        }

        let mut su2_all_eigs = Vec::new();
        for s2 in (0..=(n as i32)).step_by(2) {
            let s = s2 as f64 / 2.0;
            let basis = su2_model.build_basis(s).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_su2_heisenberg_hamiltonian(&basis, &su2_model, 1).unwrap();
            let eigs = get_all_eigenvalues(&ham);
            let degeneracy = (s2 + 1) as usize;
            for e in eigs {
                for _ in 0..degeneracy {
                    su2_all_eigs.push(e);
                }
            }
        }

        u1_all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        su2_all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            u1_all_eigs.len(),
            su2_all_eigs.len(),
            "Eigenvalue count mismatch: U(1)={}, SU(2)={}",
            u1_all_eigs.len(),
            su2_all_eigs.len()
        );

        for (i, (u1_e, su2_e)) in u1_all_eigs.iter().zip(&su2_all_eigs).enumerate() {
            assert!(
                (u1_e - su2_e).abs() < 1e-8,
                "Eigenvalue {} mismatch: U(1)={}, SU(2)={}",
                i,
                u1_e,
                su2_e
            );
        }
    }

    #[test]
    fn compare_all_eigenvalues_six_sites_spin_one() {
        let n = 6;
        let spin = 1.0;
        let two_s = 2;

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

        let mut exchange_su2 = HashMap::new();
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        for ((i, j), val) in pairs {
            exchange_su2.insert((i, j), val);
            exchange_xy.insert((i, j), val);
            exchange_z.insert((i, j), val);
        }

        let su2_model = SU2HeisenbergModel::new(vec![spin; n], exchange_su2).unwrap();
        let u1_model = HeisenbergModel {
            num_sites: n,
            two_s_list: vec![two_s; n],
            hz_list: vec![0.0; n],
            d_list: vec![0.0; n],
            exchange_xy,
            exchange_z,
        };

        let mut u1_all_eigs = Vec::new();
        for sz2 in (-(n as i32 * two_s)..=(n as i32 * two_s)).step_by(2) {
            let basis = u1_model.build_basis(&[sz2]).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_heisenberg_hamiltonian(&basis, &u1_model, 1).unwrap();
            u1_all_eigs.extend(get_all_eigenvalues(&ham));
        }

        let mut su2_all_eigs = Vec::new();
        let max_s2 = n as i32 * two_s;
        for s2 in (0..=max_s2).step_by(2) {
            let s = s2 as f64 / 2.0;
            let basis = su2_model.build_basis(s).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_su2_heisenberg_hamiltonian(&basis, &su2_model, 1).unwrap();
            let eigs = get_all_eigenvalues(&ham);
            let degeneracy = (s2 + 1) as usize;
            for e in eigs {
                for _ in 0..degeneracy {
                    su2_all_eigs.push(e);
                }
            }
        }

        u1_all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        su2_all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            u1_all_eigs.len(),
            su2_all_eigs.len(),
            "Eigenvalue count mismatch: U(1)={}, SU(2)={}",
            u1_all_eigs.len(),
            su2_all_eigs.len()
        );

        for (i, (u1_e, su2_e)) in u1_all_eigs.iter().zip(&su2_all_eigs).enumerate() {
            assert!(
                (u1_e - su2_e).abs() < 1e-8,
                "Eigenvalue {} mismatch: U(1)={}, SU(2)={}",
                i,
                u1_e,
                su2_e
            );
        }
    }
}
