use ahash::AHashMap;
use anyhow::{bail, Result};
use std::cell::RefCell;
use std::sync::Arc;

use crate::basis::SU2HeisenbergBasis;
use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::model::SU2HeisenbergModel;
use crate::utility::rayon_pool::with_pool;
use rayon::prelude::*;
use wigner_symbols::Wigner6j;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SwapKey {
    x: i32,
    j_k: i32,
    j_k1: i32,
    a: i32,
    b: i32,
}

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
    forward: Vec<SwapStep>,
    reverse: Vec<SwapStep>,
}

struct PairGroup {
    prefix_cols: Arc<Vec<usize>>,
    transitions: Vec<Vec<(usize, f64)>>,
}

struct PairTable {
    val: f64,
    row_group_ids: Arc<Vec<usize>>,
    row_prefix_ids: Arc<Vec<usize>>,
    groups: Vec<PairGroup>,
}

struct SuffixIndex {
    row_group_ids: Arc<Vec<usize>>,
    row_prefix_ids: Arc<Vec<usize>>,
    group_local_maps: Vec<AHashMap<usize, usize>>,
    group_prefix_ids: Vec<Vec<usize>>,
    prefix_cols: Vec<Arc<Vec<usize>>>,
    global_prefixes: Vec<Vec<u8>>,
    global_prefix_map: AHashMap<Vec<u8>, usize>,
    swap_mats: AHashMap<(usize, i32, i32), Vec<Vec<(usize, f64)>>>,
}

thread_local! {
    static SWAP_CACHE: RefCell<AHashMap<SwapKey, Vec<(i32, f64)>>> =
        RefCell::new(AHashMap::new());
}

fn phase_from_half_integer(sum_two: i32) -> f64 {
    if ((sum_two / 2) & 1) != 0 {
        -1.0
    } else {
        1.0
    }
}

fn swap_coeffs(x: i32, j_k: i32, j_k1: i32, a: i32, b: i32, zero_eps: f64) -> Vec<(i32, f64)> {
    let key = SwapKey { x, j_k, j_k1, a, b };

    SWAP_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(v) = cache.get(&key) {
            return v.clone();
        }

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
                    out.push((j_kp, c));
                }
            }

            j_kp += 2;
        }

        cache.insert(key, out.clone());
        out
    })
}

fn build_pair_plan(n: usize, two_s_list: &[i32], i: usize, j: usize) -> PairPlan {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    let mut order: Vec<usize> = (0..n).collect();
    let mut forward: Vec<SwapStep> = Vec::new();

    let mut pos_i = order.iter().position(|&v| v == i).unwrap();
    while pos_i > 0 {
        let pos = pos_i - 1;
        let a = two_s_list[order[pos]];
        let b = two_s_list[order[pos + 1]];
        forward.push(SwapStep { pos, a, b });
        order.swap(pos, pos + 1);
        pos_i -= 1;
    }

    let mut pos_j = order.iter().position(|&v| v == j).unwrap();
    while pos_j > 1 {
        let pos = pos_j - 1;
        let a = two_s_list[order[pos]];
        let b = two_s_list[order[pos + 1]];
        forward.push(SwapStep { pos, a, b });
        order.swap(pos, pos + 1);
        pos_j -= 1;
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
        two_si: two_s_list[i],
        two_sj: two_s_list[j],
        forward,
        reverse,
    }
}

fn build_swap_matrix(
    prefixes: &[Vec<u8>],
    prefix_map: &AHashMap<Vec<u8>, usize>,
    pos: usize,
    a: i32,
    b: i32,
    zero_eps: f64,
) -> Vec<Vec<(usize, f64)>> {
    let dim = prefixes.len();
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dim];

    for (pid, prefix) in prefixes.iter().enumerate() {
        if pos == 0 {
            let j1 = prefix[1] as i32;
            let phase = phase_from_half_integer(a + b - j1);
            let mut new_prefix = prefix.clone();
            new_prefix[0] = b as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if phase.abs() >= zero_eps {
                    out[pid].push((nid, phase));
                }
            }
            continue;
        }

        let x = prefix[pos - 1] as i32;
        let j_k = prefix[pos] as i32;
        let j_k1 = prefix[pos + 1] as i32;
        let coeffs = swap_coeffs(x, j_k, j_k1, a, b, zero_eps);

        let mut row = Vec::with_capacity(coeffs.len());
        for (j_kp, c) in coeffs {
            let mut new_prefix = prefix.clone();
            new_prefix[pos] = j_kp as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if c.abs() >= zero_eps {
                    row.push((nid, c));
                }
            }
        }
        out[pid] = row;
    }

    out
}

fn apply_step_vec(
    vec_in: &[(usize, f64)],
    step_mat: &[Vec<(usize, f64)>],
    acc: &mut [f64],
    touched: &mut Vec<usize>,
    out: &mut Vec<(usize, f64)>,
    zero_eps: f64,
) {
    touched.clear();
    out.clear();
    for &(id, coeff) in vec_in.iter() {
        if coeff.abs() < zero_eps {
            continue;
        }
        for &(nid, w) in step_mat[id].iter() {
            let v = coeff * w;
            if v.abs() < zero_eps {
                continue;
            }
            if acc[nid].abs() < zero_eps {
                touched.push(nid);
            }
            acc[nid] += v;
        }
    }
    if touched.is_empty() {
        return;
    }
    out.reserve(touched.len());
    for &nid in touched.iter() {
        let v = acc[nid];
        acc[nid] = 0.0;
        if v.abs() >= zero_eps {
            out.push((nid, v));
        }
    }
}

fn build_pair_transitions_global_dp(
    prefixes: &[Vec<u8>],
    plan: &PairPlan,
    swap_mats: &AHashMap<(usize, i32, i32), Vec<Vec<(usize, f64)>>>,
    zero_eps: f64,
) -> Vec<Vec<(usize, f64)>> {
    let dim = prefixes.len();
    let two_k_list: Vec<i32> = prefixes.iter().map(|p| p[1] as i32).collect();

    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dim];

    out.par_iter_mut().enumerate().for_each_init(
        || {
            (
                vec![0.0f64; dim],
                Vec::<usize>::new(),
                Vec::<(usize, f64)>::new(),
                Vec::<(usize, f64)>::new(),
                Vec::<(usize, f64)>::new(),
            )
        },
        |(acc, touched, buf_a, buf_b, buf_diag), (pid, out_row)| {
            buf_a.clear();
            buf_a.push((pid, 1.0));

            for step in plan.forward.iter() {
                let step_mat = swap_mats
                    .get(&(step.pos, step.a, step.b))
                    .expect("swap matrix missing");
                apply_step_vec(buf_a, step_mat, acc, touched, buf_b, zero_eps);
                std::mem::swap(buf_a, buf_b);
                if buf_a.is_empty() {
                    return;
                }
            }

            buf_diag.clear();
            buf_diag.reserve(buf_a.len());
            for &(id, coeff) in buf_a.iter() {
                let two_k = two_k_list[id];
                let eig = 0.125
                    * ((two_k * (two_k + 2)
                        - plan.two_si * (plan.two_si + 2)
                        - plan.two_sj * (plan.two_sj + 2)) as f64);
                let v = coeff * eig;
                if v.abs() >= zero_eps {
                    buf_diag.push((id, v));
                }
            }

            if buf_diag.is_empty() {
                return;
            }

            buf_a.clear();
            buf_a.extend_from_slice(buf_diag);
            for step in plan.reverse.iter() {
                let step_mat = swap_mats
                    .get(&(step.pos, step.a, step.b))
                    .expect("swap matrix missing");
                apply_step_vec(buf_a, step_mat, acc, touched, buf_b, zero_eps);
                std::mem::swap(buf_a, buf_b);
                if buf_a.is_empty() {
                    return;
                }
            }

            *out_row = std::mem::take(buf_a);
        },
    );

    out
}

fn build_suffix_index(basis: &SU2HeisenbergBasis, j: usize) -> SuffixIndex {
    let dim = basis.dim();

    let mut group_map: AHashMap<Vec<u8>, usize> = AHashMap::new();
    let mut group_local_maps: Vec<AHashMap<usize, usize>> = Vec::new();
    let mut group_prefix_ids: Vec<Vec<usize>> = Vec::new();
    let mut group_suffixes: Vec<Vec<u8>> = Vec::new();
    let mut row_group_ids = vec![0usize; dim];
    let mut row_prefix_ids = vec![0usize; dim];

    let mut global_prefix_map: AHashMap<Vec<u8>, usize> = AHashMap::new();
    let mut global_prefixes: Vec<Vec<u8>> = Vec::new();

    for (row, state) in basis.basis.iter().enumerate() {
        let suffix = state[j + 1..].to_vec();
        let gid = if let Some(&gid) = group_map.get(&suffix) {
            gid
        } else {
            let gid = group_map.len();
            group_map.insert(suffix.clone(), gid);
            group_local_maps.push(AHashMap::new());
            group_prefix_ids.push(Vec::new());
            group_suffixes.push(suffix);
            gid
        };

        let prefix = state[..=j].to_vec();
        let gpid = if let Some(&pid) = global_prefix_map.get(&prefix) {
            pid
        } else {
            let pid = global_prefixes.len();
            global_prefixes.push(prefix);
            global_prefix_map.insert(global_prefixes[pid].clone(), pid);
            pid
        };

        let local_map = &mut group_local_maps[gid];
        let local_prefixes = &mut group_prefix_ids[gid];
        let pid = if let Some(&pid) = local_map.get(&gpid) {
            pid
        } else {
            let pid = local_prefixes.len();
            local_prefixes.push(gpid);
            local_map.insert(gpid, pid);
            pid
        };

        row_group_ids[row] = gid;
        row_prefix_ids[row] = pid;
    }

    let mut prefix_cols: Vec<Arc<Vec<usize>>> = Vec::with_capacity(group_prefix_ids.len());
    for (gid, local_prefixes) in group_prefix_ids.iter().enumerate() {
        let suffix = &group_suffixes[gid];
        let mut cols: Vec<usize> = Vec::with_capacity(local_prefixes.len());
        for &gpid in local_prefixes.iter() {
            let prefix = &global_prefixes[gpid];
            let mut full = Vec::with_capacity(prefix.len() + suffix.len());
            full.extend_from_slice(prefix);
            full.extend_from_slice(suffix);
            let col = *basis
                .inverse_basis
                .get(&full)
                .expect("state not found in inverse basis");
            cols.push(col);
        }
        prefix_cols.push(Arc::new(cols));
    }

    SuffixIndex {
        row_group_ids: Arc::new(row_group_ids),
        row_prefix_ids: Arc::new(row_prefix_ids),
        group_local_maps,
        group_prefix_ids,
        prefix_cols,
        global_prefixes,
        global_prefix_map,
        swap_mats: AHashMap::new(),
    }
}

fn fill_swap_mats(index: &mut SuffixIndex, plans: &[PairPlan], zero_eps: f64) {
    let mut steps: AHashMap<(usize, i32, i32), ()> = AHashMap::new();
    for plan in plans.iter() {
        for step in plan.forward.iter().chain(plan.reverse.iter()) {
            steps.insert((step.pos, step.a, step.b), ());
        }
    }

    let step_keys: Vec<(usize, i32, i32)> = steps.into_keys().collect();
    let mats: Vec<((usize, i32, i32), Vec<Vec<(usize, f64)>>)> = step_keys
        .par_iter()
        .map(|&(pos, a, b)| {
            let mat = build_swap_matrix(
                &index.global_prefixes,
                &index.global_prefix_map,
                pos,
                a,
                b,
                zero_eps,
            );
            ((pos, a, b), mat)
        })
        .collect();

    for (key, mat) in mats {
        index.swap_mats.insert(key, mat);
    }
}

fn build_pair_table_with_index(
    index: &SuffixIndex,
    i: usize,
    j: usize,
    val: f64,
    two_s_list: &[i32],
    zero_eps: f64,
) -> PairTable {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    let plan = build_pair_plan(j + 1, two_s_list, i, j);
    let global_transitions =
        build_pair_transitions_global_dp(&index.global_prefixes, &plan, &index.swap_mats, zero_eps);

    let mut groups: Vec<PairGroup> = Vec::with_capacity(index.group_prefix_ids.len());
    for (gid, local_prefixes) in index.group_prefix_ids.iter().enumerate() {
        let local_map = &index.group_local_maps[gid];
        let mut transitions: Vec<Vec<(usize, f64)>> = Vec::with_capacity(local_prefixes.len());
        for &gpid in local_prefixes.iter() {
            let trans = &global_transitions[gpid];
            let mut mapped: Vec<(usize, f64)> = Vec::with_capacity(trans.len());
            for &(new_gpid, coeff) in trans.iter() {
                if coeff.abs() < zero_eps {
                    continue;
                }
                if let Some(&lid) = local_map.get(&new_gpid) {
                    mapped.push((lid, coeff));
                }
            }
            transitions.push(mapped);
        }

        groups.push(PairGroup {
            prefix_cols: index.prefix_cols[gid].clone(),
            transitions,
        });
    }

    PairTable {
        val,
        row_group_ids: index.row_group_ids.clone(),
        row_prefix_ids: index.row_prefix_ids.clone(),
        groups,
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

    with_pool(num_threads, || {
        let exchange: Vec<((usize, usize), f64)> = model
            .exchange
            .iter()
            .map(|(&(i, j), &val)| ((i, j), val))
            .collect();
        let mut indices: AHashMap<usize, SuffixIndex> = AHashMap::new();
        let mut plans_by_j: AHashMap<usize, Vec<PairPlan>> = AHashMap::new();
        for &((i, j), _) in exchange.iter() {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            if !indices.contains_key(&j) {
                indices.insert(j, build_suffix_index(basis, j));
            }
            let plan = build_pair_plan(j + 1, &basis.two_s_list, i, j);
            plans_by_j.entry(j).or_default().push(plan);
        }
        for (j, plans) in plans_by_j.into_iter() {
            if let Some(index) = indices.get_mut(&j) {
                fill_swap_mats(index, &plans, zero_eps);
            }
        }
        let pair_tables: Vec<PairTable> = exchange
            .iter()
            .map(|&((i, j), val)| {
                let j_idx = if i < j { j } else { i };
                let index = indices.get(&j_idx).expect("suffix index missing");
                build_pair_table_with_index(index, i, j, val, &basis.two_s_list, zero_eps)
            })
            .collect();

        let mut row_nnz = vec![0usize; dim];

        row_nnz.par_iter_mut().enumerate().for_each_init(
            || (vec![0.0f64; dim], Vec::<usize>::new()),
            |(acc, touched), (row, slot)| {
                touched.clear();

                for table in pair_tables.iter() {
                    let gid = table.row_group_ids[row];
                    let pid = table.row_prefix_ids[row];
                    let group = &table.groups[gid];
                    let transitions = &group.transitions[pid];
                    for &(new_pid, v) in transitions.iter() {
                        if v.abs() < zero_eps {
                            continue;
                        }
                        let col = group.prefix_cols[new_pid];
                        if acc[col].abs() < zero_eps {
                            touched.push(col);
                        }
                        acc[col] += table.val * v;
                    }
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    if acc[row].abs() < zero_eps {
                        touched.push(row);
                    }
                    acc[row] += model.diagonal_shift;
                }

                let mut count = 0usize;
                for &col in touched.iter() {
                    if acc[col].abs() >= zero_eps {
                        count += 1;
                    }
                    acc[col] = 0.0;
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
        let mut row_slices: Vec<(&mut [usize], &mut [f64])> = Vec::with_capacity(dim);
        for row in 0..dim {
            let len = row_nnz[row];
            let (c_head, c_tail) = cols_rem.split_at_mut(len);
            let (v_head, v_tail) = vals_rem.split_at_mut(len);
            row_slices.push((c_head, v_head));
            cols_rem = c_tail;
            vals_rem = v_tail;
        }

        row_slices.into_par_iter().enumerate().for_each_init(
            || (vec![0.0f64; dim], Vec::<usize>::new()),
            |(acc, touched), (row, (row_cols, row_vals))| {
                touched.clear();

                for table in pair_tables.iter() {
                    let gid = table.row_group_ids[row];
                    let pid = table.row_prefix_ids[row];
                    let group = &table.groups[gid];
                    let transitions = &group.transitions[pid];
                    for &(new_pid, v) in transitions.iter() {
                        if v.abs() < zero_eps {
                            continue;
                        }
                        let col = group.prefix_cols[new_pid];
                        if acc[col].abs() < zero_eps {
                            touched.push(col);
                        }
                        acc[col] += table.val * v;
                    }
                }

                if model.diagonal_shift.abs() >= zero_eps {
                    if acc[row].abs() < zero_eps {
                        touched.push(row);
                    }
                    acc[row] += model.diagonal_shift;
                }

                touched.sort_unstable();
                let mut k = 0usize;
                for &col in touched.iter() {
                    let v = acc[col];
                    acc[col] = 0.0;
                    if v.abs() < zero_eps {
                        continue;
                    }
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
