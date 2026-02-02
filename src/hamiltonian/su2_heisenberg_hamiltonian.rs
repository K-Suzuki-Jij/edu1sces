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
    /// Position where the adjacent pair matrix is applied (j-1 in original indices)
    adjacent_pos: usize,
    forward: Vec<SwapStep>,
    reverse: Vec<SwapStep>,
}

struct PairGroup {
    prefix_cols: Arc<Vec<usize>>,
    transitions: CsrMatrix,
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
    swap_mats: AHashMap<(usize, i32, i32), CsrMatrix>,
    /// Adjacent pair matrices: (adjacent_pos, two_si, two_sj) -> matrix
    adjacent_mats: AHashMap<(usize, i32, i32), CsrMatrix>,
}

thread_local! {
    static SWAP_CACHE: RefCell<AHashMap<SwapKey, Arc<[(i32, f64)]>>> =
        RefCell::new(AHashMap::new());
    static SIXJ_TABLE_CACHE: RefCell<AHashMap<(i32, i32, i32), Arc<Sixj6Table>>> =
        RefCell::new(AHashMap::new());
}

fn get_cached_table(a: i32, b: i32, max_two_j: i32) -> Arc<Sixj6Table> {
    SIXJ_TABLE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = (a, b, max_two_j);
        if let Some(table) = cache.get(&key) {
            return Arc::clone(table);
        }
        let table = Arc::new(Sixj6Table::new(a, b, max_two_j));
        cache.insert(key, Arc::clone(&table));
        table
    })
}

/// Dense lookup table for 6j symbols with fixed a, b values.
/// Stores {x, a, j1; b, j2, t} indexed as [x][j1][j2][t/2].
struct Sixj6Table {
    max_j: usize,
    max_t_idx: usize, // max_t / 2
    data: Vec<f64>,   // Flattened 4D array
}

impl Sixj6Table {
    fn new(a: i32, b: i32, max_j: i32) -> Self {
        let max_j = max_j as usize;
        let max_t = (a + b) as usize;
        let max_t_idx = max_t / 2;
        let size = (max_j + 1) * (max_j + 1) * (max_j + 1) * (max_t_idx + 1);
        let mut data = vec![0.0; size];

        for x in 0..=max_j {
            for j1 in 0..=max_j {
                for j2 in 0..=max_j {
                    for t_idx in 0..=max_t_idx {
                        let t = (t_idx * 2) as i32;
                        let idx = Self::index_impl(x, j1, j2, t_idx, max_j, max_t_idx);
                        data[idx] = f64::from(
                            Wigner6j {
                                tj1: x as i32,
                                tj2: a,
                                tj3: j1 as i32,
                                tj4: b,
                                tj5: j2 as i32,
                                tj6: t,
                            }
                            .value(),
                        );
                    }
                }
            }
        }

        Self {
            max_j,
            max_t_idx,
            data,
        }
    }

    #[inline]
    fn index_impl(
        x: usize,
        j1: usize,
        j2: usize,
        t_idx: usize,
        max_j: usize,
        max_t_idx: usize,
    ) -> usize {
        ((x * (max_j + 1) + j1) * (max_j + 1) + j2) * (max_t_idx + 1) + t_idx
    }

    #[inline]
    fn get(&self, x: i32, j1: i32, j2: i32, t: i32) -> f64 {
        let t_idx = (t / 2) as usize;
        let idx = Self::index_impl(
            x as usize,
            j1 as usize,
            j2 as usize,
            t_idx,
            self.max_j,
            self.max_t_idx,
        );
        self.data[idx]
    }
}

fn phase_from_half_integer(sum_two: i32) -> f64 {
    if ((sum_two / 2) & 1) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Compute the adjacent pair matrix elements for S_i · S_j at position `adjacent_pos`.
/// After swaps, site i is at position `adjacent_pos` and site j is at position `adjacent_pos + 1`.
/// Uses 6j recoupling to compute ⟨J''_k| S_i · S_j |J'_k⟩.
fn build_adjacent_pair_matrix(
    prefixes: &[Vec<u8>],
    prefix_map: &AHashMap<Vec<u8>, usize>,
    adjacent_pos: usize,
    two_si: i32,
    two_sj: i32,
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    zero_eps: f64,
) -> CsrMatrix {
    let dim = prefixes.len();

    let prefix_len = if prefixes.is_empty() {
        0
    } else {
        prefixes[0].len()
    };

    // K ranges from |s_i - s_j| to s_i + s_j
    let min_k = (two_si - two_sj).abs();
    let max_k = two_si + two_sj;

    // Parallel computation over all prefixes
    let out: Vec<Vec<(usize, f64)>> = (0..dim)
        .into_par_iter()
        .map(|pid| {
            let prefix = &prefixes[pid];
            let mut row = Vec::new();

            if adjacent_pos == 0 {
                // Special case: S_i · S_j at positions 0, 1.
                // The cumulative spin at position 1 is J_2 = |S_0 + S_1|, stored in prefix[1].
                // Casimir: S_i · S_j = (J_2(J_2+1) - s_i(s_i+1) - s_j(s_j+1)) / 2
                let j2 = prefix[1] as i32;
                let eig = 0.125
                    * ((j2 * (j2 + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
                if eig.abs() >= zero_eps {
                    row.push((pid, eig));
                }
                return row;
            }

            // General case: use 6j recoupling
            let j_km1 = prefix[adjacent_pos - 1] as i32;
            let j_k = prefix[adjacent_pos] as i32;
            let j_k1 = prefix[adjacent_pos + 1] as i32;

            // Possible J'_k values satisfying triangle inequalities
            let min_jk_prime = (j_km1 - two_si).abs().max((j_k1 - two_sj).abs());
            let max_jk_prime = (j_km1 + two_si).min(j_k1 + two_sj);

            let mut new_prefix = vec![0u8; prefix_len];
            new_prefix.copy_from_slice(prefix);
            let mut jk_prime = min_jk_prime;
            if ((jk_prime + j_km1 + two_si) & 1) != 0 {
                jk_prime += 1;
            }

            while jk_prime <= max_jk_prime {
                // Matrix element ⟨J'_k| S_i · S_j |J_k⟩
                // = Σ_K (2K+1) × eigenvalue(K) × {j_km1, s_i, j_k; s_j, j_k1, K} × {j_km1, s_i, j'_k; s_j, j_k1, K}
                let mut sum = 0.0;
                let mut k = min_k;
                if ((k + two_si + two_sj) & 1) != 0 {
                    k += 1;
                }
                while k <= max_k {
                    let eig_k = 0.125
                        * ((k * (k + 2) - two_si * (two_si + 2) - two_sj * (two_sj + 2)) as f64);
                    // Use 6j table if available, otherwise compute directly
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
                        new_prefix[adjacent_pos] = jk_prime as u8;
                        if let Some(&nid) = prefix_map.get(&new_prefix) {
                            row.push((nid, coeff));
                        }
                    }
                }
                jk_prime += 2;
            }

            row
        })
        .collect();

    // Convert to CsrMatrix
    let mut rows = Vec::with_capacity(dim + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    rows.push(0);
    for row_data in out {
        for (col, val) in row_data {
            cols.push(col);
            vals.push(val);
        }
        rows.push(cols.len());
    }
    CsrMatrix {
        row_dim: dim,
        col_dim: dim,
        rows,
        cols,
        vals,
    }
}

fn swap_coeffs_with_table(
    x: i32,
    j_k: i32,
    j_k1: i32,
    a: i32,
    b: i32,
    table1: &Sixj6Table,
    table2: &Sixj6Table,
    zero_eps: f64,
) -> Arc<[(i32, f64)]> {
    let key = SwapKey { x, j_k, j_k1, a, b };

    SWAP_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(v) = cache.get(&key) {
            return Arc::clone(v);
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
                // sixj1: {x, a, j_k; b, j_k1, t}
                let s1 = table1.get(x, j_k, j_k1, t);
                if s1.abs() >= zero_eps {
                    // sixj2: {x, b, j_kp; a, j_k1, t}
                    let s2 = table2.get(x, j_kp, j_k1, t);
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

        let out: Arc<[(i32, f64)]> = out.into();
        cache.insert(key, Arc::clone(&out));
        out
    })
}

fn swap_coeffs(x: i32, j_k: i32, j_k1: i32, a: i32, b: i32, zero_eps: f64) -> Arc<[(i32, f64)]> {
    let key = SwapKey { x, j_k, j_k1, a, b };

    SWAP_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(v) = cache.get(&key) {
            return Arc::clone(v);
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
                let s1 = f64::from(
                    Wigner6j {
                        tj1: x,
                        tj2: a,
                        tj3: j_k,
                        tj4: b,
                        tj5: j_k1,
                        tj6: t,
                    }
                    .value(),
                );
                if s1.abs() >= zero_eps {
                    let s2 = f64::from(
                        Wigner6j {
                            tj1: x,
                            tj2: b,
                            tj3: j_kp,
                            tj4: a,
                            tj5: j_k1,
                            tj6: t,
                        }
                        .value(),
                    );
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

        let out: Arc<[(i32, f64)]> = out.into();
        cache.insert(key, Arc::clone(&out));
        out
    })
}

/// Build a plan to compute S_i · S_j matrix elements.
///
/// New algorithm: Move site i to position j-1 (adjacent to j), then apply Casimir.
/// This requires only O(j-i-1) swaps instead of O(max(i,j)).
/// For adjacent pairs (i, i+1), this means ZERO swaps!
fn build_pair_plan(n: usize, two_s_list: &[i32], i: usize, j: usize) -> PairPlan {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    let mut order: Vec<usize> = (0..n).collect();
    let mut forward: Vec<SwapStep> = Vec::new();

    // Move site i to position j-1 by swapping it rightward
    // Initial: ... s_i, s_{i+1}, ..., s_{j-1}, s_j, ...
    // After:   ... s_{i+1}, ..., s_{j-1}, s_i, s_j, ...
    // Number of swaps = j - i - 1 (zero for adjacent pairs!)
    let mut pos_i = i;
    while pos_i < j - 1 {
        let pos = pos_i;
        let a = two_s_list[order[pos]];
        let b = two_s_list[order[pos + 1]];
        forward.push(SwapStep { pos, a, b });
        order.swap(pos, pos + 1);
        pos_i += 1;
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
        // After moving site i to position j-1, the adjacent pair is at positions (j-1, j).
        // The adjacent pair matrix is applied at position j-1 (using prefix indices).
        adjacent_pos: j - 1,
        forward,
        reverse,
    }
}

fn build_swap_matrix_with_table(
    prefixes: &[Vec<u8>],
    prefix_map: &AHashMap<Vec<u8>, usize>,
    pos: usize,
    a: i32,
    b: i32,
    table1: &Sixj6Table,
    table2: &Sixj6Table,
    zero_eps: f64,
) -> CsrMatrix {
    let dim = prefixes.len();

    // Reusable buffer to avoid allocations
    let prefix_len = if prefixes.is_empty() {
        0
    } else {
        prefixes[0].len()
    };
    let mut new_prefix = vec![0u8; prefix_len];

    let mut rows = Vec::with_capacity(dim + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    rows.push(0);

    for prefix in prefixes.iter() {
        if pos == 0 {
            let j1 = prefix[1] as i32;
            let phase = phase_from_half_integer(a + b - j1);
            new_prefix.copy_from_slice(prefix);
            new_prefix[0] = b as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if phase.abs() >= zero_eps {
                    cols.push(nid);
                    vals.push(phase);
                }
            }
            rows.push(cols.len());
            continue;
        }

        let x = prefix[pos - 1] as i32;
        let j_k = prefix[pos] as i32;
        let j_k1 = prefix[pos + 1] as i32;
        let coeffs = swap_coeffs_with_table(x, j_k, j_k1, a, b, table1, table2, zero_eps);

        new_prefix.copy_from_slice(prefix);
        let original_val = new_prefix[pos];
        for &(j_kp, c) in coeffs.iter() {
            new_prefix[pos] = j_kp as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if c.abs() >= zero_eps {
                    cols.push(nid);
                    vals.push(c);
                }
            }
        }
        new_prefix[pos] = original_val;
        rows.push(cols.len());
    }

    CsrMatrix {
        row_dim: dim,
        col_dim: dim,
        rows,
        cols,
        vals,
    }
}

fn build_swap_matrix(
    prefixes: &[Vec<u8>],
    prefix_map: &AHashMap<Vec<u8>, usize>,
    pos: usize,
    a: i32,
    b: i32,
    zero_eps: f64,
) -> CsrMatrix {
    let dim = prefixes.len();

    // Reusable buffer to avoid allocations
    let prefix_len = if prefixes.is_empty() {
        0
    } else {
        prefixes[0].len()
    };
    let mut new_prefix = vec![0u8; prefix_len];

    let mut rows = Vec::with_capacity(dim + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    rows.push(0);

    for prefix in prefixes.iter() {
        if pos == 0 {
            let j1 = prefix[1] as i32;
            let phase = phase_from_half_integer(a + b - j1);
            new_prefix.copy_from_slice(prefix);
            new_prefix[0] = b as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if phase.abs() >= zero_eps {
                    cols.push(nid);
                    vals.push(phase);
                }
            }
            rows.push(cols.len());
            continue;
        }

        let x = prefix[pos - 1] as i32;
        let j_k = prefix[pos] as i32;
        let j_k1 = prefix[pos + 1] as i32;
        let coeffs = swap_coeffs(x, j_k, j_k1, a, b, zero_eps);

        new_prefix.copy_from_slice(prefix);
        for &(j_kp, c) in coeffs.iter() {
            new_prefix[pos] = j_kp as u8;
            if let Some(&nid) = prefix_map.get(&new_prefix) {
                if c.abs() >= zero_eps {
                    cols.push(nid);
                    vals.push(c);
                }
            }
        }
        rows.push(cols.len());
    }

    CsrMatrix {
        row_dim: dim,
        col_dim: dim,
        rows,
        cols,
        vals,
    }
}

fn apply_step_vec(
    vec_in: &[(usize, f64)],
    step_mat: &CsrMatrix,
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
        let row_start = step_mat.rows[id];
        let row_end = step_mat.rows[id + 1];
        for k in row_start..row_end {
            let nid = step_mat.cols[k];
            let w = step_mat.vals[k];
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
    swap_mats: &AHashMap<(usize, i32, i32), CsrMatrix>,
    adjacent_mats: &AHashMap<(usize, i32, i32), CsrMatrix>,
    zero_eps: f64,
) -> CsrMatrix {
    let dim = prefixes.len();

    // Get the adjacent pair matrix for this plan
    let adjacent_mat = adjacent_mats
        .get(&(plan.adjacent_pos, plan.two_si, plan.two_sj))
        .expect("adjacent pair matrix missing");

    // Fast path: adjacent pairs have no swaps, so adjacent_mat IS the transition matrix
    if plan.forward.is_empty() {
        return adjacent_mat.clone();
    }

    // Slow path: non-adjacent pairs require swap chain
    let out: Vec<Vec<(usize, f64)>> = (0..dim)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0.0f64; dim],
                    Vec::<usize>::new(),
                    Vec::<(usize, f64)>::new(),
                    Vec::<(usize, f64)>::new(),
                )
            },
            |(acc, touched, buf_a, buf_b), pid| {
                buf_a.clear();
                buf_a.push((pid, 1.0));

                // Apply forward swaps (O(j-i-1) steps)
                for step in plan.forward.iter() {
                    let step_mat = swap_mats
                        .get(&(step.pos, step.a, step.b))
                        .expect("swap matrix missing");
                    apply_step_vec(buf_a, step_mat, acc, touched, buf_b, zero_eps);
                    std::mem::swap(buf_a, buf_b);
                    if buf_a.is_empty() {
                        return Vec::new();
                    }
                }

                // Apply adjacent pair matrix (S_i · S_j interaction via 6j recoupling)
                apply_step_vec(buf_a, adjacent_mat, acc, touched, buf_b, zero_eps);
                std::mem::swap(buf_a, buf_b);
                if buf_a.is_empty() {
                    return Vec::new();
                }

                // Apply reverse swaps
                for step in plan.reverse.iter() {
                    let step_mat = swap_mats
                        .get(&(step.pos, step.a, step.b))
                        .expect("swap matrix missing");
                    apply_step_vec(buf_a, step_mat, acc, touched, buf_b, zero_eps);
                    std::mem::swap(buf_a, buf_b);
                    if buf_a.is_empty() {
                        return Vec::new();
                    }
                }

                std::mem::take(buf_a)
            },
        )
        .collect();

    // Convert to CsrMatrix
    let mut rows = Vec::with_capacity(dim + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    rows.push(0);
    for row_data in out {
        for (col, val) in row_data {
            cols.push(col);
            vals.push(val);
        }
        rows.push(cols.len());
    }
    CsrMatrix {
        row_dim: dim,
        col_dim: dim,
        rows,
        cols,
        vals,
    }
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
        adjacent_mats: AHashMap::new(),
    }
}

fn fill_swap_mats(
    index: &mut SuffixIndex,
    plans: &[PairPlan],
    tables: Option<(&Sixj6Table, &Sixj6Table)>,
    zero_eps: f64,
) {
    // Collect swap steps
    let mut steps: AHashMap<(usize, i32, i32), ()> = AHashMap::new();
    for plan in plans.iter() {
        for step in plan.forward.iter().chain(plan.reverse.iter()) {
            steps.insert((step.pos, step.a, step.b), ());
        }
    }

    let step_keys: Vec<(usize, i32, i32)> = steps.into_keys().collect();

    if let Some((table1, table2)) = tables {
        // Uniform spin case: use pre-built 6j tables
        let mats: Vec<((usize, i32, i32), CsrMatrix)> = step_keys
            .par_iter()
            .map(|&(pos, a, b)| {
                let mat = build_swap_matrix_with_table(
                    &index.global_prefixes,
                    &index.global_prefix_map,
                    pos,
                    a,
                    b,
                    table1,
                    table2,
                    zero_eps,
                );
                ((pos, a, b), mat)
            })
            .collect();

        for (key, mat) in mats {
            index.swap_mats.insert(key, mat);
        }
    } else {
        // Non-uniform case: fallback to per-call 6j computation
        let mats: Vec<((usize, i32, i32), CsrMatrix)> = step_keys
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

    // Collect adjacent pair matrix keys
    let mut adjacent_keys: AHashMap<(usize, i32, i32), ()> = AHashMap::new();
    for plan in plans.iter() {
        adjacent_keys.insert((plan.adjacent_pos, plan.two_si, plan.two_sj), ());
    }

    let adjacent_key_list: Vec<(usize, i32, i32)> = adjacent_keys.into_keys().collect();

    // Build adjacent pair matrices (these use 6j symbols for recoupling)
    let adjacent_mats: Vec<((usize, i32, i32), CsrMatrix)> = adjacent_key_list
        .par_iter()
        .map(|&(pos, two_si, two_sj)| {
            let mat = build_adjacent_pair_matrix(
                &index.global_prefixes,
                &index.global_prefix_map,
                pos,
                two_si,
                two_sj,
                tables,
                zero_eps,
            );
            ((pos, two_si, two_sj), mat)
        })
        .collect();

    for (key, mat) in adjacent_mats {
        index.adjacent_mats.insert(key, mat);
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
    let global_transitions = build_pair_transitions_global_dp(
        &index.global_prefixes,
        &plan,
        &index.swap_mats,
        &index.adjacent_mats,
        zero_eps,
    );

    let mut groups: Vec<PairGroup> = Vec::with_capacity(index.group_prefix_ids.len());
    for (gid, local_prefixes) in index.group_prefix_ids.iter().enumerate() {
        let local_map = &index.group_local_maps[gid];

        // Build CSR matrix for transitions
        let row_dim = local_prefixes.len();
        let col_dim = local_map.len();
        let mut rows: Vec<usize> = Vec::with_capacity(row_dim + 1);
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        rows.push(0);
        for &gpid in local_prefixes.iter() {
            let start = global_transitions.rows[gpid];
            let end = global_transitions.rows[gpid + 1];
            for k in start..end {
                let new_gpid = global_transitions.cols[k];
                let coeff = global_transitions.vals[k];
                if coeff.abs() < zero_eps {
                    continue;
                }
                if let Some(&lid) = local_map.get(&new_gpid) {
                    cols.push(lid);
                    vals.push(coeff);
                }
            }
            rows.push(cols.len());
        }

        groups.push(PairGroup {
            prefix_cols: index.prefix_cols[gid].clone(),
            transitions: CsrMatrix {
                row_dim,
                col_dim,
                rows,
                cols,
                vals,
            },
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

        // Collect unique j values needed
        let mut j_values: Vec<usize> = exchange
            .iter()
            .map(|&((i, j), _)| if i < j { j } else { i })
            .collect();
        j_values.sort_unstable();
        j_values.dedup();

        // Build suffix indices in parallel
        let index_pairs: Vec<(usize, SuffixIndex)> = j_values
            .par_iter()
            .map(|&j| (j, build_suffix_index(basis, j)))
            .collect();
        let indices: AHashMap<usize, SuffixIndex> = index_pairs.into_iter().collect();

        // Build plans (fast, no need for parallel)
        let mut plans_by_j: AHashMap<usize, Vec<PairPlan>> = AHashMap::new();
        for &((i, j), _) in exchange.iter() {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let plan = build_pair_plan(j + 1, &basis.two_s_list, i, j);
            plans_by_j.entry(j).or_default().push(plan);
        }
        // Check if all spins are uniform (for 6j table optimization)
        let is_uniform_spin = {
            let first_s = basis.two_s_list.first().copied().unwrap_or(0);
            basis.two_s_list.iter().all(|&s| s == first_s)
        };

        // Compute max_two_j for 6j table bounds
        // j_kp can be up to x + b, where x is the largest cumulative spin
        // and b is the local spin. We add margin for safety.
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
                get_cached_table(a, b, max_two_j),
                get_cached_table(b, a, max_two_j),
            ))
        } else {
            None
        };

        // Convert to Vec for parallel processing
        let mut index_vec: Vec<(usize, SuffixIndex)> = indices.into_iter().collect();
        let tables_ref = tables.as_ref().map(|(t1, t2)| (Arc::clone(t1), Arc::clone(t2)));

        index_vec.par_iter_mut().for_each(|(j, index)| {
            if let Some(plans) = plans_by_j.get(j) {
                let tref = tables_ref
                    .as_ref()
                    .map(|(t1, t2)| (t1.as_ref(), t2.as_ref()));
                fill_swap_mats(index, plans, tref, zero_eps);
            }
        });

        let indices: AHashMap<usize, SuffixIndex> = index_vec.into_iter().collect();
        let pair_tables: Vec<PairTable> = exchange
            .par_iter()
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
                    let trans = &group.transitions;
                    let start = trans.rows[pid];
                    let end = trans.rows[pid + 1];
                    for k in start..end {
                        let new_pid = trans.cols[k];
                        let v = trans.vals[k];
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
                    let trans = &group.transitions;
                    let start = trans.rows[pid];
                    let end = trans.rows[pid + 1];
                    for k in start..end {
                        let new_pid = trans.cols[k];
                        let v = trans.vals[k];
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

        // Long-range interactions with various coupling strengths
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

        // U(1): Collect all eigenvalues from all Sz sectors
        let mut u1_all_eigs = Vec::new();
        for sz2 in (-(n as i32)..=(n as i32)).step_by(2) {
            let basis = u1_model.build_basis(&[sz2]).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_heisenberg_hamiltonian(&basis, &u1_model, 1).unwrap();
            u1_all_eigs.extend(get_all_eigenvalues(&ham));
        }

        // SU(2): Collect all eigenvalues from all S sectors (with degeneracy 2S+1)
        let mut su2_all_eigs = Vec::new();
        for s2 in (0..=(n as i32)).step_by(2) {
            let s = s2 as f64 / 2.0;
            let basis = su2_model.build_basis(s).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_su2_heisenberg_hamiltonian(&basis, &su2_model, 1).unwrap();
            let eigs = get_all_eigenvalues(&ham);
            let degeneracy = (s2 + 1) as usize; // 2S+1
            for e in eigs {
                for _ in 0..degeneracy {
                    su2_all_eigs.push(e);
                }
            }
        }

        // Sort and compare
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

        // Long-range interactions with various coupling strengths
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

        // U(1): Collect all eigenvalues from all Sz sectors
        let mut u1_all_eigs = Vec::new();
        for sz2 in (-(n as i32 * two_s)..=(n as i32 * two_s)).step_by(2) {
            let basis = u1_model.build_basis(&[sz2]).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_heisenberg_hamiltonian(&basis, &u1_model, 1).unwrap();
            u1_all_eigs.extend(get_all_eigenvalues(&ham));
        }

        // SU(2): Collect all eigenvalues from all S sectors (with degeneracy 2S+1)
        let mut su2_all_eigs = Vec::new();
        let max_s2 = n as i32 * two_s; // Maximum 2*S = n * 2*s
        for s2 in (0..=max_s2).step_by(2) {
            let s = s2 as f64 / 2.0;
            let basis = su2_model.build_basis(s).unwrap();
            if basis.dim() == 0 {
                continue;
            }
            let ham = make_su2_heisenberg_hamiltonian(&basis, &su2_model, 1).unwrap();
            let eigs = get_all_eigenvalues(&ham);
            let degeneracy = (s2 + 1) as usize; // 2S+1
            for e in eigs {
                for _ in 0..degeneracy {
                    su2_all_eigs.push(e);
                }
            }
        }

        // Sort and compare
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
