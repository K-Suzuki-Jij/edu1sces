use ahash::AHashMap;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
use wigner_symbols::{Wigner3jm, Wigner6j};

use crate::basis::SU2HeisenbergBasis;
use crate::model::operator::SpinOperator;
use crate::utility::rayon_pool::build_pool;
use crate::utility::sixj_table::{get_cached_sixj_table, Sixj6Table};
use crate::blas::MATRIX_ZERO_EPS;

#[derive(Clone, Copy, Debug)]
struct SwapStep {
    pos: usize,
    a: i32,
    b: i32,
}

#[derive(Clone, Debug)]
pub struct SinglePlan {
    two_s_site: i32,
    forward: Vec<SwapStep>,
    reverse: Vec<SwapStep>,
}

pub fn build_single_plan(
    two_s_list: &[i32],
    coupling_order: &[usize],
    site_to_pos: &[usize],
    site: usize,
) -> SinglePlan {
    let n = coupling_order.len();
    let pos_site = site_to_pos[site];

    let mut order = (0..n).collect::<Vec<_>>();
    let mut forward = Vec::new();

    let mut cur_pos = pos_site;
    while cur_pos < n - 1 {
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

    SinglePlan {
        two_s_site: two_s_list[site],
        forward,
        reverse,
    }
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

pub fn decompose_spin_operator(op: SpinOperator) -> Vec<(i32, f64)> {
    let sqrt2 = 2.0_f64.sqrt();
    match op {
        SpinOperator::Sz => vec![(0, 1.0)],
        SpinOperator::Sp => vec![(1, -sqrt2)],
        SpinOperator::Sm => vec![(-1, sqrt2)],
        SpinOperator::Sx => vec![(1, -0.5 * sqrt2), (-1, 0.5 * sqrt2)],
        SpinOperator::ISy => vec![(1, -0.5 * sqrt2), (-1, -0.5 * sqrt2)],
    }
}

pub fn adjoint_spin_operator(op: SpinOperator) -> (SpinOperator, f64) {
    match op {
        SpinOperator::Sz => (SpinOperator::Sz, 1.0),
        SpinOperator::Sx => (SpinOperator::Sx, 1.0),
        SpinOperator::Sp => (SpinOperator::Sm, 1.0),
        SpinOperator::Sm => (SpinOperator::Sp, 1.0),
        SpinOperator::ISy => (SpinOperator::ISy, -1.0),
    }
}

fn reduced_spin_melem(
    two_s_rest: i32,
    two_s_site: i32,
    two_s_in: i32,
    two_s_out: i32,
) -> f64 {
    let k = 2;
    let sixj = f64::from(
        Wigner6j {
            tj1: two_s_site,
            tj2: two_s_in,
            tj3: two_s_rest,
            tj4: two_s_out,
            tj5: two_s_site,
            tj6: k,
        }
        .value(),
    );
    if sixj.abs() < MATRIX_ZERO_EPS {
        return 0.0;
    }

    let phase = 1.0 - 2.0 * (((two_s_rest + two_s_site + two_s_out + k) / 2) & 1) as f64;
    let s = (two_s_site as f64) / 2.0;
    let reduced_site = (s * (s + 1.0) * (2.0 * s + 1.0)).sqrt();

    let norm = ((two_s_in + 1) as f64 * (two_s_out + 1) as f64).sqrt();
    phase * norm * sixj * reduced_site
}

fn local_operator_melem(
    two_s_rest: i32,
    two_s_site: i32,
    two_s_in: i32,
    two_s_out: i32,
    two_m_in: i32,
    two_m_out: i32,
    q: i32,
) -> f64 {
    let reduced = reduced_spin_melem(two_s_rest, two_s_site, two_s_in, two_s_out);
    if reduced.abs() < MATRIX_ZERO_EPS {
        return 0.0;
    }
    let w3 = f64::from(
        Wigner3jm {
            tj1: two_s_out,
            tm1: -two_m_out,
            tj2: 2,
            tm2: 2 * q,
            tj3: two_s_in,
            tm3: two_m_in,
        }
        .value(),
    );
    if w3.abs() < MATRIX_ZERO_EPS {
        return 0.0;
    }
    let phase = 1.0 - 2.0 * (((two_s_out - two_m_out) / 2) & 1) as f64;
    phase * w3 * reduced
}

pub fn apply_local_spin_op(
    in_basis: &SU2HeisenbergBasis,
    out_basis: &SU2HeisenbergBasis,
    eigenvector: &[f64],
    _site: usize,
    plan: &SinglePlan,
    two_s_in: i32,
    two_s_out: i32,
    two_m_in: i32,
    q: i32,
    op_coeff: f64,
    num_threads: usize,
) -> Result<AHashMap<usize, f64>> {
    let zero_eps = MATRIX_ZERO_EPS;

    let n = in_basis.num_sites();
    if n == 0 {
        return Ok(AHashMap::new());
    }

    let max_two_s = in_basis.two_s_list.iter().copied().max().unwrap_or(0);
    let max_in_basis = in_basis
        .basis
        .iter()
        .flat_map(|s| s.iter().map(|&v| v as i32))
        .max()
        .unwrap_or(0);
    let max_two_j = max_in_basis + max_two_s;
    let is_uniform_spin = {
        let first_s = in_basis.two_s_list.first().copied().unwrap_or(0);
        in_basis.two_s_list.iter().all(|&s| s == first_s)
    };
    let tables: Option<(Arc<Sixj6Table>, Arc<Sixj6Table>)> =
        if is_uniform_spin && max_two_s > 0 {
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

    let two_m_out = two_m_in + 2 * q;
    if two_m_out.abs() > two_s_out {
        return Ok(AHashMap::new());
    }

    let pool = build_pool(num_threads)?;

    let out = pool.install(|| {
        in_basis
            .basis
            .par_iter()
            .enumerate()
            .fold(AHashMap::new, |mut local, (idx, state)| {
                let mut cur = Vec::new();
                let mut next = Vec::new();
                cur.push((state.clone(), 1.0));

                for step in plan.forward.iter() {
                    next.clear();
                    for (s, coeff) in cur.iter() {
                        let mut tmp = Vec::new();
                        compute_swap_transitions(
                            s,
                            step.pos,
                            step.a,
                            step.b,
                            tables_ref,
                            zero_eps,
                            &mut tmp,
                        );
                        for (ns, c) in tmp.into_iter() {
                            if (coeff * c).abs() >= zero_eps {
                                next.push((ns, coeff * c));
                            }
                        }
                    }
                    cur.clear();
                    cur.extend(next.drain(..));
                }

                let mut after_op = Vec::new();
                for (s, coeff) in cur.iter() {
                    let two_s_rest = s[n - 2] as i32;
                    let two_s_site = plan.two_s_site;

                    let min_s = (two_s_rest - two_s_site).abs();
                    let max_s = two_s_rest + two_s_site;
                    if two_s_out < min_s || two_s_out > max_s {
                        continue;
                    }
                    if ((two_s_out + two_s_rest + two_s_site) & 1) != 0 {
                        continue;
                    }

        let melem = local_operator_melem(
            two_s_rest,
            two_s_site,
            two_s_in,
            two_s_out,
            two_m_in,
            two_m_out,
            q,
        );
                    if melem.abs() < zero_eps {
                        continue;
                    }

                    let mut ns = s.clone();
                    ns[n - 1] = two_s_out as u8;
                    let total_coeff = coeff * melem * op_coeff;
                    if total_coeff.abs() >= zero_eps {
                        after_op.push((ns, total_coeff));
                    }
                }

                cur = after_op;
                for step in plan.reverse.iter() {
                    next.clear();
                    for (s, coeff) in cur.iter() {
                        let mut tmp = Vec::new();
                        compute_swap_transitions(
                            s,
                            step.pos,
                            step.a,
                            step.b,
                            tables_ref,
                            zero_eps,
                            &mut tmp,
                        );
                        for (ns, c) in tmp.into_iter() {
                            if (coeff * c).abs() >= zero_eps {
                                next.push((ns, coeff * c));
                            }
                        }
                    }
                    cur.clear();
                    cur.extend(next.drain(..));
                }

                let vec_coeff = eigenvector[idx];
                if vec_coeff.abs() >= zero_eps {
                    for (s, coeff) in cur.into_iter() {
                        if let Some(&out_idx) = out_basis.inverse_basis.get(&s) {
                            *local.entry(out_idx).or_insert(0.0) += vec_coeff * coeff;
                        }
                    }
                }

                local
            })
            .reduce(AHashMap::new, |mut a, b| {
                for (k, v) in b.into_iter() {
                    *a.entry(k).or_insert(0.0) += v;
                }
                a
            })
    });

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SU2HeisenbergModel;
    use std::collections::HashMap;

    #[test]
    fn test_local_operator_melem_nonzero() {
        let two_s_rest = 1;
        let two_s_site = 1;
        let two_s_in = 0;
        let two_s_out = 2;
        let two_m_in = 0;
        let two_m_out = 0;
        let q = 0;

        let v = local_operator_melem(
            two_s_rest,
            two_s_site,
            two_s_in,
            two_s_out,
            two_m_in,
            two_m_out,
            q,
        );
        assert!(v.abs() > 1e-12, "melem unexpectedly zero: {}", v);
    }

    #[test]
    fn test_apply_local_spin_op_nonzero() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();
        let basis_in = model.build_basis(0.0).unwrap();
        let basis_out = model.build_basis(1.0).unwrap();

        let plan = build_single_plan(
            &basis_in.two_s_list,
            &basis_in.coupling_order,
            &basis_in.site_to_pos,
            0,
        );

        let vec = vec![1.0];
        let out = apply_local_spin_op(
            &basis_in,
            &basis_out,
            &vec,
            0,
            &plan,
            0,
            2,
            0,
            0,
            1.0,
            1,
        )
        .unwrap();
        let sum: f64 = out.values().map(|v| v.abs()).sum();
        assert!(sum > 1e-12, "apply_local_spin_op returned zero");
    }
}
