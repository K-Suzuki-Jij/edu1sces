use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::blas::{inverse_iteration, lanczos, InverseIterationLog, LanczosLog, LanczosParameters};
use crate::utility::rayon_pool::build_pool;
use crate::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use crate::model::SU2HeisenbergModel;
use crate::model::operator::SpinOperator;
use crate::solver::su2_correlation::{
    adjoint_spin_operator, apply_local_spin_op, build_single_plan, decompose_spin_operator,
};
use crate::solver::SolverParameters;
use crate::utility::py_log::py_print_flush;
use crate::basis::SU2HeisenbergBasis;
use std::collections::HashMap;

#[pyclass]
pub struct SU2SolverResult {
    #[pyo3(get)]
    pub energies: Vec<f64>,
    #[pyo3(get)]
    pub eigenvectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub total_s: f64,
    #[pyo3(get)]
    pub basis_dim: usize,
    #[pyo3(get)]
    pub lanczos_logs: Vec<LanczosLog>,
    #[pyo3(get)]
    pub inverse_iteration_logs: Vec<InverseIterationLog>,
    pub basis_info: SU2BasisInfo,
}

pub struct SU2BasisInfo {
    pub two_s_current: i32,
    pub model: SU2HeisenbergModel,
    pub basis_cache: HashMap<i32, SU2HeisenbergBasis>,
    pub coupling_order: Vec<usize>,
}

impl SU2BasisInfo {
    pub fn ensure_basis_exists(&mut self, two_s: i32) -> Result<()> {
        if !self.basis_cache.contains_key(&two_s) {
            let s = (two_s as f64) / 2.0;
            let basis = self.model.build_basis_with_order(s, &self.coupling_order)?;
            self.basis_cache.insert(two_s, basis);
        }
        Ok(())
    }

    pub fn get_basis(&self, two_s: i32) -> Result<&SU2HeisenbergBasis> {
        self.basis_cache
            .get(&two_s)
            .ok_or_else(|| anyhow::anyhow!("basis not found for 2S={}", two_s))
    }
}

#[pyfunction]
pub fn solve_su2_heisenberg(
    model: &SU2HeisenbergModel,
    total_s: f64,
    params: &SolverParameters,
) -> Result<SU2SolverResult> {
    let output_log = params.output_log;
    let num_threads = params.num_threads;
    let num_states = params.num_states;
    let model_clone = model.clone();

    if num_states == 0 {
        anyhow::bail!("num_states must be at least 1");
    }

    // Build basis
    if output_log {
        py_print_flush("Building basis...\n");
    }
    let basis_start = Instant::now();
    let basis = model.build_basis(total_s)?;
    let coupling_order = basis.coupling_order.clone();
    if output_log {
        py_print_flush(&format!(
            "Done in {:.1}s ({} threads)\n",
            basis_start.elapsed().as_secs_f64(),
            num_threads
        ));
    }

    if basis.dim() == 0 {
        anyhow::bail!("Basis dimension is zero for S={}", total_s);
    }

    // Build Hamiltonian
    if output_log {
        py_print_flush("Building Hamiltonian...\n");
    }
    let ham_start = Instant::now();
    let hamiltonian = make_su2_heisenberg_hamiltonian(&basis, model, num_threads)?;
    if output_log {
        py_print_flush(&format!(
            "Done in {:.1}s ({} threads)\n",
            ham_start.elapsed().as_secs_f64(),
            num_threads
        ));
    }

    // Lanczos parameters
    let lanczos_params = LanczosParameters {
        acc: params.eigenvalue_tol,
        min_step: params.min_step,
        max_step: params.max_step,
        calc_eigenvec: true,
        output_log,
    };

    let mut energies = Vec::with_capacity(num_states);
    let mut eigenvectors = Vec::with_capacity(num_states);
    let mut lanczos_logs = Vec::with_capacity(num_states);
    let mut inverse_iteration_logs = Vec::with_capacity(num_states);

    for state_idx in 0..num_states {
        if output_log {
            if state_idx == 0 {
                py_print_flush("Diagonalizing Hamiltonian (ground state)...\n");
            } else {
                py_print_flush(&format!(
                    "Diagonalizing Hamiltonian (excited state {})...\n",
                    state_idx
                ));
            }
        }

        let lanczos_start = Instant::now();
        let mut eigenvector = Vec::new();
        let mut energy = 0.0;
        let known_eigenvecs: Vec<&[f64]> = eigenvectors
            .iter()
            .map(|v: &Vec<f64>| v.as_slice())
            .collect();

        let lanczos_log = lanczos(
            &hamiltonian,
            &mut eigenvector,
            &mut energy,
            &lanczos_params,
            &known_eigenvecs,
            num_threads,
        )?;

        if output_log {
            py_print_flush(&format!(
                "\rDone in {:.1}s ({} threads)                      \n",
                lanczos_start.elapsed().as_secs_f64(),
                num_threads
            ));
        }

        // Refine eigenvector using inverse iteration
        if output_log {
            py_print_flush("Improving eigenvector...\n");
        }
        let inv_start = Instant::now();
        let inv_log = inverse_iteration(
            &hamiltonian,
            &mut eigenvector,
            energy,
            &params.inverse_iteration_params,
            num_threads,
        )?;
        if output_log {
            py_print_flush(&format!(
                "\rDone in {:.1}s ({} threads)                      \n",
                inv_start.elapsed().as_secs_f64(),
                num_threads
            ));
        }

        energies.push(energy);
        eigenvectors.push(eigenvector);
        lanczos_logs.push(lanczos_log);
        inverse_iteration_logs.push(inv_log);
    }

    let two_s_current = (2.0 * total_s).round() as i32;
    let basis_dim = basis.dim();
    let mut basis_cache = HashMap::new();
    basis_cache.insert(two_s_current, basis);

    Ok(SU2SolverResult {
        energies,
        eigenvectors,
        total_s,
        basis_dim,
        lanczos_logs,
        inverse_iteration_logs,
        basis_info: SU2BasisInfo {
            two_s_current,
            model: model_clone,
            basis_cache,
            coupling_order,
        },
    })
}

#[pymethods]
impl SU2SolverResult {
    pub fn expectation_onsite(
        &mut self,
        op: SpinOperator,
        site: usize,
        m_total: f64,
        num_threads: usize,
        state_index: usize,
    ) -> Result<f64> {
        if state_index >= self.eigenvectors.len() {
            anyhow::bail!(
                "state_index {} out of range (num_states = {})",
                state_index,
                self.eigenvectors.len()
            );
        }

        let two_s = self.basis_info.two_s_current;
        let two_m = (2.0 * m_total).round() as i32;
        if (2.0 * m_total - (two_m as f64)).abs() > 1e-12 {
            anyhow::bail!("m_total must be integer or half-integer (got {})", m_total);
        }
        if two_m.abs() > two_s {
            anyhow::bail!(
                "m_total out of range (|M| = {} > S = {})",
                m_total,
                self.total_s
            );
        }
        if ((two_s - two_m) & 1) != 0 {
            anyhow::bail!("m_total parity does not match total_s");
        }

        let op_parts = decompose_spin_operator(op);

        self.basis_info.ensure_basis_exists(two_s)?;
        let plan = {
            let basis = self.basis_info.get_basis(two_s)?;
            build_single_plan(
                &basis.two_s_list,
                &basis.coupling_order,
                &basis.site_to_pos,
                site,
            )
        };

        let mut total = 0.0;

        let pool = build_pool(num_threads)?;

        // Only q=0 component contributes to <ψ|O|ψ> (same M sector)
        for &(q, coeff) in op_parts.iter() {
            if q != 0 {
                continue;
            }

            let basis = self.basis_info.get_basis(two_s)?;

            let vec_out = apply_local_spin_op(
                basis,
                basis,
                &self.eigenvectors[state_index],
                site,
                &plan,
                two_s,
                two_s,
                two_m,
                q,
                coeff,
                num_threads,
            )?;

            // Compute <ψ|O|ψ> = Σ_i ψ_i * (O|ψ⟩)_i (parallel)
            let eigenvec = &self.eigenvectors[state_index];
            let partial: f64 = pool.install(|| {
                vec_out
                    .par_iter()
                    .map(|(idx, v)| eigenvec[*idx] * v)
                    .sum()
            });
            total += partial;
        }

        Ok(total)
    }

    pub fn correlation_function(
        &mut self,
        op1: SpinOperator,
        site1: usize,
        op2: SpinOperator,
        site2: usize,
        m_total: f64,
        num_threads: usize,
        state_index: usize,
    ) -> Result<f64> {
        if state_index >= self.eigenvectors.len() {
            anyhow::bail!(
                "state_index {} out of range (num_states = {})",
                state_index,
                self.eigenvectors.len()
            );
        }

        let two_s = self.basis_info.two_s_current;
        let two_m = (2.0 * m_total).round() as i32;
        if (2.0 * m_total - (two_m as f64)).abs() > 1e-12 {
            anyhow::bail!("m_total must be integer or half-integer (got {})", m_total);
        }
        if two_m.abs() > two_s {
            anyhow::bail!(
                "m_total out of range (|M| = {} > S = {})",
                m_total,
                self.total_s
            );
        }
        if ((two_s - two_m) & 1) != 0 {
            anyhow::bail!("m_total parity does not match total_s");
        }

        let (op1_dag, op1_dag_sign) = adjoint_spin_operator(op1);
        let op1_parts = decompose_spin_operator(op1_dag);
        let op2_parts = decompose_spin_operator(op2);

        self.basis_info.ensure_basis_exists(two_s)?;
        let (plan1, plan2) = {
            let basis_in = self.basis_info.get_basis(two_s)?;
            let plan1 = build_single_plan(
                &basis_in.two_s_list,
                &basis_in.coupling_order,
                &basis_in.site_to_pos,
                site1,
            );
            let plan2 = build_single_plan(
                &basis_in.two_s_list,
                &basis_in.coupling_order,
                &basis_in.site_to_pos,
                site2,
            );
            (plan1, plan2)
        };

        let pool = build_pool(num_threads)?;
        let mut total = 0.0;

        let possible_two_s_out = [two_s - 2, two_s, two_s + 2];

        for &(q2, coeff2) in op2_parts.iter() {
            let two_m_out = two_m + 2 * q2;
            if two_m_out.abs() > two_s + 2 {
                continue;
            }

            for &two_s_out in possible_two_s_out.iter() {
                if two_s_out < 0 {
                    continue;
                }

                self.basis_info.ensure_basis_exists(two_s_out)?;
                let (basis_in, basis_out) = {
                    let basis_in = self.basis_info.get_basis(two_s)?;
                    let basis_out = self.basis_info.get_basis(two_s_out)?;
                    (basis_in, basis_out)
                };
                if basis_out.dim() == 0 {
                    continue;
                }

                let vec2 = apply_local_spin_op(
                    basis_in,
                    basis_out,
                    &self.eigenvectors[state_index],
                    site2,
                    &plan2,
                    two_s,
                    two_s_out,
                    two_m,
                    q2,
                    coeff2,
                    num_threads,
                )?;

                if vec2.is_empty() {
                    continue;
                }

                for &(q1, coeff1) in op1_parts.iter() {
                    if q1 != q2 {
                        continue;
                    }

                    let vec1 = apply_local_spin_op(
                        basis_in,
                        basis_out,
                        &self.eigenvectors[state_index],
                        site1,
                        &plan1,
                        two_s,
                        two_s_out,
                        two_m,
                        q1,
                        coeff1 * op1_dag_sign,
                        num_threads,
                    )?;

                    if vec1.is_empty() {
                        continue;
                    }

                    let (small, large) = if vec1.len() <= vec2.len() {
                        (&vec1, &vec2)
                    } else {
                        (&vec2, &vec1)
                    };
                    let partial: f64 = pool.install(|| {
                        small
                            .par_iter()
                            .filter_map(|(idx, v)| large.get(idx).map(|w| v * w))
                            .sum()
                    });
                    total += partial;
                }
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::{ConjugateGradientParameters, InverseIterationParameters};
    use crate::blas::lapack_dsyev;
    use crate::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
    use crate::model::{HeisenbergModel, QuantumModel};
    use crate::model::operator::SpinOperator;
    use crate::solver::{BasisInfo, SolverResult};
    use crate::solver::su2_correlation::{apply_local_spin_op, build_single_plan};
    use crate::solver::solve_heisenberg::solve_heisenberg;
    use std::collections::HashMap;

    fn make_params() -> SolverParameters {
        SolverParameters {
            eigenvalue_tol: 1e-14,
            min_step: 5,
            max_step: 1000,
            num_threads: 1,
            inverse_iteration_params: InverseIterationParameters {
                diag_add: 1e-7,
                eigenvec_tol: 1e-8,
                max_step: 100,
                cg_params: ConjugateGradientParameters {
                    residual_tol: 1e-12,
                    max_step: 1000,
                    output_log: false,
                },
            },
            output_log: false,
            num_states: 1,
        }
    }

    fn apply_local_op_vector(
        in_basis: &crate::basis::Basis,
        out_basis: &crate::basis::Basis,
        local_op: &crate::blas::CsrMatrix,
        site: usize,
        vec: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; out_basis.dim()];
        let site_base = out_basis.site_base[site];
        for (out_idx, &basis_out) in out_basis.basis.iter().enumerate() {
            let local_basis_out = out_basis.find_local_basis(basis_out, site);
            let mut val = 0.0;
            for p in local_op.rows[local_basis_out]..local_op.rows[local_basis_out + 1] {
                let local_basis_in = local_op.cols[p];
                let mat_val = local_op.vals[p];
                let basis_in = basis_out
                    + ((local_basis_in as i128) - (local_basis_out as i128)) * site_base;
                if let Some(&in_idx) = in_basis.inverse_basis.get(&basis_in) {
                    val += vec[in_idx] * mat_val;
                }
            }
            out[out_idx] = val;
        }
        out
    }

    fn apply_s2_to_vector(
        basis0: &crate::basis::Basis,
        basis_p: Option<&crate::basis::Basis>,
        basis_m: Option<&crate::basis::Basis>,
        sz_ops: &[crate::blas::CsrMatrix],
        sp_ops: &[crate::blas::CsrMatrix],
        sm_ops: &[crate::blas::CsrMatrix],
        vec: &[f64],
        s2_local: f64,
    ) -> Vec<f64> {
        let n = basis0.num_sites();
        let mut out = vec![0.0; basis0.dim()];
        for i in 0..basis0.dim() {
            out[i] += s2_local * vec[i];
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let tmp_sz = apply_local_op_vector(basis0, basis0, &sz_ops[j], j, vec);
                let tmp_sz = apply_local_op_vector(basis0, basis0, &sz_ops[i], i, &tmp_sz);
                for k in 0..out.len() {
                    out[k] += 2.0 * tmp_sz[k];
                }

                if let Some(bm) = basis_m {
                    let tmp_sm = apply_local_op_vector(basis0, bm, &sm_ops[j], j, vec);
                    let tmp_sm = apply_local_op_vector(bm, basis0, &sp_ops[i], i, &tmp_sm);
                    for k in 0..out.len() {
                        out[k] += tmp_sm[k];
                    }
                }

                if let Some(bp) = basis_p {
                    let tmp_sp = apply_local_op_vector(basis0, bp, &sp_ops[j], j, vec);
                    let tmp_sp = apply_local_op_vector(bp, basis0, &sm_ops[i], i, &tmp_sp);
                    for k in 0..out.len() {
                        out[k] += tmp_sp[k];
                    }
                }
            }
        }

        out
    }

    #[test]
    fn test_2site_spin_half() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        // Singlet: E = -3/4
        let mut result = solve_su2_heisenberg(&model, 0.0, &make_params()).unwrap();
        assert!((result.energies[0] - (-0.75)).abs() < 1e-10);

        let szsz = result
            .correlation_function(SpinOperator::Sz, 0, SpinOperator::Sz, 1, 0.0, 1, 0)
            .unwrap();
        assert!((szsz - (-0.25)).abs() < 1e-8, "szsz={}", szsz);

        let sp_sm = result
            .correlation_function(SpinOperator::Sp, 0, SpinOperator::Sm, 1, 0.0, 1, 0)
            .unwrap();
        assert!((sp_sm - (-0.5)).abs() < 1e-8);

        // Triplet: E = 1/4
        let result = solve_su2_heisenberg(&model, 1.0, &make_params()).unwrap();
        assert!((result.energies[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_onsite_sz_u1_comparison() {
        let n = 4;
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n - 1 {
            exchange_xy.insert((i, i + 1), 1.0);
            exchange_z.insert((i, i + 1), 1.0);
            exchange_su2.insert((i, i + 1), 1.0);
        }

        let u1_model = HeisenbergModel::new(
            vec![0.5; n],
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();

        let su2_model = SU2HeisenbergModel::new(vec![0.5; n], exchange_su2).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 0.0, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 0.0, &params).unwrap();

        // Compare <Sz_i> for each site
        for site in 0..n {
            let sz_op = u1_model.make_local_op_sz(site).unwrap();
            let u1_exp = u1_result.expectation_onsite(&sz_op, site, 1, 0).unwrap();
            let su2_exp = su2_result
                .expectation_onsite(SpinOperator::Sz, site, 0.0, 1, 0)
                .unwrap();
            assert!(
                (u1_exp - su2_exp).abs() < 1e-8,
                "site {}: u1={} su2={}",
                site,
                u1_exp,
                su2_exp
            );
        }
    }

    #[test]
    fn test_expectation_onsite_sz_triplet() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        // Triplet S=1, M=1: <Sz_0> = <Sz_1> = 0.5
        let mut result = solve_su2_heisenberg(&model, 1.0, &make_params()).unwrap();
        let sz0 = result
            .expectation_onsite(SpinOperator::Sz, 0, 1.0, 1, 0)
            .unwrap();
        let sz1 = result
            .expectation_onsite(SpinOperator::Sz, 1, 1.0, 1, 0)
            .unwrap();
        assert!((sz0 - 0.5).abs() < 1e-8, "sz0={}", sz0);
        assert!((sz1 - 0.5).abs() < 1e-8, "sz1={}", sz1);

        // Triplet S=1, M=0: <Sz_0> = <Sz_1> = 0
        let sz0_m0 = result
            .expectation_onsite(SpinOperator::Sz, 0, 0.0, 1, 0)
            .unwrap();
        let sz1_m0 = result
            .expectation_onsite(SpinOperator::Sz, 1, 0.0, 1, 0)
            .unwrap();
        assert!(sz0_m0.abs() < 1e-8, "sz0_m0={}", sz0_m0);
        assert!(sz1_m0.abs() < 1e-8, "sz1_m0={}", sz1_m0);
    }

    #[test]
    fn test_su2_u1_sp_isy_2site_triplet_m1() {
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);
        exchange_z.insert((0, 1), 1.0);
        let u1_model = HeisenbergModel::new(
            vec![0.5, 0.5],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            exchange_xy,
            exchange_z,
        )
        .unwrap();

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let su2_model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 1.0, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 1.0, &params).unwrap();

        let sp = u1_model.make_local_op_sp(0).unwrap();
        let isy = u1_model.make_local_op_isy(1).unwrap();
        let u1_corr = u1_result
            .correlation_function(&sp, 0, &isy, 1, 1, 0)
            .unwrap();
        let su2_corr = su2_result
            .correlation_function(SpinOperator::Sp, 0, SpinOperator::ISy, 1, 1.0, 1, 0)
            .unwrap();
        assert!(
            (u1_corr - su2_corr).abs() < 1e-8,
            "u1={} su2={}",
            u1_corr,
            su2_corr
        );
    }

    #[test]
    fn test_su2_u1_szsz_2site_mixed_spins() {
        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);
        exchange_z.insert((0, 1), 1.0);
        let u1_model = HeisenbergModel::new(
            vec![1.0, 0.5],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            exchange_xy,
            exchange_z,
        )
        .unwrap();

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let su2_model = SU2HeisenbergModel::new(vec![1.0, 0.5], exchange).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 0.5, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 0.5, &params).unwrap();

        let sz0 = u1_model.make_local_op_sz(0).unwrap();
        let sz1 = u1_model.make_local_op_sz(1).unwrap();
        let u1_corr = u1_result
            .correlation_function(&sz0, 0, &sz1, 1, 1, 0)
            .unwrap();
        let su2_corr = su2_result
            .correlation_function(SpinOperator::Sz, 0, SpinOperator::Sz, 1, 0.5, 1, 0)
            .unwrap();
        assert!(
            (u1_corr - su2_corr).abs() < 1e-8,
            "u1={} su2={}",
            u1_corr,
            su2_corr
        );
    }

    #[test]
    fn test_su2_correlation_internal_szsz_2site() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        let mut result = solve_su2_heisenberg(&model, 0.0, &make_params()).unwrap();
        let two_s = result.basis_info.two_s_current;
        result.basis_info.ensure_basis_exists(two_s + 2).unwrap();
        let basis_in = result.basis_info.get_basis(two_s).unwrap();
        let basis_out = result.basis_info.get_basis(two_s + 2).unwrap();

        let plan0 = build_single_plan(
            &basis_in.two_s_list,
            &basis_in.coupling_order,
            &basis_in.site_to_pos,
            0,
        );
        let plan1 = build_single_plan(
            &basis_in.two_s_list,
            &basis_in.coupling_order,
            &basis_in.site_to_pos,
            1,
        );

        let vec0 = apply_local_spin_op(
            basis_in,
            basis_out,
            &result.eigenvectors[0],
            0,
            &plan0,
            two_s,
            two_s + 2,
            0,
            0,
            1.0,
            1,
        )
        .unwrap();
        let vec1 = apply_local_spin_op(
            basis_in,
            basis_out,
            &result.eigenvectors[0],
            1,
            &plan1,
            two_s,
            two_s + 2,
            0,
            0,
            1.0,
            1,
        )
        .unwrap();

        let (small, large) = if vec0.len() <= vec1.len() {
            (&vec0, &vec1)
        } else {
            (&vec1, &vec0)
        };
        let mut manual = 0.0;
        for (idx, v) in small.iter() {
            if let Some(w) = large.get(idx) {
                manual += v * w;
            }
        }

        let corr = result
            .correlation_function(SpinOperator::Sz, 0, SpinOperator::Sz, 1, 0.0, 1, 0)
            .unwrap();
        assert!((manual - corr).abs() < 1e-12);
        assert!(manual.abs() > 1e-12, "manual SzSz is zero");
    }

    #[test]
    fn test_su2_u1_szsz_6site_uniform_chain() {
        let n = 6;
        let j = 1.0;

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n {
            let j_site = (i + 1) % n;
            exchange_xy.insert((i, j_site), j);
            exchange_z.insert((i, j_site), j);
            exchange_su2.insert((i, j_site), j);
        }

        let spins = vec![0.5; n];

        let u1_model = HeisenbergModel::new(
            spins.clone(),
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();
        let su2_model = SU2HeisenbergModel::new(spins, exchange_su2).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 0.0, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 0.0, &params).unwrap();

        let sz_op = u1_model.make_local_op_sz(0).unwrap();

        for i in 0..n {
            for j in (i + 1)..n {
                let u1_corr = u1_result
                    .correlation_function(&sz_op, i, &sz_op, j, 1, 0)
                    .unwrap();
                let su2_corr = su2_result
                    .correlation_function(SpinOperator::Sz, i, SpinOperator::Sz, j, 0.0, 1, 0)
                    .unwrap();
                assert!(
                    (u1_corr - su2_corr).abs() < 1e-8,
                    "SzSz mismatch at ({}, {}): u1={}, su2={}",
                    i,
                    j,
                    u1_corr,
                    su2_corr
                );
            }
        }
    }

    #[test]
    fn test_su2_u1_random_ops_6site_uniform_chain() {
        let n = 6;
        let j = 1.0;

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n {
            let j_site = (i + 1) % n;
            exchange_xy.insert((i, j_site), j);
            exchange_z.insert((i, j_site), j);
            exchange_su2.insert((i, j_site), j);
        }

        let spins = vec![0.5; n];

        let u1_model = HeisenbergModel::new(
            spins.clone(),
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();
        let su2_model = SU2HeisenbergModel::new(spins, exchange_su2).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 0.0, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 0.0, &params).unwrap();

        fn local_op_for(
            model: &HeisenbergModel,
            op: SpinOperator,
            site: usize,
        ) -> crate::blas::CsrMatrix {
            match op {
                SpinOperator::Sz => model.make_local_op_sz(site).unwrap(),
                SpinOperator::Sp => model.make_local_op_sp(site).unwrap(),
                SpinOperator::Sm => model.make_local_op_sm(site).unwrap(),
                SpinOperator::Sx => model.make_local_op_sx(site).unwrap(),
                SpinOperator::ISy => model.make_local_op_isy(site).unwrap(),
            }
        }

        let ops = [
            SpinOperator::Sz,
            SpinOperator::Sp,
            SpinOperator::Sm,
            SpinOperator::Sx,
            SpinOperator::ISy,
        ];

        let mut seed: u64 = 0x9e3779b97f4a7c15;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (seed >> 32) as u32
        };

        let samples = 40usize;
        let mut saw_sx_sz = false;
        for _ in 0..samples {
            let i = (next_u32() as usize) % n;
            let mut j = (next_u32() as usize) % n;
            if i == j {
                j = (j + 1) % n;
            }
            let op1 = ops[(next_u32() as usize) % ops.len()];
            let op2 = ops[(next_u32() as usize) % ops.len()];

            let op1_u1 = local_op_for(&u1_model, op1, i);
            let op2_u1 = local_op_for(&u1_model, op2, j);

            let u1_corr = u1_result
                .correlation_function(&op1_u1, i, &op2_u1, j, 1, 0)
                .unwrap();
            let su2_corr = su2_result
                .correlation_function(op1, i, op2, j, 0.0, 1, 0)
                .unwrap();
            assert!(
                (u1_corr - su2_corr).abs() < 1e-8,
                "op mismatch at ({}, {}): {:?} {:?} u1={} su2={}",
                i,
                j,
                op1,
                op2,
                u1_corr,
                su2_corr
            );

            if (op1 == SpinOperator::Sx && op2 == SpinOperator::Sz)
                || (op1 == SpinOperator::Sz && op2 == SpinOperator::Sx)
            {
                saw_sx_sz = true;
                assert!(
                    u1_corr.abs() < 1e-8 && su2_corr.abs() < 1e-8,
                    "Sx-Sz not zero at ({}, {}): u1={} su2={}",
                    i,
                    j,
                    u1_corr,
                    su2_corr
                );
            }
        }

        if !saw_sx_sz {
            let i = 0;
            let j = 1;
            let op1 = SpinOperator::Sx;
            let op2 = SpinOperator::Sz;
            let op1_u1 = local_op_for(&u1_model, op1, i);
            let op2_u1 = local_op_for(&u1_model, op2, j);
            let u1_corr = u1_result
                .correlation_function(&op1_u1, i, &op2_u1, j, 1, 0)
                .unwrap();
            let su2_corr = su2_result
                .correlation_function(op1, i, op2, j, 0.0, 1, 0)
                .unwrap();
            assert!(u1_corr.abs() < 1e-8 && su2_corr.abs() < 1e-8);
        }
    }

    #[test]
    fn test_su2_u1_random_ops_6site_uniform_chain_s1_m1() {
        let n = 6;
        let j = 1.0;

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n {
            let j_site = (i + 1) % n;
            exchange_xy.insert((i, j_site), j);
            exchange_z.insert((i, j_site), j);
            exchange_su2.insert((i, j_site), j);
        }

        let spins = vec![0.5; n];

        let u1_model = HeisenbergModel::new(
            spins.clone(),
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();
        let su2_model = SU2HeisenbergModel::new(spins, exchange_su2).unwrap();

        let params = make_params();
        let mut u1_result = solve_heisenberg(&u1_model, 1.0, &params).unwrap();
        let mut su2_result = solve_su2_heisenberg(&su2_model, 1.0, &params).unwrap();

        fn local_op_for(
            model: &HeisenbergModel,
            op: SpinOperator,
            site: usize,
        ) -> crate::blas::CsrMatrix {
            match op {
                SpinOperator::Sz => model.make_local_op_sz(site).unwrap(),
                SpinOperator::Sp => model.make_local_op_sp(site).unwrap(),
                SpinOperator::Sm => model.make_local_op_sm(site).unwrap(),
                SpinOperator::Sx => model.make_local_op_sx(site).unwrap(),
                SpinOperator::ISy => model.make_local_op_isy(site).unwrap(),
            }
        }

        let ops = [
            SpinOperator::Sz,
            SpinOperator::Sp,
            SpinOperator::Sm,
            SpinOperator::Sx,
            SpinOperator::ISy,
        ];

        let mut seed: u64 = 0xa5a5a5a5a5a5a5a5;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (seed >> 32) as u32
        };

        let samples = 40usize;
        let mut saw_sx_sz = false;
        for _ in 0..samples {
            let i = (next_u32() as usize) % n;
            let mut j = (next_u32() as usize) % n;
            if i == j {
                j = (j + 1) % n;
            }
            let op1 = ops[(next_u32() as usize) % ops.len()];
            let op2 = ops[(next_u32() as usize) % ops.len()];

            let op1_u1 = local_op_for(&u1_model, op1, i);
            let op2_u1 = local_op_for(&u1_model, op2, j);

            let u1_corr = u1_result
                .correlation_function(&op1_u1, i, &op2_u1, j, 1, 0)
                .unwrap();
            let su2_corr = su2_result
                .correlation_function(op1, i, op2, j, 1.0, 1, 0)
                .unwrap();
            assert!(
                (u1_corr - su2_corr).abs() < 1e-8,
                "op mismatch at ({}, {}): {:?} {:?} u1={} su2={}",
                i,
                j,
                op1,
                op2,
                u1_corr,
                su2_corr
            );

            if (op1 == SpinOperator::Sx && op2 == SpinOperator::Sz)
                || (op1 == SpinOperator::Sz && op2 == SpinOperator::Sx)
            {
                saw_sx_sz = true;
                assert!(
                    u1_corr.abs() < 1e-8 && su2_corr.abs() < 1e-8,
                    "Sx-Sz not zero at ({}, {}): u1={} su2={}",
                    i,
                    j,
                    u1_corr,
                    su2_corr
                );
            }
        }

        if !saw_sx_sz {
            let i = 0;
            let j = 1;
            let op1 = SpinOperator::Sx;
            let op2 = SpinOperator::Sz;
            let op1_u1 = local_op_for(&u1_model, op1, i);
            let op2_u1 = local_op_for(&u1_model, op2, j);
            let u1_corr = u1_result
                .correlation_function(&op1_u1, i, &op2_u1, j, 1, 0)
                .unwrap();
            let su2_corr = su2_result
                .correlation_function(op1, i, op2, j, 1.0, 1, 0)
                .unwrap();
            assert!(u1_corr.abs() < 1e-8 && su2_corr.abs() < 1e-8);
        }
    }

    fn solve_heisenberg_exact(model: &HeisenbergModel, two_m: i32) -> SolverResult {
        let basis = model.build_basis(&[two_m]).unwrap();
        let dim = basis.dim();
        let h = make_heisenberg_hamiltonian(&basis, model, 1).unwrap();

        let mut a_work = vec![0.0; dim * dim];
        let mut w_work = vec![0.0; dim];
        let mut work = vec![0.0; 3 * dim];
        lapack_dsyev(&h, &mut a_work, &mut w_work, &mut work).unwrap();

        let mut eigenvectors = Vec::with_capacity(dim);
        for col in 0..dim {
            let mut v = Vec::with_capacity(dim);
            for row in 0..dim {
                v.push(a_work[row + col * dim]);
            }
            eigenvectors.push(v);
        }

        let mut basis_cache = HashMap::new();
        basis_cache.insert(vec![two_m], basis);

        let basis_info = BasisInfo {
            num_sites: model.num_sites(),
            site_base: basis_cache.get(&vec![two_m]).unwrap().site_base.clone(),
            local_dims: basis_cache.get(&vec![two_m]).unwrap().local_dims.clone(),
            current_quantum_numbers: vec![two_m],
            model: Box::new(model.clone()),
            basis_cache,
        };

        SolverResult {
            energies: w_work,
            eigenvectors,
            basis_info,
            lanczos_logs: Vec::new(),
            inverse_iteration_logs: Vec::new(),
        }
    }

    #[test]
    fn test_su2_u1_all_s_m_6site_uniform_chain() {
        let n = 6;
        let j = 1.0;

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n {
            let j_site = (i + 1) % n;
            exchange_xy.insert((i, j_site), j);
            exchange_z.insert((i, j_site), j);
            exchange_su2.insert((i, j_site), j);
        }

        let spins = vec![0.5; n];

        let u1_model = HeisenbergModel::new(
            spins.clone(),
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();
        let su2_model = SU2HeisenbergModel::new(spins, exchange_su2).unwrap();

        let params = make_params();

        fn local_op_for(
            model: &HeisenbergModel,
            op: SpinOperator,
            site: usize,
        ) -> crate::blas::CsrMatrix {
            match op {
                SpinOperator::Sz => model.make_local_op_sz(site).unwrap(),
                SpinOperator::Sp => model.make_local_op_sp(site).unwrap(),
                SpinOperator::Sm => model.make_local_op_sm(site).unwrap(),
                SpinOperator::Sx => model.make_local_op_sx(site).unwrap(),
                SpinOperator::ISy => model.make_local_op_isy(site).unwrap(),
            }
        }

        let ops = [
            SpinOperator::Sz,
            SpinOperator::Sp,
            SpinOperator::Sm,
            SpinOperator::Sx,
            SpinOperator::ISy,
        ];

        let sz_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sz(site).unwrap())
            .collect();
        let sp_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sp(site).unwrap())
            .collect();
        let sm_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sm(site).unwrap())
            .collect();

        for two_s in (0..=6).step_by(2) {
            let total_s = (two_s as f64) / 2.0;
            let mut su2_result = solve_su2_heisenberg(&su2_model, total_s, &params).unwrap();
            let su2_energy = su2_result.energies[0];
            let target_s2 = total_s * (total_s + 1.0);

            for two_m in (-two_s..=two_s).step_by(2) {
                let total_m = (two_m as f64) / 2.0;
                let mut u1_result = solve_heisenberg_exact(&u1_model, two_m);
                let mut group = Vec::new();
                for (i, &e) in u1_result.energies.iter().enumerate() {
                    if (e - su2_energy).abs() < 1e-6 {
                        group.push(i);
                    }
                }
                if group.is_empty() {
                    let mut best_state = 0usize;
                    let mut best_de = f64::INFINITY;
                    for (i, &e) in u1_result.energies.iter().enumerate() {
                        let de = (e - su2_energy).abs();
                        if de < best_de {
                            best_de = de;
                            best_state = i;
                        }
                    }
                    group.push(best_state);
                }

                let basis0 = u1_result
                    .basis_info
                    .get_basis(&vec![two_m])
                    .unwrap();
                let basis_p = if two_m + 2 <= 6 {
                    Some(u1_model.build_basis(&[two_m + 2]).unwrap())
                } else {
                    None
                };
                let basis_m = if two_m - 2 >= -6 {
                    Some(u1_model.build_basis(&[two_m - 2]).unwrap())
                } else {
                    None
                };

                let basis_p_ref = basis_p.as_ref();
                let basis_m_ref = basis_m.as_ref();

                let group_dim = group.len();
                let mut s2_mat = vec![0.0; group_dim * group_dim];
                for (col_idx, &state_idx) in group.iter().enumerate() {
                    let vec = &u1_result.eigenvectors[state_idx];
                    let s2_vec = apply_s2_to_vector(
                        basis0,
                        basis_p_ref,
                        basis_m_ref,
                        &sz_ops,
                        &sp_ops,
                        &sm_ops,
                        vec,
                        0.75 * (n as f64),
                    );
                    for (row_idx, &state_i) in group.iter().enumerate() {
                        let v = &u1_result.eigenvectors[state_i];
                        let val = v.iter().zip(s2_vec.iter()).map(|(a, b)| a * b).sum::<f64>();
                        s2_mat[row_idx + col_idx * group_dim] = val;
                    }
                }

                let mut w_work = vec![0.0; group_dim];
                let mut work = vec![0.0; 3 * group_dim.max(1)];
                let mut info: i32 = 0;
                let lwork = work.len() as i32;
                unsafe {
                    lapack::dsyev(
                        b'V',
                        b'L',
                        group_dim as i32,
                        &mut s2_mat,
                        group_dim as i32,
                        &mut w_work,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                assert!(info == 0, "dsyev failed: info={}", info);

                let mut best_col = 0usize;
                let mut best_ds2 = f64::INFINITY;
                for (i, &s2_e) in w_work.iter().enumerate() {
                    let ds2 = (s2_e - target_s2).abs();
                    if ds2 < best_ds2 {
                        best_ds2 = ds2;
                        best_col = i;
                    }
                }

                let mut combined = vec![0.0; basis0.dim()];
                for (row_idx, &state_idx) in group.iter().enumerate() {
                    let coeff = s2_mat[row_idx + best_col * group_dim];
                    let v = &u1_result.eigenvectors[state_idx];
                    for (k, &vk) in v.iter().enumerate() {
                        combined[k] += coeff * vk;
                    }
                }

                let state_index = u1_result.eigenvectors.len();
                u1_result.eigenvectors.push(combined);
                u1_result.energies.push(su2_energy);

                for i in 0..n {
                    for j in (i + 1)..n {
                        for &op1 in ops.iter() {
                            for &op2 in ops.iter() {
                                let op2_valid = match op2 {
                                    SpinOperator::Sz => true,
                                    SpinOperator::Sp => two_m + 2 <= 6,
                                    SpinOperator::Sm => two_m - 2 >= -6,
                                    SpinOperator::Sx | SpinOperator::ISy => {
                                        two_m + 2 <= 6 && two_m - 2 >= -6
                                    }
                                };
                                if !op2_valid {
                                    continue;
                                }
                                let op1_u1 = local_op_for(&u1_model, op1, i);
                                let op2_u1 = local_op_for(&u1_model, op2, j);
                                let u1_corr = u1_result
                                    .correlation_function(
                                        &op1_u1,
                                        i,
                                        &op2_u1,
                                        j,
                                        1,
                                        state_index,
                                    )
                                    .unwrap();
                                let su2_corr = su2_result
                                    .correlation_function(op1, i, op2, j, total_m, 1, 0)
                                    .unwrap();
                                assert!(
                                    (u1_corr - su2_corr).abs() < 1e-8,
                                    "S={} M={} op mismatch at ({}, {}): {:?} {:?} u1={} su2={}",
                                    total_s,
                                    total_m,
                                    i,
                                    j,
                                    op1,
                                    op2,
                                    u1_corr,
                                    su2_corr
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_su2_u1_all_s_m_4site_random() {
        let n = 4;

        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (seed >> 32) as u32
        };

        let mut spins = Vec::with_capacity(n);
        for _ in 0..n {
            let r = next_u32() % 2;
            spins.push(if r == 0 { 0.5 } else { 1.0 });
        }

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        let mut exchange_su2 = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let r = next_u32() as f64 / (u32::MAX as f64);
                let val = 2.0 * (r - 0.5);
                if val.abs() < 1e-3 {
                    continue;
                }
                exchange_xy.insert((i, j), val);
                exchange_z.insert((i, j), val);
                exchange_su2.insert((i, j), val);
            }
        }

        let u1_model = HeisenbergModel::new(
            spins.clone(),
            vec![0.0; n],
            vec![0.0; n],
            exchange_xy,
            exchange_z,
        )
        .unwrap();
        let su2_model = SU2HeisenbergModel::new(spins.clone(), exchange_su2).unwrap();

        let params = make_params();
        let max_two_m: i32 = spins.iter().map(|s| (2.0 * s) as i32).sum();

        fn local_op_for(
            model: &HeisenbergModel,
            op: SpinOperator,
            site: usize,
        ) -> crate::blas::CsrMatrix {
            match op {
                SpinOperator::Sz => model.make_local_op_sz(site).unwrap(),
                SpinOperator::Sp => model.make_local_op_sp(site).unwrap(),
                SpinOperator::Sm => model.make_local_op_sm(site).unwrap(),
                SpinOperator::Sx => model.make_local_op_sx(site).unwrap(),
                SpinOperator::ISy => model.make_local_op_isy(site).unwrap(),
            }
        }

        let ops = [
            SpinOperator::Sz,
            SpinOperator::Sp,
            SpinOperator::Sm,
            SpinOperator::Sx,
            SpinOperator::ISy,
        ];

        let sz_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sz(site).unwrap())
            .collect();
        let sp_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sp(site).unwrap())
            .collect();
        let sm_ops: Vec<_> = (0..n)
            .map(|site| u1_model.make_local_op_sm(site).unwrap())
            .collect();

        for two_s in (0..=max_two_m).step_by(2) {
            if su2_model.calc_dim_su2_sector((two_s as f64) / 2.0).unwrap() == 0 {
                continue;
            }
            let total_s = (two_s as f64) / 2.0;
            let mut su2_result = solve_su2_heisenberg(&su2_model, total_s, &params).unwrap();
            let su2_energy = su2_result.energies[0];
            let target_s2 = total_s * (total_s + 1.0);

            for two_m in (-two_s..=two_s).step_by(2) {
                let total_m = (two_m as f64) / 2.0;
                let mut u1_result = solve_heisenberg_exact(&u1_model, two_m);
                let mut group = Vec::new();
                for (i, &e) in u1_result.energies.iter().enumerate() {
                    if (e - su2_energy).abs() < 1e-6 {
                        group.push(i);
                    }
                }
                if group.is_empty() {
                    let mut best_state = 0usize;
                    let mut best_de = f64::INFINITY;
                    for (i, &e) in u1_result.energies.iter().enumerate() {
                        let de = (e - su2_energy).abs();
                        if de < best_de {
                            best_de = de;
                            best_state = i;
                        }
                    }
                    group.push(best_state);
                }

                let basis0 = u1_result
                    .basis_info
                    .get_basis(&vec![two_m])
                    .unwrap();
                let basis_p = if two_m + 2 <= max_two_m {
                    u1_model.build_basis(&[two_m + 2]).ok()
                } else {
                    None
                };
                let basis_m = if two_m - 2 >= -max_two_m {
                    u1_model.build_basis(&[two_m - 2]).ok()
                } else {
                    None
                };

                let basis_p_ref = basis_p.as_ref();
                let basis_m_ref = basis_m.as_ref();

                let group_dim = group.len();
                let mut s2_mat = vec![0.0; group_dim * group_dim];
                for (col_idx, &state_idx) in group.iter().enumerate() {
                    let vec = &u1_result.eigenvectors[state_idx];
                    let s2_vec = apply_s2_to_vector(
                        basis0,
                        basis_p_ref,
                        basis_m_ref,
                        &sz_ops,
                        &sp_ops,
                        &sm_ops,
                        vec,
                        spins.iter().map(|s| s * (s + 1.0)).sum(),
                    );
                    for (row_idx, &state_i) in group.iter().enumerate() {
                        let v = &u1_result.eigenvectors[state_i];
                        let val = v.iter().zip(s2_vec.iter()).map(|(a, b)| a * b).sum::<f64>();
                        s2_mat[row_idx + col_idx * group_dim] = val;
                    }
                }

                let mut w_work = vec![0.0; group_dim];
                let mut work = vec![0.0; 3 * group_dim.max(1)];
                let mut info: i32 = 0;
                let lwork = work.len() as i32;
                unsafe {
                    lapack::dsyev(
                        b'V',
                        b'L',
                        group_dim as i32,
                        &mut s2_mat,
                        group_dim as i32,
                        &mut w_work,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                assert!(info == 0, "dsyev failed: info={}", info);

                let mut best_col = 0usize;
                let mut best_ds2 = f64::INFINITY;
                for (i, &s2_e) in w_work.iter().enumerate() {
                    let ds2 = (s2_e - target_s2).abs();
                    if ds2 < best_ds2 {
                        best_ds2 = ds2;
                        best_col = i;
                    }
                }

                let mut combined = vec![0.0; basis0.dim()];
                for (row_idx, &state_idx) in group.iter().enumerate() {
                    let coeff = s2_mat[row_idx + best_col * group_dim];
                    let v = &u1_result.eigenvectors[state_idx];
                    for (k, &vk) in v.iter().enumerate() {
                        combined[k] += coeff * vk;
                    }
                }

                let state_index = u1_result.eigenvectors.len();
                u1_result.eigenvectors.push(combined);
                u1_result.energies.push(su2_energy);

                for i in 0..n {
                    for j in (i + 1)..n {
                        for &op1 in ops.iter() {
                            for &op2 in ops.iter() {
                                let op2_valid = match op2 {
                                    SpinOperator::Sz => true,
                                    SpinOperator::Sp => two_m + 2 <= max_two_m,
                                    SpinOperator::Sm => two_m - 2 >= -max_two_m,
                                    SpinOperator::Sx | SpinOperator::ISy => {
                                        two_m + 2 <= max_two_m && two_m - 2 >= -max_two_m
                                    }
                                };
                                if !op2_valid {
                                    continue;
                                }
                                let op1_u1 = local_op_for(&u1_model, op1, i);
                                let op2_u1 = local_op_for(&u1_model, op2, j);
                                let u1_corr = u1_result
                                    .correlation_function(
                                        &op1_u1,
                                        i,
                                        &op2_u1,
                                        j,
                                        1,
                                        state_index,
                                    )
                                    .unwrap();
                                let su2_corr = su2_result
                                    .correlation_function(op1, i, op2, j, total_m, 1, 0)
                                    .unwrap();
                                assert!(
                                    (u1_corr - su2_corr).abs() < 1e-8,
                                    "S={} M={} op mismatch at ({}, {}): {:?} {:?} u1={} su2={}",
                                    total_s,
                                    total_m,
                                    i,
                                    j,
                                    op1,
                                    op2,
                                    u1_corr,
                                    su2_corr
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
