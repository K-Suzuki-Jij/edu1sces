use anyhow::Result;
use pyo3::prelude::*;
use std::time::Instant;

use crate::blas::{inverse_iteration, lanczos, InverseIterationLog, LanczosLog, LanczosParameters};
use crate::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use crate::model::SU2HeisenbergModel;
use crate::solver::SolverParameters;
use crate::utility::py_log::py_print_flush;

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

    if num_states == 0 {
        anyhow::bail!("num_states must be at least 1");
    }

    // Build basis
    if output_log {
        py_print_flush("Building basis...\n");
    }
    let basis_start = Instant::now();
    let basis = model.build_basis(total_s)?;
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
        let known_eigenvecs: Vec<&[f64]> = eigenvectors.iter().map(|v: &Vec<f64>| v.as_slice()).collect();

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

    Ok(SU2SolverResult {
        energies,
        eigenvectors,
        total_s,
        basis_dim: basis.dim(),
        lanczos_logs,
        inverse_iteration_logs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::{ConjugateGradientParameters, InverseIterationParameters};
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

    #[test]
    fn test_2site_spin_half() {
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();

        // Singlet: E = -3/4
        let result = solve_su2_heisenberg(&model, 0.0, &make_params()).unwrap();
        assert!((result.energies[0] - (-0.75)).abs() < 1e-10);

        // Triplet: E = 1/4
        let result = solve_su2_heisenberg(&model, 1.0, &make_params()).unwrap();
        assert!((result.energies[0] - 0.25).abs() < 1e-10);
    }
}
