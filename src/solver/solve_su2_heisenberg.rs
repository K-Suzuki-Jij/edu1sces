//! Solver for SU(2) symmetric Heisenberg model.

use anyhow::Result;
use pyo3::prelude::*;

use crate::blas::{inverse_iteration, lanczos, LanczosParameters};
use crate::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use crate::model::su2_heisenberg::SU2HeisenbergModel;
use crate::solver::{SU2SolverResult, SolverParameters};
use crate::utility::py_log::py_print_flush;

/// Solve the SU(2) symmetric Heisenberg model to find the ground state.
///
/// # Arguments
/// * `model` - The SU(2) Heisenberg model
/// * `total_s` - Target total spin quantum number (integer or half-integer)
/// * `params` - Solver parameters
///
/// # Returns
/// `SU2SolverResult` containing the ground state energy and eigenvector.
#[pyfunction]
pub fn solve_su2_heisenberg(
    model: &SU2HeisenbergModel,
    total_s: f64,
    params: &SolverParameters,
) -> Result<SU2SolverResult> {
    let output_log = params.output_log;
    let num_threads = params.num_threads;

    // Build Hamiltonian (this also builds the basis internally)
    if output_log {
        py_print_flush("Building SU(2) Hamiltonian...\n");
    }
    let ham_start = std::time::Instant::now();
    let hamiltonian = make_su2_heisenberg_hamiltonian(model, total_s, num_threads)?;
    if output_log {
        py_print_flush(&format!(
            "Done in {:.1}s ({} threads, dim={})\n",
            ham_start.elapsed().as_secs_f64(),
            num_threads,
            hamiltonian.row_dim
        ));
    }

    // Prepare Lanczos parameters
    let lanczos_params = LanczosParameters {
        acc: params.eigenvalue_tol,
        min_step: params.min_step,
        max_step: params.max_step,
        calc_eigenvec: true,
        output_log,
    };

    // Run Lanczos
    if output_log {
        py_print_flush("Diagonalizing Hamiltonian...\n");
    }
    let lanczos_start = std::time::Instant::now();
    let mut eigenvector = Vec::new();
    let mut energy = 0.0;
    let lanczos_log = lanczos(
        &hamiltonian,
        &mut eigenvector,
        &mut energy,
        &lanczos_params,
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
    let inv_start = std::time::Instant::now();
    let inverse_iteration_log = inverse_iteration(
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

    Ok(SU2SolverResult {
        energy,
        eigenvector,
        lanczos_log,
        inverse_iteration_log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::{ConjugateGradientParameters, InverseIterationParameters};
    use std::collections::HashMap;

    const TOL: f64 = 1e-8;

    fn make_solver_params() -> SolverParameters {
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
        }
    }

    #[test]
    fn test_two_spin_half_singlet() {
        // Two spin-1/2 in singlet (S=0)
        // E = -3/4 for J=1
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();
        let params = make_solver_params();
        let result = solve_su2_heisenberg(&model, 0.0, &params).unwrap();

        assert_eq!(result.eigenvector.len(), 1);
        assert!(
            (result.energy - (-0.75)).abs() < TOL,
            "Expected energy -0.75, got {}",
            result.energy
        );
    }

    #[test]
    fn test_two_spin_half_triplet() {
        // Two spin-1/2 in triplet (S=1)
        // E = +1/4 for J=1
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let model = SU2HeisenbergModel::new(vec![0.5, 0.5], exchange).unwrap();
        let params = make_solver_params();
        let result = solve_su2_heisenberg(&model, 1.0, &params).unwrap();

        assert_eq!(result.eigenvector.len(), 1);
        assert!(
            (result.energy - 0.25).abs() < TOL,
            "Expected energy 0.25, got {}",
            result.energy
        );
    }

    #[test]
    fn test_four_site_chain_ground_state() {
        // 4-site Heisenberg chain: H = S_0·S_1 + S_1·S_2 + S_2·S_3
        // Ground state is in S=0 sector with E ≈ -1.616025
        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);
        exchange.insert((1, 2), 1.0);
        exchange.insert((2, 3), 1.0);

        let model = SU2HeisenbergModel::new(vec![0.5, 0.5, 0.5, 0.5], exchange).unwrap();
        let params = make_solver_params();
        let result = solve_su2_heisenberg(&model, 0.0, &params).unwrap();

        let expected_energy = -1.616025403784;
        assert!(
            (result.energy - expected_energy).abs() < 1e-6,
            "Expected energy {}, got {}",
            expected_energy,
            result.energy
        );
    }

    #[test]
    fn test_six_site_chain_ground_state() {
        // 6-site Heisenberg chain
        // Ground state is in S=0 sector with E ≈ -2.49357713
        let mut exchange = HashMap::new();
        for i in 0..5 {
            exchange.insert((i, i + 1), 1.0);
        }

        let model = SU2HeisenbergModel::new(vec![0.5; 6], exchange).unwrap();
        let params = make_solver_params();
        let result = solve_su2_heisenberg(&model, 0.0, &params).unwrap();

        let expected_energy = -2.49357713;
        assert!(
            (result.energy - expected_energy).abs() < 1e-6,
            "Expected energy {}, got {}",
            expected_energy,
            result.energy
        );
    }
}
