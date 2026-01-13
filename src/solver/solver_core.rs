use anyhow::Result;
use std::collections::HashMap;

use crate::basis::Basis;
use crate::blas::{inverse_iteration, lanczos, CsrMatrix, LanczosParameters};
use crate::model::QuantumModel;
use crate::solver::{BasisInfo, SolverParameters, SolverResult};
use crate::utility::py_log::py_print_flush;

/// Solve a quantum model to find the ground state.
pub fn solve_model<M, MakeHam>(
    model: M,
    make_hamiltonian: MakeHam,
    current_quantum_numbers: Vec<i32>,
    params: &SolverParameters,
) -> Result<SolverResult>
where
    M: QuantumModel + 'static,
    MakeHam: FnOnce(&Basis) -> Result<CsrMatrix>,
{
    let output_log = params.output_log;
    let num_threads = params.num_threads;

    // Build basis
    if output_log {
        py_print_flush("Building basis...\n");
    }
    let basis_start = std::time::Instant::now();
    let basis = model.build_basis(&current_quantum_numbers)?;
    if output_log {
        py_print_flush(&format!(
            "Done in {:.1}s ({} threads)\n",
            basis_start.elapsed().as_secs_f64(),
            num_threads
        ));
    }

    // Build Hamiltonian
    if output_log {
        py_print_flush("Building Hamiltonian...\n");
    }
    let ham_start = std::time::Instant::now();
    let hamiltonian = make_hamiltonian(&basis)?;
    if output_log {
        py_print_flush(&format!(
            "Done in {:.1}s ({} threads)\n",
            ham_start.elapsed().as_secs_f64(),
            num_threads
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

    // Extract basis info
    let num_sites = basis.num_sites();
    let site_base = basis.site_base.clone();
    let local_dims = basis.local_dims.clone();

    // Box the model for trait object storage
    let model: Box<dyn QuantumModel> = Box::new(model);

    // Initialize basis_cache with the current sector
    let mut basis_cache = HashMap::new();
    basis_cache.insert(current_quantum_numbers.clone(), basis);

    let basis_info = BasisInfo {
        num_sites,
        site_base,
        local_dims,
        current_quantum_numbers,
        model,
        basis_cache,
    };

    Ok(SolverResult {
        energy,
        eigenvector,
        basis_info,
        lanczos_log,
        inverse_iteration_log,
    })
}
