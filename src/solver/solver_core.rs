use anyhow::Result;

use crate::basis::HilbertBasis;
use crate::blas::{inverse_iteration, lanczos, CsrMatrix, LanczosParameters};
use crate::solver::{BasisInfo, SolverParameters, SolverResult};
use crate::utility::py_log::py_print_flush;

/// Generic solver that takes closures to build basis and Hamiltonian.
pub fn solve_with_basis_and_hamiltonian<B, MakeBasis, MakeHam>(
    make_basis: MakeBasis,
    make_hamiltonian: MakeHam,
    params: &SolverParameters,
) -> Result<SolverResult>
where
    B: HilbertBasis,
    MakeBasis: FnOnce() -> Result<B>,
    MakeHam: FnOnce(&B) -> Result<CsrMatrix>,
{
    let output_log = params.output_log;
    let num_threads = params.num_threads;

    // Build basis
    if output_log {
        py_print_flush("Building basis...\n");
    }
    let basis_start = std::time::Instant::now();
    let basis_obj = make_basis()?;
    let dim = basis_obj.dim();
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
    let hamiltonian = make_hamiltonian(&basis_obj)?;
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

    // Extract basis data
    let num_sites = basis_obj.num_sites();
    let mut basis = Vec::with_capacity(dim);
    for i in 0..dim {
        basis.push(basis_obj.basis_state_at(i));
    }
    let inverse_basis = basis_obj.inverse_basis().clone();
    let site_base = basis_obj.site_base().to_vec();
    let local_dims: Vec<usize> = (0..num_sites)
        .map(|site| basis_obj.local_dim(site))
        .collect();

    let basis_info = BasisInfo {
        dim,
        basis,
        inverse_basis,
        num_sites,
        site_base,
        local_dims,
    };

    Ok(SolverResult {
        energy,
        eigenvector,
        basis_info,
        lanczos_log,
        inverse_iteration_log,
    })
}
