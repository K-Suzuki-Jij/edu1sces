use ahash::AHashMap;
use anyhow::Result;

use crate::blas::{lanczos, CsrMatrix, LanczosParameters};

/// Trait for models that can be solved using the Lanczos algorithm.
///
/// A model must be able to produce a basis and a Hamiltonian matrix.
pub trait Solvable {
    type Basis: crate::basis::HilbertBasis;

    /// Build the basis for this model.
    fn build_basis(&self) -> Result<Self::Basis>;

    /// Build the Hamiltonian matrix for this model given the basis.
    fn build_hamiltonian(&self, basis: &Self::Basis, num_threads: usize) -> Result<CsrMatrix>;
}

/// Result of solving a model.
pub struct SolverResult {
    /// Hilbert space dimension
    pub dim: usize,
    /// Ground state energy
    pub energy: f64,
    /// Ground state eigenvector
    pub eigenvector: Vec<f64>,
    /// Basis states
    pub basis: Vec<i128>,
    /// Inverse basis mapping (state -> index)
    pub inverse_basis: AHashMap<i128, usize>,
}

/// Parameters for the solver.
#[derive(Debug, Clone)]
pub struct SolverParameters {
    /// Convergence threshold for eigenvalue
    pub acc: f64,
    /// Minimum Lanczos iterations
    pub min_step: usize,
    /// Maximum Lanczos iterations
    pub max_step: usize,
    /// Number of threads for parallel computation
    pub num_threads: usize,
}

impl Default for SolverParameters {
    fn default() -> Self {
        Self {
            acc: 1e-10,
            min_step: 5,
            max_step: 1000,
            num_threads: 1,
        }
    }
}

/// Solve a model to find the ground state energy and eigenvector.
pub fn solve<M: Solvable>(model: &M, params: &SolverParameters) -> Result<SolverResult> {
    // Build basis from model
    let basis_obj = model.build_basis()?;
    let dim = basis_obj.dim();

    // Build Hamiltonian
    let hamiltonian = model.build_hamiltonian(&basis_obj, params.num_threads)?;

    // Prepare Lanczos parameters
    let lanczos_params = LanczosParameters {
        acc: params.acc,
        min_step: params.min_step,
        max_step: params.max_step,
        calc_eigenvec: true,
    };

    // Run Lanczos
    let mut eigenvector = Vec::new();
    let mut energy = 0.0;
    lanczos(
        &hamiltonian,
        &mut eigenvector,
        &mut energy,
        &lanczos_params,
        params.num_threads,
    )?;

    // Extract basis data
    let mut basis = Vec::with_capacity(dim);
    for i in 0..dim {
        basis.push(basis_obj.basis_state_at(i));
    }
    let inverse_basis = basis_obj.inverse_basis().clone();

    Ok(SolverResult {
        dim,
        energy,
        eigenvector,
        basis,
        inverse_basis,
    })
}
