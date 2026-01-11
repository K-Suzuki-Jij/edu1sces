use ahash::AHashMap;

use crate::blas::{InverseIterationLog, InverseIterationParameters, LanczosLog};

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
    /// Lanczos solver log
    pub lanczos_log: LanczosLog,
    /// Inverse iteration solver log
    pub inverse_iteration_log: InverseIterationLog,
}

/// Parameters for the solver.
#[derive(Debug, Clone)]
pub struct SolverParameters {
    /// Convergence threshold for eigenvalue (Lanczos)
    pub eigenvalue_tol: f64,
    /// Minimum Lanczos iterations
    pub min_step: usize,
    /// Maximum Lanczos iterations
    pub max_step: usize,
    /// Number of threads for parallel computation
    pub num_threads: usize,
    /// Parameters for inverse iteration (eigenvector refinement)
    pub inverse_iteration_params: InverseIterationParameters,
}
