//! Solver result types for SU(2) symmetric models.

use pyo3::prelude::*;

use crate::blas::{InverseIterationLog, LanczosLog};

/// Result of solving an SU(2) symmetric model.
///
/// Unlike `SolverResult`, this does not include `BasisInfo` since SU(2) basis
/// states are represented differently (as intermediate spin quantum numbers).
#[pyclass]
pub struct SU2SolverResult {
    /// Ground state energy
    #[pyo3(get)]
    pub energy: f64,
    /// Ground state eigenvector
    #[pyo3(get)]
    pub eigenvector: Vec<f64>,
    /// Lanczos solver log
    #[pyo3(get)]
    pub lanczos_log: LanczosLog,
    /// Inverse iteration solver log
    #[pyo3(get)]
    pub inverse_iteration_log: InverseIterationLog,
}
