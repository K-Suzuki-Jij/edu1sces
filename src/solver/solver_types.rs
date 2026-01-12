use ahash::AHashMap;
use pyo3::prelude::*;

use crate::basis::find_local_basis;
use crate::blas::{CsrMatrix, InverseIterationLog, InverseIterationParameters, LanczosLog};

/// Basis information required for expectation value calculations.
pub struct BasisInfo {
    /// Hilbert space dimension
    pub dim: usize,
    /// Basis states
    pub basis: Vec<i128>,
    /// Inverse basis mapping (state -> index)
    pub inverse_basis: AHashMap<i128, usize>,
    /// Number of sites
    pub num_sites: usize,
    /// Site base for basis encoding (site_base[i] = product of local_dims[0..i])
    pub site_base: Vec<i128>,
    /// Local Hilbert space dimension for each site
    pub local_dims: Vec<usize>,
}

/// Result of solving a model.
#[pyclass]
pub struct SolverResult {
    /// Ground state energy
    #[pyo3(get)]
    pub energy: f64,
    /// Ground state eigenvector
    #[pyo3(get)]
    pub eigenvector: Vec<f64>,
    /// Basis information for expectation value calculations
    pub basis_info: BasisInfo,
    /// Lanczos solver log
    #[pyo3(get)]
    pub lanczos_log: LanczosLog,
    /// Inverse iteration solver log
    #[pyo3(get)]
    pub inverse_iteration_log: InverseIterationLog,
}

#[pymethods]
impl SolverResult {
    /// Hilbert space dimension (convenience accessor)
    pub fn dim(&self) -> usize {
        self.basis_info.dim
    }

    /// Compute expectation value of a local operator at a specific site.
    ///
    /// # Arguments
    /// * `local_op` - CSR matrix representing the local operator
    /// * `site` - Site index
    ///
    /// # Returns
    /// Expectation value `<psi| O_site |psi>`.
    pub fn expectation_onsite(&self, local_op: &CsrMatrix, site: usize) -> f64 {
        let info = &self.basis_info;
        let site_base = info.site_base[site];
        let local_dim = info.local_dims[site];

        // Compute M|psi> and then <psi|M|psi>
        let mut result = 0.0;

        for (i_out, &basis_out) in info.basis.iter().enumerate() {
            let local_basis_out = find_local_basis(basis_out, site_base, local_dim);

            let mut temp_val = 0.0;

            // Apply local operator: sum over matrix elements
            for j in local_op.rows[local_basis_out]..local_op.rows[local_basis_out + 1] {
                let local_basis_in = local_op.cols[j];
                let mat_val = local_op.vals[j];

                // Compute the input basis state
                let a_basis_in =
                    basis_out + ((local_basis_in as i128) - (local_basis_out as i128)) * site_base;

                // Look up the index of the input basis state
                if let Some(&i_in) = info.inverse_basis.get(&a_basis_in) {
                    temp_val += self.eigenvector[i_in] * mat_val;
                }
            }

            // Inner product contribution: psi[i_out] * (M|psi>)[i_out]
            result += self.eigenvector[i_out] * temp_val;
        }

        result
    }
}

/// Parameters for the solver.
#[derive(Debug, Clone)]
#[pyclass(get_all)]
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
    /// If true, print progress to stderr (overwrites same line with \r)
    pub output_log: bool,
}

#[pymethods]
impl SolverParameters {
    #[new]
    fn new(
        eigenvalue_tol: f64,
        min_step: usize,
        max_step: usize,
        num_threads: usize,
        inverse_iteration_params: InverseIterationParameters,
        output_log: bool,
    ) -> Self {
        Self {
            eigenvalue_tol,
            min_step,
            max_step,
            num_threads,
            inverse_iteration_params,
            output_log,
        }
    }
}
