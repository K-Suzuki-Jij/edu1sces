use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::basis::Basis;
use crate::blas::{
    csr_transpose, dot, CsrMatrix, InverseIterationLog, InverseIterationParameters, LanczosLog,
};
use crate::model::QuantumModel;
use crate::utility::rayon_pool::build_pool;

/// Basis information required for expectation value calculations.
pub struct BasisInfo {
    /// Number of sites
    pub num_sites: usize,
    /// Site base for basis encoding (site_base[i] = product of local_dims[0..i])
    pub site_base: Vec<i128>,
    /// Local Hilbert space dimension for each site
    pub local_dims: Vec<usize>,
    /// Current sector's quantum numbers
    pub current_quantum_numbers: Vec<i32>,
    /// Model for constructing bases in different sectors
    pub model: Box<dyn QuantumModel>,
    /// Cache of bases for different quantum number sectors
    pub basis_cache: HashMap<Vec<i32>, Basis>,
}

impl BasisInfo {
    /// Ensure a basis exists for the specified quantum numbers.
    /// This builds the basis if it doesn't exist yet.
    pub fn ensure_basis_exists(&mut self, quantum_numbers: &[i32]) -> Result<()> {
        let qn_vec = quantum_numbers.to_vec();
        if !self.basis_cache.contains_key(&qn_vec) {
            let new_basis = self.model.build_basis(quantum_numbers)?;
            self.basis_cache.insert(qn_vec, new_basis);
        }
        Ok(())
    }

    /// Get a reference to a cached basis.
    /// Call `ensure_basis_exists` first.
    pub fn get_basis(&self, quantum_numbers: &[i32]) -> Result<&Basis> {
        self.basis_cache
            .get(quantum_numbers)
            .ok_or_else(|| anyhow::anyhow!("basis not found in cache for {:?}", quantum_numbers))
    }
}

/// Result of solving a model.
#[pyclass]
pub struct SolverResult {
    /// Eigenvalues (energies) for each computed state
    #[pyo3(get)]
    pub energies: Vec<f64>,
    /// Eigenvectors for each computed state
    #[pyo3(get)]
    pub eigenvectors: Vec<Vec<f64>>,
    /// Basis information for expectation value calculations
    pub basis_info: BasisInfo,
    /// Lanczos solver logs for each state
    #[pyo3(get)]
    pub lanczos_logs: Vec<LanczosLog>,
    /// Inverse iteration solver logs for each state
    #[pyo3(get)]
    pub inverse_iteration_logs: Vec<InverseIterationLog>,
}

#[pymethods]
impl SolverResult {
    /// Compute expectation value of a local operator at a specific site.
    ///
    /// # Arguments
    /// * `local_op` - CSR matrix representing the local operator
    /// * `site` - Site index
    /// * `num_threads` - Number of threads for parallel computation
    /// * `state_index` - Index of the eigenstate (0 = ground state)
    ///
    /// # Returns
    /// Expectation value `<psi| O_site |psi>`.
    pub fn expectation_onsite(
        &mut self,
        local_op: &CsrMatrix,
        site: usize,
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

        let current_qn = self.basis_info.current_quantum_numbers.clone();

        // Ensure basis exists (should already be cached from solve)
        self.basis_info.ensure_basis_exists(&current_qn)?;

        // Use rayon thread pool with specified number of threads
        let pool = build_pool(num_threads)?;

        // Compute M|psi>
        let m_psi = self.apply_local_op_to_eigenvector(
            state_index,
            local_op,
            site,
            &current_qn,
            &current_qn,
            &pool,
        )?;

        // Compute <psi|M|psi>
        dot(&pool, &self.eigenvectors[state_index], &m_psi)
    }

    /// Compute two-point correlation function <psi|O1_{site1} O2_{site2}|psi>.
    ///
    /// When the operators change quantum numbers (e.g., S+, S-), the intermediate
    /// basis is automatically constructed and cached.
    ///
    /// # Arguments
    /// * `op1` - CSR matrix representing local operator 1
    /// * `site1` - Site index for operator 1
    /// * `op2` - CSR matrix representing local operator 2
    /// * `site2` - Site index for operator 2
    /// * `num_threads` - Number of threads for parallel computation
    /// * `state_index` - Index of the eigenstate (0 = ground state)
    ///
    /// # Returns
    /// Correlation value <psi|O1_{site1} O2_{site2}|psi>.
    pub fn correlation_function(
        &mut self,
        op1: &CsrMatrix,
        site1: usize,
        op2: &CsrMatrix,
        site2: usize,
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

        // Compute all possible quantum number transitions for each operator
        let transitions_op1 = self.compute_all_quantum_number_transitions(op1, site1);
        let transitions_op2 = self.compute_all_quantum_number_transitions(op2, site2);

        let current_qn = self.basis_info.current_quantum_numbers.clone();

        // Collect all intermediate quantum numbers that will be needed
        let mut intermediate_qns: Vec<Vec<i32>> = Vec::new();
        for delta_qn2 in &transitions_op2 {
            let intermediate_qn: Vec<i32> = current_qn
                .iter()
                .zip(delta_qn2.iter())
                .map(|(q, dq)| q + dq)
                .collect();

            let needed_delta_qn1: Vec<i32> = delta_qn2.iter().map(|d| -d).collect();
            if transitions_op1.contains(&needed_delta_qn1) {
                intermediate_qns.push(intermediate_qn);
            }
        }

        // Build all required bases upfront (mutable borrow ends here)
        self.basis_info.ensure_basis_exists(&current_qn)?;
        for qn in &intermediate_qns {
            self.basis_info.ensure_basis_exists(qn)?;
        }

        // Pre-compute transpose of op1 for efficient adjoint application
        // O1† = (O1)^T for real matrices
        let op1_transpose = csr_transpose(1.0, op1)?;

        // Use rayon thread pool with specified number of threads
        let pool = build_pool(num_threads)?;

        // Now compute with immutable borrows only
        let mut total = 0.0;

        for intermediate_qn in &intermediate_qns {
            // O2|psi>: current -> intermediate
            let vec_o2_psi = self.apply_local_op_to_eigenvector(
                state_index,
                op2,
                site2,
                &current_qn,
                intermediate_qn,
                &pool,
            )?;

            // O1†|psi>: current -> intermediate
            // Using transpose: O1† = O1^T, so we apply the transposed matrix
            let vec_o1dag_psi = self.apply_local_op_to_eigenvector(
                state_index,
                &op1_transpose,
                site1,
                &current_qn,
                intermediate_qn,
                &pool,
            )?;

            // Inner product in intermediate sector: <O1†psi | O2 psi>
            total += vec_o1dag_psi
                .iter()
                .zip(vec_o2_psi.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }

        Ok(total)
    }
}

impl SolverResult {
    /// Compute all possible quantum number transitions from a local operator.
    /// Returns a set of unique delta_qn vectors.
    fn compute_all_quantum_number_transitions(
        &self,
        local_op: &CsrMatrix,
        site: usize,
    ) -> Vec<Vec<i32>> {
        let mut transitions = Vec::new();

        for row in 0..local_op.row_dim {
            for p in local_op.rows[row]..local_op.rows[row + 1] {
                let col = local_op.cols[p];
                if local_op.vals[p].abs() > 1e-14 {
                    // Found transition: col -> row
                    let qn_in = self.basis_info.model.quantum_numbers(site, col);
                    let qn_out = self.basis_info.model.quantum_numbers(site, row);
                    let delta_qn: Vec<i32> = qn_out
                        .iter()
                        .zip(qn_in.iter())
                        .map(|(o, i)| o - i)
                        .collect();

                    if !transitions.contains(&delta_qn) {
                        transitions.push(delta_qn);
                    }
                }
            }
        }

        if transitions.is_empty() {
            // Zero operator or no transitions
            transitions.push(vec![0; self.basis_info.current_quantum_numbers.len()]);
        }

        transitions
    }

    /// Apply a local operator to the eigenvector.
    /// Returns the result vector in the output basis.
    /// Both bases must already exist in the cache.
    fn apply_local_op_to_eigenvector(
        &self,
        state_index: usize,
        local_op: &CsrMatrix,
        site: usize,
        in_qn: &[i32],
        out_qn: &[i32],
        pool: &rayon::ThreadPool,
    ) -> Result<Vec<f64>> {
        let out_basis = self.basis_info.get_basis(out_qn)?;
        let in_basis = self.basis_info.get_basis(in_qn)?;
        let eigenvector = &self.eigenvectors[state_index];
        let site_base = out_basis.site_base[site];

        Ok(pool.install(|| {
            out_basis
                .basis
                .par_iter()
                .map(|&basis_out| {
                    let local_basis_out = out_basis.find_local_basis(basis_out, site);

                    let mut temp_val = 0.0;

                    for p in local_op.rows[local_basis_out]..local_op.rows[local_basis_out + 1] {
                        let local_basis_in = local_op.cols[p];
                        let mat_val = local_op.vals[p];

                        let a_basis_in = basis_out
                            + ((local_basis_in as i128) - (local_basis_out as i128)) * site_base;

                        if let Some(&i_in) = in_basis.inverse_basis.get(&a_basis_in) {
                            temp_val += eigenvector[i_in] * mat_val;
                        }
                    }

                    temp_val
                })
                .collect()
        }))
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
    /// Number of eigenstates to compute (1 = ground state only)
    pub num_states: usize,
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
        num_states: usize,
    ) -> Self {
        Self {
            eigenvalue_tol,
            min_step,
            max_step,
            num_threads,
            inverse_iteration_params,
            output_log,
            num_states,
        }
    }
}
