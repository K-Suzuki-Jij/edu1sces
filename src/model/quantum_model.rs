use anyhow::Result;

use crate::basis::Basis;

/// Trait for quantum many-body models.
///
/// This trait provides the common interface for building bases in different
/// quantum number sectors and computing expectation values.
pub trait QuantumModel: Send + Sync {
    /// Number of lattice sites.
    fn num_sites(&self) -> usize;

    /// Local Hilbert space dimension at the specified site.
    fn local_dim(&self, site: usize) -> usize;

    /// Return the quantum numbers for a local state at a given site.
    /// For Hubbard: [n, 2*sz]
    /// For Heisenberg: [2*sz]
    fn quantum_numbers(&self, site: usize, local_state: usize) -> Vec<i32>;

    /// Build a basis for the specified quantum number sector.
    fn build_basis(&self, target_quantum_numbers: &[i32]) -> Result<Basis>;
}
