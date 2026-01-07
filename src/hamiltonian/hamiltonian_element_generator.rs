use anyhow::Result;

use crate::hamiltonian::transition_state_holder::TransitionStateHolder;

/// Generate all Hamiltonian nonzero elements for a given basis state (row).
pub trait HamiltonianElementGenerator<Basis>: Sync {
    /// Generate transition elements from `basis_state` and store them into `holder`.
    fn make_elements(
        &self,
        basis_state: i128,
        basis: &Basis,
        holder: &mut TransitionStateHolder,
    ) -> Result<()>;
}
