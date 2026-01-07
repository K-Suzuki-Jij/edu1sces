use ahash::AHashMap;
use anyhow::{bail, Result};

pub trait HilbertBasis: Sync {
    fn dim(&self) -> usize;
    fn basis_state_at(&self, row: usize) -> i128;
    fn inverse_basis(&self) -> &AHashMap<i128, usize>;
    fn site_base(&self) -> &[i128];

    /// Map basis state to CSR column index.
    fn inverse_basis_at(&self, basis_state: i128) -> Result<usize> {
        let Some(&col) = self.inverse_basis().get(&basis_state) else {
            bail!("basis_state not found in inverse_basis");
        };
        Ok(col)
    }
}
