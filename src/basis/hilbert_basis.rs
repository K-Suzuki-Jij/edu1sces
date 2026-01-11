use ahash::AHashMap;
use anyhow::{bail, Result};

/// Return the local basis index (digit) at `site` from packed `basis`.
#[inline]
pub fn find_local_basis(basis: i128, site_base: i128, local_dim: usize) -> usize {
    ((basis / site_base).rem_euclid(local_dim as i128)) as usize
}

pub trait HilbertBasis: Sync {
    fn dim(&self) -> usize;
    fn basis_state_at(&self, row: usize) -> i128;
    fn inverse_basis(&self) -> &AHashMap<i128, usize>;
    fn site_base(&self) -> &[i128];

    /// Return the number of sites.
    fn num_sites(&self) -> usize {
        self.site_base().len()
    }

    /// Return the local Hilbert space dimension at `site`.
    fn local_dim(&self, site: usize) -> usize;

    /// Map basis state to CSR column index.
    fn inverse_basis_at(&self, basis_state: i128) -> Result<usize> {
        let Some(&col) = self.inverse_basis().get(&basis_state) else {
            bail!("basis_state not found in inverse_basis");
        };
        Ok(col)
    }

    /// Return the local basis index (digit) at `site` from packed `basis`.
    fn find_local_basis(&self, basis: i128, site: usize) -> usize {
        find_local_basis(basis, self.site_base()[site], self.local_dim(site))
    }
}
