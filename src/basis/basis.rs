use ahash::AHashMap;

/// Unified basis structure for all quantum models.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Basis states as packed integers
    pub basis: Vec<i128>,
    /// Inverse mapping: basis state -> index
    pub inverse_basis: AHashMap<i128, usize>,
    /// Site base for basis encoding (site_base[i] = product of local_dims[0..i])
    pub site_base: Vec<i128>,
    /// Local Hilbert space dimension for each site
    pub local_dims: Vec<usize>,
}

impl Basis {
    /// Create a new Basis from components.
    pub fn new(
        basis: Vec<i128>,
        inverse_basis: AHashMap<i128, usize>,
        site_base: Vec<i128>,
        local_dims: Vec<usize>,
    ) -> Self {
        Self {
            basis,
            inverse_basis,
            site_base,
            local_dims,
        }
    }

    /// Hilbert space dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Number of sites.
    #[inline]
    pub fn num_sites(&self) -> usize {
        self.site_base.len()
    }

    /// Get basis state at index.
    #[inline]
    pub fn basis_state_at(&self, row: usize) -> i128 {
        self.basis[row]
    }

    /// Local dimension at site.
    #[inline]
    pub fn local_dim(&self, site: usize) -> usize {
        self.local_dims[site]
    }

    /// Return the local basis index (digit) at `site` from packed `basis`.
    #[inline]
    pub fn find_local_basis(&self, basis: i128, site: usize) -> usize {
        let site_base = self.site_base[site];
        let local_dim = self.local_dims[site];
        ((basis / site_base).rem_euclid(local_dim as i128)) as usize
    }
}
