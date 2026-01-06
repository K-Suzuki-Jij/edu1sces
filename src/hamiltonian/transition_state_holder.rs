use ahash::AHashMap;

/// Thread-local holder for Hamiltonian transitions generated
/// from a single source basis state.
pub struct TransitionStateHolder {
    /// Matrix element values keyed by destination basis
    pub vals: AHashMap<i128, f64>,

    /// Per-site stride (site constant)
    /// site_base[site] = product of local dimensions of sites < site
    pub site_base: Vec<i128>,

    /// Local basis index at each site for the current source basis
    /// local_basis[site] in 0..local_dim(site)
    pub local_basis: Vec<usize>,

    /// Threshold for treating values as zero
    pub zero_eps: f64,
}
