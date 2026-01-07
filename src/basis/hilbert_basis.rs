use ahash::AHashMap;

pub trait HilbertBasis: Sync {
    fn dim(&self) -> usize;
    fn basis_state_at(&self, row: usize) -> i128;
    fn inverse_basis(&self) -> &AHashMap<i128, usize>;
    fn site_base(&self) -> &[i128];
}