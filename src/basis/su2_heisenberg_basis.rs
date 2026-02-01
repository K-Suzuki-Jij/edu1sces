use ahash::AHashMap;

/// SU(2) symmetry-adapted basis for Heisenberg model.
///
/// Each basis state is represented as a path of intermediate total spins:
/// `[2*J_1, 2*J_2, ..., 2*J_N]` where:
/// - `J_1 = S_1` (first site's spin)
/// - `J_k` = total spin after coupling sites 0..k-1
/// - `J_N = S_total` (target total spin)
#[derive(Debug, Clone)]
pub struct SU2HeisenbergBasis {
    /// Basis states: basis[i] = [2*J_1, ..., 2*J_N]
    pub basis: Vec<Vec<u8>>,
    /// Inverse mapping: state -> index
    pub inverse_basis: AHashMap<Vec<u8>, usize>,
    /// 2*S_i for each site
    pub two_s_list: Vec<i32>,
    /// 2*S_total (target total spin)
    pub two_s_total: i32,
}

impl SU2HeisenbergBasis {
    pub fn new(basis: Vec<Vec<u8>>, two_s_list: Vec<i32>, two_s_total: i32) -> Self {
        let inverse_basis: AHashMap<Vec<u8>, usize> = basis
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Self {
            basis,
            inverse_basis,
            two_s_list,
            two_s_total,
        }
    }

    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    pub fn num_sites(&self) -> usize {
        self.two_s_list.len()
    }
}
