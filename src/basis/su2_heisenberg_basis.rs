use ahash::AHashMap;

/// SU(2) symmetry-adapted basis for Heisenberg model.
///
/// Each basis state is represented as a path of intermediate total spins:
/// `[2*J_1, 2*J_2, ..., 2*J_N]` where:
/// - `J_k` = total spin after coupling the first k sites (according to `coupling_order`)
/// - `J_N = S_total` (target total spin)
///
/// The `coupling_order` specifies the order in which sites are coupled:
/// - `coupling_order[k]` = original site index of the (k+1)-th coupled site
/// - For left-coupling basis with natural order: `coupling_order = [0, 1, 2, ..., N-1]`
#[derive(Debug, Clone)]
pub struct SU2HeisenbergBasis {
    /// Basis states: basis[i] = [2*J_1, ..., 2*J_N]
    pub basis: Vec<Vec<u8>>,
    /// Inverse mapping: state -> index
    pub inverse_basis: AHashMap<Vec<u8>, usize>,
    /// 2*S_i for each site in coupling order (i.e., two_s_list[k] = 2*S_{coupling_order[k]})
    pub two_s_list: Vec<i32>,
    /// 2*S_total (target total spin)
    pub two_s_total: i32,
    /// Site coupling order: coupling_order[k] = original site index at position k
    pub coupling_order: Vec<usize>,
}

impl SU2HeisenbergBasis {
    pub fn new(
        basis: Vec<Vec<u8>>,
        two_s_list: Vec<i32>,
        two_s_total: i32,
        coupling_order: Vec<usize>,
    ) -> Self {
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
            coupling_order,
        }
    }

    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    pub fn num_sites(&self) -> usize {
        self.two_s_list.len()
    }

    /// Get the coupling position of an original site index.
    /// Returns the position in the coupling order (0-indexed).
    pub fn site_to_position(&self, site: usize) -> Option<usize> {
        self.coupling_order.iter().position(|&s| s == site)
    }
}
