use ahash::AHashMap;
use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::basis::SU2HeisenbergBasis;

/// Spin-S Heisenberg model with SU(2) symmetry
///
/// Only isotropic exchange interactions J * S_i · S_j are allowed.
/// No Zeeman field or single-ion anisotropy (they break SU(2)).
#[pyclass]
#[derive(Debug, Clone)]
pub struct SU2HeisenbergModel {
    /// number of lattice sites
    #[pyo3(get)]
    pub num_sites: usize,

    /// 2S_i for each site i (integer), length = num_sites
    #[pyo3(get)]
    pub two_s_list: Vec<i32>,

    /// isotropic exchange J_ij (normalized: i < j):
    /// J_ij * S_i · S_j = J_ij * (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)
    ///                  = J_ij * ((1/2)(S_i^+ S_j^- + S_i^- S_j^+) + Sz_i Sz_j)
    #[pyo3(get)]
    pub exchange: HashMap<(usize, usize), f64>,

    /// Diagonal shift from self-interactions: Σ_i J_ii * S_i(S_i+1)
    #[pyo3(get)]
    pub diagonal_shift: f64,
}

impl SU2HeisenbergModel {
    fn check_pairs(name: &str, num_sites: usize, map: &HashMap<(usize, usize), f64>) -> Result<()> {
        for (&(i, j), _) in map.iter() {
            if i >= num_sites || j >= num_sites {
                bail!(
                    "{} ({}, {}) refers to out-of-range site (num_sites = {})",
                    name,
                    i,
                    j,
                    num_sites
                );
            }
        }
        Ok(())
    }

    /// Build SU(2) basis for a fixed total spin S_total.
    ///
    /// Automatically optimizes the coupling order using Cuthill-McKee algorithm
    /// to minimize the number of swap operations in Hamiltonian construction.
    pub fn build_basis(&self, total_s: f64) -> Result<SU2HeisenbergBasis> {
        let order = self.optimal_coupling_order();
        self.build_basis_with_order(total_s, &order)
    }

    /// Build SU(2) basis with a specified coupling order.
    ///
    /// Each basis state is a Vec<u8> of length N:
    ///   [2*J_1, 2*J_2, ..., 2*J_N]
    /// where
    ///   J_k = total spin after coupling the first k sites (according to coupling_order)
    ///   J_N = S_total (target).
    ///
    /// `coupling_order[k]` specifies the original site index at coupling position k.
    /// For example, `coupling_order = [2, 0, 1]` means:
    ///   - First couple site 2
    ///   - Then couple site 0
    ///   - Finally couple site 1
    ///
    /// The stored values are 2*J (integers) in u8, so we require sum(2S_i) <= 255.
    pub fn build_basis_with_order(
        &self,
        total_s: f64,
        coupling_order: &[usize],
    ) -> Result<SU2HeisenbergBasis> {
        let two_s_f = 2.0 * total_s;
        let two_s_target = two_s_f.round() as i32;

        if (two_s_f - (two_s_target as f64)).abs() > 1e-12 {
            bail!("total_s must be integer or half-integer (got {})", total_s);
        }
        if two_s_target < 0 {
            bail!("total_s must be non-negative (got {})", total_s);
        }

        let n = self.num_sites;

        // Validate coupling_order
        if coupling_order.len() != n {
            bail!(
                "coupling_order length {} does not match num_sites {}",
                coupling_order.len(),
                n
            );
        }
        let mut seen = vec![false; n];
        for &site in coupling_order {
            if site >= n {
                bail!("coupling_order contains invalid site index {}", site);
            }
            if seen[site] {
                bail!("coupling_order contains duplicate site index {}", site);
            }
            seen[site] = true;
        }

        // For basis construction, we need spins in coupling order
        let two_s_in_order: Vec<i32> = coupling_order
            .iter()
            .map(|&site| self.two_s_list[site])
            .collect();

        let sum_two_s: i32 = two_s_in_order.iter().sum();
        if sum_two_s > (u8::MAX as i32) {
            bail!(
                "sum(2S_i) = {} exceeds u8::MAX (=255); cannot store 2J in u8",
                sum_two_s
            );
        }

        // Parity + range checks
        if ((sum_two_s - two_s_target) & 1) != 0 {
            return Ok(SU2HeisenbergBasis::new(
                Vec::new(),
                self.two_s_list.clone(),
                two_s_target,
                coupling_order.to_vec(),
            ));
        }
        if two_s_target > sum_two_s {
            return Ok(SU2HeisenbergBasis::new(
                Vec::new(),
                self.two_s_list.clone(),
                two_s_target,
                coupling_order.to_vec(),
            ));
        }

        // N=1: only possible if S_tot == S_1
        if n == 1 {
            let two_s1 = two_s_in_order[0];
            if two_s1 == two_s_target {
                let state = vec![two_s1 as u8];
                return Ok(SU2HeisenbergBasis::new(
                    vec![state],
                    self.two_s_list.clone(),
                    two_s_target,
                    coupling_order.to_vec(),
                ));
            }
            return Ok(SU2HeisenbergBasis::new(
                Vec::new(),
                self.two_s_list.clone(),
                two_s_target,
                coupling_order.to_vec(),
            ));
        }

        // suffix_sum_two_s[i] = sum_{k=i}^{n-1} two_s_in_order[k]
        let mut suffix_sum_two_s = vec![0i32; n + 1];
        for i in (0..n).rev() {
            suffix_sum_two_s[i] = suffix_sum_two_s[i + 1] + two_s_in_order[i];
        }

        // Reachability check: can we reach target from current 2J with remaining spins?
        #[inline]
        fn can_reach(cur: i32, rem_sum: i32, target: i32) -> bool {
            if target > cur + rem_sum {
                return false;
            }
            ((target - (cur + rem_sum)) & 1) == 0
        }

        let two_s1 = two_s_in_order[0];
        if !can_reach(two_s1, suffix_sum_two_s[1], two_s_target) {
            return Ok(SU2HeisenbergBasis::new(
                Vec::new(),
                self.two_s_list.clone(),
                two_s_target,
                coupling_order.to_vec(),
            ));
        }

        let mut basis: Vec<Vec<u8>> = Vec::new();
        let mut path: Vec<u8> = Vec::with_capacity(n);
        path.push(two_s1 as u8); // J_1 = S_1 (fixed)

        fn dfs(
            site: usize,
            n: usize,
            two_s_list: &[i32],
            suffix_sum_two_s: &[i32],
            two_s_target: i32,
            cur_two_j: i32,
            path: &mut Vec<u8>,
            out: &mut Vec<Vec<u8>>,
        ) {
            #[inline]
            fn can_reach(cur: i32, rem_sum: i32, target: i32) -> bool {
                if target > cur + rem_sum {
                    return false;
                }
                ((target - (cur + rem_sum)) & 1) == 0
            }

            if site == n {
                if cur_two_j == two_s_target {
                    out.push(path.clone());
                }
                return;
            }

            let two_s_site = two_s_list[site];
            let min_two_j = (cur_two_j - two_s_site).abs();
            let max_two_j = cur_two_j + two_s_site;
            let rem_sum = suffix_sum_two_s[site + 1];

            let mut two_j_new = min_two_j;
            while two_j_new <= max_two_j {
                if can_reach(two_j_new, rem_sum, two_s_target) {
                    path.push(two_j_new as u8);
                    dfs(
                        site + 1,
                        n,
                        two_s_list,
                        suffix_sum_two_s,
                        two_s_target,
                        two_j_new,
                        path,
                        out,
                    );
                    path.pop();
                }
                two_j_new += 2;
            }
        }

        dfs(
            1,
            n,
            &two_s_in_order,
            &suffix_sum_two_s,
            two_s_target,
            two_s1,
            &mut path,
            &mut basis,
        );

        basis.sort_unstable();

        Ok(SU2HeisenbergBasis::new(
            basis,
            self.two_s_list.clone(),
            two_s_target,
            coupling_order.to_vec(),
        ))
    }

    /// Cuthill-McKee algorithm to find a site ordering that minimizes bandwidth.
    ///
    /// The algorithm:
    /// 1. Build adjacency list from bonds
    /// 2. Start BFS from a peripheral node (node with minimum degree)
    /// 3. Visit neighbors in order of increasing degree
    ///
    /// Returns a permutation `order` where `order[position] = original_site_index`.
    ///
    /// For a 1D chain with nearest-neighbor bonds, this produces the natural
    /// ordering [0, 1, 2, ..., n-1] (starting from an endpoint).
    pub fn cuthill_mckee_order(num_sites: usize, bonds: &[(usize, usize)]) -> Vec<usize> {
        if num_sites == 0 {
            return Vec::new();
        }
        if num_sites == 1 {
            return vec![0];
        }

        // Build adjacency list
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_sites];
        for &(i, j) in bonds {
            if i != j && i < num_sites && j < num_sites {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }

        let mut order = Vec::with_capacity(num_sites);
        let mut visited = vec![false; num_sites];

        // Process all connected components
        while order.len() < num_sites {
            // Find the unvisited node with minimum degree (peripheral node)
            let start = (0..num_sites)
                .filter(|&v| !visited[v])
                .min_by_key(|&v| adj[v].len())
                .unwrap();

            // BFS from the starting node
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            order.push(start);

            while let Some(u) = queue.pop_front() {
                // Get unvisited neighbors sorted by degree (ascending)
                let mut neighbors: Vec<usize> =
                    adj[u].iter().copied().filter(|&v| !visited[v]).collect();
                neighbors.sort_by_key(|&v| adj[v].len());

                for v in neighbors {
                    if !visited[v] {
                        visited[v] = true;
                        order.push(v);
                        queue.push_back(v);
                    }
                }
            }
        }

        order
    }

    /// Returns an optimal coupling order for the Hamiltonian construction.
    ///
    /// Uses Cuthill-McKee algorithm to find an ordering that minimizes the
    /// bandwidth of the interaction graph, which reduces the number of swap
    /// operations needed in the Hamiltonian construction.
    pub fn optimal_coupling_order(&self) -> Vec<usize> {
        let bonds: Vec<(usize, usize)> = self.exchange.keys().copied().collect();
        Self::cuthill_mckee_order(self.num_sites, &bonds)
    }
}

#[pymethods]
impl SU2HeisenbergModel {
    #[pyo3(text_signature = "(spin_list, exchange)")]
    #[new]
    pub fn new(spin_list: Vec<f64>, exchange: HashMap<(usize, usize), f64>) -> Result<Self> {
        let num_sites = spin_list.len();
        if num_sites == 0 {
            bail!("num_sites must be positive");
        }

        let mut two_s_list = vec![0; num_sites];
        for (i, &s) in spin_list.iter().enumerate() {
            let two_s_f = (2.0 * s).round();
            if (2.0 * s - two_s_f).abs() > 1e-12 {
                bail!(
                    "Spin at site {} must be integer or half-integer (got {})",
                    i,
                    s
                );
            }
            if two_s_f < 0.0 {
                bail!("Spin at site {} must be non-negative (got {})", i, s);
            }
            two_s_list[i] = two_s_f as i32;
        }

        Self::check_pairs("exchange", num_sites, &exchange)?;

        // Normalize exchange: separate self-interactions and pair interactions
        // S_i · S_j = S_j · S_i, so (i, j) and (j, i) should be merged
        let mut normalized_exchange = HashMap::new();
        let mut diagonal_shift = 0.0;
        for (&(i, j), &v) in exchange.iter() {
            if i == j {
                let two_si = two_s_list[i];
                let si_si_plus_1 = (two_si as f64 / 2.0) * (two_si as f64 / 2.0 + 1.0);
                diagonal_shift += v * si_si_plus_1;
            } else {
                let key = if i < j { (i, j) } else { (j, i) };
                *normalized_exchange.entry(key).or_insert(0.0) += v;
            }
        }

        Ok(Self {
            num_sites,
            two_s_list,
            exchange: normalized_exchange,
            diagonal_shift,
        })
    }

    /// Dimension of the SU(2) sector with fixed total spin S.
    ///
    /// This returns the dimension of the subspace with total spin S,
    /// which equals multiplicity × (2S + 1).
    ///
    /// `total_s` must be integer or half-integer and non-negative.
    /// Returns 0 if the sector is forbidden for the given local spins.
    #[pyo3(text_signature = "(self, total_s)")]
    pub fn calc_dim_su2_sector(&self, total_s: f64) -> Result<i128> {
        let two_s_f = 2.0 * total_s;
        let two_s_target = two_s_f.round() as i32;

        if (two_s_f - two_s_target as f64).abs() > 1e-12 {
            bail!("total_s must be integer or half-integer (got {total_s})");
        }
        if two_s_target < 0 {
            return Ok(0);
        }

        // dp[two_s] = multiplicity g(S)
        // start from the trivial representation S=0
        let mut dp: AHashMap<i32, i128> = AHashMap::new();
        dp.insert(0, 1);

        for &two_s_site in self.two_s_list.iter() {
            let mut next = AHashMap::<i32, i128>::new();

            for (&two_s_prev, &ways) in dp.iter() {
                // Clebsch-Gordan: |S_prev - S_i| .. S_prev + S_i
                let min_two = (two_s_prev - two_s_site).abs();
                let max_two = two_s_prev + two_s_site;

                let mut two_s_new = min_two;
                while two_s_new <= max_two {
                    *next.entry(two_s_new).or_insert(0) += ways;
                    two_s_new += 2;
                }
            }

            dp = next;
        }

        let mult = *dp.get(&two_s_target).unwrap_or(&0);

        // SU(2) sector dimension = (2S + 1) * multiplicity
        let dim = mult
            .checked_mul((two_s_target as i128) + 1)
            .ok_or_else(|| anyhow::anyhow!("i128 overflow"))?;

        Ok(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_valid_input() {
        let spins = vec![0.5, 1.0, 1.5];

        let mut exchange = HashMap::new();
        exchange.insert((0, 1), 1.0);

        let m = SU2HeisenbergModel::new(spins, exchange).unwrap();
        assert_eq!(m.num_sites, 3);
        assert_eq!(m.two_s_list, vec![1, 2, 3]);
    }

    #[test]
    fn reject_empty_sites() {
        let spins: Vec<f64> = vec![];
        assert!(SU2HeisenbergModel::new(spins, HashMap::new()).is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let spins = vec![0.5, 0.5];

        let mut exchange = HashMap::new();
        exchange.insert((0, 2), 1.0);

        assert!(SU2HeisenbergModel::new(spins, exchange).is_err());
    }

    #[test]
    fn su2_sector_two_spin_half() {
        // Two spin-1/2: 1/2 ⊗ 1/2 = 0 ⊕ 1
        // S=0: mult=1, dim = 1 × (2×0+1) = 1
        // S=1: mult=1, dim = 1 × (2×1+1) = 3
        // Total: 1 + 3 = 4 = 2^2 ✓
        let m = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 1); // singlet
        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 3); // triplet
        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 0); // forbidden (parity)
        assert_eq!(m.calc_dim_su2_sector(2.0).unwrap(), 0); // too large
    }

    #[test]
    fn su2_sector_three_spin_half() {
        // Three spin-1/2: (1/2 ⊗ 1/2) ⊗ 1/2 = (0 ⊕ 1) ⊗ 1/2
        //   = 1/2 ⊕ (1/2 ⊕ 3/2) = 2×(1/2) ⊕ 3/2
        // S=1/2: mult=2, dim = 2 × (2×0.5+1) = 2 × 2 = 4
        // S=3/2: mult=1, dim = 1 × (2×1.5+1) = 1 × 4 = 4
        // Total: 4 + 4 = 8 = 2^3 ✓
        let m = SU2HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 4);
        assert_eq!(m.calc_dim_su2_sector(1.5).unwrap(), 4);
        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 0); // forbidden (parity)
        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 0); // forbidden (parity)
    }

    #[test]
    fn su2_sector_four_spin_half() {
        // Four spin-1/2: 1/2 ⊗ 1/2 ⊗ 1/2 ⊗ 1/2 = 2×(0) ⊕ 3×(1) ⊕ 1×(2)
        // S=0: mult=2, dim = 2 × 1 = 2
        // S=1: mult=3, dim = 3 × 3 = 9
        // S=2: mult=1, dim = 1 × 5 = 5
        // Total: 2 + 9 + 5 = 16 = 2^4 ✓
        let m = SU2HeisenbergModel {
            num_sites: 4,
            two_s_list: vec![1, 1, 1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 2);
        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 9);
        assert_eq!(m.calc_dim_su2_sector(2.0).unwrap(), 5);
        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 0); // forbidden (parity)
    }

    #[test]
    fn su2_sector_spin_half_and_one() {
        // 1/2 ⊗ 1 = 1/2 ⊕ 3/2
        // S=1/2: mult=1, dim = 1 × 2 = 2
        // S=3/2: mult=1, dim = 1 × 4 = 4
        // Total: 2 + 4 = 6 = 2 × 3 ✓
        let m = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 2],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 2);
        assert_eq!(m.calc_dim_su2_sector(1.5).unwrap(), 4);
        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 0); // forbidden (parity)
        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 0); // forbidden (parity)
        assert_eq!(m.calc_dim_su2_sector(2.5).unwrap(), 0); // too large
    }

    #[test]
    fn su2_sector_single_spin() {
        // Single spin-1/2: only S=1/2, dim = 1 × 2 = 2
        let m = SU2HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 2);
        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 0);
        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 0);

        // Single spin-1: only S=1, dim = 1 × 3 = 3
        let m = SU2HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        assert_eq!(m.calc_dim_su2_sector(1.0).unwrap(), 3);
        assert_eq!(m.calc_dim_su2_sector(0.0).unwrap(), 0);
        assert_eq!(m.calc_dim_su2_sector(0.5).unwrap(), 0);
    }

    #[test]
    fn su2_sector_negative_spin() {
        let m = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // Negative spin returns 0 (not an error)
        assert_eq!(m.calc_dim_su2_sector(-0.5).unwrap(), 0);
    }

    #[test]
    fn cuthill_mckee_1d_chain() {
        // 1D chain: 0-1-2-3-4
        // Nearest-neighbor bonds
        let bonds = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let order = SU2HeisenbergModel::cuthill_mckee_order(5, &bonds);

        // Should produce either [0, 1, 2, 3, 4] or [4, 3, 2, 1, 0]
        // (depending on which endpoint is chosen as the start)
        assert_eq!(order.len(), 5);

        // Check that consecutive elements in the order are neighbors
        for i in 0..4 {
            let diff = (order[i] as i32 - order[i + 1] as i32).abs();
            assert_eq!(
                diff,
                1,
                "Sites at positions {} and {} are not neighbors",
                i,
                i + 1
            );
        }
    }

    #[test]
    fn cuthill_mckee_square_lattice() {
        // 2x2 square lattice:
        //  0 - 1
        //  |   |
        //  2 - 3
        // Bonds: (0,1), (0,2), (1,3), (2,3)
        let bonds = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        let order = SU2HeisenbergModel::cuthill_mckee_order(4, &bonds);

        assert_eq!(order.len(), 4);
        // Should be a valid permutation
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn build_basis_with_order_keeps_site_order() {
        // Create a model with specific spins
        let m = SU2HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 2, 3], // spins 1/2, 1, 3/2
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // Reverse the order: coupling_order[k] = original site at position k
        let order = vec![2, 1, 0];
        let basis = m.build_basis_with_order(0.5, &order).unwrap();

        // two_s_list stays in original site order (NOT coupling order)
        assert_eq!(basis.two_s_list, vec![1, 2, 3]); // same as model
        assert_eq!(basis.coupling_order, vec![2, 1, 0]);
    }

    #[test]
    fn optimal_coupling_order_1d() {
        // 1D chain with sites in random order
        let mut exchange = HashMap::new();
        exchange.insert((0, 3), 1.0); // 0-3
        exchange.insert((3, 1), 1.0); // 3-1
        exchange.insert((1, 4), 1.0); // 1-4
        exchange.insert((4, 2), 1.0); // 4-2
                                      // Actual chain: 0-3-1-4-2

        let m = SU2HeisenbergModel {
            num_sites: 5,
            two_s_list: vec![1; 5],
            exchange,
            diagonal_shift: 0.0,
        };

        let order = m.optimal_coupling_order();

        // Build inverse mapping: site_to_pos[site] = position
        let mut site_to_pos = vec![0usize; 5];
        for (pos, &site) in order.iter().enumerate() {
            site_to_pos[site] = pos;
        }

        // After optimization, bonds should be between consecutive positions
        for &(i, j) in m.exchange.keys() {
            let pos_i = site_to_pos[i];
            let pos_j = site_to_pos[j];
            let diff = (pos_i as i32 - pos_j as i32).abs();
            assert_eq!(
                diff, 1,
                "Bond ({}, {}) at positions ({}, {}) is not adjacent",
                i, j, pos_i, pos_j
            );
        }
    }

    #[test]
    fn build_basis_single_site() {
        // Single spin-1/2
        let m = SU2HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // S=1/2: one basis state [2*J_1] = [1]
        let basis = m.build_basis(0.5).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8]);

        // S=0: forbidden
        let basis = m.build_basis(0.0).unwrap();
        assert_eq!(basis.dim(), 0);
    }

    #[test]
    fn build_basis_two_spin_half() {
        // Two spin-1/2: 1/2 ⊗ 1/2 = 0 ⊕ 1
        let m = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // S=0: one basis [2*J_1, 2*J_2] = [1, 0]
        let basis = m.build_basis(0.0).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 0]);

        // S=1: one basis [2*J_1, 2*J_2] = [1, 2]
        let basis = m.build_basis(1.0).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 2]);
    }

    #[test]
    fn build_basis_three_spin_half() {
        // Three spin-1/2: 2×(1/2) ⊕ 3/2
        let m = SU2HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // S=1/2: two bases
        // [2*J_1, 2*J_2, 2*J_3] where J_3 = 1/2
        // Path 1: J_1=1/2, J_2=0, then 0 ⊗ 1/2 = 1/2 → [1, 0, 1]
        // Path 2: J_1=1/2, J_2=1, then 1 ⊗ 1/2 = 1/2 or 3/2, pick 1/2 → [1, 2, 1]
        let basis = m.build_basis(0.5).unwrap();
        assert_eq!(basis.dim(), 2);
        assert!(basis.basis.contains(&vec![1u8, 0, 1]));
        assert!(basis.basis.contains(&vec![1u8, 2, 1]));

        // S=3/2: one basis
        // J_1=1/2, J_2=1, then 1 ⊗ 1/2 = 3/2 → [1, 2, 3]
        let basis = m.build_basis(1.5).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 2, 3]);
    }

    #[test]
    fn build_basis_four_spin_half() {
        // Four spin-1/2: 2×(0) ⊕ 3×(1) ⊕ 1×(2)
        let m = SU2HeisenbergModel {
            num_sites: 4,
            two_s_list: vec![1, 1, 1, 1],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // S=0: 2 bases
        let basis = m.build_basis(0.0).unwrap();
        assert_eq!(basis.dim(), 2);

        // S=1: 3 bases
        let basis = m.build_basis(1.0).unwrap();
        assert_eq!(basis.dim(), 3);

        // S=2: 1 basis
        // [2*J_1, 2*J_2, 2*J_3, 2*J_4] = [1, 2, 3, 4]
        // J_1=1/2, J_2=1, J_3=3/2, J_4=2
        let basis = m.build_basis(2.0).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 2, 3, 4]);
    }

    #[test]
    fn build_basis_mixed_spins() {
        // 1/2 ⊗ 1 = 1/2 ⊕ 3/2
        let m = SU2HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 2],
            exchange: HashMap::new(),
            diagonal_shift: 0.0,
        };

        // S=1/2: [2*J_1, 2*J_2] = [1, 1]
        let basis = m.build_basis(0.5).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 1]);

        // S=3/2: [2*J_1, 2*J_2] = [1, 3]
        let basis = m.build_basis(1.5).unwrap();
        assert_eq!(basis.dim(), 1);
        assert_eq!(basis.basis[0], vec![1u8, 3]);
    }
}
