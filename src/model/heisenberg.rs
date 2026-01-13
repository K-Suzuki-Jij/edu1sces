use ahash::AHashMap;
use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::basis::Basis;
use crate::blas::{csr_add, csr_mul, csr_transpose, CsrMatrix};
use crate::model::quantum_model::QuantumModel;

/// Spin-S Heisenberg model with Sz conservation (U(1))
#[pyclass]
#[derive(Debug, Clone)]
pub struct HeisenbergModel {
    /// number of lattice sites
    #[pyo3(get)]
    pub num_sites: usize,

    /// 2S_i for each site i (integer), length = num_sites
    #[pyo3(get)]
    pub two_s_list: Vec<i32>,

    /// Zeeman field along z, length = num_sites
    #[pyo3(get)]
    pub hz_list: Vec<f64>,

    /// single-ion anisotropy D_i (Sz_i)^2, length = num_sites
    #[pyo3(get)]
    pub d_list: Vec<f64>,

    /// exchange (xy) part:
    /// (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
    #[pyo3(get)]
    pub exchange_xy: HashMap<(usize, usize), f64>,

    /// exchange (z) part:
    /// S_i^z S_j^z
    #[pyo3(get)]
    pub exchange_z: HashMap<(usize, usize), f64>,
}

impl HeisenbergModel {
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
}

#[pymethods]
impl HeisenbergModel {
    #[pyo3(text_signature = "(spin_list, hz_list, d_list, exchange_xy, exchange_z)")]
    #[new]
    pub fn new(
        spin_list: Vec<f64>,
        hz_list: Vec<f64>,
        d_list: Vec<f64>,
        exchange_xy: HashMap<(usize, usize), f64>,
        exchange_z: HashMap<(usize, usize), f64>,
    ) -> Result<Self> {
        let num_sites = spin_list.len();
        if hz_list.len() != num_sites {
            bail!(
                "length mismatch: hz_list.len() = {}, spin_list.len() = {}",
                hz_list.len(),
                num_sites
            );
        }
        if d_list.len() != num_sites {
            bail!(
                "length mismatch: d_list.len() = {}, spin_list.len() = {}",
                d_list.len(),
                num_sites
            );
        }
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

        Self::check_pairs("exchange_xy", num_sites, &exchange_xy)?;
        Self::check_pairs("exchange_z", num_sites, &exchange_z)?;

        Ok(Self {
            num_sites,
            two_s_list,
            hz_list,
            d_list,
            exchange_xy,
            exchange_z,
        })
    }

    /// Dimension of the U(1) sector with fixed total Sz.
    ///
    /// `total_sz` must be integer or half-integer.
    /// Returns 0 if the sector is forbidden for the given local spins.
    #[pyo3(text_signature = "(self, total_sz)")]
    pub fn calc_dim_u1_sector(&self, total_sz: f64) -> Result<i128> {
        let two_m_f = 2.0 * total_sz;
        let two_m = two_m_f.round() as i32;

        if (two_m_f - two_m as f64).abs() > 1e-12 {
            bail!("total_sz must be integer or half-integer (got {total_sz})");
        }

        let sum_two_s: i32 = self.two_s_list.iter().sum();

        if ((sum_two_s - two_m) & 1) != 0 {
            return Ok(0);
        }

        let min_two_m: i32 = self.two_s_list.iter().map(|&s| -s).sum();
        let max_two_m: i32 = self.two_s_list.iter().map(|&s| s).sum();

        if two_m < min_two_m || two_m > max_two_m {
            return Ok(0);
        }

        let offset = -min_two_m;
        let size = (max_two_m - min_two_m + 1) as usize;

        let mut dp = vec![0i128; size];
        dp[offset as usize] = 1;

        for &two_s in self.two_s_list.iter() {
            let mut next = vec![0i128; size];

            for two_m_site in (-two_s..=two_s).step_by(2) {
                for m in min_two_m..=max_two_m {
                    let idx = (m + offset) as usize;
                    let ways = dp[idx];
                    if ways == 0 {
                        continue;
                    }

                    let new_m = m + two_m_site;
                    if new_m < min_two_m || new_m > max_two_m {
                        continue;
                    }

                    let new_idx = (new_m + offset) as usize;
                    next[new_idx] += ways;
                }
            }

            dp = next;
        }

        Ok(dp[(two_m + offset) as usize])
    }

    pub fn make_local_op_sz(&self, site: usize) -> Result<CsrMatrix> {
        let two_sz = self.two_s_list[site];
        let dim = (two_sz as usize) + 1;

        let mut m = CsrMatrix::new();
        m.row_dim = dim;
        m.col_dim = dim;

        m.rows = Vec::with_capacity(dim + 1);
        m.rows.push(0);

        for k in 0..dim {
            let m2 = two_sz - 2 * (k as i32);
            if m2 != 0 {
                m.cols.push(k);
                m.vals.push(0.5 * (m2 as f64));
            }
            m.rows.push(m.vals.len());
        }

        Ok(m)
    }

    pub fn make_local_op_sp(&self, site: usize) -> Result<CsrMatrix> {
        let two_sz = self.two_s_list[site];
        let dim = (two_sz as usize) + 1;

        let s = 0.5 * (two_sz as f64);

        let mut m = CsrMatrix::new();
        m.row_dim = dim;
        m.col_dim = dim;

        m.rows = Vec::with_capacity(dim + 1);
        m.rows.push(0);

        for row in 0..dim {
            if row + 1 >= dim {
                m.rows.push(m.vals.len());
                continue;
            }

            let col = row + 1;

            let m2 = two_sz - 2 * (col as i32);
            let mm = 0.5 * (m2 as f64);

            let v2 = s * (s + 1.0) - mm * (mm + 1.0);
            let v = if v2 <= 0.0 { 0.0 } else { v2.sqrt() };

            if v != 0.0 {
                m.cols.push(col);
                m.vals.push(v);
            }

            m.rows.push(m.vals.len());
        }

        Ok(m)
    }

    pub fn make_local_op_sm(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_sp(site)?)
    }

    pub fn make_local_op_sx(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.make_local_op_sp(site)?;
        let sm = self.make_local_op_sm(site)?;
        csr_add(0.5, &sp, 0.5, &sm)
    }

    pub fn make_local_op_isy(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.make_local_op_sp(site)?;
        let sm = self.make_local_op_sm(site)?;
        csr_add(0.5, &sp, -0.5, &sm)
    }

    /// H_i = hz_i * Sz_i + d_i * (Sz_i)^2
    pub fn make_local_hamiltonian(&self, site: usize) -> Result<CsrMatrix> {
        let sz = self.make_local_op_sz(site)?;
        let hz = self.hz_list[site];
        let d = self.d_list[site];

        let szsz = csr_mul(1.0, &sz, 1.0, &sz)?;
        csr_add(hz, &sz, d, &szsz)
    }
}

impl QuantumModel for HeisenbergModel {
    fn num_sites(&self) -> usize {
        self.num_sites
    }

    fn local_dim(&self, site: usize) -> usize {
        (self.two_s_list[site] as usize) + 1
    }

    /// Return quantum numbers [2*sz] for a local state at a given site.
    /// For spin S, local states are 0..=2S corresponding to sz = S, S-1, ..., -S
    /// local_state=0 → sz=+S, local_state=2S → sz=-S
    fn quantum_numbers(&self, site: usize, local_state: usize) -> Vec<i32> {
        let two_s = self.two_s_list[site];
        // sz = S - local_state, so 2*sz = 2S - 2*local_state
        let sz2 = two_s - 2 * (local_state as i32);
        vec![sz2]
    }

    /// Build basis for sector with quantum numbers [2*total_sz].
    fn build_basis(&self, target_quantum_numbers: &[i32]) -> Result<Basis> {
        let total_sz2 = target_quantum_numbers[0];

        // Validate total_sz
        let sum_two_s: i32 = self.two_s_list.iter().sum();
        if ((sum_two_s - total_sz2) & 1) != 0 {
            bail!(
                "parity mismatch: sum(2S) = {} but 2*total_sz = {}",
                sum_two_s,
                total_sz2
            );
        }

        let min_two_m: i32 = self.two_s_list.iter().map(|&s| -s).sum();
        let max_two_m: i32 = self.two_s_list.iter().map(|&s| s).sum();

        if total_sz2 < min_two_m || total_sz2 > max_two_m {
            bail!(
                "total_sz = {} out of range [{}, {}]",
                total_sz2 as f64 / 2.0,
                min_two_m as f64 / 2.0,
                max_two_m as f64 / 2.0
            );
        }

        let num_sites = self.num_sites;

        // Build site_base and local_dims
        let mut site_base = Vec::with_capacity(num_sites);
        let mut local_dims = Vec::with_capacity(num_sites);
        let mut site_stride: i128 = 1;
        for &two_s in self.two_s_list.iter() {
            site_base.push(site_stride);
            let local_dim = (two_s as usize) + 1;
            local_dims.push(local_dim);
            site_stride = site_stride
                .checked_mul(local_dim as i128)
                .ok_or_else(|| anyhow::anyhow!("i128 overflow"))?;
        }

        // Precompute suffix bounds for pruning
        let mut suffix_min_sz2 = vec![0; num_sites + 1];
        let mut suffix_max_sz2 = vec![0; num_sites + 1];
        for i in (0..num_sites).rev() {
            let two_s = self.two_s_list[i];
            suffix_min_sz2[i] = suffix_min_sz2[i + 1] - two_s;
            suffix_max_sz2[i] = suffix_max_sz2[i + 1] + two_s;
        }

        let mut basis = Vec::new();

        fn dfs(
            site: usize,
            two_s_list: &[i32],
            site_base: &[i128],
            suffix_min_sz2: &[i32],
            suffix_max_sz2: &[i32],
            total_sz2_target: i32,
            sz2_sum: i32,
            basis_code: i128,
            out: &mut Vec<i128>,
        ) {
            if site == two_s_list.len() {
                if sz2_sum == total_sz2_target {
                    out.push(basis_code);
                }
                return;
            }

            let two_s = two_s_list[site];
            let min_sz2 = -two_s;
            let max_sz2 = two_s;

            let remain_min = suffix_min_sz2[site + 1];
            let remain_max = suffix_max_sz2[site + 1];

            let need = total_sz2_target - sz2_sum;
            if need < min_sz2 + remain_min || need > max_sz2 + remain_max {
                return;
            }

            let base = site_base[site];

            let mut sz2 = min_sz2;
            while sz2 <= max_sz2 {
                let digit = ((two_s - sz2) / 2) as i128;
                dfs(
                    site + 1,
                    two_s_list,
                    site_base,
                    suffix_min_sz2,
                    suffix_max_sz2,
                    total_sz2_target,
                    sz2_sum + sz2,
                    basis_code + digit * base,
                    out,
                );
                sz2 += 2;
            }
        }

        dfs(
            0,
            &self.two_s_list,
            &site_base,
            &suffix_min_sz2,
            &suffix_max_sz2,
            total_sz2,
            0,
            0i128,
            &mut basis,
        );

        basis.sort_unstable();

        let mut inverse_basis = AHashMap::with_capacity(basis.len());
        for (i, &basis_code) in basis.iter().enumerate() {
            inverse_basis.insert(basis_code, i);
        }

        Ok(Basis::new(basis, inverse_basis, site_base, local_dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::MATRIX_ZERO_EPS;

    #[test]
    fn accept_valid_input() {
        let spins = vec![0.5, 1.0, 1.5];
        let hz_list = vec![0.0, 0.0, 0.0];
        let d_list = vec![0.0, 0.0, 0.0];

        let mut exchange_xy = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);

        let mut exchange_z = HashMap::new();
        exchange_z.insert((0, 1), 1.0);

        let m = HeisenbergModel::new(spins, hz_list, d_list, exchange_xy, exchange_z).unwrap();
        assert_eq!(m.num_sites, 3);
        assert_eq!(m.two_s_list, vec![1, 2, 3]);
    }

    #[test]
    fn reject_length_mismatch() {
        let spins = vec![0.5, 0.5];
        let hz_list = vec![0.0];
        let d_list = vec![0.0, 0.0];

        assert!(
            HeisenbergModel::new(spins, hz_list, d_list, HashMap::new(), HashMap::new()).is_err()
        );
    }

    #[test]
    fn reject_invalid_index() {
        let spins = vec![0.5, 0.5];
        let hz_list = vec![0.0, 0.0];
        let d_list = vec![0.0, 0.0];

        let mut exchange_xy = HashMap::new();
        exchange_xy.insert((0, 2), 1.0);

        assert!(HeisenbergModel::new(spins, hz_list, d_list, exchange_xy, HashMap::new()).is_err());
    }

    #[test]
    fn mixed_spins_three_sites_all_key_properties() {
        let m = HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 2, 1], // 1/2, 1, 1/2
            hz_list: vec![0.0, 0.0, 0.0],
            d_list: vec![0.0, 0.0, 0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        // nontrivial sector dimensions
        assert_eq!(m.calc_dim_u1_sector(0.0).unwrap(), 4);
        assert_eq!(m.calc_dim_u1_sector(1.0).unwrap(), 3);
        assert_eq!(m.calc_dim_u1_sector(2.0).unwrap(), 1);

        // symmetry
        assert_eq!(m.calc_dim_u1_sector(-1.0).unwrap(), 3);
        assert_eq!(m.calc_dim_u1_sector(-2.0).unwrap(), 1);

        // out of range must be zero
        assert_eq!(m.calc_dim_u1_sector(3.0).unwrap(), 0);
        assert_eq!(m.calc_dim_u1_sector(-3.0).unwrap(), 0);
    }

    #[test]
    fn single_spin_half_reachable_and_unreachable_sectors() {
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1], // 1/2
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        // reachable
        assert_eq!(m.calc_dim_u1_sector(0.5).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(-0.5).unwrap(), 1);

        // parity mismatch or unreachable sector must be zero
        assert_eq!(m.calc_dim_u1_sector(0.0).unwrap(), 0);
    }

    #[test]
    fn non_half_integer_total_sz_is_error() {
        let m = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![0.0, 0.0],
            d_list: vec![0.0, 0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        assert!(m.calc_dim_u1_sector(0.1).is_err());
    }

    #[test]
    fn local_op_sz_s_half_and_one() {
        // S = 1/2, order: m=+1/2 (0), -1/2 (1)
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sz = m.make_local_op_sz(0).unwrap();
        assert_eq!(sz.row_dim, 2);
        assert_eq!(sz.col_dim, 2);
        assert_eq!(sz.rows, vec![0, 1, 2]);
        assert_eq!(sz.cols, vec![0, 1]);
        assert_eq!(sz.vals.len(), 2);
        assert!((sz.vals[0] - 0.5).abs() <= MATRIX_ZERO_EPS);
        assert!((sz.vals[1] - (-0.5)).abs() <= MATRIX_ZERO_EPS);

        // S = 1, order: m=+1 (0), 0 (1), -1 (2)
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sz = m.make_local_op_sz(0).unwrap();
        assert_eq!(sz.row_dim, 3);
        assert_eq!(sz.col_dim, 3);
        assert_eq!(sz.rows, vec![0, 1, 1, 2]);
        assert_eq!(sz.cols, vec![0, 2]);
        assert_eq!(sz.vals.len(), 2);
        assert!((sz.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((sz.vals[1] - (-1.0)).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sp_s_half_and_one() {
        let rt2 = 2.0_f64.sqrt();

        // S = 1/2, order: m=+1/2 (0), -1/2 (1)
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sp = m.make_local_op_sp(0).unwrap();
        assert_eq!(sp.row_dim, 2);
        assert_eq!(sp.col_dim, 2);
        assert_eq!(sp.rows, vec![0, 1, 1]);
        assert_eq!(sp.cols, vec![1]);
        assert_eq!(sp.vals.len(), 1);
        assert!((sp.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);

        // S = 1, order: m=+1 (0), 0 (1), -1 (2)
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sp = m.make_local_op_sp(0).unwrap();
        assert_eq!(sp.row_dim, 3);
        assert_eq!(sp.col_dim, 3);
        assert_eq!(sp.rows, vec![0, 1, 2, 2]);
        assert_eq!(sp.cols, vec![1, 2]);
        assert_eq!(sp.vals.len(), 2);
        assert!((sp.vals[0] - rt2).abs() <= MATRIX_ZERO_EPS);
        assert!((sp.vals[1] - rt2).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sm_s_half_and_one() {
        let rt2 = 2.0_f64.sqrt();

        // S = 1/2
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sm = m.make_local_op_sm(0).unwrap();
        assert_eq!(sm.row_dim, 2);
        assert_eq!(sm.col_dim, 2);
        assert_eq!(sm.rows, vec![0, 0, 1]);
        assert_eq!(sm.cols, vec![0]);
        assert_eq!(sm.vals.len(), 1);
        assert!((sm.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);

        // S = 1
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sm = m.make_local_op_sm(0).unwrap();
        assert_eq!(sm.row_dim, 3);
        assert_eq!(sm.col_dim, 3);
        assert_eq!(sm.rows, vec![0, 0, 1, 2]);
        assert_eq!(sm.cols, vec![0, 1]);
        assert_eq!(sm.vals.len(), 2);
        assert!((sm.vals[0] - rt2).abs() <= MATRIX_ZERO_EPS);
        assert!((sm.vals[1] - rt2).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sx_s_half_and_one() {
        let rt2h = 0.5 * 2.0_f64.sqrt();

        // S = 1/2 : Sx = 0.5 Sp + 0.5 Sm
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sx = m.make_local_op_sx(0).unwrap();
        assert_eq!(sx.row_dim, 2);
        assert_eq!(sx.col_dim, 2);
        assert_eq!(sx.rows, vec![0, 1, 2]);
        assert_eq!(sx.cols, vec![1, 0]);
        assert_eq!(sx.vals, vec![0.5, 0.5]);

        // S = 1
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let sx = m.make_local_op_sx(0).unwrap();
        assert_eq!(sx.row_dim, 3);
        assert_eq!(sx.col_dim, 3);
        assert_eq!(sx.rows, vec![0, 1, 3, 4]);
        assert_eq!(sx.cols, vec![1, 0, 2, 1]);
        assert_eq!(sx.vals.len(), 4);
        assert!((sx.vals[0] - rt2h).abs() <= MATRIX_ZERO_EPS);
        assert!((sx.vals[1] - rt2h).abs() <= MATRIX_ZERO_EPS);
        assert!((sx.vals[2] - rt2h).abs() <= MATRIX_ZERO_EPS);
        assert!((sx.vals[3] - rt2h).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_isy_s_half_and_one() {
        let rt2h = 0.5 * 2.0_f64.sqrt();

        // S = 1/2
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let isy = m.make_local_op_isy(0).unwrap();
        assert_eq!(isy.row_dim, 2);
        assert_eq!(isy.col_dim, 2);
        assert_eq!(isy.rows, vec![0, 1, 2]);
        assert_eq!(isy.cols, vec![1, 0]);
        assert_eq!(isy.vals, vec![0.5, -0.5]);

        // S = 1
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![0.0],
            d_list: vec![0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let isy = m.make_local_op_isy(0).unwrap();
        assert_eq!(isy.row_dim, 3);
        assert_eq!(isy.col_dim, 3);
        assert_eq!(isy.rows, vec![0, 1, 3, 4]);
        assert_eq!(isy.cols, vec![1, 0, 2, 1]);
        assert_eq!(isy.vals.len(), 4);
        assert!((isy.vals[0] - (rt2h)).abs() <= MATRIX_ZERO_EPS);
        assert!((isy.vals[1] - (-rt2h)).abs() <= MATRIX_ZERO_EPS);
        assert!((isy.vals[2] - (rt2h)).abs() <= MATRIX_ZERO_EPS);
        assert!((isy.vals[3] - (-rt2h)).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_onsite_hamiltonian_s_half_and_one() {
        // S = 1/2, hz=2, d=3
        // basis: |S>, |-S> so Sz diag = [ +1/2, -1/2 ]
        // Sz^2 diag = [ 1/4, 1/4 ]
        // H diag = 2*Sz + 3*Sz^2 = [ 1 + 3/4, -1 + 3/4 ] = [ 1.75, -0.25 ]
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![1],
            hz_list: vec![2.0],
            d_list: vec![3.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let h = m.make_local_hamiltonian(0).unwrap();
        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert_eq!(h.rows, vec![0, 1, 2]);
        assert_eq!(h.cols, vec![0, 1]);
        assert_eq!(h.vals.len(), 2);
        assert!((h.vals[0] - 1.75).abs() <= MATRIX_ZERO_EPS);
        assert!((h.vals[1] - (-0.25)).abs() <= MATRIX_ZERO_EPS);

        // S = 1, hz=2, d=3
        // basis: |1>, |0>, |-1| so Sz diag = [ +1, 0, -1 ]
        // Sz^2 diag = [ 1, 0, 1 ]
        // H diag = [ 2*1 + 3*1, 0, 2*(-1) + 3*1 ] = [ 5, 0, 1 ]
        let m = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![2.0],
            d_list: vec![3.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };
        let h = m.make_local_hamiltonian(0).unwrap();
        assert_eq!(h.row_dim, 3);
        assert_eq!(h.col_dim, 3);
        assert_eq!(h.rows, vec![0, 1, 1, 2]);
        assert_eq!(h.cols, vec![0, 2]);
        assert_eq!(h.vals.len(), 2);
        assert!((h.vals[0] - 5.0).abs() <= MATRIX_ZERO_EPS);
        assert!((h.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }
}
