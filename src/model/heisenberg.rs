use crate::blas::{csr_add, csr_mul, csr_transpose, CsrMatrix};
use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

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

    fn get_two_sz_at(&self, site: usize) -> Result<i32> {
        let two_sz = *self
            .two_s_list
            .get(site)
            .ok_or_else(|| anyhow::anyhow!("site out of range"))?;
        if two_sz < 0 {
            bail!("two_s_list must be non-negative");
        }
        Ok(two_sz)
    }

    pub fn local_op_sz(&self, site: usize) -> Result<CsrMatrix> {
        let two_sz = self.get_two_sz_at(site)?;
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

    pub fn local_op_sp(&self, site: usize) -> Result<CsrMatrix> {
        let two_sz = self.get_two_sz_at(site)?;
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

    pub fn local_op_sm(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.local_op_sp(site)?)
    }

    pub fn local_op_sx(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.local_op_sp(site)?;
        let sm = self.local_op_sm(site)?;
        csr_add(0.5, &sp, 0.5, &sm)
    }

    pub fn local_op_isy(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.local_op_sp(site)?;
        let sm = self.local_op_sm(site)?;
        csr_add(0.5, &sp, -0.5, &sm)
    }

    /// H_i = hz_i * Sz_i + d_i * (Sz_i)^2
    pub fn local_onsite_hamiltonian(&self, site: usize) -> Result<CsrMatrix> {
        let sz = self.local_op_sz(site)?;
        let hz = self.hz_list[site];
        let d = self.d_list[site];

        let szsz = csr_mul(1.0, &sz, 1.0, &sz)?;
        csr_add(hz, &sz, d, &szsz)
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
    pub fn calc_dim_u1_sector(&self, total_sz: f64) -> Result<u128> {
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

        let mut dp = vec![0u128; size];
        dp[offset as usize] = 1;

        for &two_s in self.two_s_list.iter() {
            let mut next = vec![0u128; size];

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
        let sz = m.local_op_sz(0).unwrap();
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
        let sz = m.local_op_sz(0).unwrap();
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
        let sp = m.local_op_sp(0).unwrap();
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
        let sp = m.local_op_sp(0).unwrap();
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
        let sm = m.local_op_sm(0).unwrap();
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
        let sm = m.local_op_sm(0).unwrap();
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
        let sx = m.local_op_sx(0).unwrap();
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
        let sx = m.local_op_sx(0).unwrap();
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
        let isy = m.local_op_isy(0).unwrap();
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
        let isy = m.local_op_isy(0).unwrap();
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
        let h = m.local_onsite_hamiltonian(0).unwrap();
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
        let h = m.local_onsite_hamiltonian(0).unwrap();
        assert_eq!(h.row_dim, 3);
        assert_eq!(h.col_dim, 3);
        assert_eq!(h.rows, vec![0, 1, 1, 2]);
        assert_eq!(h.cols, vec![0, 2]);
        assert_eq!(h.vals.len(), 2);
        assert!((h.vals[0] - 5.0).abs() <= MATRIX_ZERO_EPS);
        assert!((h.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }
}
