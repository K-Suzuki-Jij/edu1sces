use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Hubbard model with U(1) particle-number conservation and Sz conservation
#[pyclass]
#[derive(Debug, Clone)]
pub struct HubbardModel {
    /// number of lattice sites
    #[pyo3(get)]
    pub num_sites: usize,

    /// hopping terms: (i, j) -> t_ij
    #[pyo3(get)]
    pub hopping: HashMap<(usize, usize), f64>,

    /// onsite interaction U_i, length = num_sites
    #[pyo3(get)]
    pub u_list: Vec<f64>,

    /// onsite potential / chemical potential mu_i, length = num_sites
    #[pyo3(get)]
    pub mu_list: Vec<f64>,

    /// Zeeman field along z, length = num_sites
    /// couples to (n_up - n_down) (equivalently 2 Sz)
    #[pyo3(get)]
    pub hz_list: Vec<f64>,

    /// density-density interaction: (i, j) -> V_ij
    #[pyo3(get)]
    pub density_density: HashMap<(usize, usize), f64>,

    /// exchange (xy) part:
    /// (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
    #[pyo3(get)]
    pub exchange_xy: HashMap<(usize, usize), f64>,

    /// exchange (z) part:
    /// S_i^z S_j^z
    #[pyo3(get)]
    pub exchange_z: HashMap<(usize, usize), f64>,
}

impl HubbardModel {
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
impl HubbardModel {
    #[new]
    #[pyo3(
        text_signature = "(hopping, u_list, mu_list, hz_list, density_density, exchange_xy, exchange_z)"
    )]
    pub fn new(
        hopping: HashMap<(usize, usize), f64>,
        u_list: Vec<f64>,
        mu_list: Vec<f64>,
        hz_list: Vec<f64>,
        density_density: HashMap<(usize, usize), f64>,
        exchange_xy: HashMap<(usize, usize), f64>,
        exchange_z: HashMap<(usize, usize), f64>,
    ) -> Result<Self> {
        if u_list.len() != mu_list.len() {
            bail!(
                "length mismatch: u_list.len() = {}, mu_list.len() = {}",
                u_list.len(),
                mu_list.len()
            );
        }
        if u_list.len() != hz_list.len() {
            bail!(
                "length mismatch: u_list.len() = {}, hz_list.len() = {}",
                u_list.len(),
                hz_list.len()
            );
        }

        let num_sites = u_list.len();
        if num_sites == 0 {
            bail!("num_sites must be positive");
        }

        Self::check_pairs("hopping", num_sites, &hopping)?;
        Self::check_pairs("density_density", num_sites, &density_density)?;
        Self::check_pairs("exchange_xy", num_sites, &exchange_xy)?;
        Self::check_pairs("exchange_z", num_sites, &exchange_z)?;

        Ok(Self {
            num_sites,
            hopping,
            u_list,
            mu_list,
            hz_list,
            density_density,
            exchange_xy,
            exchange_z,
        })
    }

    /// Dimension of the fixed-(N, total Sz) sector.
    ///
    /// Local basis:
    /// 0 -> |vac>, 1 -> |up>, 2 -> |down>, 3 -> |up down>
    ///
    /// N_up = (N + 2Sz)/2, N_dn = (N - 2Sz)/2
    #[pyo3(text_signature = "(self, num_electrons, total_sz)")]
    pub fn calc_dim_u1_sector(&self, num_electrons: usize, total_sz: f64) -> Result<i128> {
        // Parse total_sz as integer/half-integer -> two_m = 2*Sz (integer)
        let two_m_f = 2.0 * total_sz;
        let two_m = two_m_f.round() as i32;
        if (two_m_f - two_m as f64).abs() > 1e-12 {
            bail!("total_sz must be integer or half-integer (got {total_sz})");
        }

        let l = self.num_sites;
        if num_electrons > 2 * l {
            return Ok(0);
        }

        let n = num_electrons as i32;

        // N_up, N_dn must be integers
        if ((n + two_m) & 1) != 0 || ((n - two_m) & 1) != 0 {
            return Ok(0);
        }

        let n_up_i32 = (n + two_m) / 2;
        let n_dn_i32 = (n - two_m) / 2;
        if n_up_i32 < 0 || n_dn_i32 < 0 {
            return Ok(0);
        }

        let n_up = n_up_i32 as usize;
        let n_dn = n_dn_i32 as usize;
        if n_up > l || n_dn > l {
            return Ok(0);
        }

        // local exact binomial C(n,k) in i128 (overflow-checked)
        fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            a.abs()
        }

        fn binom_i128(n: usize, k: usize) -> Result<i128> {
            if k > n {
                return Ok(0);
            }
            let k = k.min(n - k);

            // multiplicative formula with gcd reduction
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            for i in 1..=k {
                let a = (n - k + i) as i128;
                let b = i as i128;

                num = num
                    .checked_mul(a)
                    .ok_or_else(|| anyhow::anyhow!("i128 overflow while building binom"))?;
                den = den
                    .checked_mul(b)
                    .ok_or_else(|| anyhow::anyhow!("i128 overflow while building binom"))?;

                let g = gcd_i128(num, den);
                num /= g;
                den /= g;
            }

            debug_assert!(den == 1);
            Ok(num)
        }

        let c_up = binom_i128(l, n_up)?;
        let c_dn = binom_i128(l, n_dn)?;

        c_up.checked_mul(c_dn)
            .ok_or_else(|| anyhow::anyhow!("i128 overflow in dim = C(L,N_up)*C(L,N_dn)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn accept_valid_input() {
        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let u_list = vec![2.0, 2.0, 2.0];
        let mu_list = vec![0.0, 0.0, 0.0];
        let hz_list = vec![0.1, 0.1, 0.1];

        let m = HubbardModel::new(
            hopping,
            u_list,
            mu_list,
            hz_list,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        assert_eq!(m.num_sites, 3);
    }

    #[test]
    fn reject_length_mismatch_mu() {
        assert!(HubbardModel::new(
            HashMap::new(),
            vec![1.0, 1.0],
            vec![0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_length_mismatch_hz() {
        assert!(HubbardModel::new(
            HashMap::new(),
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.1],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_zero_sites() {
        assert!(HubbardModel::new(
            HashMap::new(),
            vec![],
            vec![],
            vec![],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let mut hopping = HashMap::new();
        hopping.insert((0, 3), 1.0);

        assert!(HubbardModel::new(
            hopping,
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn calc_dim_u1_sector_cases() {
        let m = HubbardModel::new(
            HashMap::new(),
            vec![1.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        // Valid cases: dim = C(L, n_up) * C(L, n_dn)
        // L=4, n=2, Sz=0 => n_up=1, n_dn=1 => C(4,1)*C(4,1) = 16
        assert_eq!(m.calc_dim_u1_sector(2, 0.0).unwrap(), 16);
        // L=4, n=4, Sz=0 => n_up=2, n_dn=2 => C(4,2)*C(4,2) = 36
        assert_eq!(m.calc_dim_u1_sector(4, 0.0).unwrap(), 36);
        // L=4, n=2, Sz=1 => n_up=2, n_dn=0 => C(4,2)*C(4,0) = 6
        assert_eq!(m.calc_dim_u1_sector(2, 1.0).unwrap(), 6);
        // L=4, n=0, Sz=0 => n_up=0, n_dn=0 => 1
        assert_eq!(m.calc_dim_u1_sector(0, 0.0).unwrap(), 1);
        // L=4, n=8, Sz=0 => n_up=4, n_dn=4 => 1
        assert_eq!(m.calc_dim_u1_sector(8, 0.0).unwrap(), 1);

        // Zero cases (unreachable sector -> 0)
        assert_eq!(m.calc_dim_u1_sector(9, 0.0).unwrap(), 0); // too many electrons
        assert_eq!(m.calc_dim_u1_sector(2, 2.0).unwrap(), 0); // Sz out of range
        assert_eq!(m.calc_dim_u1_sector(3, 0.0).unwrap(), 0); // parity mismatch

        // Error case: non-half-integer Sz
        assert!(m.calc_dim_u1_sector(2, 0.3).is_err());
    }
}
