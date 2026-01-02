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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
