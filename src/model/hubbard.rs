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
}
