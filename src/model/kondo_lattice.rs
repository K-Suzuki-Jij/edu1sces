use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Kondo lattice model with
/// - U(1) particle-number conservation
/// - U(1) spin symmetry (Sz conservation)
#[pyclass]
#[derive(Debug, Clone)]
pub struct KondoLatticeModel {
    /// number of lattice sites
    #[pyo3(get)]
    pub num_sites: usize,

    /// 2S_i for localized spins, length = num_sites
    #[pyo3(get)]
    pub two_s_list: Vec<i32>,

    /// conduction-electron hopping terms: (i, j) -> t_ij
    #[pyo3(get)]
    pub hopping: HashMap<(usize, usize), f64>,

    /// onsite Hubbard interaction for conduction electrons: U_i
    #[pyo3(get)]
    pub u_list: Vec<f64>,

    /// onsite potential / chemical potential for conduction electrons: mu_i
    #[pyo3(get)]
    pub mu_list: Vec<f64>,

    /// Zeeman field (z) for conduction electrons: hz_c_i
    /// couples to (n_up - n_down)
    #[pyo3(get)]
    pub hz_c_list: Vec<f64>,

    /// Zeeman field (z) for localized spins: hz_f_i
    /// couples to Sz
    #[pyo3(get)]
    pub hz_f_list: Vec<f64>,

    /// single-ion anisotropy for localized spins: D_i (Sz)^2
    #[pyo3(get)]
    pub d_list: Vec<f64>,

    /// density-density interaction for conduction electrons: (i, j) -> V_ij
    #[pyo3(get)]
    pub density_density: HashMap<(usize, usize), f64>,

    /// Kondo coupling (xy) on each site:
    /// (1/2) K_xy_i (S^+ s^- + S^- s^+)
    #[pyo3(get)]
    pub kondo_xy_list: Vec<f64>,

    /// Kondo coupling (z) on each site:
    /// Kz_i Sz * sz
    #[pyo3(get)]
    pub kondo_z_list: Vec<f64>,

    /// localized-spin exchange (xy):
    /// (1/2) J_xy_ij (S_i^+ S_j^- + S_i^- S_j^+)
    #[pyo3(get)]
    pub ff_exchange_xy: HashMap<(usize, usize), f64>,

    /// localized-spin exchange (z):
    /// J_z_ij S_i^z S_j^z
    #[pyo3(get)]
    pub ff_exchange_z: HashMap<(usize, usize), f64>,
}

impl KondoLatticeModel {
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
impl KondoLatticeModel {
    #[new]
    #[pyo3(
        text_signature = "(spin_list, hopping, u_list, mu_list, hz_c_list, hz_f_list, d_list, density_density, kondo_xy_list, kondo_z_list, ff_exchange_xy, ff_exchange_z)"
    )]
    pub fn new(
        spin_list: Vec<f64>,
        hopping: HashMap<(usize, usize), f64>,
        u_list: Vec<f64>,
        mu_list: Vec<f64>,
        hz_c_list: Vec<f64>,
        hz_f_list: Vec<f64>,
        d_list: Vec<f64>,
        density_density: HashMap<(usize, usize), f64>,
        kondo_xy_list: Vec<f64>,
        kondo_z_list: Vec<f64>,
        ff_exchange_xy: HashMap<(usize, usize), f64>,
        ff_exchange_z: HashMap<(usize, usize), f64>,
    ) -> Result<Self> {
        let num_sites = spin_list.len();
        if num_sites == 0 {
            bail!("num_sites must be non-zero");
        }

        // length consistency
        for (name, len) in [
            ("u_list", u_list.len()),
            ("mu_list", mu_list.len()),
            ("hz_c_list", hz_c_list.len()),
            ("hz_f_list", hz_f_list.len()),
            ("d_list", d_list.len()),
            ("kondo_xy_list", kondo_xy_list.len()),
            ("kondo_z_list", kondo_z_list.len()),
        ] {
            if len != num_sites {
                bail!(
                    "length mismatch: {}.len() = {}, spin_list.len() = {}",
                    name,
                    len,
                    num_sites
                );
            }
        }

        // convert spin_list -> two_s_list
        let mut two_s_list = vec![0; num_sites];
        for (i, &s) in spin_list.iter().enumerate() {
            let two_s_f = (2.0 * s).round();
            if (2.0 * s - two_s_f).abs() > 1e-12 {
                bail!("Spin at site {} = {} is not a half-integer", i, s);
            }
            if two_s_f < 0.0 {
                bail!("Spin at site {} must be non-negative (got {})", i, s);
            }
            two_s_list[i] = two_s_f as i32;
        }

        // index checks
        Self::check_pairs("hopping", num_sites, &hopping)?;
        Self::check_pairs("density_density", num_sites, &density_density)?;
        Self::check_pairs("ff_exchange_xy", num_sites, &ff_exchange_xy)?;
        Self::check_pairs("ff_exchange_z", num_sites, &ff_exchange_z)?;

        Ok(Self {
            num_sites,
            two_s_list,
            hopping,
            u_list,
            mu_list,
            hz_c_list,
            hz_f_list,
            d_list,
            density_density,
            kondo_xy_list,
            kondo_z_list,
            ff_exchange_xy,
            ff_exchange_z,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_valid_input() {
        let spin_list = vec![0.5, 0.5, 1.0];

        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let u_list = vec![0.0, 0.0, 0.0];
        let mu_list = vec![0.0, 0.0, 0.0];
        let hz_c_list = vec![0.0, 0.0, 0.0];
        let hz_f_list = vec![0.0, 0.0, 0.0];
        let d_list = vec![0.0, 0.0, 0.0];

        let density_density = HashMap::new();

        let kondo_xy_list = vec![1.0, 1.0, 1.0];
        let kondo_z_list = vec![1.0, 1.0, 1.0];

        let ff_exchange_xy = HashMap::new();
        let ff_exchange_z = HashMap::new();

        let m = KondoLatticeModel::new(
            spin_list,
            hopping,
            u_list,
            mu_list,
            hz_c_list,
            hz_f_list,
            d_list,
            density_density,
            kondo_xy_list,
            kondo_z_list,
            ff_exchange_xy,
            ff_exchange_z,
        )
        .unwrap();

        assert_eq!(m.num_sites, 3);
        assert_eq!(m.two_s_list, vec![1, 1, 2]);
    }

    #[test]
    fn reject_length_mismatch() {
        let spin_list = vec![0.5, 0.5];
        let hopping = HashMap::new();

        assert!(KondoLatticeModel::new(
            spin_list,
            hopping,
            vec![0.0, 0.0],
            vec![0.0], // mismatch
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let spin_list = vec![0.5, 0.5];
        let mut hopping = HashMap::new();
        hopping.insert((0, 2), 1.0);

        assert!(KondoLatticeModel::new(
            spin_list,
            hopping,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            HashMap::new(),
            HashMap::new(),
        )
        .is_err());
    }
}
