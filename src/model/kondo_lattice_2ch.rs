use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Two-channel Kondo lattice model with
/// - U(1) particle-number conservation in each channel (N0 and N1)
/// - U(1) spin symmetry (total Sz conservation)
#[pyclass]
#[derive(Debug, Clone)]
pub struct KondoLattice2ChModel {
    /// number of lattice sites
    #[pyo3(get)]
    pub num_sites: usize,

    /// 2S_i for localized spins, length = num_sites
    #[pyo3(get)]
    pub two_s_list: Vec<i32>,

    // -------------------------
    // Conduction electrons (ch=0,1)
    // -------------------------
    /// hopping terms for channel 0: (i, j) -> t_ij^(0)
    #[pyo3(get)]
    pub hopping_0: HashMap<(usize, usize), f64>,

    /// hopping terms for channel 1: (i, j) -> t_ij^(1)
    #[pyo3(get)]
    pub hopping_1: HashMap<(usize, usize), f64>,

    /// onsite Hubbard U for channel 0, length = num_sites
    #[pyo3(get)]
    pub u_list_0: Vec<f64>,

    /// onsite Hubbard U for channel 1, length = num_sites
    #[pyo3(get)]
    pub u_list_1: Vec<f64>,

    /// chemical potential / onsite potential for channel 0, length = num_sites
    #[pyo3(get)]
    pub mu_list_0: Vec<f64>,

    /// chemical potential / onsite potential for channel 1, length = num_sites
    #[pyo3(get)]
    pub mu_list_1: Vec<f64>,

    /// Zeeman field (z) for channel 0, length = num_sites
    /// couples to (n_up - n_down)
    #[pyo3(get)]
    pub hz_c_list_0: Vec<f64>,

    /// Zeeman field (z) for channel 1, length = num_sites
    /// couples to (n_up - n_down)
    #[pyo3(get)]
    pub hz_c_list_1: Vec<f64>,

    /// density-density interaction within channel 0: (i, j) -> V_ij^(0)
    #[pyo3(get)]
    pub density_density_0: HashMap<(usize, usize), f64>,

    /// density-density interaction within channel 1: (i, j) -> V_ij^(1)
    #[pyo3(get)]
    pub density_density_1: HashMap<(usize, usize), f64>,

    /// density-density interaction between channels: (i, j) -> V_ij^(01) for n_{i,0} n_{j,1}
    #[pyo3(get)]
    pub density_density_01: HashMap<(usize, usize), f64>,

    // -------------------------
    // Localized spins
    // -------------------------
    /// Zeeman field (z) for localized spins, length = num_sites
    #[pyo3(get)]
    pub hz_f_list: Vec<f64>,

    /// single-ion anisotropy D_i (Sz)^2, length = num_sites
    #[pyo3(get)]
    pub d_list: Vec<f64>,

    /// localized-spin exchange (xy): (1/2) J_xy_ij (S_i^+ S_j^- + S_i^- S_j^+)
    #[pyo3(get)]
    pub ff_exchange_xy: HashMap<(usize, usize), f64>,

    /// localized-spin exchange (z): J_z_ij S_i^z S_j^z
    #[pyo3(get)]
    pub ff_exchange_z: HashMap<(usize, usize), f64>,

    // -------------------------
    // Kondo couplings (channel-resolved)
    // -------------------------
    /// Kondo coupling (xy) for channel 0:
    /// (1/2) K_xy_i^(0) (S^+ s^-_{i0} + S^- s^+_{i0}), length = num_sites
    #[pyo3(get)]
    pub kondo_xy_list_0: Vec<f64>,

    /// Kondo coupling (z) for channel 0:
    /// Kz_i^(0) Sz * sz_{i0}, length = num_sites
    #[pyo3(get)]
    pub kondo_z_list_0: Vec<f64>,

    /// Kondo coupling (xy) for channel 1:
    /// (1/2) K_xy_i^(1) (S^+ s^-_{i1} + S^- s^+_{i1}), length = num_sites
    #[pyo3(get)]
    pub kondo_xy_list_1: Vec<f64>,

    /// Kondo coupling (z) for channel 1:
    /// Kz_i^(1) Sz * sz_{i1}, length = num_sites
    #[pyo3(get)]
    pub kondo_z_list_1: Vec<f64>,
}

impl KondoLattice2ChModel {
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
impl KondoLattice2ChModel {
    #[new]
    #[pyo3(
        text_signature = "(spin_list, hopping_0, hopping_1, u_list_0, u_list_1, mu_list_0, mu_list_1, hz_c_list_0, hz_c_list_1, density_density_0, density_density_1, density_density_01, hz_f_list, d_list, ff_exchange_xy, ff_exchange_z, kondo_xy_list_0, kondo_z_list_0, kondo_xy_list_1, kondo_z_list_1)"
    )]
    pub fn new(
        spin_list: Vec<f64>,
        hopping_0: HashMap<(usize, usize), f64>,
        hopping_1: HashMap<(usize, usize), f64>,
        u_list_0: Vec<f64>,
        u_list_1: Vec<f64>,
        mu_list_0: Vec<f64>,
        mu_list_1: Vec<f64>,
        hz_c_list_0: Vec<f64>,
        hz_c_list_1: Vec<f64>,
        density_density_0: HashMap<(usize, usize), f64>,
        density_density_1: HashMap<(usize, usize), f64>,
        density_density_01: HashMap<(usize, usize), f64>,
        hz_f_list: Vec<f64>,
        d_list: Vec<f64>,
        ff_exchange_xy: HashMap<(usize, usize), f64>,
        ff_exchange_z: HashMap<(usize, usize), f64>,
        kondo_xy_list_0: Vec<f64>,
        kondo_z_list_0: Vec<f64>,
        kondo_xy_list_1: Vec<f64>,
        kondo_z_list_1: Vec<f64>,
    ) -> Result<Self> {
        let num_sites = spin_list.len();
        if num_sites == 0 {
            bail!("num_sites must be non-zero");
        }

        for (name, len) in [
            ("u_list_0", u_list_0.len()),
            ("u_list_1", u_list_1.len()),
            ("mu_list_0", mu_list_0.len()),
            ("mu_list_1", mu_list_1.len()),
            ("hz_c_list_0", hz_c_list_0.len()),
            ("hz_c_list_1", hz_c_list_1.len()),
            ("hz_f_list", hz_f_list.len()),
            ("d_list", d_list.len()),
            ("kondo_xy_list_0", kondo_xy_list_0.len()),
            ("kondo_z_list_0", kondo_z_list_0.len()),
            ("kondo_xy_list_1", kondo_xy_list_1.len()),
            ("kondo_z_list_1", kondo_z_list_1.len()),
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

        Self::check_pairs("hopping_0", num_sites, &hopping_0)?;
        Self::check_pairs("hopping_1", num_sites, &hopping_1)?;
        Self::check_pairs("density_density_0", num_sites, &density_density_0)?;
        Self::check_pairs("density_density_1", num_sites, &density_density_1)?;
        Self::check_pairs("density_density_01", num_sites, &density_density_01)?;
        Self::check_pairs("ff_exchange_xy", num_sites, &ff_exchange_xy)?;
        Self::check_pairs("ff_exchange_z", num_sites, &ff_exchange_z)?;

        Ok(Self {
            num_sites,
            two_s_list,
            hopping_0,
            hopping_1,
            u_list_0,
            u_list_1,
            mu_list_0,
            mu_list_1,
            hz_c_list_0,
            hz_c_list_1,
            density_density_0,
            density_density_1,
            density_density_01,
            hz_f_list,
            d_list,
            ff_exchange_xy,
            ff_exchange_z,
            kondo_xy_list_0,
            kondo_z_list_0,
            kondo_xy_list_1,
            kondo_z_list_1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_valid_input() {
        let spin_list = vec![0.5, 0.5, 1.0];

        let mut hopping_0 = HashMap::new();
        hopping_0.insert((0, 1), 1.0);
        let hopping_1 = HashMap::new();

        let u_list_0 = vec![0.0, 0.0, 0.0];
        let u_list_1 = vec![0.0, 0.0, 0.0];
        let mu_list_0 = vec![0.0, 0.0, 0.0];
        let mu_list_1 = vec![0.0, 0.0, 0.0];
        let hz_c_list_0 = vec![0.0, 0.0, 0.0];
        let hz_c_list_1 = vec![0.0, 0.0, 0.0];

        let density_density_0 = HashMap::new();
        let density_density_1 = HashMap::new();
        let density_density_01 = HashMap::new();

        let hz_f_list = vec![0.0, 0.0, 0.0];
        let d_list = vec![0.0, 0.0, 0.0];

        let ff_exchange_xy = HashMap::new();
        let ff_exchange_z = HashMap::new();

        let kondo_xy_list_0 = vec![1.0, 1.0, 1.0];
        let kondo_z_list_0 = vec![1.0, 1.0, 1.0];
        let kondo_xy_list_1 = vec![1.0, 1.0, 1.0];
        let kondo_z_list_1 = vec![1.0, 1.0, 1.0];

        let m = KondoLattice2ChModel::new(
            spin_list,
            hopping_0,
            hopping_1,
            u_list_0,
            u_list_1,
            mu_list_0,
            mu_list_1,
            hz_c_list_0,
            hz_c_list_1,
            density_density_0,
            density_density_1,
            density_density_01,
            hz_f_list,
            d_list,
            ff_exchange_xy,
            ff_exchange_z,
            kondo_xy_list_0,
            kondo_z_list_0,
            kondo_xy_list_1,
            kondo_z_list_1,
        )
        .unwrap();

        assert_eq!(m.num_sites, 3);
        assert_eq!(m.two_s_list, vec![1, 1, 2]);
    }

    #[test]
    fn reject_length_mismatch() {
        let spin_list = vec![0.5, 0.5];
        let hopping_0 = HashMap::new();
        let hopping_1 = HashMap::new();

        assert!(KondoLattice2ChModel::new(
            spin_list,
            hopping_0,
            hopping_1,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0], // mismatch
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let spin_list = vec![0.5, 0.5];
        let mut hopping_0 = HashMap::new();
        hopping_0.insert((0, 2), 1.0);

        assert!(KondoLattice2ChModel::new(
            spin_list,
            hopping_0,
            HashMap::new(),
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .is_err());
    }
}
