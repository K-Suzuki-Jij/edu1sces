use ahash::AHashMap;

/// Kondo lattice model with
/// - U(1) particle-number conservation
/// - U(1) spin symmetry (Sz conservation)
#[derive(Debug, Clone)]
pub struct KondoLatticeModel {
    /// number of lattice sites
    pub num_sites: usize,

    /// 2S_i for localized spins, length = num_sites
    pub two_s_list: Vec<i32>,

    /// conduction-electron hopping terms: (i, j) -> t_ij
    pub hopping: AHashMap<(usize, usize), f64>,

    /// onsite Hubbard interaction for conduction electrons: U_i
    pub u_list: Vec<f64>,

    /// onsite potential / chemical potential for conduction electrons: mu_i
    pub mu_list: Vec<f64>,

    /// Zeeman field (z) for conduction electrons: hz_c_i
    /// couples to (n_up - n_down)
    pub hz_c_list: Vec<f64>,

    /// Zeeman field (z) for localized spins: hz_f_i
    /// couples to Sz
    pub hz_f_list: Vec<f64>,

    /// single-ion anisotropy for localized spins: D_i (Sz)^2
    pub d_list: Vec<f64>,

    /// density-density interaction for conduction electrons: (i, j) -> V_ij
    pub density_density: AHashMap<(usize, usize), f64>,

    /// Kondo coupling (xy) on each site:
    /// (1/2) K_xy_i (S^+ s^- + S^- s^+)
    pub kondo_xy_list: Vec<f64>,

    /// Kondo coupling (z) on each site:
    /// Kz_i Sz * sz
    pub kondo_z_list: Vec<f64>,

    /// localized-spin exchange (xy):
    /// (1/2) J_xy_ij (S_i^+ S_j^- + S_i^- S_j^+)
    pub ff_exchange_xy: AHashMap<(usize, usize), f64>,

    /// localized-spin exchange (z):
    /// J_z_ij S_i^z S_j^z
    pub ff_exchange_z: AHashMap<(usize, usize), f64>,
}

impl KondoLatticeModel {
    pub fn new(
        spin_list: Vec<f64>,
        hopping: AHashMap<(usize, usize), f64>,
        u_list: Vec<f64>,
        mu_list: Vec<f64>,
        hz_c_list: Vec<f64>,
        hz_f_list: Vec<f64>,
        d_list: Vec<f64>,
        density_density: AHashMap<(usize, usize), f64>,
        kondo_xy_list: Vec<f64>,
        kondo_z_list: Vec<f64>,
        ff_exchange_xy: AHashMap<(usize, usize), f64>,
        ff_exchange_z: AHashMap<(usize, usize), f64>,
    ) -> Result<Self, String> {
        let num_sites = spin_list.len();
        if num_sites == 0 {
            return Err("num_sites must be non-zero".to_string());
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
                return Err(format!(
                    "length mismatch: {}.len() = {}, spin_list.len() = {}",
                    name, len, num_sites
                ));
            }
        }

        // convert spin_list -> two_s_list
        let mut two_s_list = vec![0; num_sites];
        for (i, &s) in spin_list.iter().enumerate() {
            let two_s_f = (2.0 * s).round();
            if (2.0 * s - two_s_f).abs() > 1e-12 {
                return Err(format!("Spin at site {} = {} is not a half-integer", i, s));
            }
            if two_s_f < 0.0 {
                return Err(format!(
                    "Spin at site {} must be non-negative (got {})",
                    i, s
                ));
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

    fn check_pairs(
        name: &str,
        num_sites: usize,
        map: &AHashMap<(usize, usize), f64>,
    ) -> Result<(), String> {
        for (&(i, j), _) in map.iter() {
            if i >= num_sites || j >= num_sites {
                return Err(format!(
                    "{} ({}, {}) refers to out-of-range site (num_sites = {})",
                    name, i, j, num_sites
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    #[test]
    fn accept_valid_input() {
        let spin_list = vec![0.5, 0.5, 1.0];

        let mut hopping = AHashMap::new();
        hopping.insert((0, 1), 1.0);

        let u_list = vec![0.0, 0.0, 0.0];
        let mu_list = vec![0.0, 0.0, 0.0];
        let hz_c_list = vec![0.0, 0.0, 0.0];
        let hz_f_list = vec![0.0, 0.0, 0.0];
        let d_list = vec![0.0, 0.0, 0.0];

        let density_density = AHashMap::new();

        let kondo_xy_list = vec![1.0, 1.0, 1.0];
        let kondo_z_list = vec![1.0, 1.0, 1.0];

        let ff_exchange_xy = AHashMap::new();
        let ff_exchange_z = AHashMap::new();

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
        let hopping = AHashMap::new();

        assert!(KondoLatticeModel::new(
            spin_list,
            hopping,
            vec![0.0, 0.0],
            vec![0.0], // mismatch
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            AHashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            AHashMap::new(),
            AHashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let spin_list = vec![0.5, 0.5];
        let mut hopping = AHashMap::new();
        hopping.insert((0, 2), 1.0);

        assert!(KondoLatticeModel::new(
            spin_list,
            hopping,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            AHashMap::new(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            AHashMap::new(),
            AHashMap::new(),
        )
        .is_err());
    }
}
