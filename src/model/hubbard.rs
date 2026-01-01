use ahash::AHashMap;

/// Hubbard model with U(1) particle-number conservation and Sz conservation
#[derive(Debug, Clone)]
pub struct HubbardModel {
    /// number of lattice sites
    pub num_sites: usize,

    /// hopping terms: (i, j) -> t_ij
    pub hopping: AHashMap<(usize, usize), f64>,

    /// onsite interaction U_i, length = num_sites
    pub u_list: Vec<f64>,

    /// onsite potential / chemical potential mu_i, length = num_sites
    pub mu_list: Vec<f64>,

    /// Zeeman field along z, length = num_sites
    /// couples to (n_up - n_down) (equivalently 2 Sz)
    pub hz_list: Vec<f64>,

    /// density-density interaction: (i, j) -> V_ij
    pub density_density: AHashMap<(usize, usize), f64>,

    /// exchange (xy) part:
    /// (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
    pub exchange_xy: AHashMap<(usize, usize), f64>,

    /// exchange (z) part:
    /// S_i^z S_j^z
    pub exchange_z: AHashMap<(usize, usize), f64>,
}

impl HubbardModel {
    pub fn new(
        hopping: AHashMap<(usize, usize), f64>,
        u_list: Vec<f64>,
        mu_list: Vec<f64>,
        hz_list: Vec<f64>,
        density_density: AHashMap<(usize, usize), f64>,
        exchange_xy: AHashMap<(usize, usize), f64>,
        exchange_z: AHashMap<(usize, usize), f64>,
    ) -> Result<Self, String> {
        if u_list.len() != mu_list.len() {
            return Err(format!(
                "length mismatch: u_list.len() = {}, mu_list.len() = {}",
                u_list.len(),
                mu_list.len()
            ));
        }
        if u_list.len() != hz_list.len() {
            return Err(format!(
                "length mismatch: u_list.len() = {}, hz_list.len() = {}",
                u_list.len(),
                hz_list.len()
            ));
        }

        let num_sites = u_list.len();
        if num_sites == 0 {
            return Err("num_sites must be non-zero".to_string());
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
        let mut hopping = AHashMap::new();
        hopping.insert((0, 1), 1.0);

        let u_list = vec![2.0, 2.0, 2.0];
        let mu_list = vec![0.0, 0.0, 0.0];
        let hz_list = vec![0.1, 0.1, 0.1];

        let m = HubbardModel::new(
            hopping,
            u_list,
            mu_list,
            hz_list,
            AHashMap::new(),
            AHashMap::new(),
            AHashMap::new(),
        )
        .unwrap();

        assert_eq!(m.num_sites, 3);
    }

    #[test]
    fn reject_length_mismatch_hz() {
        assert!(HubbardModel::new(
            AHashMap::new(),
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.1],
            AHashMap::new(),
            AHashMap::new(),
            AHashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_zero_sites() {
        assert!(HubbardModel::new(
            AHashMap::new(),
            vec![],
            vec![],
            vec![],
            AHashMap::new(),
            AHashMap::new(),
            AHashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn reject_invalid_index() {
        let mut hopping = AHashMap::new();
        hopping.insert((0, 3), 1.0);

        assert!(HubbardModel::new(
            hopping,
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            AHashMap::new(),
            AHashMap::new(),
            AHashMap::new(),
        )
        .is_err());
    }
}
