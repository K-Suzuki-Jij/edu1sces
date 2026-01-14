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

    /// Dimension of the U(1) sector specified by:
    /// - total number of conduction electrons `total_num_electrons`
    /// - total Sz (conduction + localized) `total_sz`
    #[pyo3(text_signature = "(self, total_num_electrons, total_sz)")]
    pub fn calc_dim_u1_sector(&self, total_num_electrons: i32, total_sz: f64) -> Result<i128> {
        let num_sites = self.num_sites;

        if total_num_electrons < 0 || total_num_electrons > (2 * num_sites) as i32 {
            bail!(
                "total_num_electrons out of range: {} (valid: 0..={})",
                total_num_electrons,
                2 * num_sites
            );
        }

        let total_two_sz = (2.0 * total_sz).round() as i32;
        if ((2.0 * total_sz) - (total_two_sz as f64)).abs() > 1e-12 {
            bail!("total_sz must be integer or half-integer");
        }

        // global min/max possible (2*Sz) to size the DP array
        // For each site: localized two_sz ∈ [-two_s, -two_s+2, ..., two_s]
        // conduction contribution two_sz_c ∈ {0, +1, -1, 0} (empty, up, down, double)
        // => per-site two_sz_total ∈ [-(two_s+1), +(two_s+1)]
        let mut min_two_sz_total = 0i32;
        let mut max_two_sz_total = 0i32;
        for &two_s in self.two_s_list.iter() {
            if two_s < 0 {
                bail!("two_s_list contains negative value: {}", two_s);
            }
            min_two_sz_total -= two_s + 1;
            max_two_sz_total += two_s + 1;
        }

        if total_two_sz < min_two_sz_total || total_two_sz > max_two_sz_total {
            return Ok(0);
        }

        let two_sz_offset = -min_two_sz_total;
        let two_sz_range = (max_two_sz_total - min_two_sz_total + 1) as usize;

        // dp[n][idx] = number of ways after processing some sites
        // where n = total conduction electrons, idx encodes total two_sz via offset.
        let n_max = 2 * num_sites;
        let mut dp = vec![vec![0i128; two_sz_range]; n_max + 1];
        dp[0][two_sz_offset as usize] = 1;

        // Conduction-electron local states: (dn, two_sz_c)
        let conduction_states = [(0i32, 0i32), (1, 1), (1, -1), (2, 0)];

        for &two_s in self.two_s_list.iter() {
            // Per-site aggregate: (dn, two_sz_site) -> multiplicity
            let mut local = HashMap::new();

            let mut two_sz_f = -two_s;
            while two_sz_f <= two_s {
                for &(dn, two_sz_c) in conduction_states.iter() {
                    let key = (dn, two_sz_c + two_sz_f);
                    *local.entry(key).or_insert(0i128) += 1;
                }
                two_sz_f += 2;
            }

            let mut next = vec![vec![0i128; two_sz_range]; n_max + 1];

            for n in 0..=n_max {
                for idx in 0..two_sz_range {
                    let cur = dp[n][idx];
                    if cur == 0 {
                        continue;
                    }

                    let cur_two_sz = (idx as i32) - two_sz_offset;

                    for (&(dn, two_sz_site), &mult) in local.iter() {
                        let n2_i32 = (n as i32) + dn;
                        if n2_i32 < 0 || n2_i32 > n_max as i32 {
                            continue;
                        }
                        let n2 = n2_i32 as usize;

                        let two_sz2 = cur_two_sz + two_sz_site;
                        if two_sz2 < min_two_sz_total || two_sz2 > max_two_sz_total {
                            continue;
                        }
                        let idx2 = (two_sz2 + two_sz_offset) as usize;

                        let add = match cur.checked_mul(mult) {
                            Some(v) => v,
                            None => bail!("i128 overflow"),
                        };
                        next[n2][idx2] = match next[n2][idx2].checked_add(add) {
                            Some(v) => v,
                            None => bail!("i128 overflow"),
                        };
                    }
                }
            }

            dp = next;
        }

        Ok(dp[total_num_electrons as usize][(total_two_sz + two_sz_offset) as usize])
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

    #[test]
    fn two_sites_s_half_n2_nontrivial_exact_counts() {
        let m = KondoLatticeModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hopping: HashMap::new(),
            u_list: vec![0.0, 0.0],
            mu_list: vec![0.0, 0.0],
            hz_c_list: vec![0.0, 0.0],
            hz_f_list: vec![0.0, 0.0],
            d_list: vec![0.0, 0.0],
            density_density: HashMap::new(),
            kondo_xy_list: vec![0.0, 0.0],
            kondo_z_list: vec![0.0, 0.0],
            ff_exchange_xy: HashMap::new(),
            ff_exchange_z: HashMap::new(),
        };

        assert_eq!(m.calc_dim_u1_sector(2, -2.0).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(2, -1.0).unwrap(), 6);
        assert_eq!(m.calc_dim_u1_sector(2, 0.0).unwrap(), 10);
        assert_eq!(m.calc_dim_u1_sector(2, 1.0).unwrap(), 6);
        assert_eq!(m.calc_dim_u1_sector(2, 2.0).unwrap(), 1);
    }

    #[test]
    fn one_site_s_one_exact_counts() {
        // 1 site, S = 1 (two_s = 2)
        let m = KondoLatticeModel {
            num_sites: 1,
            two_s_list: vec![2],
            hopping: HashMap::new(),
            u_list: vec![0.0],
            mu_list: vec![0.0],
            hz_c_list: vec![0.0],
            hz_f_list: vec![0.0],
            d_list: vec![0.0],
            density_density: HashMap::new(),
            kondo_xy_list: vec![0.0],
            kondo_z_list: vec![0.0],
            ff_exchange_xy: HashMap::new(),
            ff_exchange_z: HashMap::new(),
        };

        // n = 0 : localized Sz = -1,0,1
        assert_eq!(m.calc_dim_u1_sector(0, -1.0).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(0, 0.0).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(0, 1.0).unwrap(), 1);

        // n = 1
        // Sz = {-3/2,-1/2,1/2,3/2} with multiplicities {1,2,2,1}
        assert_eq!(m.calc_dim_u1_sector(1, -1.5).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(1, -0.5).unwrap(), 2);
        assert_eq!(m.calc_dim_u1_sector(1, 0.5).unwrap(), 2);
        assert_eq!(m.calc_dim_u1_sector(1, 1.5).unwrap(), 1);

        // n = 2 : same pattern as n=0
        assert_eq!(m.calc_dim_u1_sector(2, -1.0).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(2, 0.0).unwrap(), 1);
        assert_eq!(m.calc_dim_u1_sector(2, 1.0).unwrap(), 1);
    }
}
