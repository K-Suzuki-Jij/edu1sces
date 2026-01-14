use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::blas::{csr_add, csr_mul, csr_transpose, CsrMatrix};

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

    // =========================================================================
    // Local operators
    //
    // Local basis ordering (for spin S):
    //   0  -> |S>|vac>
    //   1  -> |S>|up>
    //   2  -> |S>|down>
    //   3  -> |S>|updown>
    //   4  -> |S-1>|vac>
    //   5  -> |S-1>|up>
    //   6  -> |S-1>|down>
    //   7  -> |S-1>|updown>
    //   ...
    //   4k + 0 -> |S-k>|vac>
    //   4k + 1 -> |S-k>|up>
    //   4k + 2 -> |S-k>|down>
    //   4k + 3 -> |S-k>|updown>
    //
    // where k = 0, 1, ..., 2S
    // Total local dimension = 4 * (2S + 1)
    // =========================================================================

    /// c_up (annihilation): acts only on the conduction electron part
    /// |m>|up> -> |m>|vac>, |m>|updown> -> |m>|down>
    /// With fermion sign: c_up |updown> = +|down> (up is first)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_up(&self, site: usize) -> Result<CsrMatrix> {
        if site >= self.num_sites {
            bail!(
                "site {} out of range (num_sites = {})",
                site,
                self.num_sites
            );
        }

        let two_s = self.two_s_list[site];
        let dim_spin = (two_s as usize) + 1;
        let dim = dim_spin * 4;

        let mut rows = Vec::with_capacity(dim + 1);
        rows.push(0);
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for row in 0..dim {
            let e_row = row % 4;
            let base = row - e_row;

            match e_row {
                // |vac><up|
                0 => {
                    cols.push(base + 1);
                    vals.push(1.0);
                }
                // |down><updown|
                2 => {
                    cols.push(base + 3);
                    vals.push(1.0);
                }
                _ => {}
            }

            rows.push(vals.len());
        }

        Ok(CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows,
            cols,
            vals,
        })
    }

    /// c_down (annihilation): acts only on the conduction electron part
    /// |m>|down> -> |m>|vac>, |m>|updown> -> -|m>|up>
    /// With fermion sign: c_down |updown> = -|up> (down must pass up)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_down(&self, site: usize) -> Result<CsrMatrix> {
        if site >= self.num_sites {
            bail!(
                "site {} out of range (num_sites = {})",
                site,
                self.num_sites
            );
        }

        let two_s = self.two_s_list[site];
        let dim_spin = (two_s as usize) + 1;
        let dim = dim_spin * 4;

        let mut rows = Vec::with_capacity(dim + 1);
        rows.push(0);
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for row in 0..dim {
            let e_row = row % 4;
            let base = row - e_row;

            match e_row {
                // |vac><down|
                0 => {
                    cols.push(base + 2);
                    vals.push(1.0);
                }
                // -|up><updown|
                1 => {
                    cols.push(base + 3);
                    vals.push(-1.0);
                }
                _ => {}
            }

            rows.push(vals.len());
        }

        Ok(CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows,
            cols,
            vals,
        })
    }

    /// c_up^dag (creation) = transpose(c_up)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_up_dag(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_c_up(site)?)
    }

    /// c_down^dag (creation) = transpose(c_down)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_down_dag(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_c_down(site)?)
    }

    /// n_up = c_up^dag c_up
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_n_up(&self, site: usize) -> Result<CsrMatrix> {
        let c_up = self.make_local_op_c_up(site)?;
        let c_up_dag = self.make_local_op_c_up_dag(site)?;
        csr_mul(1.0, &c_up_dag, 1.0, &c_up)
    }

    /// n_down = c_down^dag c_down
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_n_down(&self, site: usize) -> Result<CsrMatrix> {
        let c_down = self.make_local_op_c_down(site)?;
        let c_down_dag = self.make_local_op_c_down_dag(site)?;
        csr_mul(1.0, &c_down_dag, 1.0, &c_down)
    }

    /// n = n_up + n_down (conduction electron number)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_n(&self, site: usize) -> Result<CsrMatrix> {
        let n_up = self.make_local_op_n_up(site)?;
        let n_down = self.make_local_op_n_down(site)?;
        csr_add(1.0, &n_up, 1.0, &n_down)
    }

    /// Conduction electron Sz = (n_up - n_down) / 2
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_sz(&self, site: usize) -> Result<CsrMatrix> {
        let n_up = self.make_local_op_n_up(site)?;
        let n_down = self.make_local_op_n_down(site)?;
        csr_add(0.5, &n_up, -0.5, &n_down)
    }

    /// Conduction electron S+ = c_up^dag c_down
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_sp(&self, site: usize) -> Result<CsrMatrix> {
        let c_up_dag = self.make_local_op_c_up_dag(site)?;
        let c_down = self.make_local_op_c_down(site)?;
        csr_mul(1.0, &c_up_dag, 1.0, &c_down)
    }

    /// Conduction electron S- = c_down^dag c_up = transpose(S+)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_c_sm(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_c_sp(site)?)
    }

    // =========================================================================
    // Localized spin operators
    // =========================================================================

    /// Localized spin Sz operator
    /// S_z |m>|e> = m |m>|e>
    /// where m = S, S-1, ..., -S and |e> is the electron state
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_l_sz(&self, site: usize) -> Result<CsrMatrix> {
        if site >= self.num_sites {
            bail!(
                "site {} out of range (num_sites = {})",
                site,
                self.num_sites
            );
        }

        let two_s = self.two_s_list[site];
        let dim_spin = (two_s as usize) + 1;
        let dim = dim_spin * 4;

        let mut rows = Vec::with_capacity(dim + 1);
        rows.push(0);
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for row in 0..dim {
            let k = row / 4;
            let two_m = two_s - 2 * (k as i32); // 2m = 2S - 2k

            // Skip if m = 0 (i.e., two_m == 0)
            if two_m != 0 {
                let m = (two_m as f64) / 2.0;
                cols.push(row);
                vals.push(m);
            }

            rows.push(vals.len());
        }

        Ok(CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows,
            cols,
            vals,
        })
    }

    /// Localized spin S+ operator
    /// S^+ |m>|e> = sqrt(S(S+1) - m(m+1)) |m+1>|e>
    /// In our basis: k -> k-1 (since m = S - k, so m+1 corresponds to k-1)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_l_sp(&self, site: usize) -> Result<CsrMatrix> {
        if site >= self.num_sites {
            bail!(
                "site {} out of range (num_sites = {})",
                site,
                self.num_sites
            );
        }

        let two_s = self.two_s_list[site];
        let dim_spin = (two_s as usize) + 1;
        let dim = dim_spin * 4;
        let s = (two_s as f64) / 2.0;

        let mut rows = Vec::with_capacity(dim + 1);
        rows.push(0);
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for row in 0..dim {
            let k = row / 4;
            let e = row % 4;

            // S^+ raises m by 1, which means k decreases by 1
            if k > 0 {
                let m = s - (k as f64); // current m
                // coefficient: sqrt(S(S+1) - m(m+1))
                let coeff = (s * (s + 1.0) - m * (m + 1.0)).sqrt();
                let col = (k - 1) * 4 + e; // target: k-1 block, same electron state
                cols.push(col);
                vals.push(coeff);
            }

            rows.push(vals.len());
        }

        Ok(CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows,
            cols,
            vals,
        })
    }

    /// Localized spin S- operator = transpose(S+)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_l_sm(&self, site: usize) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_l_sp(site)?)
    }

    /// Localized spin Sx operator = (S+ + S-) / 2
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_l_sx(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.make_local_op_l_sp(site)?;
        let sm = self.make_local_op_l_sm(site)?;
        csr_add(0.5, &sp, 0.5, &sm)
    }

    /// Localized spin i*Sy operator = (S+ - S-) / 2
    /// Returns i*Sy to avoid complex numbers
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_op_l_isy(&self, site: usize) -> Result<CsrMatrix> {
        let sp = self.make_local_op_l_sp(site)?;
        let sm = self.make_local_op_l_sm(site)?;
        csr_add(0.5, &sp, -0.5, &sm)
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

    // =========================================================================
    // Local operator tests
    // =========================================================================

    fn make_model_s_half() -> KondoLatticeModel {
        // S = 1/2, local dim = 4 * 2 = 8
        KondoLatticeModel {
            num_sites: 1,
            two_s_list: vec![1],
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
        }
    }

    fn make_model_s_one() -> KondoLatticeModel {
        // S = 1, local dim = 4 * 3 = 12
        KondoLatticeModel {
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
        }
    }

    #[test]
    fn local_op_c_up_s_half() {
        // S = 1/2: local dim = 8
        // Basis: |+1/2>|vac>, |+1/2>|up>, |+1/2>|down>, |+1/2>|updown>,
        //        |-1/2>|vac>, |-1/2>|up>, |-1/2>|down>, |-1/2>|updown>
        //
        // c_up acts on electron part only:
        //   row 0: c_up |+1/2>|up> = |+1/2>|vac>   => col=1
        //   row 2: c_up |+1/2>|updown> = |+1/2>|down> => col=3
        //   row 4: c_up |-1/2>|up> = |-1/2>|vac>   => col=5
        //   row 6: c_up |-1/2>|updown> = |-1/2>|down> => col=7
        let m = make_model_s_half();
        let c_up = m.make_local_op_c_up(0).unwrap();

        assert_eq!(c_up.row_dim, 8);
        assert_eq!(c_up.col_dim, 8);
        assert_eq!(c_up.rows, vec![0, 1, 1, 2, 2, 3, 3, 4, 4]);
        assert_eq!(c_up.cols, vec![1, 3, 5, 7]);
        assert_eq!(c_up.vals, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn local_op_c_down_s_half() {
        // c_down acts on electron part only:
        //   row 0: c_down |+1/2>|down> = |+1/2>|vac>   => col=2
        //   row 1: c_down |+1/2>|updown> = -|+1/2>|up> => col=3, val=-1
        //   row 4: c_down |-1/2>|down> = |-1/2>|vac>   => col=6
        //   row 5: c_down |-1/2>|updown> = -|-1/2>|up> => col=7, val=-1
        let m = make_model_s_half();
        let c_down = m.make_local_op_c_down(0).unwrap();

        assert_eq!(c_down.row_dim, 8);
        assert_eq!(c_down.col_dim, 8);
        assert_eq!(c_down.rows, vec![0, 1, 2, 2, 2, 3, 4, 4, 4]);
        assert_eq!(c_down.cols, vec![2, 3, 6, 7]);
        assert_eq!(c_down.vals, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn local_op_l_sz_s_half() {
        // S = 1/2: m = +1/2, -1/2
        // Diagonal: rows 0-3 have m=+1/2, rows 4-7 have m=-1/2
        let m = make_model_s_half();
        let l_sz = m.make_local_op_l_sz(0).unwrap();

        assert_eq!(l_sz.row_dim, 8);
        assert_eq!(l_sz.col_dim, 8);
        // m=+1/2 for rows 0,1,2,3; m=-1/2 for rows 4,5,6,7
        assert_eq!(l_sz.rows, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(l_sz.cols, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(l_sz.vals, vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]);
    }

    #[test]
    fn local_op_l_sz_s_one() {
        // S = 1: m = +1, 0, -1
        // Diagonal: rows 0-3 have m=+1, rows 4-7 have m=0, rows 8-11 have m=-1
        // m=0 rows should be skipped
        let m = make_model_s_one();
        let l_sz = m.make_local_op_l_sz(0).unwrap();

        assert_eq!(l_sz.row_dim, 12);
        assert_eq!(l_sz.col_dim, 12);
        // rows 0-3: m=+1, rows 4-7: m=0 (skipped), rows 8-11: m=-1
        assert_eq!(l_sz.rows, vec![0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8]);
        assert_eq!(l_sz.cols, vec![0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(l_sz.vals, vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn local_op_l_sp_s_half() {
        // S = 1/2: S^+ |m=-1/2> = sqrt(1/2 * 3/2 - (-1/2)(1/2)) |m=+1/2>
        //                      = sqrt(3/4 + 1/4) = 1
        // S^+ raises m by 1, i.e., k decreases by 1 (k=1 -> k=0)
        // So rows 4-7 (k=1) -> cols 0-3 (k=0)
        let m = make_model_s_half();
        let l_sp = m.make_local_op_l_sp(0).unwrap();

        assert_eq!(l_sp.row_dim, 8);
        assert_eq!(l_sp.col_dim, 8);
        // rows 0-3: k=0, cannot raise
        // rows 4-7: k=1, raise to k=0
        assert_eq!(l_sp.rows, vec![0, 0, 0, 0, 0, 1, 2, 3, 4]);
        assert_eq!(l_sp.cols, vec![0, 1, 2, 3]);
        assert_eq!(l_sp.vals, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn local_op_l_sp_s_one() {
        // S = 1:
        // k=0: m=+1, cannot raise further
        // k=1: m=0, S^+ |0> = sqrt(1*2 - 0*1) |+1> = sqrt(2)
        // k=2: m=-1, S^+ |-1> = sqrt(1*2 - (-1)*0) |0> = sqrt(2)
        let m = make_model_s_one();
        let l_sp = m.make_local_op_l_sp(0).unwrap();

        let sqrt2 = 2.0_f64.sqrt();

        assert_eq!(l_sp.row_dim, 12);
        assert_eq!(l_sp.col_dim, 12);
        // rows 0-3: k=0, no contribution
        // rows 4-7: k=1 -> k=0, coeff=sqrt(2)
        // rows 8-11: k=2 -> k=1, coeff=sqrt(2)
        assert_eq!(l_sp.rows, vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(l_sp.cols, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        for &v in &l_sp.vals {
            assert!((v - sqrt2).abs() < 1e-12);
        }
    }
}
