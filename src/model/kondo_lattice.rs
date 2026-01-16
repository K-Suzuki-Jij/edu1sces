use ahash::AHashMap;
use anyhow::{bail, Result};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::basis::Basis;
use crate::blas::{csr_add, csr_mul, csr_transpose, CsrMatrix};
use crate::model::quantum_model::QuantumModel;

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
    /// Matrix element H[row, col] = <row| S^+ |col>
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
            let k_row = row / 4;
            let e = row % 4;

            // S^+ raises m by 1, which means k decreases by 1
            // <row| S^+ |col> is nonzero when row has k-1 and col has k
            // So col = row + 4 (k_col = k_row + 1)
            let k_col = k_row + 1;
            if k_col < dim_spin {
                let col = k_col * 4 + e; // input state with higher k (lower m)
                let m_col = s - (k_col as f64); // m of input state
                // coefficient: sqrt(S(S+1) - m(m+1))
                let coeff = (s * (s + 1.0) - m_col * (m_col + 1.0)).sqrt();
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

    // =========================================================================
    // Local Hamiltonian
    // =========================================================================

    /// Local Hamiltonian at site i:
    ///   H_i = U_i n_up n_down
    ///       - mu_i n
    ///       + hz_c_i (n_up - n_down)
    ///       + hz_f_i Sz_loc
    ///       + d_i (Sz_loc)^2
    ///       + Kz_i Sz_loc * sz_cond
    ///       + (1/2) K_xy_i (S^+ s^- + S^- s^+)
    #[pyo3(text_signature = "(self, site)")]
    pub fn make_local_hamiltonian(&self, site: usize) -> Result<CsrMatrix> {
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

        // Build operators
        let n_up = self.make_local_op_n_up(site)?;
        let n_down = self.make_local_op_n_down(site)?;
        let l_sz = self.make_local_op_l_sz(site)?;
        let c_sz = self.make_local_op_c_sz(site)?;
        let l_sp = self.make_local_op_l_sp(site)?;
        let l_sm = self.make_local_op_l_sm(site)?;
        let c_sp = self.make_local_op_c_sp(site)?;
        let c_sm = self.make_local_op_c_sm(site)?;

        // Start with zero matrix
        let mut h = CsrMatrix {
            row_dim: dim,
            col_dim: dim,
            rows: vec![0; dim + 1],
            cols: Vec::new(),
            vals: Vec::new(),
        };

        // U n_up n_down
        let u = self.u_list[site];
        if u.abs() > 1e-15 {
            let n_up_n_down = csr_mul(1.0, &n_up, 1.0, &n_down)?;
            h = csr_add(1.0, &h, u, &n_up_n_down)?;
        }

        // -mu n = -mu (n_up + n_down)
        let mu = self.mu_list[site];
        if mu.abs() > 1e-15 {
            h = csr_add(1.0, &h, -mu, &n_up)?;
            h = csr_add(1.0, &h, -mu, &n_down)?;
        }

        // hz_c (n_up - n_down)
        let hz_c = self.hz_c_list[site];
        if hz_c.abs() > 1e-15 {
            h = csr_add(1.0, &h, hz_c, &n_up)?;
            h = csr_add(1.0, &h, -hz_c, &n_down)?;
        }

        // hz_f Sz_loc
        let hz_f = self.hz_f_list[site];
        if hz_f.abs() > 1e-15 {
            h = csr_add(1.0, &h, hz_f, &l_sz)?;
        }

        // d (Sz_loc)^2
        let d = self.d_list[site];
        if d.abs() > 1e-15 {
            let l_sz_sq = csr_mul(1.0, &l_sz, 1.0, &l_sz)?;
            h = csr_add(1.0, &h, d, &l_sz_sq)?;
        }

        // Kz Sz_loc * sz_cond
        let kz = self.kondo_z_list[site];
        if kz.abs() > 1e-15 {
            let l_sz_c_sz = csr_mul(1.0, &l_sz, 1.0, &c_sz)?;
            h = csr_add(1.0, &h, kz, &l_sz_c_sz)?;
        }

        // (1/2) K_xy (S^+ s^- + S^- s^+)
        let kxy = self.kondo_xy_list[site];
        if kxy.abs() > 1e-15 {
            let sp_sm = csr_mul(1.0, &l_sp, 1.0, &c_sm)?;
            let sm_sp = csr_mul(1.0, &l_sm, 1.0, &c_sp)?;
            h = csr_add(1.0, &h, 0.5 * kxy, &sp_sm)?;
            h = csr_add(1.0, &h, 0.5 * kxy, &sm_sp)?;
        }

        Ok(h)
    }
}

impl QuantumModel for KondoLatticeModel {
    fn num_sites(&self) -> usize {
        self.num_sites
    }

    fn local_dim(&self, site: usize) -> usize {
        // (2S + 1) * 4, where 4 = {|vac>, |up>, |down>, |updown>}
        let two_s = self.two_s_list[site];
        ((two_s as usize) + 1) * 4
    }

    /// Return quantum numbers [n, 2*total_sz] for a local state.
    /// Local state index = k * 4 + e, where:
    ///   k = 0, 1, ..., 2S (spin block index, m = S - k)
    ///   e = 0, 1, 2, 3 (electron state: vac, up, down, updown)
    fn quantum_numbers(&self, site: usize, local_state: usize) -> Vec<i32> {
        let two_s = self.two_s_list[site];
        let dim_spin = (two_s as usize) + 1;

        let k = local_state / 4;
        let e = local_state % 4;

        if k >= dim_spin {
            panic!(
                "invalid local_state: {} (site {} has dim {})",
                local_state,
                site,
                dim_spin * 4
            );
        }

        // Localized spin contribution: two_m = 2S - 2k
        let two_m_loc = two_s - 2 * (k as i32);

        // Electron contribution: (n, 2*sz)
        let (n_e, two_sz_e) = match e {
            0 => (0, 0),  // |vac>
            1 => (1, 1),  // |up>
            2 => (1, -1), // |down>
            3 => (2, 0),  // |updown>
            _ => unreachable!(),
        };

        // Total: [n, 2*(sz_loc + sz_e)]
        vec![n_e, two_m_loc + two_sz_e]
    }

    /// Build basis for sector with quantum numbers:
    ///   target_quantum_numbers[0] = num_electrons
    ///   target_quantum_numbers[1] = 2*total_sz
    fn build_basis(&self, target_quantum_numbers: &[i32]) -> Result<Basis> {
        let num_electrons = target_quantum_numbers[0] as usize;
        let total_sz2 = target_quantum_numbers[1];

        let num_sites = self.num_sites;
        if num_sites == 0 {
            bail!("num_sites must be non-zero");
        }
        if target_quantum_numbers.len() < 2 {
            bail!("target_quantum_numbers must have length >= 2");
        }
        if target_quantum_numbers[0] < 0 {
            bail!(
                "num_electrons must be non-negative (got {})",
                target_quantum_numbers[0]
            );
        }
        if num_electrons > 2 * num_sites {
            bail!(
                "num_electrons out of range: {} (valid: 0..={})",
                num_electrons,
                2 * num_sites
            );
        }

        // Parity condition:
        // total_sz2 = sum(two_s) + (n_up - n_down), and (n_up - n_down) ≡ num_electrons (mod 2)
        let sum_two_s: i32 = self.two_s_list.iter().sum();
        if (((sum_two_s + (num_electrons as i32)) - total_sz2) & 1) != 0 {
            bail!(
                "parity mismatch: sum(2S)+N = {} but 2*total_sz = {}",
                sum_two_s + (num_electrons as i32),
                total_sz2
            );
        }

        // Coarse range check for total_sz2 (tighter than +/-1 per site using fixed N):
        // electron contribution to 2*sz is in [-min(N,L), +min(N,L)]
        let e_max = (num_electrons.min(num_sites)) as i32;
        let min_two_m: i32 = self.two_s_list.iter().map(|&s| -s).sum::<i32>() - e_max;
        let max_two_m: i32 = self.two_s_list.iter().map(|&s| s).sum::<i32>() + e_max;
        if total_sz2 < min_two_m || total_sz2 > max_two_m {
            bail!(
                "2*total_sz = {} out of range [{}, {}]",
                total_sz2,
                min_two_m,
                max_two_m
            );
        }

        // Build site_base and local_dims (local dim = 4*(2S+1))
        let mut site_base = Vec::with_capacity(num_sites);
        let mut local_dims = Vec::with_capacity(num_sites);
        let mut site_stride: i128 = 1;
        for &two_s in self.two_s_list.iter() {
            if two_s < 0 {
                bail!("two_s_list contains negative value: {}", two_s);
            }
            site_base.push(site_stride);
            let local_dim = 4usize
                .checked_mul((two_s as usize) + 1)
                .ok_or_else(|| anyhow::anyhow!("usize overflow"))?;
            local_dims.push(local_dim);
            site_stride = site_stride
                .checked_mul(local_dim as i128)
                .ok_or_else(|| anyhow::anyhow!("i128 overflow"))?;
        }

        // Suffix bounds for pruning:
        // per site:
        //   electrons in [0,2]
        //   two_sz in [-(two_s+1), +(two_s+1)]  (localized in [-two_s,+two_s] plus electron in [-1,+1])
        let mut suffix_min_n = vec![0i32; num_sites + 1];
        let mut suffix_max_n = vec![0i32; num_sites + 1];
        let mut suffix_min_sz2 = vec![0i32; num_sites + 1];
        let mut suffix_max_sz2 = vec![0i32; num_sites + 1];
        for i in (0..num_sites).rev() {
            suffix_min_n[i] = suffix_min_n[i + 1];
            suffix_max_n[i] = suffix_max_n[i + 1] + 2;

            let two_s = self.two_s_list[i];
            suffix_min_sz2[i] = suffix_min_sz2[i + 1] - (two_s + 1);
            suffix_max_sz2[i] = suffix_max_sz2[i + 1] + (two_s + 1);
        }

        // Reserve basis capacity if possible (optional fast path).
        // Using the same DP-style method you already have (if present).
        let mut basis = Vec::new();
        if let Ok(dim) = self.calc_dim_u1_sector(num_electrons as i32, (total_sz2 as f64) / 2.0) {
            if dim > 0 && dim <= (usize::MAX as i128) {
                basis = Vec::with_capacity(dim as usize);
            }
        }

        // Electron local states in fixed order:
        // e=0 vac  -> (dn=0, e_sz2=0)
        // e=1 up   -> (dn=1, e_sz2=+1)
        // e=2 down -> (dn=1, e_sz2=-1)
        // e=3 updn -> (dn=2, e_sz2=0)
        const DN: [i32; 4] = [0, 1, 1, 2];
        const ESZ2: [i32; 4] = [0, 1, -1, 0];

        fn dfs(
            site: usize,
            two_s_list: &[i32],
            site_base: &[i128],
            suffix_min_n: &[i32],
            suffix_max_n: &[i32],
            suffix_min_sz2: &[i32],
            suffix_max_sz2: &[i32],
            n_target: i32,
            sz2_target: i32,
            n_sum: i32,
            sz2_sum: i32,
            basis_code: i128,
            out: &mut Vec<i128>,
        ) {
            if site == two_s_list.len() {
                if n_sum == n_target && sz2_sum == sz2_target {
                    out.push(basis_code);
                }
                return;
            }

            let remain_min_n = suffix_min_n[site + 1];
            let remain_max_n = suffix_max_n[site + 1];
            let remain_min_sz2 = suffix_min_sz2[site + 1];
            let remain_max_sz2 = suffix_max_sz2[site + 1];

            // quick prune by remaining possible ranges
            let need_n = n_target - n_sum;
            if need_n < 0 + remain_min_n || need_n > 2 + remain_max_n {
                return;
            }

            let two_s = two_s_list[site];
            let need_sz2 = sz2_target - sz2_sum;
            if need_sz2 < (-(two_s + 1)) + remain_min_sz2 || need_sz2 > (two_s + 1) + remain_max_sz2
            {
                return;
            }

            let base = site_base[site];

            // local basis digit = 4*k + e, where k=0..2S (m=S-k), e=0..3 (vac,up,down,updn)
            let dim_spin = (two_s as usize) + 1;
            let mut k = 0usize;
            while k < dim_spin {
                let two_m = two_s - 2 * (k as i32); // 2m
                let block = (4 * k) as i128;

                let mut e = 0usize;
                while e < 4 {
                    let dn = DN[e];
                    let sz2_site = two_m + ESZ2[e];

                    // extra prune (tight, cheap)
                    let n2 = n_sum + dn;
                    if n2 <= n_target {
                        let sz2_2 = sz2_sum + sz2_site;

                        let need_n2 = n_target - n2;
                        if need_n2 >= remain_min_n && need_n2 <= remain_max_n {
                            let need_sz2_2 = sz2_target - sz2_2;
                            if need_sz2_2 >= remain_min_sz2 && need_sz2_2 <= remain_max_sz2 {
                                let digit = block + (e as i128);
                                dfs(
                                    site + 1,
                                    two_s_list,
                                    site_base,
                                    suffix_min_n,
                                    suffix_max_n,
                                    suffix_min_sz2,
                                    suffix_max_sz2,
                                    n_target,
                                    sz2_target,
                                    n2,
                                    sz2_2,
                                    basis_code + digit * base,
                                    out,
                                );
                            }
                        }
                    }

                    e += 1;
                }
                k += 1;
            }
        }

        dfs(
            0,
            &self.two_s_list,
            &site_base,
            &suffix_min_n,
            &suffix_max_n,
            &suffix_min_sz2,
            &suffix_max_sz2,
            num_electrons as i32,
            total_sz2,
            0,
            0,
            0i128,
            &mut basis,
        );

        basis.sort_unstable();

        let mut inverse_basis = AHashMap::with_capacity(basis.len());
        for (i, &basis_code) in basis.iter().enumerate() {
            inverse_basis.insert(basis_code, i);
        }

        Ok(Basis::new(basis, inverse_basis, site_base, local_dims))
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
        // Matrix element <row|S^+|col> is nonzero when:
        //   col is in k=1 block (states 4-7), row is in k=0 block (states 0-3)
        // CSR: rows 0-3 have entries pointing to cols 4-7
        let m = make_model_s_half();
        let l_sp = m.make_local_op_l_sp(0).unwrap();

        assert_eq!(l_sp.row_dim, 8);
        assert_eq!(l_sp.col_dim, 8);
        // row 0: <0|S^+|4> = 1, row 1: <1|S^+|5> = 1, etc.
        assert_eq!(l_sp.rows, vec![0, 1, 2, 3, 4, 4, 4, 4, 4]);
        assert_eq!(l_sp.cols, vec![4, 5, 6, 7]);
        assert_eq!(l_sp.vals, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn local_op_l_sp_s_one() {
        // S = 1:
        // k=0: m=+1, can be reached from k=1 (m=0)
        // k=1: m=0, can be reached from k=2 (m=-1)
        // k=2: m=-1, cannot be reached by S^+ (would need m=-2)
        //
        // S^+ |m=0> = sqrt(1*2 - 0*1) |m=+1> = sqrt(2)
        // S^+ |m=-1> = sqrt(1*2 - (-1)*0) |m=0> = sqrt(2)
        //
        // Matrix element <row|S^+|col>:
        //   rows 0-3 (k=0): <k=0|S^+|k=1> = sqrt(2) -> cols 4-7
        //   rows 4-7 (k=1): <k=1|S^+|k=2> = sqrt(2) -> cols 8-11
        let m = make_model_s_one();
        let l_sp = m.make_local_op_l_sp(0).unwrap();

        let sqrt2 = 2.0_f64.sqrt();

        assert_eq!(l_sp.row_dim, 12);
        assert_eq!(l_sp.col_dim, 12);
        // rows 0-3: entry at cols 4-7
        // rows 4-7: entry at cols 8-11
        // rows 8-11: no entries
        assert_eq!(l_sp.rows, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8]);
        assert_eq!(l_sp.cols, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        for &v in &l_sp.vals {
            assert!((v - sqrt2).abs() < 1e-12);
        }
    }

    // =========================================================================
    // build_basis tests
    // =========================================================================

    #[test]
    fn build_basis_one_site_s_half() {
        // 1 site, S = 1/2
        // Local states (index = 4*k + e):
        //   0: |+1/2>|vac>   -> [n=0, 2*Sz=1]
        //   1: |+1/2>|up>    -> [n=1, 2*Sz=2]
        //   2: |+1/2>|down>  -> [n=1, 2*Sz=0]
        //   3: |+1/2>|updown>-> [n=2, 2*Sz=1]
        //   4: |-1/2>|vac>   -> [n=0, 2*Sz=-1]
        //   5: |-1/2>|up>    -> [n=1, 2*Sz=0]
        //   6: |-1/2>|down>  -> [n=1, 2*Sz=-2]
        //   7: |-1/2>|updown>-> [n=2, 2*Sz=-1]
        let m = make_model_s_half();

        // (N=0, Sz=0.5): state 0 only
        let b = m.build_basis(&[0, 1]).unwrap();
        assert_eq!(b.basis, vec![0]);

        // (N=0, Sz=-0.5): state 4 only
        let b = m.build_basis(&[0, -1]).unwrap();
        assert_eq!(b.basis, vec![4]);

        // (N=1, Sz=1.0): state 1 only
        let b = m.build_basis(&[1, 2]).unwrap();
        assert_eq!(b.basis, vec![1]);

        // (N=1, Sz=0.0): states 2 and 5
        let b = m.build_basis(&[1, 0]).unwrap();
        assert_eq!(b.basis, vec![2, 5]);

        // (N=1, Sz=-1.0): state 6 only
        let b = m.build_basis(&[1, -2]).unwrap();
        assert_eq!(b.basis, vec![6]);

        // (N=2, Sz=0.5): state 3 only
        let b = m.build_basis(&[2, 1]).unwrap();
        assert_eq!(b.basis, vec![3]);

        // (N=2, Sz=-0.5): state 7 only
        let b = m.build_basis(&[2, -1]).unwrap();
        assert_eq!(b.basis, vec![7]);
    }

    #[test]
    fn build_basis_one_site_s_one() {
        // 1 site, S = 1
        // Local states (index = 4*k + e):
        //   k=0 (m=+1):  0-3
        //   k=1 (m=0):   4-7
        //   k=2 (m=-1):  8-11
        //
        //   0: |+1>|vac>   -> [n=0, 2*Sz=2]
        //   1: |+1>|up>    -> [n=1, 2*Sz=3]
        //   2: |+1>|down>  -> [n=1, 2*Sz=1]
        //   3: |+1>|updown>-> [n=2, 2*Sz=2]
        //   4: |0>|vac>    -> [n=0, 2*Sz=0]
        //   5: |0>|up>     -> [n=1, 2*Sz=1]
        //   6: |0>|down>   -> [n=1, 2*Sz=-1]
        //   7: |0>|updown> -> [n=2, 2*Sz=0]
        //   8: |-1>|vac>   -> [n=0, 2*Sz=-2]
        //   9: |-1>|up>    -> [n=1, 2*Sz=-1]
        //  10: |-1>|down>  -> [n=1, 2*Sz=-3]
        //  11: |-1>|updown>-> [n=2, 2*Sz=-2]
        let m = make_model_s_one();

        // (N=0, Sz=1.0): state 0 only
        let b = m.build_basis(&[0, 2]).unwrap();
        assert_eq!(b.basis, vec![0]);

        // (N=0, Sz=0.0): state 4 only
        let b = m.build_basis(&[0, 0]).unwrap();
        assert_eq!(b.basis, vec![4]);

        // (N=0, Sz=-1.0): state 8 only
        let b = m.build_basis(&[0, -2]).unwrap();
        assert_eq!(b.basis, vec![8]);

        // (N=1, Sz=1.5): state 1 only
        let b = m.build_basis(&[1, 3]).unwrap();
        assert_eq!(b.basis, vec![1]);

        // (N=1, Sz=0.5): states 2 and 5
        let b = m.build_basis(&[1, 1]).unwrap();
        assert_eq!(b.basis, vec![2, 5]);

        // (N=1, Sz=-0.5): states 6 and 9
        let b = m.build_basis(&[1, -1]).unwrap();
        assert_eq!(b.basis, vec![6, 9]);

        // (N=1, Sz=-1.5): state 10 only
        let b = m.build_basis(&[1, -3]).unwrap();
        assert_eq!(b.basis, vec![10]);

        // (N=2, Sz=1.0): state 3 only
        let b = m.build_basis(&[2, 2]).unwrap();
        assert_eq!(b.basis, vec![3]);

        // (N=2, Sz=0.0): state 7 only
        let b = m.build_basis(&[2, 0]).unwrap();
        assert_eq!(b.basis, vec![7]);

        // (N=2, Sz=-1.0): state 11 only
        let b = m.build_basis(&[2, -2]).unwrap();
        assert_eq!(b.basis, vec![11]);
    }

    #[test]
    fn build_basis_two_sites_s_half() {
        // 2 sites, S = 1/2 each
        // site_base[0] = 1, site_base[1] = 8
        // basis_code = local_state[0] + 8 * local_state[1]
        //
        // For (N=1, Sz=0): total n=1, total 2*Sz=0
        // Possible combinations:
        //   site0: n=0, 2*sz_tot=a  =>  site1: n=1, 2*sz_tot=-a
        //   site0: n=1, 2*sz_tot=a  =>  site1: n=0, 2*sz_tot=-a
        //
        // site0 n=0, 2*sz=1 (state 0): site1 n=1, 2*sz=-1 => no such state (need 2*sz_loc=-1 from electron impossible with n=1)
        // Actually let's enumerate carefully:
        //
        // Local states for S=1/2:
        //   0: |+1/2>|vac>   -> n=0, 2*sz=1
        //   1: |+1/2>|up>    -> n=1, 2*sz=2
        //   2: |+1/2>|down>  -> n=1, 2*sz=0
        //   3: |+1/2>|updown>-> n=2, 2*sz=1
        //   4: |-1/2>|vac>   -> n=0, 2*sz=-1
        //   5: |-1/2>|up>    -> n=1, 2*sz=0
        //   6: |-1/2>|down>  -> n=1, 2*sz=-2
        //   7: |-1/2>|updown>-> n=2, 2*sz=-1
        //
        // (N=1, Sz=0): n_tot=1, 2*sz_tot=0
        // Combinations:
        //   site0=0 (n=0, 2*sz=1): need site1 n=1, 2*sz=-1 => no match
        //   site0=4 (n=0, 2*sz=-1): need site1 n=1, 2*sz=1 => no match
        //   site0=2 (n=1, 2*sz=0): need site1 n=0, 2*sz=0 => no match
        //   site0=5 (n=1, 2*sz=0): need site1 n=0, 2*sz=0 => no match
        //
        // Hmm, for 2*sz=0 at site1 with n=0, we need localized spin m=0, but S=1/2 only has m=+1/2,-1/2.
        // So (N=1, Sz=0) sector should be empty for 2-site S=1/2.
        //
        // Let's try (N=2, Sz=0): n_tot=2, 2*sz_tot=0
        //   site0=0 (n=0, 2*sz=1): need site1 n=2, 2*sz=-1 => state 7 => code = 0 + 8*7 = 56
        //   site0=4 (n=0, 2*sz=-1): need site1 n=2, 2*sz=1 => state 3 => code = 4 + 8*3 = 28
        //   site0=2 (n=1, 2*sz=0): need site1 n=1, 2*sz=0 => states 2,5
        //       code = 2 + 8*2 = 18
        //       code = 2 + 8*5 = 42
        //   site0=5 (n=1, 2*sz=0): need site1 n=1, 2*sz=0 => states 2,5
        //       code = 5 + 8*2 = 21
        //       code = 5 + 8*5 = 45
        //   site0=3 (n=2, 2*sz=1): need site1 n=0, 2*sz=-1 => state 4 => code = 3 + 8*4 = 35
        //   site0=7 (n=2, 2*sz=-1): need site1 n=0, 2*sz=1 => state 0 => code = 7 + 8*0 = 7
        //   site0=1 (n=1, 2*sz=2): need site1 n=1, 2*sz=-2 => state 6 => code = 1 + 8*6 = 49
        //   site0=6 (n=1, 2*sz=-2): need site1 n=1, 2*sz=2 => state 1 => code = 6 + 8*1 = 14
        //
        // Sorted: [7, 14, 18, 21, 28, 35, 42, 45, 49, 56]
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

        let b = m.build_basis(&[2, 0]).unwrap();
        assert_eq!(b.basis, vec![7, 14, 18, 21, 28, 35, 42, 45, 49, 56]);
    }

    // =========================================================================
    // make_local_hamiltonian tests
    // =========================================================================

    #[test]
    fn local_hamiltonian_kondo_xy_s_half() {
        // S = 1/2, Kxy = 2
        // Local Hamiltonian should have off-diagonal Kondo xy term
        // (1/2) Kxy (S^+ s^- + S^- s^+)
        //
        // Let's first understand what S^+ s^- and S^- s^+ do:
        //
        // S^+ (localized): raises m by 1 (k decreases by 1)
        //   S^+ |m=-1/2> = |m=+1/2>  (coefficient = 1 for S=1/2)
        //   S^+ |m=+1/2> = 0
        //
        // s^- (conduction): lowers conduction spin
        //   s^- |up> = |down>, s^- |down> = 0
        //   s^- = c_down^dag c_up
        //
        // S^+ s^-: Raises localized spin AND lowers conduction spin
        //   |m=-1/2>|up> -> |m=+1/2>|down>  i.e., state 5 -> state 2
        //   <2| S^+ s^- |5> = 1
        //
        // S^- s^+: Lowers localized spin AND raises conduction spin
        //   |m=+1/2>|down> -> |m=-1/2>|up>  i.e., state 2 -> state 5
        //   <5| S^- s^+ |2> = 1
        //
        // So H[2,5] = (1/2)*Kxy*1 = 1 and H[5,2] = (1/2)*Kxy*1 = 1
        let m = KondoLatticeModel {
            num_sites: 1,
            two_s_list: vec![1],
            hopping: HashMap::new(),
            u_list: vec![0.0],
            mu_list: vec![0.0],
            hz_c_list: vec![0.0],
            hz_f_list: vec![0.0],
            d_list: vec![0.0],
            density_density: HashMap::new(),
            kondo_xy_list: vec![2.0],
            kondo_z_list: vec![0.0],
            ff_exchange_xy: HashMap::new(),
            ff_exchange_z: HashMap::new(),
        };

        let h = m.make_local_hamiltonian(0).unwrap();

        // Should have off-diagonal elements at (2, 5) and (5, 2)
        let mut found_2_5 = false;
        let mut found_5_2 = false;
        let tol = 1e-12;

        for row in 0..h.row_dim {
            for k in h.rows[row]..h.rows[row + 1] {
                let col = h.cols[k];
                let val = h.vals[k];
                if row == 2 && col == 5 {
                    assert!((val - 1.0).abs() < tol);
                    found_2_5 = true;
                }
                if row == 5 && col == 2 {
                    assert!((val - 1.0).abs() < tol);
                    found_5_2 = true;
                }
            }
        }

        assert!(found_2_5 && found_5_2);
    }
}
