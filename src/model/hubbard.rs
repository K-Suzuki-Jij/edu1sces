use crate::blas::{csr_add, csr_mul, csr_transpose, CsrMatrix};
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
    /// couples to Sz = (n_up - n_down) / 2
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

    // =========================================================================
    // Local operators (4x4 matrices on local Fock space)
    //
    // Local basis:
    //   0 -> |vac>     (n_up=0, n_down=0)
    //   1 -> |up>      (n_up=1, n_down=0)
    //   2 -> |down>    (n_up=0, n_down=1)
    //   3 -> |updown>  (n_up=1, n_down=1)
    //
    // Fundamental operators: c_up, c_down (annihilation)
    // All other operators are derived from these.
    // =========================================================================

    /// c_up (annihilation): |up> -> |vac>, |updown> -> |down>
    /// With fermion sign: c_up |updown> = +|down> (up is first)
    pub fn make_local_op_c_up(&self) -> CsrMatrix {
        // row 0 (|vac>):  c_up |up> = |vac>      => col=1, val=1
        // row 2 (|down>): c_up |updown> = |down> => col=3, val=1
        CsrMatrix {
            row_dim: 4,
            col_dim: 4,
            rows: vec![0, 1, 1, 2, 2],
            cols: vec![1, 3],
            vals: vec![1.0, 1.0],
        }
    }

    /// c_down (annihilation): |down> -> |vac>, |updown> -> |up>
    /// With fermion sign: c_down |updown> = -|up> (down is second, must pass up)
    pub fn make_local_op_c_down(&self) -> CsrMatrix {
        // row 0 (|vac>): c_down |down> = |vac>   => col=2, val=1
        // row 1 (|up>):  c_down |updown> = -|up> => col=3, val=-1
        CsrMatrix {
            row_dim: 4,
            col_dim: 4,
            rows: vec![0, 1, 2, 2, 2],
            cols: vec![2, 3],
            vals: vec![1.0, -1.0],
        }
    }

    /// c_up^dag (creation) = transpose(c_up)
    pub fn make_local_op_c_up_dag(&self) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_c_up())
    }

    /// c_down^dag (creation) = transpose(c_down)
    pub fn make_local_op_c_down_dag(&self) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_c_down())
    }

    /// n_up = c_up^dag c_up
    pub fn make_local_op_n_up(&self) -> Result<CsrMatrix> {
        let c_up = self.make_local_op_c_up();
        let c_up_dag = self.make_local_op_c_up_dag()?;
        csr_mul(1.0, &c_up_dag, 1.0, &c_up)
    }

    /// n_down = c_down^dag c_down
    pub fn make_local_op_n_down(&self) -> Result<CsrMatrix> {
        let c_down = self.make_local_op_c_down();
        let c_down_dag = self.make_local_op_c_down_dag()?;
        csr_mul(1.0, &c_down_dag, 1.0, &c_down)
    }

    /// n = n_up + n_down
    pub fn make_local_op_n(&self) -> Result<CsrMatrix> {
        let n_up = self.make_local_op_n_up()?;
        let n_down = self.make_local_op_n_down()?;
        csr_add(1.0, &n_up, 1.0, &n_down)
    }

    /// Sz = (n_up - n_down) / 2
    pub fn make_local_op_sz(&self) -> Result<CsrMatrix> {
        let n_up = self.make_local_op_n_up()?;
        let n_down = self.make_local_op_n_down()?;
        csr_add(0.5, &n_up, -0.5, &n_down)
    }

    /// S+ = c_up^dag c_down
    pub fn make_local_op_sp(&self) -> Result<CsrMatrix> {
        let c_up_dag = self.make_local_op_c_up_dag()?;
        let c_down = self.make_local_op_c_down();
        csr_mul(1.0, &c_up_dag, 1.0, &c_down)
    }

    /// S- = c_down^dag c_up = transpose(S+)
    pub fn make_local_op_sm(&self) -> Result<CsrMatrix> {
        csr_transpose(1.0, &self.make_local_op_sp()?)
    }

    /// Local onsite Hamiltonian:
    /// H_i = U n_up n_down - mu (n_up + n_down) + hz Sz
    ///     = U n_up n_down - mu (n_up + n_down) + hz (n_up - n_down) / 2
    pub fn make_local_hamiltonian(&self, site: usize) -> Result<CsrMatrix> {
        let u = self.u_list[site];
        let mu = self.mu_list[site];
        let hz = self.hz_list[site];

        let n_up = self.make_local_op_n_up()?;
        let n_down = self.make_local_op_n_down()?;

        // n_up * n_down
        let n_up_n_down = csr_mul(1.0, &n_up, 1.0, &n_down)?;

        // U * n_up * n_down
        // - mu * (n_up + n_down) = -mu * n_up - mu * n_down
        // + hz * Sz = hz * (n_up - n_down) / 2 = (hz/2) * n_up - (hz/2) * n_down
        // = U * n_up_n_down + (-mu + hz/2) * n_up + (-mu - hz/2) * n_down

        let coeff_up = -mu + 0.5 * hz;
        let coeff_down = -mu - 0.5 * hz;

        let term1 = csr_add(u, &n_up_n_down, coeff_up, &n_up)?;
        csr_add(1.0, &term1, coeff_down, &n_down)
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

    const MATRIX_ZERO_EPS: f64 = 1e-12;

    fn make_model() -> HubbardModel {
        HubbardModel::new(
            HashMap::new(),
            vec![4.0],
            vec![1.0],
            vec![2.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn local_op_c_up() {
        // c_up: |up> -> |vac>, |updown> -> |down>
        // Local basis: 0=|vac>, 1=|up>, 2=|down>, 3=|updown>
        // Matrix (row=final, col=initial):
        //   row 0: c_up |up> = |vac>      => col=1
        //   row 2: c_up |updown> = |down> => col=3
        let m = make_model();
        let c_up = m.make_local_op_c_up();

        assert_eq!(c_up.row_dim, 4);
        assert_eq!(c_up.col_dim, 4);
        assert_eq!(c_up.rows, vec![0, 1, 1, 2, 2]);
        assert_eq!(c_up.cols, vec![1, 3]);
        assert_eq!(c_up.vals.len(), 2);
        assert!((c_up.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((c_up.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_c_down() {
        // c_down: |down> -> |vac>, |updown> -> -|up> (fermion sign)
        // Matrix:
        //   row 0: c_down |down> = |vac>    => col=2, val=1
        //   row 1: c_down |updown> = -|up>  => col=3, val=-1
        let m = make_model();
        let c_down = m.make_local_op_c_down();

        assert_eq!(c_down.row_dim, 4);
        assert_eq!(c_down.col_dim, 4);
        assert_eq!(c_down.rows, vec![0, 1, 2, 2, 2]);
        assert_eq!(c_down.cols, vec![2, 3]);
        assert_eq!(c_down.vals.len(), 2);
        assert!((c_down.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((c_down.vals[1] - (-1.0)).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_c_up_dag() {
        // c_up^dag: |vac> -> |up>, |down> -> |updown>
        // Matrix:
        //   row 1: c_up^dag |vac> = |up>       => col=0
        //   row 3: c_up^dag |down> = |updown>  => col=2
        let m = make_model();
        let c_up_dag = m.make_local_op_c_up_dag().unwrap();

        assert_eq!(c_up_dag.row_dim, 4);
        assert_eq!(c_up_dag.col_dim, 4);
        assert_eq!(c_up_dag.rows, vec![0, 0, 1, 1, 2]);
        assert_eq!(c_up_dag.cols, vec![0, 2]);
        assert_eq!(c_up_dag.vals.len(), 2);
        assert!((c_up_dag.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((c_up_dag.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_c_down_dag() {
        // c_down^dag: |vac> -> |down>, |up> -> -|updown>
        // Matrix:
        //   row 2: c_down^dag |vac> = |down>     => col=0, val=1
        //   row 3: c_down^dag |up> = -|updown>   => col=1, val=-1
        let m = make_model();
        let c_down_dag = m.make_local_op_c_down_dag().unwrap();

        assert_eq!(c_down_dag.row_dim, 4);
        assert_eq!(c_down_dag.col_dim, 4);
        assert_eq!(c_down_dag.rows, vec![0, 0, 0, 1, 2]);
        assert_eq!(c_down_dag.cols, vec![0, 1]);
        assert_eq!(c_down_dag.vals.len(), 2);
        assert!((c_down_dag.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((c_down_dag.vals[1] - (-1.0)).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_n_up() {
        // n_up = c_up^dag c_up: diagonal
        // n_up[0]=0, n_up[1]=1, n_up[2]=0, n_up[3]=1
        let m = make_model();
        let n_up = m.make_local_op_n_up().unwrap();

        assert_eq!(n_up.row_dim, 4);
        assert_eq!(n_up.col_dim, 4);
        assert_eq!(n_up.rows, vec![0, 0, 1, 1, 2]);
        assert_eq!(n_up.cols, vec![1, 3]);
        assert_eq!(n_up.vals.len(), 2);
        assert!((n_up.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((n_up.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_n_down() {
        // n_down = c_down^dag c_down: diagonal
        // n_down[0]=0, n_down[1]=0, n_down[2]=1, n_down[3]=1
        let m = make_model();
        let n_down = m.make_local_op_n_down().unwrap();

        assert_eq!(n_down.row_dim, 4);
        assert_eq!(n_down.col_dim, 4);
        assert_eq!(n_down.rows, vec![0, 0, 0, 1, 2]);
        assert_eq!(n_down.cols, vec![2, 3]);
        assert_eq!(n_down.vals.len(), 2);
        assert!((n_down.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((n_down.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_n() {
        // n = n_up + n_down: diagonal
        // n[0]=0, n[1]=1, n[2]=1, n[3]=2
        let m = make_model();
        let n = m.make_local_op_n().unwrap();

        assert_eq!(n.row_dim, 4);
        assert_eq!(n.col_dim, 4);
        assert_eq!(n.rows, vec![0, 0, 1, 2, 3]);
        assert_eq!(n.cols, vec![1, 2, 3]);
        assert_eq!(n.vals.len(), 3);
        assert!((n.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((n.vals[1] - 1.0).abs() <= MATRIX_ZERO_EPS);
        assert!((n.vals[2] - 2.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sz() {
        // Sz = (n_up - n_down) / 2: diagonal
        // sz[0]=0, sz[1]=+0.5, sz[2]=-0.5, sz[3]=0
        let m = make_model();
        let sz = m.make_local_op_sz().unwrap();

        assert_eq!(sz.row_dim, 4);
        assert_eq!(sz.col_dim, 4);
        assert_eq!(sz.rows, vec![0, 0, 1, 2, 2]);
        assert_eq!(sz.cols, vec![1, 2]);
        assert_eq!(sz.vals.len(), 2);
        assert!((sz.vals[0] - 0.5).abs() <= MATRIX_ZERO_EPS);
        assert!((sz.vals[1] - (-0.5)).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sp() {
        // S+ = c_up^dag c_down: |down> -> |up>
        // Matrix:
        //   row 1: S+ |down> = |up> => col=2, val=1
        let m = make_model();
        let sp = m.make_local_op_sp().unwrap();

        assert_eq!(sp.row_dim, 4);
        assert_eq!(sp.col_dim, 4);
        assert_eq!(sp.rows, vec![0, 0, 1, 1, 1]);
        assert_eq!(sp.cols, vec![2]);
        assert_eq!(sp.vals.len(), 1);
        assert!((sp.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_op_sm() {
        // S- = c_down^dag c_up: |up> -> |down>
        // Matrix:
        //   row 2: S- |up> = |down> => col=1, val=1
        let m = make_model();
        let sm = m.make_local_op_sm().unwrap();

        assert_eq!(sm.row_dim, 4);
        assert_eq!(sm.col_dim, 4);
        assert_eq!(sm.rows, vec![0, 0, 0, 1, 1]);
        assert_eq!(sm.cols, vec![1]);
        assert_eq!(sm.vals.len(), 1);
        assert!((sm.vals[0] - 1.0).abs() <= MATRIX_ZERO_EPS);
    }

    #[test]
    fn local_hamiltonian() {
        // H_i = U n_up n_down - mu (n_up + n_down) + hz Sz
        // With U=4, mu=1, hz=2:
        //   |vac>:    0
        //   |up>:     -mu + hz/2 = -1 + 1 = 0
        //   |down>:   -mu - hz/2 = -1 - 1 = -2
        //   |updown>: U - 2*mu = 4 - 2 = 2
        // Non-zero diagonal: row 2 -> -2, row 3 -> 2
        let m = make_model();
        let h = m.make_local_hamiltonian(0).unwrap();

        assert_eq!(h.row_dim, 4);
        assert_eq!(h.col_dim, 4);
        assert_eq!(h.rows, vec![0, 0, 0, 1, 2]);
        assert_eq!(h.cols, vec![2, 3]);
        assert_eq!(h.vals.len(), 2);
        assert!((h.vals[0] - (-2.0)).abs() <= MATRIX_ZERO_EPS);
        assert!((h.vals[1] - 2.0).abs() <= MATRIX_ZERO_EPS);
    }
}
