use crate::basis::{HilbertBasis, HubbardBasis};
use crate::blas::CsrMatrix;
use crate::hamiltonian::{
    make_hamiltonian, make_intersite_elements, make_onsite_elements, HamiltonianElementGenerator,
    TransitionStateHolder,
};
use crate::model::HubbardModel;
use anyhow::Result;

#[derive(Debug)]
struct HubbardLocalOps {
    pub onsite_ham: Vec<CsrMatrix>, // H_i = U n_up n_down - mu n + hz (n_up - n_down)
    pub n_up: CsrMatrix,
    pub n_down: CsrMatrix,
    pub sz: CsrMatrix,
    pub sp: CsrMatrix,
    pub sm: CsrMatrix,
    pub c_up: CsrMatrix,
    pub c_up_dag: CsrMatrix,
    pub c_down: CsrMatrix,
    pub c_down_dag: CsrMatrix,
}

impl HubbardLocalOps {
    pub fn new(model: &HubbardModel) -> Result<Self> {
        let n = model.num_sites;

        let mut onsite_ham = Vec::with_capacity(n);
        for site in 0..n {
            onsite_ham.push(model.make_local_hamiltonian(site)?);
        }

        // Local operators are site-independent (same 4x4 matrix for all sites)
        let n_up = model.make_local_op_n_up()?;
        let n_down = model.make_local_op_n_down()?;
        let sz = model.make_local_op_sz()?;
        let sp = model.make_local_op_sp()?;
        let sm = model.make_local_op_sm()?;
        let c_up = model.make_local_op_c_up();
        let c_up_dag = model.make_local_op_c_up_dag()?;
        let c_down = model.make_local_op_c_down();
        let c_down_dag = model.make_local_op_c_down_dag()?;

        Ok(Self {
            onsite_ham,
            n_up,
            n_down,
            sz,
            sp,
            sm,
            c_up,
            c_up_dag,
            c_down,
            c_down_dag,
        })
    }
}

pub struct HubbardHamiltonianElementGenerator {
    model: HubbardModel,
    local_ops: HubbardLocalOps,
}

impl HubbardHamiltonianElementGenerator {
    pub fn new(model: HubbardModel) -> Result<Self> {
        let local_ops = HubbardLocalOps::new(&model)?;
        Ok(Self { model, local_ops })
    }

    pub fn model(&self) -> &HubbardModel {
        &self.model
    }
}

impl HamiltonianElementGenerator<HubbardBasis> for HubbardHamiltonianElementGenerator {
    fn make_elements(
        &self,
        basis_state: i128,
        basis: &HubbardBasis,
        holder: &mut TransitionStateHolder,
    ) -> Result<()> {
        holder.vals.clear();

        // Local basis for the current input basis
        for site in 0..self.model.num_sites {
            holder.local_basis[site] = basis.find_local_basis(basis_state, site);
        }

        // =====================================================================
        // Onsite terms: H_i = U n_up n_down - mu n + hz (n_up - n_down)
        // =====================================================================
        for site in 0..self.model.num_sites {
            make_onsite_elements(
                basis_state,
                site,
                &self.local_ops.onsite_ham[site],
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Hopping terms: -t (c_i^dag c_j + c_j^dag c_i) for both spins
        //
        // Note: For fermions on different sites, we need to account for the
        // Jordan-Wigner string (fermion sign from intervening occupied sites).
        // This is handled by computing the parity of occupied sites between i and j.
        // =====================================================================
        for (&(i, j), &t) in self.model.hopping.iter() {
            // Compute fermion sign for hopping between sites i and j
            // Sign = (-1)^(number of electrons between sites min(i,j)+1 and max(i,j)-1)
            let (site_min, site_max) = if i < j { (i, j) } else { (j, i) };

            // Count electrons between site_min and site_max (exclusive)
            let mut parity = 0usize;
            for k in (site_min + 1)..site_max {
                let local_state = holder.local_basis[k];
                // Count electrons: |vac>=0, |up>=1, |down>=1, |updown>=2
                parity += match local_state {
                    0 => 0,
                    1 => 1,
                    2 => 1,
                    3 => 2,
                    _ => 0,
                };
            }
            let sign = if parity % 2 == 0 { 1.0 } else { -1.0 };

            // -t * sign * (c_i_up^dag c_j_up + c_j_up^dag c_i_up)
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.c_up_dag,
                &self.local_ops.c_up,
                -t * sign,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                j,
                i,
                &self.local_ops.c_up_dag,
                &self.local_ops.c_up,
                -t * sign,
                1.0,
                holder,
            );

            // -t * sign * (c_i_down^dag c_j_down + c_j_down^dag c_i_down)
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.c_down_dag,
                &self.local_ops.c_down,
                -t * sign,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                j,
                i,
                &self.local_ops.c_down_dag,
                &self.local_ops.c_down,
                -t * sign,
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Density-density interaction: V_ij n_i n_j
        // =====================================================================
        for (&(i, j), &v) in self.model.density_density.iter() {
            // n_i n_j = (n_up_i + n_down_i)(n_up_j + n_down_j)
            // = n_up_i n_up_j + n_up_i n_down_j + n_down_i n_up_j + n_down_i n_down_j
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_up,
                &self.local_ops.n_up,
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_up,
                &self.local_ops.n_down,
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_down,
                &self.local_ops.n_up,
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_down,
                &self.local_ops.n_down,
                v,
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Exchange z term: Jz Sz_i Sz_j
        // =====================================================================
        for (&(i, j), &jz) in self.model.exchange_z.iter() {
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sz,
                &self.local_ops.sz,
                jz,
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Exchange xy term: (Jxy/2) (S+_i S-_j + S-_i S+_j)
        // =====================================================================
        for (&(i, j), &jxy) in self.model.exchange_xy.iter() {
            let c = 0.5 * jxy;

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sp,
                &self.local_ops.sm,
                c,
                1.0,
                holder,
            );

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sm,
                &self.local_ops.sp,
                c,
                1.0,
                holder,
            );
        }

        Ok(())
    }
}

pub fn make_hubbard_hamiltonian(
    basis: &HubbardBasis,
    model: &HubbardModel,
    num_threads: usize,
) -> Result<CsrMatrix> {
    make_hamiltonian(
        basis,
        &HubbardHamiltonianElementGenerator::new(model.clone())?,
        num_threads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::HilbertBasis;
    use std::collections::HashMap;

    fn make_model(num_sites: usize) -> HubbardModel {
        HubbardModel::new(
            HashMap::new(),
            vec![0.0; num_sites],
            vec![0.0; num_sites],
            vec![0.0; num_sites],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn ham_two_sites_no_interaction() {
        // Two sites, N=2, Sz=0, no hopping, no U
        // All states should have zero energy
        let model = make_model(2);
        let basis = HubbardBasis::new(model.clone(), 2, 0.0).unwrap();

        let h = make_hubbard_hamiltonian(&basis, &model, 1).unwrap();

        assert_eq!(h.row_dim, 4);
        assert_eq!(h.col_dim, 4);

        // All matrix elements should be zero (empty Hamiltonian)
        assert!(h.vals.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn ham_two_sites_onsite_u() {
        let tol = 1e-12;

        // Two sites, U=4 on each site, N=2, Sz=0
        // States: |updown,vac>, |up,down>, |down,up>, |vac,updown>
        // Energies: U, 0, 0, U
        let mut model = make_model(2);
        model.u_list = vec![4.0, 4.0];

        let basis = HubbardBasis::new(model.clone(), 2, 0.0).unwrap();
        let h = make_hubbard_hamiltonian(&basis, &model, 1).unwrap();

        assert_eq!(h.row_dim, 4);

        // basis codes: [3, 6, 9, 12]
        // 3 = |updown,vac>  -> E = U = 4
        // 6 = |down,up>     -> E = 0
        // 9 = |up,down>     -> E = 0
        // 12 = |vac,updown> -> E = U = 4
        let diag: Vec<f64> = (0..4)
            .map(|row| {
                for k in h.rows[row]..h.rows[row + 1] {
                    if h.cols[k] == row {
                        return h.vals[k];
                    }
                }
                0.0
            })
            .collect();

        assert!((diag[0] - 4.0).abs() < tol); // |updown,vac>
        assert!(diag[1].abs() < tol); // |down,up>
        assert!(diag[2].abs() < tol); // |up,down>
        assert!((diag[3] - 4.0).abs() < tol); // |vac,updown>
    }

    #[test]
    fn ham_two_sites_hopping() {
        let tol = 1e-12;

        // Two sites with hopping t=1, N=1, Sz=0.5 (one up electron)
        // Basis: |up,vac>, |vac,up> => codes [1, 4]
        // H = -t (c1^dag c2 + c2^dag c1)
        // Matrix: [[0, -t], [-t, 0]]
        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let model = HubbardModel::new(
            hopping,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let basis = HubbardBasis::new(model.clone(), 1, 0.5).unwrap();
        assert_eq!(basis.dim(), 2);

        let h = make_hubbard_hamiltonian(&basis, &model, 1).unwrap();

        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);

        // Check off-diagonal elements are -1
        // Row 0: should have element at col 1
        // Row 1: should have element at col 0
        let mut found_01 = false;
        let mut found_10 = false;

        for row in 0..2 {
            for k in h.rows[row]..h.rows[row + 1] {
                let col = h.cols[k];
                let val = h.vals[k];
                if row == 0 && col == 1 {
                    assert!((val + 1.0).abs() < tol);
                    found_01 = true;
                }
                if row == 1 && col == 0 {
                    assert!((val + 1.0).abs() < tol);
                    found_10 = true;
                }
            }
        }

        assert!(found_01 && found_10);
        assert!(h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_hubbard_dimer_half_filling() {
        let tol = 1e-12;

        // Hubbard dimer: 2 sites, t=1, U=2, N=2, Sz=0
        // This is a well-known exactly solvable case
        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let model = HubbardModel::new(
            hopping,
            vec![2.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let basis = HubbardBasis::new(model.clone(), 2, 0.0).unwrap();
        assert_eq!(basis.dim(), 4);

        let h = make_hubbard_hamiltonian(&basis, &model, 1).unwrap();

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());

        // The Hamiltonian should be:
        // Basis: |updown,vac>=3, |down,up>=6, |up,down>=9, |vac,updown>=12
        // |updown,vac> and |vac,updown> have energy U=2 and hop to |up,down>, |down,up>
        // Matrix structure (after ordering):
        // [U, -t, -t, 0]
        // [-t, 0, 0, -t]
        // [-t, 0, 0, -t]
        // [0, -t, -t, U]
    }
}
