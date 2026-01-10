use crate::basis::HeisenbergBasis;
use crate::blas::CsrMatrix;
use crate::hamiltonian::{
    make_hamiltonian, make_intersite_elements, make_onsite_elements, HamiltonianElementGenerator,
    TransitionStateHolder,
};
use crate::model::HeisenbergModel;
use anyhow::Result;

#[derive(Debug)]
struct HeisenbergLocalOps {
    pub onsite_ham: Vec<CsrMatrix>, // H_i = hz_i Sz_i + d_i Sz_i^2
    pub sz: Vec<CsrMatrix>,
    pub sp: Vec<CsrMatrix>,
    pub sm: Vec<CsrMatrix>,
}

impl HeisenbergLocalOps {
    pub fn new(model: &HeisenbergModel) -> Result<Self> {
        let n = model.num_sites;

        let mut sz = Vec::with_capacity(n);
        let mut sp = Vec::with_capacity(n);
        let mut sm = Vec::with_capacity(n);
        let mut onsite_ham = Vec::with_capacity(n);

        for site in 0..n {
            let sz_i = model.make_local_op_sz(site)?;
            let sp_i = model.make_local_op_sp(site)?;
            let sm_i = model.make_local_op_sm(site)?; // transpose(sp)

            let on_i = model.make_local_hamiltonian(site)?; // hz*Sz + d*Sz^2

            sz.push(sz_i);
            sp.push(sp_i);
            sm.push(sm_i);
            onsite_ham.push(on_i);
        }

        Ok(Self {
            onsite_ham,
            sz,
            sp,
            sm,
        })
    }
}

pub struct HeisenbergHamiltonianElementGenerator {
    model: HeisenbergModel,
    local_ops: HeisenbergLocalOps,
}

impl HeisenbergHamiltonianElementGenerator {
    pub fn new(model: HeisenbergModel) -> Result<Self> {
        let local_ops = HeisenbergLocalOps::new(&model)?;
        Ok(Self { model, local_ops })
    }

    pub fn model(&self) -> &HeisenbergModel {
        &self.model
    }
}

impl HamiltonianElementGenerator<HeisenbergBasis> for HeisenbergHamiltonianElementGenerator {
    fn make_elements(
        &self,
        basis_state: i128,
        basis: &HeisenbergBasis,
        holder: &mut TransitionStateHolder,
    ) -> Result<()> {
        holder.vals.clear();

        // Local basis for the current input basis
        for site in 0..self.model.num_sites {
            holder.local_basis[site] = basis.find_local_basis(basis_state, site);
        }

        // Onsite terms: H_i = hz_i * Sz_i + d_i * Sz_i^2
        for site in 0..self.model.num_sites {
            make_onsite_elements(
                basis_state,
                site,
                &self.local_ops.onsite_ham[site],
                1.0,
                holder,
            );
        }

        // Intersite terms: Jz * Sz_i Sz_j
        for (&(i, j), &val) in self.model.exchange_z.iter() {
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sz[i],
                &self.local_ops.sz[j],
                val,
                1.0,
                holder,
            );
        }

        // Intersite terms: (Jxy/2) * (Sp_i Sm_j + Sm_i Sp_j)
        for (&(i, j), &val) in self.model.exchange_xy.iter() {
            let c = 0.5 * val;

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sp[i],
                &self.local_ops.sm[j],
                c,
                1.0,
                holder,
            );

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.sm[i],
                &self.local_ops.sp[j],
                c,
                1.0,
                holder,
            );
        }

        Ok(())
    }
}

pub fn make_heisenberg_hamiltonian(
    basis: &HeisenbergBasis,
    model: &HeisenbergModel,
    num_threads: usize,
) -> Result<CsrMatrix> {
    make_hamiltonian(
        basis,
        &HeisenbergHamiltonianElementGenerator::new(model.clone())?,
        num_threads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn ham_two_spin_half_two_sites_sz0_with_onsite() {
        let tol = 1e-12;

        // Two sites, S=1/2 each, total Sz = 0 sector
        //
        // Basis order (sorted):
        // |↓↑>, |↑↓>
        //
        // Parameters:
        //   Jz = 1
        //   Jxy = 1
        //   hz = [2, 2]
        //   d  = [3, 3]
        //
        // Onsite contribution:
        //   Sz total = 0  -> hz term cancels
        //   Sz^2 total = 1/4 + 1/4 = 1/2
        //   onsite = d * 1/2 = 1.5
        //
        // Intersite:
        //   Sz1 Sz2 = -1/4
        //   flip-flop = 1/2
        //
        // Total matrix:
        // [ 1.5 - 0.25,  0.5 ]
        // [ 0.5,  1.5 - 0.25 ]
        // =
        // [ 1.25, 0.5 ]
        // [ 0.5,  1.25 ]

        let mut exchange_xy = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);

        let mut exchange_z = HashMap::new();
        exchange_z.insert((0, 1), 1.0);

        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![2.0, 2.0],
            d_list: vec![3.0, 3.0],
            exchange_xy,
            exchange_z,
        };

        let basis = HeisenbergBasis::new(model.clone(), 0.0).unwrap();

        let h = make_heisenberg_hamiltonian(&basis, &model, 2).unwrap();

        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert_eq!(h.rows, vec![0, 2, 4]);

        // CSR is sorted by column index
        assert_eq!(h.cols, vec![0, 1, 0, 1]);

        assert!((h.vals[0] - 1.25).abs() <= tol);
        assert!((h.vals[1] - 0.5).abs() <= tol);
        assert!((h.vals[2] - 0.5).abs() <= tol);
        assert!((h.vals[3] - 1.25).abs() <= tol);

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_one_site_onsite_only_s1_total_sz_plus_one() {
        let tol = 1e-12;

        // One site, S=1, total Sz = +1 sector
        //
        // H = hz*Sz + d*Sz^2
        // hz = 2, d = 3
        //
        // Sz = +1, Sz^2 = 1
        // -> H = 2*1 + 3*1 = 5

        let model = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![2.0],
            d_list: vec![3.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        let basis = HeisenbergBasis::new(model.clone(), 1.0).unwrap();

        let h = make_heisenberg_hamiltonian(&basis, &model, 2).unwrap();

        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        assert_eq!(h.rows, vec![0, 1]);
        assert_eq!(h.cols, vec![0]);

        assert!((h.vals[0] - 5.0).abs() <= tol);

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());
    }
}
