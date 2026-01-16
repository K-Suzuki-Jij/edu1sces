use crate::basis::Basis;
use crate::blas::CsrMatrix;
use crate::hamiltonian::{
    make_hamiltonian, make_intersite_elements, make_onsite_elements, HamiltonianElementGenerator,
    TransitionStateHolder,
};
use crate::model::KondoLatticeModel;
use anyhow::Result;

#[derive(Debug)]
struct KondoLatticeLocalOps {
    // Onsite Hamiltonian (site-dependent)
    pub onsite_ham: Vec<CsrMatrix>,

    // Conduction electron operators (site-dependent due to varying local spin S)
    pub c_up: Vec<CsrMatrix>,
    pub c_up_dag: Vec<CsrMatrix>,
    pub c_down: Vec<CsrMatrix>,
    pub c_down_dag: Vec<CsrMatrix>,
    pub n_up: Vec<CsrMatrix>,
    pub n_down: Vec<CsrMatrix>,

    // Localized spin operators
    pub l_sz: Vec<CsrMatrix>,
    pub l_sp: Vec<CsrMatrix>,
    pub l_sm: Vec<CsrMatrix>,
}

impl KondoLatticeLocalOps {
    pub fn new(model: &KondoLatticeModel) -> Result<Self> {
        let n = model.num_sites;

        let mut onsite_ham = Vec::with_capacity(n);

        let mut c_up = Vec::with_capacity(n);
        let mut c_up_dag = Vec::with_capacity(n);
        let mut c_down = Vec::with_capacity(n);
        let mut c_down_dag = Vec::with_capacity(n);
        let mut n_up = Vec::with_capacity(n);
        let mut n_down = Vec::with_capacity(n);

        let mut l_sz = Vec::with_capacity(n);
        let mut l_sp = Vec::with_capacity(n);
        let mut l_sm = Vec::with_capacity(n);

        for site in 0..n {
            onsite_ham.push(model.make_local_hamiltonian(site)?);

            c_up.push(model.make_local_op_c_up(site)?);
            c_up_dag.push(model.make_local_op_c_up_dag(site)?);
            c_down.push(model.make_local_op_c_down(site)?);
            c_down_dag.push(model.make_local_op_c_down_dag(site)?);
            n_up.push(model.make_local_op_n_up(site)?);
            n_down.push(model.make_local_op_n_down(site)?);

            l_sz.push(model.make_local_op_l_sz(site)?);
            l_sp.push(model.make_local_op_l_sp(site)?);
            l_sm.push(model.make_local_op_l_sm(site)?);
        }

        Ok(Self {
            onsite_ham,
            c_up,
            c_up_dag,
            c_down,
            c_down_dag,
            n_up,
            n_down,
            l_sz,
            l_sp,
            l_sm,
        })
    }
}

pub struct KondoLatticeHamiltonianElementGenerator {
    model: KondoLatticeModel,
    local_ops: KondoLatticeLocalOps,
}

impl KondoLatticeHamiltonianElementGenerator {
    pub fn new(model: KondoLatticeModel) -> Result<Self> {
        let local_ops = KondoLatticeLocalOps::new(&model)?;
        Ok(Self { model, local_ops })
    }

    pub fn model(&self) -> &KondoLatticeModel {
        &self.model
    }
}

impl HamiltonianElementGenerator for KondoLatticeHamiltonianElementGenerator {
    fn make_elements(
        &self,
        basis_state: i128,
        basis: &Basis,
        holder: &mut TransitionStateHolder,
    ) -> Result<()> {
        holder.vals.clear();

        // Local basis for the current input basis
        for site in 0..self.model.num_sites {
            holder.local_basis[site] = basis.find_local_basis(basis_state, site);
        }

        // =====================================================================
        // Onsite terms (includes U, mu, hz_c, hz_f, d, Kondo z, Kondo xy)
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
        // =====================================================================
        for (&(i, j), &t) in self.model.hopping.iter() {
            // Compute fermion sign for hopping between sites i and j
            let (site_min, site_max) = if i < j { (i, j) } else { (j, i) };

            // Count electrons between site_min and site_max (exclusive)
            let mut parity = 0usize;
            for k in (site_min + 1)..site_max {
                let local_state = holder.local_basis[k];
                let e = local_state % 4;
                parity += match e {
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
                &self.local_ops.c_up_dag[i],
                &self.local_ops.c_up[j],
                -t * sign,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                j,
                i,
                &self.local_ops.c_up_dag[j],
                &self.local_ops.c_up[i],
                -t * sign,
                1.0,
                holder,
            );

            // -t * sign * (c_i_down^dag c_j_down + c_j_down^dag c_i_down)
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.c_down_dag[i],
                &self.local_ops.c_down[j],
                -t * sign,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                j,
                i,
                &self.local_ops.c_down_dag[j],
                &self.local_ops.c_down[i],
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
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_up[i],
                &self.local_ops.n_up[j],
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_up[i],
                &self.local_ops.n_down[j],
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_down[i],
                &self.local_ops.n_up[j],
                v,
                1.0,
                holder,
            );
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.n_down[i],
                &self.local_ops.n_down[j],
                v,
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Localized spin exchange (z): J_z S_i^z S_j^z
        // =====================================================================
        for (&(i, j), &jz) in self.model.ff_exchange_z.iter() {
            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.l_sz[i],
                &self.local_ops.l_sz[j],
                jz,
                1.0,
                holder,
            );
        }

        // =====================================================================
        // Localized spin exchange (xy): (J_xy/2) (S_i^+ S_j^- + S_i^- S_j^+)
        // =====================================================================
        for (&(i, j), &jxy) in self.model.ff_exchange_xy.iter() {
            let c = 0.5 * jxy;

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.l_sp[i],
                &self.local_ops.l_sm[j],
                c,
                1.0,
                holder,
            );

            make_intersite_elements(
                basis_state,
                i,
                j,
                &self.local_ops.l_sm[i],
                &self.local_ops.l_sp[j],
                c,
                1.0,
                holder,
            );
        }

        Ok(())
    }
}

pub fn make_kondo_lattice_hamiltonian(
    basis: &Basis,
    model: &KondoLatticeModel,
    num_threads: usize,
) -> Result<CsrMatrix> {
    make_hamiltonian(
        basis,
        &KondoLatticeHamiltonianElementGenerator::new(model.clone())?,
        num_threads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QuantumModel;
    use std::collections::HashMap;

    fn make_model(num_sites: usize, two_s: i32) -> KondoLatticeModel {
        KondoLatticeModel {
            num_sites,
            two_s_list: vec![two_s; num_sites],
            hopping: HashMap::new(),
            u_list: vec![0.0; num_sites],
            mu_list: vec![0.0; num_sites],
            hz_c_list: vec![0.0; num_sites],
            hz_f_list: vec![0.0; num_sites],
            d_list: vec![0.0; num_sites],
            density_density: HashMap::new(),
            kondo_xy_list: vec![0.0; num_sites],
            kondo_z_list: vec![0.0; num_sites],
            ff_exchange_xy: HashMap::new(),
            ff_exchange_z: HashMap::new(),
        }
    }

    #[test]
    fn ham_one_site_no_interaction() {
        // 1 site, S=1/2, N=1, Sz=0 (one electron)
        // No interactions -> all energies should be zero
        let model = make_model(1, 1);
        let basis = model.build_basis(&[1, 0]).unwrap(); // N=1, 2*Sz=0

        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();

        assert_eq!(h.row_dim, 2); // |+1/2>|down> and |-1/2>|up>
        assert!(h.vals.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn ham_one_site_u_term() {
        let tol = 1e-12;

        // 1 site, S=1/2, U=4, N=2, any Sz
        // Only states with |updown> have energy U
        let mut model = make_model(1, 1);
        model.u_list = vec![4.0];

        // N=2, Sz=0.5: |+1/2>|updown>
        let basis = model.build_basis(&[2, 1]).unwrap();
        assert_eq!(basis.dim(), 1);

        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert_eq!(h.row_dim, 1);
        assert!((h.vals[0] - 4.0).abs() < tol);
    }

    #[test]
    fn ham_one_site_mu_term() {
        let tol = 1e-12;

        // 1 site, S=1/2, mu=2, N=1
        // Energy = -mu * n = -2 * 1 = -2
        let mut model = make_model(1, 1);
        model.mu_list = vec![2.0];

        let basis = model.build_basis(&[1, 0]).unwrap(); // N=1, 2*Sz=0
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();

        for &v in &h.vals {
            if v.abs() > tol {
                assert!((v + 2.0).abs() < tol);
            }
        }
    }

    #[test]
    fn ham_one_site_hz_c_term() {
        let tol = 1e-12;

        // 1 site, S=1/2, hz_c=1
        // Energy = hz_c * (n_up - n_down)
        // |up>: E = 1, |down>: E = -1
        let mut model = make_model(1, 1);
        model.hz_c_list = vec![1.0];

        // N=1, 2*Sz=2: |+1/2>|up>
        let basis = model.build_basis(&[1, 2]).unwrap();
        assert_eq!(basis.dim(), 1);
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] - 1.0).abs() < tol);

        // N=1, 2*Sz=-2: |-1/2>|down>
        let basis = model.build_basis(&[1, -2]).unwrap();
        assert_eq!(basis.dim(), 1);
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] + 1.0).abs() < tol);
    }

    #[test]
    fn ham_one_site_hz_f_term() {
        let tol = 1e-12;

        // 1 site, S=1/2, hz_f=2, N=0
        // Energy = hz_f * m_loc
        // |+1/2>: E = 1, |-1/2>: E = -1
        let mut model = make_model(1, 1);
        model.hz_f_list = vec![2.0];

        // N=0, 2*Sz=1: |+1/2>|vac>
        let basis = model.build_basis(&[0, 1]).unwrap();
        assert_eq!(basis.dim(), 1);
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] - 1.0).abs() < tol);

        // N=0, 2*Sz=-1: |-1/2>|vac>
        let basis = model.build_basis(&[0, -1]).unwrap();
        assert_eq!(basis.dim(), 1);
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] + 1.0).abs() < tol);
    }

    #[test]
    fn ham_one_site_d_term() {
        let tol = 1e-12;

        // 1 site, S=1, d=2, N=0
        // Energy = d * m^2
        // |+1>: E = 2, |0>: E = 0, |-1>: E = 2
        let mut model = make_model(1, 2); // S=1
        model.d_list = vec![2.0];

        // N=0, 2*Sz=2: |+1>|vac>
        let basis = model.build_basis(&[0, 2]).unwrap();
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] - 2.0).abs() < tol);

        // N=0, 2*Sz=0: |0>|vac>
        let basis = model.build_basis(&[0, 0]).unwrap();
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!(h.vals.is_empty() || h.vals.iter().all(|&v| v.abs() < tol));

        // N=0, 2*Sz=-2: |-1>|vac>
        let basis = model.build_basis(&[0, -2]).unwrap();
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] - 2.0).abs() < tol);
    }

    #[test]
    fn ham_one_site_kondo_z() {
        let tol = 1e-12;

        // 1 site, S=1/2, Kz=2, N=1
        // Energy = Kz * m_loc * sz_cond
        // |+1/2>|up>: E = 2 * 0.5 * 0.5 = 0.5
        // |+1/2>|down>: E = 2 * 0.5 * (-0.5) = -0.5
        // |-1/2>|up>: E = 2 * (-0.5) * 0.5 = -0.5
        // |-1/2>|down>: E = 2 * (-0.5) * (-0.5) = 0.5
        let mut model = make_model(1, 1);
        model.kondo_z_list = vec![2.0];

        // N=1, 2*Sz=2: |+1/2>|up>
        let basis = model.build_basis(&[1, 2]).unwrap();
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        assert!((h.vals[0] - 0.5).abs() < tol);

        // N=1, 2*Sz=0: |+1/2>|down> and |-1/2>|up>
        let basis = model.build_basis(&[1, 0]).unwrap();
        assert_eq!(basis.dim(), 2);
        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();
        // Both should have E = -0.5
        for row in 0..2 {
            for k in h.rows[row]..h.rows[row + 1] {
                if h.cols[k] == row {
                    assert!((h.vals[k] + 0.5).abs() < tol);
                }
            }
        }
    }

    #[test]
    fn ham_one_site_kondo_xy() {
        let tol = 1e-12;

        // 1 site, S=1/2, Kxy=2, N=1, 2*Sz=0
        // Basis: |+1/2>|down> (state 2), |-1/2>|up> (state 5)
        //
        // Kondo xy term: (1/2) Kxy (S^+ s^- + S^- s^+)
        // S^+ s^- : |+1/2>|down> <- |-1/2>|up>
        // S^- s^+ : |-1/2>|up> <- |+1/2>|down>
        //
        // For S=1/2:
        //   S^+ |m=-1/2> = sqrt(1/2*3/2 - (-1/2)*1/2) |m=+1/2> = sqrt(3/4+1/4) = 1
        //   S^- |m=+1/2> = sqrt(1/2*3/2 - 1/2*(-1/2)) |m=-1/2> = 1
        //
        // Matrix element = (1/2) * Kxy * 1 = 1
        let mut model = make_model(1, 1);
        model.kondo_xy_list = vec![2.0];

        let basis = model.build_basis(&[1, 0]).unwrap();
        assert_eq!(basis.dim(), 2);

        // Verify basis states: should be state 2 and state 5
        assert_eq!(basis.basis, vec![2, 5]);

        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();

        // Check off-diagonal elements
        let mut found_01 = false;
        let mut found_10 = false;

        for row in 0..2 {
            for k in h.rows[row]..h.rows[row + 1] {
                let col = h.cols[k];
                let val = h.vals[k];
                if row == 0 && col == 1 {
                    assert!((val - 1.0).abs() < tol);
                    found_01 = true;
                }
                if row == 1 && col == 0 {
                    assert!((val - 1.0).abs() < tol);
                    found_10 = true;
                }
            }
        }

        assert!(found_01 && found_10);
        assert!(h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_two_sites_hopping() {
        let tol = 1e-12;

        // 2 sites, S=1/2, t=1, N=1, 2*Sz=0 (one electron)
        // Hopping: -t (c_1^dag c_2 + h.c.)
        let mut model = make_model(2, 1);
        model.hopping.insert((0, 1), 1.0);

        // N=1, 2*Sz=0: total Sz = 0
        // Possibilities:
        //   m0 + m1 + sz_e = 0
        // With one up electron (sz_e = 1/2): m0 + m1 = -1/2
        //   -> (m0,m1) = (+1/2,-1) invalid, (-1/2,0) invalid for S=1/2
        // With one down electron (sz_e = -1/2): m0 + m1 = +1/2
        //   -> (m0,m1) = (+1/2,0) invalid, (0,+1/2) invalid for S=1/2
        // Actually for S=1/2: m = +1/2 or -1/2 only
        // So for 2*Sz=0 (Sz=0):
        //   up electron at site 0: (+1/2,-1/2,up,vac) or (-1/2,+1/2,up,vac) -> 2*Sz = 1-1+1 = 1 or -1+1+1=1 (not 0)
        // Let me recalculate: 2*(m0 + m1 + sz_e) = 0
        //   with up electron: 2*sz_e = 1, so 2*(m0+m1) = -1 -> m0+m1 = -1/2
        //     (m0,m1) = (+1/2,-1) or (-1/2,0) - invalid for S=1/2
        //   with down electron: 2*sz_e = -1, so 2*(m0+m1) = 1 -> m0+m1 = +1/2
        //     (m0,m1) = (+1/2,0) or (0,+1/2) - invalid for S=1/2
        // So 2*Sz=0 is not achievable with N=1 for two S=1/2 sites
        // Let's use N=2, 2*Sz=0 instead
        let basis = model.build_basis(&[2, 0]).unwrap();

        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_two_sites_ff_exchange() {
        let tol = 1e-12;

        // 2 sites, S=1/2, J=1, N=0, 2*Sz=0
        // Heisenberg-like exchange between localized spins
        // Basis: |+1/2,-1/2>|vac,vac>, |-1/2,+1/2>|vac,vac>
        let mut model = make_model(2, 1);
        model.ff_exchange_xy.insert((0, 1), 1.0);
        model.ff_exchange_z.insert((0, 1), 1.0);

        let basis = model.build_basis(&[0, 0]).unwrap();
        assert_eq!(basis.dim(), 2);

        let h = make_kondo_lattice_hamiltonian(&basis, &model, 1).unwrap();

        // This should be the same as a 2-site S=1/2 Heisenberg model in Sz=0 sector
        // H = J (Sz1 Sz2 + 0.5 (S+1 S-2 + S-1 S+2))
        // Diagonal: Sz1 Sz2 = (+1/2)(-1/2) = -1/4
        // Off-diagonal: 0.5 * 1 = 0.5
        assert_eq!(h.rows, vec![0, 2, 4]);
        assert_eq!(h.cols, vec![0, 1, 0, 1]);
        assert!((h.vals[0] + 0.25).abs() < tol); // diagonal
        assert!((h.vals[1] - 0.5).abs() < tol); // off-diagonal
        assert!((h.vals[2] - 0.5).abs() < tol); // off-diagonal
        assert!((h.vals[3] + 0.25).abs() < tol); // diagonal

        assert!(h.is_symmetric(tol).unwrap());
    }
}
